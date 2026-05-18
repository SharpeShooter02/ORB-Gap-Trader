"""
signals/pre_market.py — Pre-market qualification job (two phases).

PHASE 1 (~9:20am ET, before market open):
  For each symbol in live_cfg.symbols:
    1. Get today's reference price (pre-market quote or Alpaca fallback).
    2. Compute gap using daily bars up to yesterday.
    3. Apply gap filter (per-instrument or default).
    4. Apply day-of-week exclusion.
    5. Apply direction filter.
    6. Check prior session filter.
    7. Record to state_store.gap_scan and ps_filter_result.
  Returns list[Phase1Result] of symbols passing all phase-1 gates.

PHASE 2 (~10:01am ET, after ORB closes):
  For each phase-1 candidate:
    1. Compute opening range from first 30 minutes of intraday bars.
    2. Compute RTG value and percentile rank.
    3. Apply RTG gap exclusion.
    4. Apply RTG pair routing.
    5. Run pre-flight liquidity checks.
    6. Record to state_store.candidates.
  Returns list[Phase2Result] — only is_candidate=True symbols should be watched.

CRITICAL CONSTRAINT:
  Which instrument fires is unknowable at 9:30am.  Phase 1 produces a
  candidate list; phase 2 produces the final list.  Only at 10:01am does
  the executor know which symbols are live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Optional, TYPE_CHECKING

import pandas as pd

from orb_live.signals.strategy_signals import (
    compute_gap,
    check_prior_session_filter,
    compute_opening_range,
)
from orb_live.signals.rtg import RtgHistoryStore, compute_rtg_val, decide_rtg_targets
from orb_live.signals.routing import PairRouter, SIZE_MULT
from orb_live.signals.liquidity import PreFlightCheck, CandidateDecision

if TYPE_CHECKING:
    from orb_live.config.live_config import LiveConfig
    from orb_live.core.state_store import StateStore
    from orb_live.data.alpaca_client import AlpacaClient
    from orb_live.data.underlying_data import UnderlyingDataStore


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Phase1Result:
    symbol: str
    gap_abs: float
    gap_direction: int
    prior_close: float
    ps_filter_passed: bool
    ps_filter_warning: bool = False


@dataclass
class Phase2Result:
    symbol: str
    gap_abs: float
    gap_direction: int          # +1 gap-up / -1 gap-down
    prior_close: float
    first_open: Optional[float] # close of the 9:30 bar — gap reference price;
                                # also the initial EMA seed input for the session runner
    orb: Optional[dict]         # keys: high, low, midpoint, size_pct, n_bars, ema
    rtg_val: Optional[float]
    rtg_pct: Optional[float]
    tp1_mult: float             # pass as tp1_mult_override to compute_entry
    tp2_mult: float             # pass as tp2_mult_override to compute_entry
    rtg_excluded: bool
    routing_action: str         # "normal" | "double" | "skip"
    size_mult: float            # 1.0 (normal), 2.0 (double), 0.0 (skip)
    preflight: Optional[CandidateDecision]
    is_candidate: bool          # True iff symbol should be watched by session executor
    exclusion_reason: str = ""


# ── Warning-capturing logger shim ─────────────────────────────────────────────

class _WarnCapture:
    """Captures whether check_prior_session_filter emitted a data warning."""
    def __init__(self):
        self.warned = False

    def warning(self, event, **kw):  # noqa: ANN
        self.warned = True


# ── PreMarketJob ──────────────────────────────────────────────────────────────

class PreMarketJob:
    """
    Orchestrates pre-market qualification for one trading session.

    Instantiate fresh each session.
    """

    def __init__(
        self,
        live_cfg: "LiveConfig",
        store: "StateStore",
        client: "AlpacaClient",
        underlying_store: "UnderlyingDataStore",
        logger=None,
    ):
        self._cfg    = live_cfg
        self._store  = store
        self._client = client
        self._ul     = underlying_store
        self._log    = logger

        self._rtg_store = RtgHistoryStore(store)
        self._router    = PairRouter(live_cfg.strategy_config)
        self._preflight = PreFlightCheck(live_cfg, store, client, logger)

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def run_phase1(
        self,
        trade_date: date,
        ref_prices: Optional[dict[str, float]] = None,
        daily_bars: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[Phase1Result]:
        """
        Gap scan and prior-session filter for all symbols.

        ref_prices  — {symbol: float} pre-market reference prices.  If absent
                      for a symbol, falls back to Alpaca latest quote mid.
        daily_bars  — {symbol: DataFrame} daily bars for gap computation.
                      If absent for a symbol, fetched from Alpaca (lookback=10d).

        Writes to state_store.gap_scan and state_store.ps_filter_result.
        Returns Phase1Result list for symbols that pass all gates.
        """
        cfg   = self._cfg
        s_cfg = cfg.strategy_config

        underlying_data = self._load_underlying_data(trade_date)
        results: list[Phase1Result] = []

        for symbol in cfg.symbols:
            effective_gap_filter = s_cfg.instrument_gap_filters.get(
                symbol, s_cfg.gap_filter_pct
            )

            # 1. Reference price.
            ref_price = self._get_ref_price(symbol, ref_prices)
            if ref_price is None or ref_price <= 0:
                self._store.save_gap_scan(
                    trade_date, symbol,
                    qualifies=False, filter_reason="no_ref_price",
                )
                continue

            # 2. Daily bars.
            d_bars = self._get_daily_bars(symbol, daily_bars)
            if d_bars is None or d_bars.empty:
                self._store.save_gap_scan(
                    trade_date, symbol,
                    qualifies=False, filter_reason="no_daily_bars",
                )
                continue

            # 3. Gap.
            gap_result = compute_gap(trade_date, d_bars, ref_price)
            if gap_result is None:
                self._store.save_gap_scan(
                    trade_date, symbol,
                    qualifies=False, filter_reason="no_prior_close",
                )
                continue

            gap_abs, gap_direction, prior_close = gap_result

            # 4. Gap size filter.
            if gap_abs < effective_gap_filter:
                self._store.save_gap_scan(
                    trade_date, symbol,
                    prev_close=prior_close, open_price=ref_price,
                    gap_pct=gap_abs, gap_dir=gap_direction,
                    qualifies=False, filter_reason="gap_too_small",
                )
                continue

            # 5. Day-of-week exclusion.
            excl_dow = s_cfg.day_of_week_exclusions.get(symbol)
            if excl_dow and trade_date.weekday() in excl_dow:
                self._store.save_gap_scan(
                    trade_date, symbol,
                    prev_close=prior_close, open_price=ref_price,
                    gap_pct=gap_abs, gap_dir=gap_direction,
                    qualifies=False, filter_reason="dow_excluded",
                )
                continue

            # 6. Direction filter.
            allowed_dir = s_cfg.direction_filters.get(symbol)
            if allowed_dir is not None and gap_direction != allowed_dir:
                self._store.save_gap_scan(
                    trade_date, symbol,
                    prev_close=prior_close, open_price=ref_price,
                    gap_pct=gap_abs, gap_dir=gap_direction,
                    qualifies=False, filter_reason="direction_filtered",
                )
                continue

            # Gap qualifies — record before PS filter.
            self._store.save_gap_scan(
                trade_date, symbol,
                prev_close=prior_close, open_price=ref_price,
                gap_pct=gap_abs, gap_dir=gap_direction,
                qualifies=True,
            )

            # 7. Prior session filter.
            # Pass cfg (LiveConfig) not s_cfg so sigma_override.yaml thresholds apply.
            warn_cap = _WarnCapture()
            ps_passed = check_prior_session_filter(
                symbol, trade_date, gap_direction,
                cfg, underlying_data,
                logger=warn_cap,
            )

            # Resolve the underlying symbol for the DB record.
            ps_spec = cfg.prior_session_filters.get(symbol)
            ul_sym = ps_spec[0] if ps_spec else None

            self._store.save_ps_filter(
                trade_date, symbol,
                underlying=ul_sym,
                passed=ps_passed,
            )

            if not ps_passed:
                continue

            results.append(Phase1Result(
                symbol=symbol,
                gap_abs=gap_abs,
                gap_direction=gap_direction,
                prior_close=prior_close,
                ps_filter_passed=True,
                ps_filter_warning=warn_cap.warned,
            ))

        return results

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def run_phase2(
        self,
        trade_date: date,
        phase1_results: list[Phase1Result],
        intraday_bars: Optional[dict[str, pd.DataFrame]] = None,
        current_equity: float = 100_000.0,
    ) -> list[Phase2Result]:
        """
        Post-ORB RTG computation, pair routing, and pre-flight checks.

        intraday_bars — {symbol: DataFrame with DatetimeIndex} covering 9:30-10:00.
                        If absent for a symbol, fetched from Alpaca.

        Returns Phase2Result list; only results with is_candidate=True should
        be watched by the session executor.
        Writes RTG history and candidates to state_store.
        """
        cfg   = self._cfg
        s_cfg = cfg.strategy_config

        # First pass: ORB + RTG for all phase-1 candidates.
        pre_decisions: dict[str, dict] = {}

        for p1 in phase1_results:
            symbol = p1.symbol
            bars   = self._get_intraday_bars(symbol, trade_date, intraday_bars)

            if bars is None or bars.empty:
                pre_decisions[symbol] = {
                    "qualifies": False, "reason": "no_intraday_bars",
                }
                continue

            orb = compute_opening_range(bars, s_cfg, symbol=symbol)
            if orb is None:
                pre_decisions[symbol] = {
                    "qualifies": False, "reason": "orb_invalid",
                }
                continue

            first_open = float(bars.iloc[0]["close"])
            rtg_val = compute_rtg_val(orb, p1.gap_abs, first_open)

            rtg_pct = None
            if rtg_val is not None:
                rtg_pct = self._rtg_store.compute_rtg_pct(
                    symbol, trade_date, rtg_val, s_cfg, gap_abs=p1.gap_abs,
                )

            tp1_mult, tp2_mult, excluded = decide_rtg_targets(
                symbol, p1.gap_abs, rtg_pct, s_cfg,
            )

            pre_decisions[symbol] = {
                "qualifies":     not excluded,
                "rtg_val":       rtg_val,
                "rtg_pct":       rtg_pct,
                "tp1_mult":      tp1_mult,
                "tp2_mult":      tp2_mult,
                "excluded":      excluded,
                "orb":           orb,
                "first_open":    first_open,
                "gap_abs":       p1.gap_abs,
                "gap_direction": p1.gap_direction,
            }

            # Persist RTG for future sessions.  Must happen after ORB is known
            # and only for days with a valid RTG value (no lookahead).
            if rtg_val is not None:
                self._rtg_store.update_history(
                    symbol, trade_date, rtg_val,
                    gap_abs=p1.gap_abs, gap_direction=p1.gap_direction,
                )

        # Routing decisions — must see all pre_decisions to compare pairs.
        routing = self._router.decide(trade_date, pre_decisions)

        # Second pass: pre-flight + final candidate list.
        results: list[Phase2Result] = []

        for p1 in phase1_results:
            symbol = p1.symbol
            pd_    = pre_decisions.get(symbol, {})

            orb          = pd_.get("orb")
            first_open   = pd_.get("first_open")   # 9:30 bar close; None if ORB failed
            rtg_val      = pd_.get("rtg_val")
            rtg_pct      = pd_.get("rtg_pct")
            tp1_mult     = pd_.get("tp1_mult", s_cfg.tp1_target_multiple)
            tp2_mult     = pd_.get("tp2_mult", s_cfg.tp2_target_multiple)
            rtg_excluded = pd_.get("excluded", False)
            qualifies    = pd_.get("qualifies", False)
            no_orb_rsn   = pd_.get("reason", "")

            routing_action = routing.get(symbol, "normal")
            size_mult      = SIZE_MULT[routing_action]

            # Resolve intended_direction from direction_filters explicitly.
            # Phase 1 already excludes wrong-direction gaps, but we must record
            # and pass the authoritative direction (not just the gap direction)
            # so the short/HTB check in PreFlightCheck.check() is always correct.
            intended_dir = s_cfg.direction_filters.get(symbol) or p1.gap_direction

            # Routing losers and RTG-excluded symbols skip pre-flight.
            if routing_action == "skip":
                res = Phase2Result(
                    symbol=symbol, gap_abs=p1.gap_abs,
                    gap_direction=p1.gap_direction, prior_close=p1.prior_close,
                    first_open=first_open,
                    orb=orb, rtg_val=rtg_val, rtg_pct=rtg_pct,
                    tp1_mult=tp1_mult, tp2_mult=tp2_mult,
                    rtg_excluded=rtg_excluded, routing_action=routing_action,
                    size_mult=size_mult, preflight=None,
                    is_candidate=False, exclusion_reason="routing_skip",
                )
                self._record_candidate(trade_date, p1, orb, rtg_val, rtg_pct,
                                       routing_action, preflight=None,
                                       decision="routing_skip",
                                       intended_direction=intended_dir)
                results.append(res)
                continue

            if not qualifies:
                reason = "rtg_excluded" if rtg_excluded else (no_orb_rsn or "orb_invalid")
                res = Phase2Result(
                    symbol=symbol, gap_abs=p1.gap_abs,
                    gap_direction=p1.gap_direction, prior_close=p1.prior_close,
                    first_open=first_open,
                    orb=orb, rtg_val=rtg_val, rtg_pct=rtg_pct,
                    tp1_mult=tp1_mult, tp2_mult=tp2_mult,
                    rtg_excluded=rtg_excluded, routing_action=routing_action,
                    size_mult=size_mult, preflight=None,
                    is_candidate=False, exclusion_reason=reason,
                )
                self._record_candidate(trade_date, p1, orb, rtg_val, rtg_pct,
                                       routing_action, preflight=None,
                                       decision=reason,
                                       intended_direction=intended_dir)
                results.append(res)
                continue

            # Pre-flight check uses intended_dir (from direction_filters) so
            # the short/HTB gate is evaluated against the correct trade side.
            preflight = self._preflight.check(
                symbol, trade_date, intended_dir, current_equity,
            )

            decision_str = "candidate" if preflight.passed else preflight.reason
            res = Phase2Result(
                symbol=symbol, gap_abs=p1.gap_abs,
                gap_direction=p1.gap_direction, prior_close=p1.prior_close,
                first_open=first_open,
                orb=orb, rtg_val=rtg_val, rtg_pct=rtg_pct,
                tp1_mult=tp1_mult, tp2_mult=tp2_mult,
                rtg_excluded=rtg_excluded, routing_action=routing_action,
                size_mult=size_mult, preflight=preflight,
                is_candidate=preflight.passed,
                exclusion_reason="" if preflight.passed else preflight.reason,
            )
            self._record_candidate(trade_date, p1, orb, rtg_val, rtg_pct,
                                   routing_action, preflight=preflight,
                                   decision=decision_str,
                                   intended_direction=intended_dir)
            results.append(res)

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_underlying_data(self, trade_date: date) -> dict[str, pd.DataFrame]:
        """Load all PS-filter underlying DataFrames from the parquet store."""
        cfg  = self._cfg
        data: dict[str, pd.DataFrame] = {}
        seen: set[str] = set()

        for spec in cfg.prior_session_filters.values():
            if spec is None:
                continue
            ul_sym = spec[0]
            if ul_sym in seen:
                continue
            seen.add(ul_sym)
            try:
                df = self._ul.get(ul_sym)
                if not df.empty:
                    data[ul_sym] = df
                else:
                    warn = self._ul.warn_if_stale(ul_sym, trade_date)
                    if self._log and warn:
                        self._log.warning("underlying_stale",
                                          ul_sym=ul_sym, msg=warn)
            except Exception as exc:
                if self._log:
                    self._log.warning("underlying_load_error",
                                      ul_sym=ul_sym, exc=str(exc))

        return data

    def _get_ref_price(
        self,
        symbol: str,
        ref_prices: Optional[dict[str, float]],
    ) -> Optional[float]:
        if ref_prices:
            p = ref_prices.get(symbol)
            if p and float(p) > 0:
                return float(p)
        try:
            q = self._client.get_latest_quote(symbol)
            bid, ask = float(q.get("bid", 0)), float(q.get("ask", 0))
            if ask > 0:
                return (bid + ask) / 2.0
        except Exception:
            pass
        return None

    def _get_daily_bars(
        self,
        symbol: str,
        daily_bars: Optional[dict[str, pd.DataFrame]],
    ) -> Optional[pd.DataFrame]:
        if daily_bars and symbol in daily_bars:
            return daily_bars[symbol]
        try:
            return self._client.get_daily_bars(symbol, lookback_days=10)
        except Exception:
            return None

    def _get_intraday_bars(
        self,
        symbol: str,
        trade_date: date,
        intraday_bars: Optional[dict[str, pd.DataFrame]],
    ) -> Optional[pd.DataFrame]:
        """
        Return intraday bars with a DatetimeIndex (required by compute_opening_range).

        If fetching from Alpaca, sets index from the 'timestamp' column.
        """
        if intraday_bars and symbol in intraday_bars:
            return intraday_bars[symbol]
        try:
            import pytz
            et    = pytz.timezone("America/New_York")
            start = et.localize(datetime.combine(trade_date, dtime(9, 30)))
            end   = et.localize(datetime.combine(trade_date, dtime(10, 5)))
            df = self._client.get_intraday_bars(symbol, start, end, timeframe="1Min")
            if df.empty:
                return df
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            return df
        except Exception:
            return None

    def _record_candidate(
        self,
        trade_date: date,
        p1: Phase1Result,
        orb: Optional[dict],
        rtg_val: Optional[float],
        rtg_pct: Optional[float],
        routing_action: str,
        preflight: Optional[CandidateDecision],
        decision: str,
        intended_direction: Optional[int] = None,
    ) -> None:
        self._store.save_candidate(
            session_date=trade_date,
            symbol=p1.symbol,
            phase=2,
            gap_abs=p1.gap_abs,
            gap_direction=p1.gap_direction,
            prior_close=p1.prior_close,
            ps_filter_passed=p1.ps_filter_passed,
            ps_filter_warning=p1.ps_filter_warning,
            preflight_passed=preflight.passed if preflight else None,
            preflight_reason=preflight.reason if preflight else None,
            decision=decision,
            intended_direction=intended_direction if intended_direction is not None
                               else p1.gap_direction,
        )
