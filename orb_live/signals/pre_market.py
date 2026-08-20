"""
signals/pre_market.py — Pre-market qualification job (two phases).

PHASE 1 (~09:31 ET, after the 9:30 bar closes):
  1. For each symbol in the v1 universe, fetch the 9:30 bar close as the
     gap reference price.
  2. Compute the ETF overnight gap → convert to UL-equivalent gap
     (÷ leverage × direction_sign).
  3. Apply |UL gap| ≥ 2% threshold and direction filter.
  4. Apply PS filter (k=1.00 σ) using UL daily data.
  5. Call plan_session() to get the final candidate list + multipliers
     (includes skip-cheap-top-2 pruning, regime, and cap_factor).
  Returns list[Phase1Result] (symbols in plan.candidates).
  Stores the SessionPlan internally for Phase 2 to read.

PHASE 2 (~10:01 ET, after ORB window closes):
  For each Phase 1 candidate:
    1. Compute opening range from the 9:30-10:00 intraday bars.
    2. Run pre-flight liquidity check.
    3. Set size_mult = plan.multipliers[sym]; tp1_mult from config
       (tp1_target_multiple, 2.0 = 2× ORB in v1), tp2_mult=0.0.
  Returns list[Phase2Result] — only is_candidate=True symbols are watched.

CRITICAL CONSTRAINT: every input to plan_session() is causal at 9:30 ET.
  No fired-trade counts, no intraday data, no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Optional, TYPE_CHECKING

import pandas as pd

from orb_live.signals.strategy_signals import compute_opening_range
from orb_live.signals.gap_scan import scan_gaps
from orb_live.signals.liquidity import PreFlightCheck, CandidateDecision
from orb_live.strategy.session_fixture import (
    serialise_raw_inputs,
    write_fixture as write_session_fixture,
)
from orb_live.strategy.v1_strategy import (
    plan_session,
    SessionPlan,
    classify,
)
if TYPE_CHECKING:
    from orb_live.config.live_config import LiveConfig
    from orb_live.core.state_store import StateStore
    from orb_live.data.broker_client import BrokerClient
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
    gap_direction: int
    prior_close: float
    first_open: Optional[float]
    orb: Optional[dict]
    tp1_mult: float              # v1: config tp1_target_multiple (2.0 = 2× ORB)
    tp2_mult: float              # always 0.0 in v1 (TP1-only)
    rtg_val: Optional[float]     # always None in v1 (kept for interface compat)
    rtg_pct: Optional[float]     # always None in v1
    rtg_excluded: bool           # always False in v1
    routing_action: str          # always "normal" in v1
    size_mult: float             # v1 multiplier from SessionPlan
    preflight: Optional[CandidateDecision]
    is_candidate: bool
    exclusion_reason: str = ""


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
        client: "BrokerClient",
        underlying_store: "UnderlyingDataStore",
        logger=None,
    ):
        self._cfg    = live_cfg
        self._store  = store
        self._client = client
        self._ul     = underlying_store
        self._log    = logger
        self._preflight = PreFlightCheck(live_cfg, store, client, logger)

        # Set by run_phase1; consumed by run_phase2
        self._session_plan: Optional[SessionPlan] = None

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def run_phase1(
        self,
        trade_date: date,
        ref_prices: Optional[dict[str, float]] = None,
        daily_bars: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[Phase1Result]:
        """
        Gap scan, PS filter, and plan_session() at 09:31 ET.

        ref_prices  — {symbol: float} pre-fetched reference prices.  If absent,
                      fetched via get_intraday_bars (9:30 bar close).
        daily_bars  — {symbol: DataFrame} daily bars.  If absent, fetched from broker.

        Calls plan_session() internally and stores the result in self._session_plan.
        Returns Phase1Result list for the symbols in plan.candidates.
        """
        cfg         = self._cfg
        instruments = cfg.instruments   # dict[str, Instrument] from v1

        underlying_data = self._load_underlying_data(trade_date)

        # ── Gather per-ETF market data ─────────────────────────────────────────
        # I/O only. Everything that decides anything lives in scan_gaps(), so
        # that the backtest can replay this session from the raw inputs rather
        # than from the UL gaps we derive from them.
        fetched_refs:  dict[str, Optional[float]] = {}
        fetched_daily: dict[str, Optional[pd.DataFrame]] = {}
        for symbol in cfg.symbols:
            if symbol not in instruments:
                continue
            fetched_refs[symbol]  = self._get_ref_price(symbol, ref_prices, trade_date)
            fetched_daily[symbol] = self._get_daily_bars(symbol, daily_bars)

        scan_result = scan_gaps(
            trade_date,
            symbols=list(cfg.symbols),
            instruments=instruments,
            ref_prices=fetched_refs,
            etf_daily=fetched_daily,
            ul_daily=underlying_data,
            config=cfg,
        )

        overnight_gaps   = scan_result.overnight_gaps
        prior_two_closes = scan_result.prior_two_closes
        prior_etf_close  = scan_result.prior_etf_close

        # ── Persist and log what the scan decided ─────────────────────────────
        preliminary_p1: list[Phase1Result] = []

        for scan in scan_result.scans:
            if scan.prior_close is None:
                # Never got as far as a gap — nothing but the reason to record.
                self._store.save_gap_scan(
                    trade_date, scan.symbol,
                    qualifies=False, filter_reason=scan.filter_reason,
                )
                continue

            # The gap scan row records gap qualification only; the PS filter
            # gets its own row below, as it did when this was one loop.
            gap_qualified = scan.filter_reason not in ("gap_too_small", "direction_filtered")
            self._store.save_gap_scan(
                trade_date, scan.symbol,
                prev_close=scan.prior_close, open_price=scan.ref_price,
                gap_pct=scan.etf_gap_abs, gap_dir=scan.gap_direction,
                qualifies=gap_qualified,
                **({} if gap_qualified else {"filter_reason": scan.filter_reason}),
            )
            if not gap_qualified:
                continue

            self._store.save_ps_filter(
                trade_date, scan.symbol,
                underlying=scan.ps_underlying,
                ul_move_pct=scan.ul_move_pct,
                threshold_pct=scan.threshold_pct,
                passed=scan.ps_passed,
            )

            if self._log:
                self._log.info(
                    "ps_filter",
                    symbol=scan.symbol, ul=scan.ps_underlying,
                    ul_move=round(scan.ul_move_pct, 4) if scan.ul_move_pct is not None else None,
                    threshold=round(scan.threshold_pct, 4) if scan.threshold_pct is not None else None,
                    passed=scan.ps_passed,
                )

            if not scan.qualifies:
                continue

            preliminary_p1.append(Phase1Result(
                symbol=scan.symbol,
                gap_abs=scan.etf_gap_abs,
                gap_direction=scan.gap_direction,
                prior_close=scan.prior_close,
                ps_filter_passed=True,
                ps_filter_warning=scan.ps_warned,
            ))

        # ── Call plan_session() to get the definitive candidate list ───────────
        # plan_session() re-runs gap + PS + direction + skip-cheap-top-2 using
        # the UL-level inputs we just assembled.  Its candidate set == what we
        # expect (modulo the ETF-gap approximation of UL gap).
        qualified_symbols = [r.symbol for r in preliminary_p1]
        self._session_plan = plan_session(
            universe=qualified_symbols,
            instruments=instruments,
            sigmas=cfg.sigmas,
            overnight_gaps=overnight_gaps,
            prior_two_closes=prior_two_closes,
            prior_etf_close=prior_etf_close,
        )

        # Filter preliminary list to plan candidates (plan already applied
        # skip-cheap and zero-weight drops).
        plan_set = set(self._session_plan.candidates)
        results  = [r for r in preliminary_p1 if r.symbol in plan_set]

        # Golden-session fixture: the backtest replays these exact inputs
        # through the same plan_session() and asserts an identical plan.
        # Never allowed to break the session — write_fixture swallows errors.
        write_session_fixture(
            trade_date,
            universe=qualified_symbols,
            instruments=instruments,
            sigmas=cfg.sigmas,
            overnight_gaps=overnight_gaps,
            prior_two_closes=prior_two_closes,
            prior_etf_close=prior_etf_close,
            plan=self._session_plan,
            raw=serialise_raw_inputs(
                symbols=list(cfg.symbols),
                ref_prices=fetched_refs,
                etf_daily=fetched_daily,
                ul_daily=underlying_data,
                prior_session_filters=cfg.prior_session_filters,
            ),
            log=self._log,
        )

        if self._log:
            for _ul, _gap in sorted(overnight_gaps.items()):
                _ptc  = prior_two_closes.get(_ul)
                _c_t1 = round(_ptc[0], 4) if _ptc else None
                _c_t2 = round(_ptc[1], 4) if _ptc else None
                _ul_df = underlying_data.get(_ul)
                _d_t1 = _d_t2 = None
                if _ul_df is not None:
                    _pr = _ul_df[_ul_df["date"] < pd.Timestamp(trade_date)].tail(2)
                    if len(_pr) >= 2:
                        _d_t1 = str(_pr.iloc[-1]["date"].date())
                        _d_t2 = str(_pr.iloc[-2]["date"].date())
                self._log.info(
                    "ul_overnight_gap",
                    ul=_ul, ul_gap=round(_gap, 4),
                    c_t1=_c_t1, d_t1=_d_t1,
                    c_t2=_c_t2, d_t2=_d_t2,
                )
            self._log.info(
                "plan_session_complete",
                n_candidates=len(self._session_plan.candidates),
                regime=self._session_plan.regime,
                n_uls=self._session_plan.n_uls,
                cap_factor=round(self._session_plan.cap_factor, 4),
                overnight_uls=sorted(overnight_gaps.keys()),
            )

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
        Post-ORB: compute opening range and pre-flight for each Phase 1 candidate.

        size_mult for each symbol is taken from self._session_plan.multipliers,
        so ORB sizing is v1-based (weight × cap_factor).
        """
        cfg   = self._cfg
        scfg  = cfg.strategy_config
        plan  = self._session_plan

        results: list[Phase2Result] = []

        for p1 in phase1_results:
            symbol = p1.symbol

            bars = self._get_intraday_bars(symbol, trade_date, intraday_bars)
            if bars is None or bars.empty:
                results.append(self._make_p2(
                    p1, orb=None, first_open=None, size_mult=0.0,
                    preflight=None, is_candidate=False,
                    exclusion_reason="no_intraday_bars",
                ))
                continue

            orb = compute_opening_range(bars, scfg, symbol=symbol)
            if orb is None:
                results.append(self._make_p2(
                    p1, orb=None, first_open=None, size_mult=0.0,
                    preflight=None, is_candidate=False,
                    exclusion_reason="orb_invalid",
                ))
                continue

            first_open = float(bars.iloc[0]["close"])

            # v1 multiplier — already includes cap_factor, regime, and class weight.
            size_mult = plan.multipliers.get(symbol, 0.0) if plan else 0.0
            if size_mult == 0.0:
                results.append(self._make_p2(
                    p1, orb=orb, first_open=first_open, size_mult=0.0,
                    preflight=None, is_candidate=False,
                    exclusion_reason="zero_multiplier",
                ))
                continue

            intended_dir = cfg.direction_filters.get(symbol) or p1.gap_direction
            preflight    = self._preflight.check(
                symbol, trade_date, intended_dir, current_equity,
            )

            is_candidate   = preflight.passed
            exclusion_rsn  = "" if preflight.passed else preflight.reason

            self._store.save_candidate(
                session_date=trade_date,
                symbol=symbol, phase=2,
                gap_abs=p1.gap_abs,
                gap_direction=p1.gap_direction,
                prior_close=p1.prior_close,
                ps_filter_passed=p1.ps_filter_passed,
                ps_filter_warning=p1.ps_filter_warning,
                preflight_passed=preflight.passed,
                preflight_reason=preflight.reason,
                decision="candidate" if preflight.passed else preflight.reason,
                intended_direction=intended_dir,
            )

            results.append(self._make_p2(
                p1, orb=orb, first_open=first_open, size_mult=size_mult,
                preflight=preflight, is_candidate=is_candidate,
                exclusion_reason=exclusion_rsn,
            ))

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _tp1_mult_for(self, symbol: str) -> float:
        """TP1 target multiple for `symbol`: per-class map (C1→2×, C2/C3→1×) if
        the class is present, else the scalar tp1_target_multiple fallback."""
        scfg = self._cfg.strategy_config
        by_class = getattr(scfg, "tp1_target_multiple_by_class", None) or {}
        return by_class.get(classify(symbol), scfg.tp1_target_multiple)

    def _make_p2(
        self,
        p1: Phase1Result,
        orb: Optional[dict],
        first_open: Optional[float],
        size_mult: float,
        preflight: Optional[CandidateDecision],
        is_candidate: bool,
        exclusion_reason: str = "",
    ) -> Phase2Result:
        return Phase2Result(
            symbol=p1.symbol,
            gap_abs=p1.gap_abs,
            gap_direction=p1.gap_direction,
            prior_close=p1.prior_close,
            first_open=first_open,
            orb=orb,
            # TP1 target multiple, per candidate. Class map (C1→2×, C2/C3→1×)
            # takes precedence over the scalar tp1_target_multiple.
            # Must NOT be hardcoded: it is passed to compute_entry as an override
            # and so silently beats the config if wrong (was pinned to 1.0, which
            # cut every winner to 1× ORB — half the intended target).
            tp1_mult=self._tp1_mult_for(p1.symbol),
            tp2_mult=0.0,   # v1 is TP1-only (exit_ratio_tp2=0); TP2 leg unused
            rtg_val=None,
            rtg_pct=None,
            rtg_excluded=False,
            routing_action="normal",
            size_mult=size_mult,
            preflight=preflight,
            is_candidate=is_candidate,
            exclusion_reason=exclusion_reason,
        )

    def _load_underlying_data(self, trade_date: date) -> dict[str, pd.DataFrame]:
        """Load UL DataFrames from parquet store for all underlyings in universe."""
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
                        self._log.warning("underlying_stale", ul_sym=ul_sym, msg=warn)
            except Exception as exc:
                if self._log:
                    self._log.warning("underlying_load_error", ul_sym=ul_sym, exc=str(exc))

        return data

    def _get_ref_price(
        self,
        symbol: str,
        ref_prices: Optional[dict[str, float]],
        trade_date: date,
    ) -> Optional[float]:
        if ref_prices:
            p = ref_prices.get(symbol)
            if p and float(p) > 0:
                return float(p)
        try:
            from zoneinfo import ZoneInfo
            _et = ZoneInfo("America/New_York")
            start = datetime.combine(trade_date, dtime(9, 30)).replace(tzinfo=_et)
            end   = datetime.combine(trade_date, dtime(9, 32)).replace(tzinfo=_et)
            df = self._client.get_intraday_bars(symbol, start, end, timeframe="1Min")
            if df.empty:
                return None
            return float(df.iloc[0]["close"])
        except Exception:
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
