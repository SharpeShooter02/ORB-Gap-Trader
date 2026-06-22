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
    3. Set size_mult = plan.multipliers[sym], tp1_mult=1.0, tp2_mult=0.0.
  Returns list[Phase2Result] — only is_candidate=True symbols are watched.

CRITICAL CONSTRAINT: every input to plan_session() is causal at 9:30 ET.
  No fired-trade counts, no intraday data, no look-ahead.
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
from orb_live.signals.liquidity import PreFlightCheck, CandidateDecision
from orb_live.strategy.v1_strategy import (
    plan_session,
    SessionPlan,
    GAP_THRESHOLD,
)
from orb_live.core.calendar import is_trading_day

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
    tp1_mult: float              # always 1.0 in v1
    tp2_mult: float              # always 0.0 in v1
    rtg_val: Optional[float]     # always None in v1 (kept for interface compat)
    rtg_pct: Optional[float]     # always None in v1
    rtg_excluded: bool           # always False in v1
    routing_action: str          # always "normal" in v1
    size_mult: float             # v1 multiplier from SessionPlan
    preflight: Optional[CandidateDecision]
    is_candidate: bool
    exclusion_reason: str = ""


# ── Warning-capturing logger shim ─────────────────────────────────────────────

class _WarnCapture:
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
        sigmas      = cfg.sigmas        # dict[UL, float]

        underlying_data = self._load_underlying_data(trade_date)

        # ── Gather per-ETF market data ─────────────────────────────────────────
        # We accumulate inputs for plan_session() as we scan each ETF.
        overnight_gaps:    dict[str, float]                  = {}  # UL → gap
        prior_two_closes:  dict[str, tuple[float, float]]    = {}  # UL → (c_t-1, c_t-2)
        prior_etf_close:   dict[str, float]                  = {}  # sym → last ETF close

        preliminary_p1: list[Phase1Result] = []

        for symbol in cfg.symbols:
            inst = instruments.get(symbol)
            if inst is None:
                continue

            effective_gap_filter = inst.leverage * GAP_THRESHOLD  # e.g. 0.04 for 2x

            # 1. Reference price (9:30 bar close).
            ref_price = self._get_ref_price(symbol, ref_prices, trade_date)
            if ref_price is None or ref_price <= 0:
                self._store.save_gap_scan(
                    trade_date, symbol, qualifies=False, filter_reason="no_ref_price"
                )
                continue

            # 2. Daily bars (for prior ETF close).
            d_bars = self._get_daily_bars(symbol, daily_bars)
            if d_bars is None or d_bars.empty:
                self._store.save_gap_scan(
                    trade_date, symbol, qualifies=False, filter_reason="no_daily_bars"
                )
                continue

            # 3. ETF gap and prior close.
            gap_result = compute_gap(trade_date, d_bars, ref_price)
            if gap_result is None:
                self._store.save_gap_scan(
                    trade_date, symbol, qualifies=False, filter_reason="no_prior_close"
                )
                continue

            etf_gap_abs, gap_direction, prior_close = gap_result
            prior_etf_close[symbol] = prior_close

            # 4. Convert ETF gap to UL-equivalent gap.
            #    Non-inverse: ETF_gap =  UL_gap × leverage → UL_gap =  ETF_signed / leverage
            #    Inverse:     ETF_gap = -UL_gap × leverage → UL_gap = -ETF_signed / leverage
            etf_gap_signed = etf_gap_abs * gap_direction
            ul_gap = (-etf_gap_signed if inst.inverse else etf_gap_signed) / inst.leverage

            # 5. ETF-level gap size check (same as |ul_gap| ≥ GAP_THRESHOLD).
            if etf_gap_abs < effective_gap_filter:
                self._store.save_gap_scan(
                    trade_date, symbol,
                    prev_close=prior_close, open_price=ref_price,
                    gap_pct=etf_gap_abs, gap_dir=gap_direction,
                    qualifies=False, filter_reason="gap_too_small",
                )
                continue

            # Direction filter (LABU/LABD).
            allowed_dir = cfg.direction_filters.get(symbol)
            if allowed_dir is not None and gap_direction != allowed_dir:
                self._store.save_gap_scan(
                    trade_date, symbol,
                    prev_close=prior_close, open_price=ref_price,
                    gap_pct=etf_gap_abs, gap_dir=gap_direction,
                    qualifies=False, filter_reason="direction_filtered",
                )
                continue

            # Gap qualifies — record before PS filter.
            self._store.save_gap_scan(
                trade_date, symbol,
                prev_close=prior_close, open_price=ref_price,
                gap_pct=etf_gap_abs, gap_dir=gap_direction,
                qualifies=True,
            )

            # 6. Prior-session filter (uses UL data, matches check_prior_session_filter).
            warn_cap   = _WarnCapture()
            ps_passed  = check_prior_session_filter(
                symbol, trade_date, gap_direction,
                cfg, underlying_data,
                logger=warn_cap,
            )
            ps_spec       = cfg.prior_session_filters.get(symbol)
            ul_sym_for_db = ps_spec[0] if ps_spec else None

            # Compute PS metrics for the state store (mirrors check_prior_session_filter).
            ul_move_pct   = None
            threshold_pct = float(ps_spec[1]) if ps_spec else None
            if ul_sym_for_db:
                _ul_df = underlying_data.get(ul_sym_for_db)
                if _ul_df is not None:
                    _pr = _ul_df[_ul_df["date"] < pd.Timestamp(trade_date)].tail(2)
                    if len(_pr) >= 2:
                        _c1, _c2 = float(_pr.iloc[-1]["close"]), float(_pr.iloc[-2]["close"])
                        if _c2 > 0:
                            _ps_raw = (_c1 - _c2) / _c2
                            _is_inv = len(ps_spec) == 3 and ps_spec[2] is True
                            _eff    = -gap_direction if _is_inv else gap_direction
                            ul_move_pct = _ps_raw * _eff

            self._store.save_ps_filter(
                trade_date, symbol,
                underlying=ul_sym_for_db,
                ul_move_pct=ul_move_pct,
                threshold_pct=threshold_pct,
                passed=ps_passed,
            )

            if self._log:
                self._log.info(
                    "ps_filter",
                    symbol=symbol, ul=ul_sym_for_db,
                    ul_move=round(ul_move_pct, 4) if ul_move_pct is not None else None,
                    threshold=round(threshold_pct, 4) if threshold_pct is not None else None,
                    passed=ps_passed,
                )

            if not ps_passed:
                continue

            # Accumulate plan_session() inputs.
            ul = inst.underlying
            # UL gap for plan_session() — causal overnight gap on the underlying.
            # We use our ETF-derived approximation.
            if ul not in overnight_gaps:
                overnight_gaps[ul] = ul_gap
            else:
                # Keep the most extreme gap if multiple ETFs share a UL.
                if abs(ul_gap) > abs(overnight_gaps[ul]):
                    overnight_gaps[ul] = ul_gap

            # Prior two UL closes from underlying_data store.
            if ul not in prior_two_closes:
                ul_df = underlying_data.get(ul)
                if ul_df is not None:
                    prior_rows = ul_df[ul_df["date"] < pd.Timestamp(trade_date)].tail(2)
                    if len(prior_rows) >= 2:
                        c_t1 = float(prior_rows.iloc[-1]["close"])
                        c_t2 = float(prior_rows.iloc[-2]["close"])
                        prior_two_closes[ul] = (c_t1, c_t2)

            preliminary_p1.append(Phase1Result(
                symbol=symbol,
                gap_abs=etf_gap_abs,
                gap_direction=gap_direction,
                prior_close=prior_close,
                ps_filter_passed=True,
                ps_filter_warning=warn_cap.warned,
            ))

        # ── Call plan_session() to get the definitive candidate list ───────────
        # plan_session() re-runs gap + PS + direction + skip-cheap-top-2 using
        # the UL-level inputs we just assembled.  Its candidate set == what we
        # expect (modulo the ETF-gap approximation of UL gap).
        self._session_plan = plan_session(
            universe=cfg.symbols,
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
            tp1_mult=1.0,
            tp2_mult=0.0,
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
                    # Filter to NYSE trading days so crypto weekend/holiday bars
                    # do not contaminate the PS-filter prior-two-closes window.
                    # Aligns live behaviour with the backtester, which sources UL
                    # data from equity-calendar feeds with no weekend rows.
                    df = df[df["date"].dt.date.apply(is_trading_day)].copy()
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
