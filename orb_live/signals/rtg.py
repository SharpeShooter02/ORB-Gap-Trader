"""
signals/rtg.py — RTG (ORB-range-to-gap-dollar ratio) history and decisions.

RTG = (orb_high - orb_low) / (gap_abs * first_open)
    — knowable at 10:00am before entry.

High RTG = wide ORB relative to gap = "spring loaded fully".
Low RTG  = narrow ORB relative to gap = fast momentum trade.

PERCENTILE RANK:
    Rolling 252-session window of prior RTG values.
    Uses scipy.stats.percentileofscore if available, else manual count.
    Only applied when gap_abs < config.rtg_scale_gap_cap (default 0.12).
    Requires >= config.rtg_min_history (default 60) prior values.

SCALING (when config.use_rtg_scaling is True):
    Below rtg_scale_threshold (60th pct): standard tp1/tp2 targets.
    Above threshold: linear compression toward rtg_tp1_min / rtg_tp2_min.

EXCLUSION (when config.rtg_gap_exclusion is True):
    Skip days where RTG pct >= threshold AND gap_abs < exclusion_threshold.
    Only applies to rtg_gap_exclusion_symbols (or all if empty tuple).

All logic is identical to backtester lines 1828-1870.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from orb_live.core.state_store import StateStore


# ── Raw RTG computation ───────────────────────────────────────────────────────

def compute_rtg_val(orb: dict, gap_abs: float, first_open: float) -> Optional[float]:
    """
    Compute raw RTG = (orb_high - orb_low) / (gap_abs * first_open).

    Returns None if gap_dollar is zero or non-positive.
    Identical to backtester line 1840.
    """
    if gap_abs <= 0 or first_open <= 0:
        return None
    gap_dollar = gap_abs * first_open
    if gap_dollar <= 0:
        return None
    return (orb["high"] - orb["low"]) / gap_dollar


# ── RTG target decision ───────────────────────────────────────────────────────

def decide_rtg_targets(
    symbol: str,
    gap_abs: float,
    rtg_pct: Optional[float],
    config,                    # StrategyConfig
) -> tuple[float, float, bool]:
    """
    Given RTG percentile rank, decide TP multipliers and exclusion flag.

    Returns (tp1_mult_used, tp2_mult_used, excluded).

    Identical to backtester lines 1834-1870.
    """
    tp1_mult_used = config.tp1_target_multiple
    tp2_mult_used = config.tp2_target_multiple
    excluded = False

    if rtg_pct is None:
        return tp1_mult_used, tp2_mult_used, excluded

    thr = config.rtg_scale_threshold

    # Continuous scaling: compress TP targets for high-RTG trades.
    if config.use_rtg_scaling and rtg_pct >= thr:
        _sp = (rtg_pct - thr) / (1.0 - thr)
        tp1_mult_used = (config.rtg_tp1_max
                         + _sp * (config.rtg_tp1_min - config.rtg_tp1_max))
        tp2_mult_used = (config.rtg_tp2_max
                         + _sp * (config.rtg_tp2_min - config.rtg_tp2_max))

    # Exclusion: skip high-RTG low-gap days.
    if (config.rtg_gap_exclusion
            and rtg_pct >= thr
            and gap_abs < config.rtg_gap_exclusion_threshold
            and (len(config.rtg_gap_exclusion_symbols) == 0
                 or symbol in config.rtg_gap_exclusion_symbols)):
        excluded = True

    return tp1_mult_used, tp2_mult_used, excluded


# ── Persistent RTG history ────────────────────────────────────────────────────

class RtgHistoryStore:
    """
    Persists RTG values via the state_store rtg_history table.

    All reads exclude the current session (before_date strict) to prevent
    lookahead bias — identical to the backtest 252-window pre-scan.

    Usage:
        store = RtgHistoryStore(state_store)

        # After ORB is known (10:00am), persist today's value:
        store.update_history(symbol, today, rtg_val, gap_abs, gap_direction)

        # Before entry decision, compute percentile rank:
        pct = store.compute_rtg_pct(symbol, today, rtg_val, strategy_config, gap_abs)
    """

    _WINDOW = 252

    def __init__(self, state_store: "StateStore"):
        self._store = state_store

    def update_history(
        self,
        symbol: str,
        bar_date: date,
        rtg_val: float,
        gap_abs: Optional[float] = None,
        gap_direction: Optional[int] = None,
    ) -> None:
        """Persist today's RTG after the ORB is known. Call once per session."""
        self._store.save_rtg(
            symbol, bar_date, rtg_val,
            gap_abs=gap_abs, gap_direction=gap_direction,
        )

    def get_history(self, symbol: str, before_date: date) -> list[float]:
        """Return up to 252 RTG values strictly before before_date (no lookahead)."""
        return self._store.get_rtg_history(symbol, before_date, window=self._WINDOW)

    def compute_rtg_pct(
        self,
        symbol: str,
        today: date,
        rtg_val: float,
        config,                    # StrategyConfig
        gap_abs: Optional[float] = None,
    ) -> Optional[float]:
        """
        Compute percentile rank of rtg_val in the rolling prior history.

        Returns None if:
          - History has fewer than config.rtg_min_history entries.
          - gap_abs is provided and >= config.rtg_scale_gap_cap
            (large gaps are exempt from RTG adjustment).

        Identical logic to backtester lines 1843-1851.
        """
        if gap_abs is not None and gap_abs >= config.rtg_scale_gap_cap:
            return None

        hist = self.get_history(symbol, before_date=today)
        if len(hist) < config.rtg_min_history:
            return None

        try:
            from scipy import stats as _sc
            return _sc.percentileofscore(hist, rtg_val, kind="weak") / 100.0
        except ImportError:
            return sum(1 for h in hist if h <= rtg_val) / len(hist)
