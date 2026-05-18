"""
execution/indicators.py — Rolling per-symbol indicators for the live session.

ONLY ONE EMA IS IMPLEMENTED: the full-session midpoint EMA used for TP3
EMA-crossback decisions.  This is EMA #2 in the system:

    Input:  (high + low) / 2 MIDPOINT of every session bar from 9:30
    Seed:   midpoint of the 9:30 bar
    Update: ema_t = mid_t * k + ema_{t-1} * (1 - k)
            k = 2 / (ema_length + 1)

This is NOT orb["ema"] (EMA #1, frozen close-EMA over the ORB window used
for breakout detection).  Do not confuse them.  TestIndicatorParity in
test_signal_parity.py proves they are numerically distinct.

PARITY CONTRACT (verified by test_indicator_parity.py):
  seed_from_orb_bars produces the SAME EMA value that the backtest would
  hold at the end of the last ORB bar (orb_backtester.py:1882-1891 applied
  to only the ORB bars).  For each subsequent bar, on_bar produces the same
  value as iterating the backtest's ema_map forward by one step.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from orb_live.core.state_store import StateStore


class RollingIndicators:
    """
    Maintains the full-session midpoint EMA for one symbol.

    Lifecycle (called by the session runner):
      1. Instantiate at 10:00 ET (after ORB closes).
      2. Call seed_from_orb_bars(orb_bars) ONCE with the 9:30-10:00 bars.
      3. For every subsequent 1-min bar, call on_bar(bar_dict).
      4. Read self.ema for the current EMA value.
      5. Optionally call persist_to() to snapshot EMA to the DB.

    The runner must call on_bar BEFORE passing the bar to
    position_manager.on_bar, so the position manager always reads the
    already-updated EMA value.
    """

    def __init__(self, symbol: str, config):
        """
        config — StrategyConfig or duck-typed object with `ema_length` field.
        """
        self._symbol  = symbol
        self._k       = 2.0 / (config.ema_length + 1)
        self._ema: Optional[float] = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def ema(self) -> Optional[float]:
        """Current EMA value; None until seed_from_orb_bars has been called."""
        return self._ema

    @property
    def is_seeded(self) -> bool:
        return self._ema is not None

    # ── Seeding (called once at 10:00 ET) ──────────────────────────────────────

    def seed_from_orb_bars(self, orb_bars: pd.DataFrame) -> None:
        """
        Replay the 9:30-10:00 ORB bars to initialise the full-session midpoint
        EMA to the state the backtest holds after processing the last ORB bar.

        Mirrors orb_backtester.py:1882-1891 applied to the ORB window only.

        orb_bars must be a DataFrame with a DatetimeIndex covering 9:30-9:59
        (the same DataFrame passed to compute_opening_range).  Columns 'high'
        and 'low' are used; if absent, 'close' is substituted for both (matches
        the backtest's fallback at line 1883-1884).

        Must be called BEFORE any post-ORB bar is delivered via on_bar().
        Calling on_bar() before seed_from_orb_bars() raises RuntimeError.
        """
        if orb_bars.empty:
            raise ValueError(f"RollingIndicators.seed_from_orb_bars: empty bars for {self._symbol}")

        k = self._k

        hi_col = orb_bars["high"]  if "high" in orb_bars.columns else orb_bars["close"]
        lo_col = orb_bars["low"]   if "low"  in orb_bars.columns else orb_bars["close"]

        mids = ((hi_col + lo_col) / 2.0).values
        ema  = float(mids[0])           # seed from first bar's midpoint
        for mid in mids[1:]:
            ema = float(mid) * k + ema * (1.0 - k)

        self._ema = ema

    # ── Incremental update (called every post-ORB bar) ─────────────────────────

    def on_bar(self, bar: dict) -> float:
        """
        Update EMA with this bar's midpoint and return the updated value.

        bar must have 'high' and 'low' keys (or 'close' as fallback, matching
        the backtest's behaviour at lines 1883-1884).

        Raises RuntimeError if called before seed_from_orb_bars.
        """
        if self._ema is None:
            raise RuntimeError(
                f"RollingIndicators for {self._symbol} has not been seeded. "
                "Call seed_from_orb_bars() before on_bar()."
            )
        hi  = float(bar.get("high",  bar["close"]))
        lo  = float(bar.get("low",   bar["close"]))
        mid = (hi + lo) / 2.0
        self._ema = mid * self._k + self._ema * (1.0 - self._k)
        return self._ema

    # ── Persistence (optional, for audit / restart) ────────────────────────────

    def persist_to(
        self,
        state_store: "StateStore",
        session_date,
        symbol: str,
        ts: datetime,
    ) -> None:
        """Snapshot current EMA to indicator_state table (non-critical, best-effort)."""
        if self._ema is None:
            return
        try:
            state_store.save_indicator_state(session_date, symbol, ts, self._ema)
        except Exception:
            pass  # indicator state is audit-only; never let it crash the session
