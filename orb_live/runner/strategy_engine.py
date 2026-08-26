"""
runner/strategy_engine.py — Per-symbol ORB entry state machine.

State transitions (one-way, no re-entry within a session):
    WAITING_FOR_ORB   → ORB_COMPLETE        on_orb_complete() with is_candidate=True
    WAITING_FOR_ORB   → EXITED_OR_SKIPPED   on_orb_complete() with is_candidate=False
    ORB_COMPLETE      → IN_POSITION         breakout fires + open_position() succeeds
    ORB_COMPLETE      → EXITED_OR_SKIPPED   past latest_entry_minute OR entry rejected
    IN_POSITION       — terminal for entry detection; exits handled by position_manager

runner directly (not through this engine) to avoid double-calling.
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime
from enum import Enum
from typing import Optional, TYPE_CHECKING

import pandas as pd

from orb_live.signals.strategy_signals import check_breakout, compute_entry

if TYPE_CHECKING:
    from orb_live.config.live_config import LiveConfig
    from orb_live.core.state_store import StateStore
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.signals.pre_market import Phase2Result


class SymbolState(str, Enum):
    WAITING_FOR_ORB   = "waiting_for_orb"
    ORB_COMPLETE      = "orb_complete"
    IN_POSITION       = "in_position"
    EXITED_OR_SKIPPED = "exited_or_skipped"
    RESTING           = "resting"  # pre-placed stop-limit resting; no bar-watch entry


class StrategyEngine:
    """
    Per-symbol ORB entry state machine for one trading session.

    The runner owns this engine and calls:
      - new_session(trade_date)                          at session start
      - on_orb_complete(symbol, p2_result, orb_bars_df)  at 10:00 ET
      - on_bar(symbol, bar, ts)                          every post-ORB bar

    Entry logic only: once a position is opened, position_manager handles
    all bar-level exit decisions independently.
    """

    def __init__(
        self,
        position_manager: "LivePositionManager",
        config: "LiveConfig",
        state_store: "StateStore",
        broker,
        logger=None,
    ):
        self._mgr              = position_manager
        self._cfg              = config
        self._store            = state_store
        self._broker           = broker
        self._log              = logger
        self._v1_base_notional = getattr(config, "v1_base_notional", 0.0)
        self._base_notional_pct = getattr(config, "base_notional_pct", 0.0)

        self._states:     dict[str, SymbolState] = {}
        self._p2:         dict[str, "Phase2Result"] = {}
        self._trade_date: Optional[date] = None

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def new_session(self, trade_date: date) -> None:
        self._states.clear()
        self._p2.clear()
        self._trade_date = trade_date

    # ── ORB completion ─────────────────────────────────────────────────────────

    def on_orb_complete(
        self,
        symbol: str,
        p2_result: "Phase2Result",
        orb_bars_df,    # pd.DataFrame with ORB bars (unused here; held by runner)
    ) -> None:
        """
        Called by the runner at 10:00 ET for each phase-2 symbol.
        Sets terminal EXITED_OR_SKIPPED if the symbol was excluded; otherwise
        transitions to ORB_COMPLETE for breakout watching.
        """
        if not p2_result.is_candidate:
            self._states[symbol] = SymbolState.EXITED_OR_SKIPPED
            if self._log:
                self._log.info(
                    "symbol_skipped", symbol=symbol,
                    reason=p2_result.exclusion_reason or "not_candidate",
                )
            return

        if p2_result.size_mult == 0.0:
            self._states[symbol] = SymbolState.EXITED_OR_SKIPPED
            if self._log:
                self._log.info("routing_skip", symbol=symbol)
            return

        if p2_result.orb is None:
            self._states[symbol] = SymbolState.EXITED_OR_SKIPPED
            if self._log:
                self._log.info("orb_invalid", symbol=symbol)
            return

        # Pre-placed mode: rest a stop-limit at the ORB boundary now instead of
        # watching bars for the break. Entry fills arrive via IB fill events.
        if getattr(self._cfg, "use_resting_entries", False):
            self._place_resting_entry(symbol, p2_result, orb_bars_df)
            self._states[symbol] = SymbolState.RESTING
            return

        self._states[symbol] = SymbolState.ORB_COMPLETE
        self._p2[symbol] = p2_result
        if self._log:
            self._log.info(
                "watching_for_breakout", symbol=symbol,
                orb_high=p2_result.orb["high"], orb_low=p2_result.orb["low"],
            )

    def _place_resting_entry(self, symbol: str, p2: "Phase2Result", orb_bars_df) -> None:
        """Compute the boundary entry and submit a resting stop-limit for it.

        Called once per candidate at 10:00 when use_resting_entries is on. The
        stop trigger is the ORB boundary (entry_at_boundary sizing); the order
        rests at IB and fills the instant price crosses. Buying-power overload
        from simultaneous breaks is resolved at fill time in position_manager.
        """
        scfg = self._cfg.strategy_config
        if orb_bars_df is not None and not orb_bars_df.empty:
            bar_series = orb_bars_df.iloc[-1]
        else:
            ref = p2.orb["high"] if p2.gap_direction == 1 else p2.orb["low"]
            bar_series = pd.Series(
                {"high": p2.orb["high"], "low": p2.orb["low"], "close": ref}
            )
        current_equity = float(self._broker.get_account().get("equity", 100_000.0))
        entry = compute_entry(
            bar_series, p2.orb, p2.gap_direction, scfg,
            current_equity=current_equity, symbol=symbol,
            tp1_mult_override=p2.tp1_mult, tp2_mult_override=p2.tp2_mult,
            size_mult=p2.size_mult, v1_base_notional=self._base_notional(current_equity),
        )
        if entry.get("shares", 0) == 0:
            if self._log:
                self._log.info("resting_zero_shares", symbol=symbol)
            return
        order_id = self._mgr.place_entry_order(
            entry=entry, symbol=symbol,
            direction=p2.gap_direction, session_date=self._trade_date,
        )
        if self._log:
            self._log.info(
                "resting_entry_placed", symbol=symbol, order_id=order_id,
                shares=entry["shares"], boundary=round(entry["entry_price"], 4),
            )

    def _base_notional(self, current_equity: float) -> Optional[float]:
        """Per-unit sizing notional. base_notional_pct (fraction of live equity)
        takes precedence so sizing compounds with the account; falls back to the
        fixed v1_base_notional. Returns None if neither is usable."""
        if self._base_notional_pct > 0 and current_equity > 0:
            return self._base_notional_pct * current_equity
        return self._v1_base_notional or None

    def get_state(self, symbol: str) -> SymbolState:
        return self._states.get(symbol, SymbolState.WAITING_FOR_ORB)

    # ── Per-bar entry detection ────────────────────────────────────────────────

    def on_bar(self, symbol: str, bar: dict, ts: datetime) -> None:
        """
        Check one post-ORB bar for a valid breakout entry.

        No-op unless state is ORB_COMPLETE.  Does NOT call
        position_manager.on_bar() — the session runner does that directly,
        before calling this method.
        """
        if self._states.get(symbol) != SymbolState.ORB_COMPLETE:
            return

        p2 = self._p2.get(symbol)
        if p2 is None:
            return

        scfg = self._cfg.strategy_config

        # Latest-entry cutoff: minutes after 9:30 market open
        if scfg.latest_entry_minute is not None:
            market_open = ts.replace(
                hour=scfg.market_open_hour,
                minute=scfg.market_open_minute,
                second=0, microsecond=0,
            )
            mins_after_open = int((ts - market_open).total_seconds() // 60)
            if mins_after_open >= scfg.latest_entry_minute:
                self._states[symbol] = SymbolState.EXITED_OR_SKIPPED
                if self._log:
                    self._log.info(
                        "past_latest_entry", symbol=symbol,
                        mins_after_open=mins_after_open,
                        cutoff=scfg.latest_entry_minute,
                    )
                return

        bar_series = pd.Series(bar)
        if not check_breakout(bar_series, p2.orb, p2.gap_direction, scfg):
            return

        self._store.save_breakout_signal(
            trade_date=self._trade_date,
            symbol=symbol,
            direction=p2.gap_direction,
            breakout_price=float(bar["close"]),
            orb_high=float(p2.orb["high"]),
            orb_low=float(p2.orb["low"]),
        )
        if self._log:
            self._log.info(
                "breakout_detected", symbol=symbol,
                direction=p2.gap_direction, price=float(bar["close"]),
            )

        # max_entry_price gate
        entry_price_raw = float(bar["close"])
        if entry_price_raw > scfg.max_entry_price:
            self._states[symbol] = SymbolState.EXITED_OR_SKIPPED
            if self._log:
                self._log.warning(
                    "max_entry_price_exceeded", symbol=symbol,
                    price=entry_price_raw, limit=scfg.max_entry_price,
                )
            return

        current_equity = float(self._broker.get_account().get("equity", 100_000.0))
        entry = compute_entry(
            bar_series, p2.orb, p2.gap_direction, scfg,
            current_equity=current_equity,
            symbol=symbol,
            tp1_mult_override=p2.tp1_mult,
            tp2_mult_override=p2.tp2_mult,
            size_mult=p2.size_mult,
            v1_base_notional=self._base_notional(current_equity),
        )

        if entry.get("shares", 0) == 0:
            self._states[symbol] = SymbolState.EXITED_OR_SKIPPED
            if self._log:
                self._log.info("zero_shares_computed", symbol=symbol)
            return

        pos = self._mgr.open_position(
            entry=entry,
            symbol=symbol,
            gap_direction=p2.gap_direction,
            session_date=self._trade_date,
        )

        if pos is not None:
            self._states[symbol] = SymbolState.IN_POSITION
            if self._log:
                self._log.info(
                    "position_opened", symbol=symbol,
                    actual_entry=pos.actual_entry_price,
                    shares=pos.entry_shares,
                )
        else:
            self._states[symbol] = SymbolState.EXITED_OR_SKIPPED
            if self._log:
                self._log.info("entry_rejected_or_unfilled", symbol=symbol)
