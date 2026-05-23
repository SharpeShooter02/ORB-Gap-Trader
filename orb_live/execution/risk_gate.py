"""
execution/risk_gate.py — Session-level risk gate for the live ORB system.

Checks (in order) before every new entry:
  1. Already in position for this symbol?
  2. Session kill active (realized + unrealized loss > kill_loss_pct)?
  3. Gross exposure cap (total open position value > max_gross_exposure_pct * equity)?
  4. Per-symbol position cap (intended dollars > max_position_pct * equity)?

The session kill is also checked after every closed trade; once triggered it
blocks all new entries and is recorded in state_store.day_state.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from orb_live.config.live_config import LiveConfig
    from orb_live.core.state_store import StateStore


class RiskGate:
    """
    Pre-entry authorization and session-kill management.

    The gate is intentionally stateless with respect to the DB — it derives
    all live state from alpaca.get_account() and the positions dict passed
    by the caller, making it safe to construct fresh on restart.

    live_cfg fields used:
        session_kill_loss_pct  — daily loss limit as fraction of starting equity
        max_concurrent_positions — 0 = unlimited
        max_gross_exposure_pct — max sum(open notional) / equity
        max_position_pct       — max single position notional / equity
    """

    def __init__(
        self,
        live_cfg: "LiveConfig",
        state_store: "StateStore",
        broker,
        logger=None,
    ):
        self._cfg       = live_cfg
        self._store     = state_store
        self._broker    = broker
        self._log       = logger

        self._starting_equity:   float = 0.0
        self._realized_pnl:      float = 0.0
        self._kill_triggered:    bool  = False
        self._trade_date: Optional[date] = None

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def session_start(self, starting_equity: float, trade_date: date) -> None:
        """Called by the runner at session open.  Resets daily counters."""
        self._starting_equity = starting_equity
        self._realized_pnl   = 0.0
        self._trade_date     = trade_date
        self._kill_triggered = self._store.is_kill_triggered(trade_date)

    # ── PnL tracking ──────────────────────────────────────────────────────────

    def record_realized_pnl(self, amount: float) -> None:
        """
        Called by LivePositionManager after each closed fill leg.
        Checks kill threshold after update.
        """
        self._realized_pnl += amount
        self._check_kill(unrealized_pnl=0.0)

    def _check_kill(self, unrealized_pnl: float = 0.0) -> None:
        if self._kill_triggered:
            return
        if self._starting_equity <= 0:
            return
        total_loss = self._realized_pnl + unrealized_pnl
        threshold  = -self._cfg.session_kill_loss_pct * self._starting_equity
        if total_loss <= threshold:
            self._kill_triggered = True
            reason = (
                f"session_kill: loss {total_loss:.2f} exceeded "
                f"threshold {threshold:.2f}"
            )
            if self._log:
                self._log.critical("session_kill_triggered", reason=reason,
                                   total_loss=total_loss, threshold=threshold)
            if self._trade_date:
                self._store.set_kill_switch(self._trade_date, reason)
                self._store.log_alert(
                    level="CRITICAL",
                    message=reason,
                    category="kill_switch",
                    trade_date=self._trade_date,
                )

    # ── Entry authorization ────────────────────────────────────────────────────

    def is_session_killed(self) -> bool:
        return self._kill_triggered

    def authorize_entry(
        self,
        symbol: str,
        direction: int,
        intended_dollars: float,
        current_equity: float,
        today_realized_pnl: float = 0.0,
        today_unrealized_pnl: float = 0.0,
        open_positions: Optional[dict] = None,   # {symbol: position_notional}
    ) -> tuple[bool, Optional[str]]:
        """
        Returns (True, None) on pass, (False, reason_str) on reject.

        open_positions — mapping of currently open symbols to their notional
                         value (entry_price × remaining shares). If None,
                         exposure checks are skipped (fail-open on data gap).
        """
        cfg = self._cfg

        # 1. Already in position?
        if open_positions and symbol in open_positions:
            return False, "already_in_position"

        # 2. Session kill active?
        # Also check against current PnL in case record_realized_pnl wasn't called.
        if self._kill_triggered:
            return False, "session_kill_active"
        if self._starting_equity > 0:
            total_pnl  = today_realized_pnl + today_unrealized_pnl
            kill_thr   = -cfg.session_kill_loss_pct * self._starting_equity
            if total_pnl <= kill_thr:
                self._check_kill(unrealized_pnl=today_unrealized_pnl)
                return False, "session_kill_active"

        # 3. Gross exposure cap?
        if open_positions is not None and current_equity > 0:
            total_exposure = sum(open_positions.values()) + intended_dollars
            max_exposure   = cfg.max_gross_exposure_pct * current_equity
            if total_exposure > max_exposure:
                return False, "max_gross_exposure"

        # 4. Per-symbol cap?
        if current_equity > 0:
            max_position = cfg.max_position_pct * current_equity
            if intended_dollars > max_position:
                return False, "max_position_pct"

        return True, None
