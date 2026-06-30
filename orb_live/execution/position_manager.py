"""
execution/position_manager.py — Live equivalent of simulate_trade.

LivePositionManager mirrors the exact order-of-operations from
orb_backtester.py:simulate_trade (lines 1376-1487):

    Per-bar sequence (on_bar):
      1.  Update max_fav and post_tp2_mfe          (lines 1381-1388)
      2.  EOD exit check                           (lines 1390-1396)
      3.  TP1 fixed-price check                    (lines 1398-1411)
          → moves current_stop to breakeven (or trail if use_trail_atp1)
          → if tp2_shares==0: sets tp2_hit=True (TP3 fires on next bar)
      4.  trail_after_tp1 peak/stop update         (lines 1413-1419)
      5.  TP2 fixed-price check ("if" not "elif")  (lines 1421-1426)
      6.  TP3 EMA-crossback check                  (lines 1462-1473)
      7.  Stop-loss check                          (lines 1475-1486)

PRODUCTION NOTES:
  - atr_trail TP3 mode is NOT implemented (raises NotImplementedError at init).
  - EOD exit at eod_exit_hour=16 NEVER fires on a delivered RTH bar (last bar
    is 15:59). The runner calls flatten_all('eod_sweep') at 16:00:30 ET instead.
  - Position sizing uses shared account equity (not per-symbol pools as in the
    backtest). See HANDOFF_PROMPT_3.md §5 for the intentional divergence note.
  - Every state change is persisted to state_store BEFORE the order is placed,
    so a crash mid-bar leaves a recoverable state.

KEY DISTINCTION — TWO DIFFERENT EMAs:
  on_bar reads indicators_store[symbol].ema for TP3 decisions.  This is the
  full-session MIDPOINT EMA (EMA #2), NOT orb["ema"] (EMA #1, the frozen
  close-EMA used by check_breakout).  The runner owns the indicators_store
  and seeds / updates it; position_manager only reads it.
"""

from __future__ import annotations

import math
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from orb_live.execution.order_policy import Fill, MarketableLimitPolicy
    from orb_live.execution.indicators import RollingIndicators
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.core.state_store import StateStore

UTC = timezone.utc


# ── Position dataclass ─────────────────────────────────────────────────────────

@dataclass
class Position:
    """
    Full mutable state of one open trade.

    Mirrors every variable in simulate_trade's inner loop so that an
    in-memory crash recovery can be seeded from a state_store row.
    """
    symbol:              str
    session_date:        date
    direction:           int           # +1 long, -1 short

    # Entry prices
    entry_price:         float         # limit price from compute_entry
    actual_entry_price:  float         # actual fill avg_price (slippage-adjusted)

    # Share counts
    entry_shares:        int           # full allocation from compute_entry
    remaining:           int           # decrements as TP legs fill

    # ORB geometry
    orb_range:           float

    # Stop
    stop_price:          float         # original stop (never changes)
    current_stop:        float         # moves to breakeven after TP1, then trails

    # TP targets
    tp1_price:           float
    tp2_price:           float

    # TP share allocations (reflect actual filled qty, not compute_entry output)
    tp1_shares:          int
    tp2_shares:          int
    tp3_shares:          int

    # TP state flags
    tp1_hit:             bool = False
    tp2_hit:             bool = False
    tp3_hit:             bool = False

    # Trailing stop after TP1 (trail_after_tp1 exit override)
    use_trail_atp1:      bool  = False
    trail_atp1_peak:     Optional[float] = None
    trail_atp1_dist:     float = 0.0

    # Running MFE trackers (mirrors lines 1381-1388)
    max_fav:             float = 0.0
    post_tp2_mfe:        float = 0.0

    # Exchange stop order
    stop_order_id:       Optional[str] = None  # IB order_id of resting stop-market
    tp1_order_id:        Optional[str] = None  # IB order_id of OCA TP1 limit leg

    # OCA diagnostic / partial-fill tracking (in-memory only; crash recovery via reconcile)
    tp1_crossed_unfilled_bars: int = 0  # consecutive bars price past TP1 but fill unconfirmed
    tp1_filled_qty_booked:     int = 0  # cumulative OCA-filled shares booked (partial-fill guard)

    # Lifecycle
    status:              str  = "open"   # entering/open/unfilled/closed
    decision_reason:     str  = ""

    # Exit (populated when closed)
    exit_reason:         str  = "EOD"
    exit_price:          Optional[float] = None
    exit_time:           Optional[datetime] = None

    # Aliases consumed by some helper callers
    @property
    def qty(self) -> int:
        return self.entry_shares


# ── LivePositionManager ────────────────────────────────────────────────────────

class LivePositionManager:
    """
    Manages open positions for a single trading session.

    indicators_store — dict[symbol → RollingIndicators], owned by the runner.
    The runner seeds each indicator at 10:00 ET and calls indicator.on_bar()
    BEFORE calling this manager's on_bar(), so the EMA is always current when
    TP3 is evaluated.
    """

    def __init__(
        self,
        broker,
        policy: "MarketableLimitPolicy",
        state_store: "StateStore",
        risk_gate: "RiskGate",
        indicators_store: dict,
        config,                  # StrategyConfig
        logger=None,
    ):
        if config.tp3_mode != "ema_crossback":
            raise NotImplementedError(
                f"tp3_mode={config.tp3_mode!r} is not implemented in live execution. "
                "Production uses 'ema_crossback'."
            )

        self._broker     = broker
        self._policy     = policy
        self._store      = state_store
        self._gate       = risk_gate
        self._indicators = indicators_store
        self._config     = config
        self._log        = logger

        # In-memory position cache — source of truth for hot-path decisions.
        # Persisted to state_store on every mutation.
        self._positions: dict[str, Position] = {}

        # Pending entry orders (submitted but not yet filled/cancelled).
        # Keyed by symbol; polled in on_bar until terminal status.
        self._pending_entries: dict[str, dict] = {}

        # Universe of watched symbols for EOD flatten safety net.
        self._universe: list[str] = []

    # ── Entry ──────────────────────────────────────────────────────────────────

    def open_position(
        self,
        entry: dict,
        symbol: str,
        gap_direction: int,
        session_date: date,
    ) -> Optional[Position]:
        """
        Called when check_breakout fires on a post-ORB bar close.

        Submits the entry order non-blocking (safe to call from inside an
        ib_async event callback — no time.sleep, no ib.sleep).

        If the broker fills immediately (MockBroker / already-filled order),
        returns the open Position.  If the order is still pending, stores it
        in _pending_entries, returns None, and on_bar will detect the fill on
        the next bar and place the bracket then.
        """
        cfg = self._config

        # Build open_positions notional for exposure check (include 'entering')
        open_notional = {
            sym: (pos.actual_entry_price * pos.remaining)
            for sym, pos in self._positions.items()
            if pos.status in ("open", "entering")
        }

        # Gate check
        current_equity   = float(self._broker.get_account().get("equity", 0))
        intended_dollars = entry["entry_price"] * entry["shares"]
        ok, reason = self._gate.authorize_entry(
            symbol=symbol,
            direction=gap_direction,
            intended_dollars=intended_dollars,
            current_equity=current_equity,
            open_positions=open_notional,
        )
        if not ok:
            self._store.save_candidate(
                session_date=session_date, symbol=symbol, phase=3,
                decision=reason, gap_direction=gap_direction,
                gap_abs=0.0, prior_close=0.0,
                ps_filter_passed=None, intended_direction=gap_direction,
            )
            if self._log:
                self._log.warning("entry_rejected_by_risk_gate",
                                  symbol=symbol, reason=reason)
            return None

        # Block if a pending entry or in-memory position already exists
        if symbol in self._pending_entries or symbol in self._positions:
            if self._log:
                self._log.warning("entry_blocked_in_flight", symbol=symbol)
            return None

        # Block if DB has a real 'open' row (restart with live broker position)
        existing_db = self._store.get_open_position(symbol)
        if existing_db and existing_db.get("status") == "open":
            if self._log:
                self._log.warning(
                    "open_position_skipped_db_already_open",
                    symbol=symbol,
                    db_status=existing_db.get("status"),
                )
            return None

        # Fix 3: block re-entry if broker already carries a position
        try:
            broker_pos = self._broker.get_position(symbol)
            if broker_pos and abs(float(broker_pos.get("qty", 0))) > 0:
                if self._log:
                    self._log.warning("entry_blocked_by_broker_position",
                                      symbol=symbol, broker_qty=broker_pos.get("qty"))
                return None
        except Exception as exc:
            if self._log:
                self._log.warning("entry_broker_check_failed",
                                  symbol=symbol, exc=str(exc))

        # Build position in 'entering' state and persist BEFORE order
        override  = entry.get("exit_override", {})
        use_trail = (override.get("method") == "trail_after_tp1")
        trail_mult = float(override.get("mult", 1.0)) if use_trail else 0.0
        trail_dist = trail_mult * entry["orb_range"]

        pos = Position(
            symbol=symbol,
            session_date=session_date,
            direction=gap_direction,
            entry_price=entry["entry_price"],
            actual_entry_price=entry["entry_price"],
            entry_shares=entry["shares"],
            remaining=entry["shares"],
            orb_range=entry["orb_range"],
            stop_price=entry["stop_price"],
            current_stop=entry["stop_price"],
            tp1_price=entry["tp1_price"],
            tp2_price=entry["tp2_price"],
            tp1_shares=entry["tp1_shares"],
            tp2_shares=entry["tp2_shares"],
            tp3_shares=entry["tp3_shares"],
            use_trail_atp1=use_trail,
            trail_atp1_dist=trail_dist,
            status="entering",
        )
        self._persist_new(pos, session_date)

        # Submit entry order non-blocking (no polling / no sleep)
        side = "buy" if gap_direction == 1 else "sell"
        submitted_at = datetime.now(UTC)
        try:
            limit_price = self._policy.compute_entry_limit(
                side, symbol, entry["entry_price"]
            )
            client_id = str(uuid.uuid4())
            order     = self._broker.submit_limit_order(
                symbol, side, entry["shares"], limit_price, client_id
            )
            order_id = order.get("id", client_id)
        except Exception as exc:
            if self._log:
                self._log.critical("entry_order_exception", symbol=symbol, exc=str(exc))
            self._store.close_position(symbol)
            return None

        # Check fill status immediately — no sleep; safe inside asyncio callbacks.
        # For MockBroker / already-filled orders this resolves synchronously.
        # For real IB at PendingSubmit this goes to the async path below.
        try:
            order_state  = self._broker.get_order(order_id)
            broker_status = order_state.get("status", "new")
            fill_qty     = int(float(order_state.get("filled_qty", 0) or 0))
            fill_price   = float(order_state.get("filled_avg_price", 0) or 0)
        except Exception:
            broker_status = "new"
            fill_qty      = 0
            fill_price    = 0.0

        if broker_status in ("filled", "partially_filled") and fill_qty > 0:
            # Immediate fill (MockBroker or pre-filled order) — complete inline.
            return self._complete_entry_fill(
                symbol=symbol, fill_qty=fill_qty, fill_price=fill_price,
                pos=pos, entry=entry, side=side, session_date=session_date,
                attempt=1, order_id=order_id, submitted_at=submitted_at,
            )

        if broker_status in ("cancelled", "expired", "rejected", "inactive"):
            # Immediate rejection — no position.
            self._store.close_position(symbol)
            return None

        # Order is still pending (PendingSubmit / Submitted at IB).
        # Store in memory and poll in on_bar; the event loop will update the
        # order cache via _on_order_status before the next bar arrives.
        self._positions[symbol] = pos
        self._pending_entries[symbol] = {
            "order_id":         order_id,
            "entry":            entry,
            "side":             side,
            "direction":        gap_direction,
            "session_date":     session_date,
            "pos":              pos,
            "attempt":          1,
            "submit_time":      _time.monotonic(),
            "first_submit_time": _time.monotonic(),
            "submitted_at":     submitted_at,
            "limit_price":      limit_price,
            "original_qty":     entry["shares"],
        }
        return None

    # ── Per-bar processing ─────────────────────────────────────────────────────

    def on_bar(self, symbol: str, bar: dict, ts: datetime) -> None:
        """
        Process one closed 1-min bar for an open position.

        bar — dict with keys: high, low, close (high/low may be absent,
              in which case close is substituted, matching the backtest at
              lines 1377-1379).

        ts  — bar's opening timestamp (start-of-bar convention, per
              HANDOFF_PROMPT_2.md §6).

        The runner MUST call indicators_store[symbol].on_bar(bar) BEFORE
        calling this method.
        """
        # Poll any pending entry order first; may promote pos to 'open'.
        if symbol in self._pending_entries:
            self._check_pending_entry(symbol, ts)

        pos = self._positions.get(symbol)
        if pos is None or pos.status != "open":
            return

        # ── Poll exchange-resident stop status ───────────────────────────────
        # If IB fired the stop since the last bar, early-return so no TP/EMA
        # logic runs against a position that's already closed at the exchange.
        # On polling failure, log a warning and continue — the bar-by-bar stop
        # check below acts as backup (removed in the next prompt).
        if pos.stop_order_id:
            try:
                stop_state  = self._broker.get_order(pos.stop_order_id)
                stop_status = stop_state.get("status", "")
                if stop_status in ("filled", "partially_filled"):
                    self._handle_stop_fired(pos, symbol, stop_state)
                    return
            except Exception as exc:
                if self._log:
                    self._log.warning(
                        "stop_poll_failed",
                        symbol=symbol,
                        stop_order_id=pos.stop_order_id,
                        error=str(exc),
                    )

        cfg = self._config
        lo  = float(bar.get("low",   bar["close"]))
        hi  = float(bar.get("high",  bar["close"]))
        cl  = float(bar["close"])

        # ── 1. Update max_fav and post_tp2_mfe ───────────────────────────────
        bar_fav = (hi - pos.actual_entry_price) * pos.direction
        if bar_fav > pos.max_fav:
            pos.max_fav = bar_fav
        if pos.tp2_hit:
            bar_post = (hi - pos.tp2_price) * pos.direction
            if bar_post > pos.post_tp2_mfe:
                pos.post_tp2_mfe = bar_post

        # ── 2. EOD check ──────────────────────────────────────────────────────
        eod_time = dtime(cfg.eod_exit_hour, cfg.eod_exit_minute)
        if ts.time() >= eod_time:
            exit_reason = ("TP2"      if pos.tp2_hit else
                           "TP1_ONLY" if pos.tp1_hit else "EOD")
            self._exit_all(pos, symbol, leg="eod", ref_price=cl,
                           exit_reason=exit_reason, exit_time=ts,
                           session_date=pos.session_date)
            return

        # ── 3. TP1 check ──────────────────────────────────────────────────────
        if not pos.tp1_hit and pos.tp1_shares > 0:
            tp1_realized_price = pos.tp1_price   # updated from confirmed fill below
            tp1_hit            = False
            _tp1_qty           = pos.tp1_shares  # shares to book; OCA partials reduce this
            _tp1_terminal      = True            # False for OCA partially_filled

            if pos.tp1_order_id:
                # OCA bracket: IB placed a resting TP1 limit at entry.
                # Only commit on confirmed fill — never synthesize from bar price.
                # A wick to TP1 that reverses before IB confirms causes double-booking
                # and a zombie position with remaining=0 but status="open".
                try:
                    tp1_state  = self._broker.get_order(pos.tp1_order_id)
                    tp1_status = tp1_state.get("status", "")
                    if tp1_status in ("filled", "partially_filled"):
                        current_filled = int(float(tp1_state.get("filled_qty", 0) or 0))
                        newly_filled   = current_filled - pos.tp1_filled_qty_booked
                        if newly_filled > 0:
                            raw_float = float(tp1_state.get("filled_avg_price", 0) or 0)
                            if raw_float == 0.0:
                                # Re-poll once: filled_avg_price may lag the status
                                # event by one TWS round-trip (~50 ms).
                                try:
                                    import time
                                    time.sleep(0.05)
                                    tp1_state2 = self._broker.get_order(pos.tp1_order_id)
                                    raw_float  = float(
                                        tp1_state2.get("filled_avg_price", 0) or 0
                                    )
                                except Exception:
                                    pass
                            tp1_realized_price        = raw_float if raw_float > 0 else pos.tp1_price
                            _tp1_qty                  = newly_filled
                            _tp1_terminal             = (tp1_status == "filled")
                            tp1_hit                   = True
                            pos.tp1_filled_qty_booked = current_filled
                        # else: no new fills since last poll; wait for next bar
                    else:
                        # Order still working — count bars where price has already
                        # crossed so a thin-book / mis-priced limit surfaces as an alert.
                        bar_crossed = (
                            (pos.direction == 1  and hi >= pos.tp1_price) or
                            (pos.direction == -1 and lo <= pos.tp1_price)
                        )
                        if bar_crossed:
                            pos.tp1_crossed_unfilled_bars += 1
                            if pos.tp1_crossed_unfilled_bars >= 2:
                                self._store.log_alert(
                                    level="CRITICAL",
                                    message=(
                                        f"OCA TP1 limit not filling after "
                                        f"{pos.tp1_crossed_unfilled_bars} bars past "
                                        f"target for {symbol} "
                                        f"(tp1_order_id={pos.tp1_order_id}, "
                                        f"tp1_price={pos.tp1_price:.4f})"
                                    ),
                                    category="tp1_limit_not_filling",
                                    symbol=symbol,
                                    trade_date=pos.session_date,
                                )
                        else:
                            pos.tp1_crossed_unfilled_bars = 0
                except Exception as exc:
                    if self._log:
                        self._log.warning(
                            "tp1_order_poll_failed",
                            symbol=symbol,
                            tp1_order_id=pos.tp1_order_id,
                            error=str(exc),
                        )
                    # Poll failure: wait for next bar; do not synthesize a fill.
            else:
                # Non-OCA path: bar-price trigger submits exit via _exit_partial.
                tp1_hit = (
                    (pos.direction == 1  and hi >= pos.tp1_price) or
                    (pos.direction == -1 and lo <= pos.tp1_price)
                )

            if tp1_hit:
                if pos.tp1_order_id:
                    # ── OCA path: IB already executed the exit ────────────────
                    # Book only the newly-confirmed shares (_tp1_qty ≤ tp1_shares).
                    pnl = (
                        (tp1_realized_price - pos.actual_entry_price)
                        * _tp1_qty * pos.direction
                    )
                    self._gate.record_realized_pnl(pnl)
                    pos.remaining -= _tp1_qty
                    if _tp1_terminal:
                        # Full fill: advance state flags and move stop to breakeven.
                        pos.tp1_hit = True
                        if pos.use_trail_atp1:
                            pos.trail_atp1_peak = pos.tp1_price
                            if pos.direction == 1:
                                pos.current_stop = pos.tp1_price - pos.trail_atp1_dist
                            else:
                                pos.current_stop = pos.tp1_price + pos.trail_atp1_dist
                        else:
                            pos.current_stop = pos.actual_entry_price  # breakeven
                        if pos.tp2_shares == 0:
                            pos.tp2_hit = True
                    self._update_pos(pos)

                    # ── All shares exited at TP1 (v1 TP1-only mode) ──────────
                    if pos.remaining == 0:
                        # Verify IB cancelled the stop sibling via OCA.
                        stop_cancelled = False
                        try:
                            stop_state = self._broker.get_order(pos.stop_order_id)
                            stop_cancelled = stop_state.get("status") in (
                                "cancelled", "filled", "ApiCancelled", "Inactive"
                            )
                        except Exception as exc:
                            if self._log:
                                self._log.warning(
                                    "stop_poll_failed_after_tp1",
                                    symbol=symbol,
                                    stop_order_id=pos.stop_order_id,
                                    error=str(exc),
                                )
                        if not stop_cancelled:
                            self._store.log_alert(
                                level="CRITICAL",
                                message=(
                                    f"OCA stop sibling not cancelled after TP1 fill "
                                    f"for {symbol} (stop_order_id={pos.stop_order_id}) "
                                    "— position left for reconcile_from_broker"
                                ),
                                category="oca_cancel_failure",
                                symbol=symbol,
                                trade_date=pos.session_date,
                            )
                            if self._log:
                                self._log.critical(
                                    "oca_stop_not_cancelled_after_tp1",
                                    symbol=symbol,
                                    stop_order_id=pos.stop_order_id,
                                )
                            return   # position retained; reconcile will close it

                        pos.status      = "closed"
                        pos.exit_reason = "TP1_ONLY"
                        pos.exit_time   = ts
                        pos.exit_price  = pos.tp1_price
                        self._update_pos(pos)
                        self._store.save_closed_trade(
                            trade_date=pos.session_date, symbol=symbol,
                            direction=pos.direction,
                            entry_price=pos.actual_entry_price,
                            exit_price=pos.tp1_price,
                            realized_exit_price=tp1_realized_price,
                            qty=pos.entry_shares,
                            exit_reason="TP1_ONLY",
                            opened_at=datetime.now(UTC),
                        )
                        self._store.close_position(symbol)
                        del self._positions[symbol]
                        return

                    # Partial OCA fill (remaining > 0): the resting OCA stop leg
                    # remains exchange-resident protection — do not modify it.
                    return

                else:
                    # ── Non-OCA path: submit marketable-limit exit now ────────
                    fill = self._exit_partial(pos, symbol, leg="tp1", qty=pos.tp1_shares,
                                              ref_price=pos.tp1_price,
                                              session_date=pos.session_date)
                    tp1_realized_price = fill.avg_price

                    pos.remaining -= pos.tp1_shares
                    pos.tp1_hit   = True
                    if pos.use_trail_atp1:
                        pos.trail_atp1_peak = pos.tp1_price
                        if pos.direction == 1:
                            pos.current_stop = pos.tp1_price - pos.trail_atp1_dist
                        else:
                            pos.current_stop = pos.tp1_price + pos.trail_atp1_dist
                    else:
                        pos.current_stop = pos.actual_entry_price  # breakeven
                    if pos.tp2_shares == 0:
                        pos.tp2_hit = True   # TP3 fires directly after TP1
                    self._update_pos(pos)

                    # ── All shares exited at TP1 (non-OCA TP1-only mode) ──────
                    if pos.remaining == 0:
                        # Explicitly cancel the now-unneeded stop.
                        cancelled = False
                        try:
                            if pos.stop_order_id:
                                cancelled = bool(
                                    self._broker.cancel_order(pos.stop_order_id)
                                )
                        except Exception as exc:
                            if self._log:
                                self._log.warning(
                                    "stop_cancel_exception_after_tp1",
                                    symbol=symbol,
                                    stop_order_id=pos.stop_order_id,
                                    error=str(exc),
                                )
                        if pos.stop_order_id and not cancelled:
                            self._store.log_alert(
                                level="CRITICAL",
                                message=(
                                    f"cancel_order failed for stop {pos.stop_order_id} "
                                    f"after TP1 full exit ({symbol}) — position left "
                                    "for reconcile_from_broker; live stop may still be active"
                                ),
                                category="stop_cancel_failure",
                                symbol=symbol,
                                trade_date=pos.session_date,
                            )
                            if self._log:
                                self._log.critical(
                                    "stop_cancel_failed_after_tp1_leaving_for_reconcile",
                                    symbol=symbol,
                                    stop_order_id=pos.stop_order_id,
                                )
                            return   # position retained; reconcile will close it

                        pos.status      = "closed"
                        pos.exit_reason = "TP1_ONLY"
                        pos.exit_time   = ts
                        pos.exit_price  = pos.tp1_price
                        self._update_pos(pos)
                        self._store.save_closed_trade(
                            trade_date=pos.session_date, symbol=symbol,
                            direction=pos.direction,
                            entry_price=pos.actual_entry_price,
                            exit_price=pos.tp1_price,
                            realized_exit_price=tp1_realized_price,
                            qty=pos.entry_shares,
                            exit_reason="TP1_ONLY",
                            opened_at=datetime.now(UTC),
                        )
                        self._store.close_position(symbol)
                        del self._positions[symbol]
                        return

                    # Non-OCA multi-leg: propagate post-TP1 stop change to IB.
                    if pos.stop_order_id:
                        try:
                            self._broker.modify_stop_order(
                                order_id=pos.stop_order_id,
                                new_qty=pos.remaining,
                                new_stop_price=pos.current_stop,
                                timeout=5.0,
                            )
                            if self._log:
                                self._log.info(
                                    "stop_modified_after_tp1",
                                    symbol=symbol,
                                    new_qty=pos.remaining,
                                    new_stop_price=pos.current_stop,
                                )
                        except Exception as exc:
                            if self._log:
                                self._log.error(
                                    "stop_modify_failed_attempting_recovery",
                                    symbol=symbol,
                                    stop_order_id=pos.stop_order_id,
                                    error=str(exc),
                                )
                            self._recover_stop_after_modify_failure(pos, symbol)

        # ── 4. trail_after_tp1 peak/stop update ──────────────────────────────
        if pos.tp1_hit and pos.use_trail_atp1 and pos.remaining > 0:
            if pos.direction == 1:
                new_peak = max(pos.trail_atp1_peak, hi)
            else:
                new_peak = min(pos.trail_atp1_peak, lo)
            if new_peak != pos.trail_atp1_peak:
                pos.trail_atp1_peak = new_peak
                if pos.direction == 1:
                    pos.current_stop = new_peak - pos.trail_atp1_dist
                else:
                    pos.current_stop = new_peak + pos.trail_atp1_dist
                self._update_pos(pos)

        # ── 5. TP2 check ("if" not "elif") ───────────────────────────────────
        if pos.tp1_hit and not pos.tp2_hit and pos.tp2_shares > 0 \
                and not pos.use_trail_atp1:
            tp2_hit = (
                (pos.direction == 1  and hi >= pos.tp2_price) or
                (pos.direction == -1 and lo <= pos.tp2_price)
            )
            if tp2_hit:
                self._exit_partial(pos, symbol, leg="tp2", qty=pos.tp2_shares,
                                   ref_price=pos.tp2_price,
                                   session_date=pos.session_date)
                pos.remaining -= pos.tp2_shares
                pos.tp2_hit    = True
                self._update_pos(pos)

        # ── 6. TP3 EMA-crossback check ────────────────────────────────────────
        if pos.tp2_hit and not pos.tp3_hit and pos.remaining > 0 \
                and not pos.use_trail_atp1:
            ind = self._indicators.get(symbol)
            if ind is None or not ind.is_seeded:
                raise RuntimeError(
                    f"RollingIndicators for {symbol} is not seeded — "
                    "the runner must call seed_from_orb_bars() before "
                    "delivering post-ORB bars to on_bar()."
                )
            ema          = ind.ema
            is_profitable = (ema > pos.actual_entry_price) if pos.direction == 1 \
                            else (ema < pos.actual_entry_price)
            crossed_back  = (cl < ema) if pos.direction == 1 else (cl > ema)
            if is_profitable and crossed_back:
                self._exit_all(pos, symbol, leg="tp3", ref_price=ema,
                               exit_reason="TP3", exit_time=ts,
                               session_date=pos.session_date)
                pos.tp3_hit = True
                return

    # ── Universe registration ──────────────────────────────────────────────────

    def set_universe(self, symbols: list) -> None:
        """Register the session's candidate symbols for EOD flatten safety net."""
        self._universe = list(symbols)

    # ── Async entry fill detection ─────────────────────────────────────────────

    def _complete_entry_fill(
        self,
        symbol: str,
        fill_qty: int,
        fill_price: float,
        pos: "Position",
        entry: dict,
        side: str,
        session_date: date,
        attempt: int,
        order_id: str,
        submitted_at: datetime,
    ) -> Optional["Position"]:
        """Finalise a detected entry fill: persist, place bracket, return Position."""
        cfg = self._config

        if fill_qty == 0:
            self._store.close_position(symbol)
            self._positions.pop(symbol, None)
            self._pending_entries.pop(symbol, None)
            return None

        # Save fill record
        try:
            self._store.save_fill(
                session_date=session_date,
                symbol=symbol,
                side=side,
                qty=fill_qty,
                avg_price=fill_price if fill_price > 0 else entry["entry_price"],
                order_id=order_id,
                leg="entry",
                attempts=attempt,
                reason="filled" if fill_qty >= entry["shares"] else "partial_unfilled",
                submitted_at=submitted_at,
                filled_at=datetime.now(UTC),
                raw_response_json="{}",
            )
        except Exception as exc:
            if self._log:
                self._log.warning("entry_fill_record_failed", symbol=symbol, exc=str(exc))

        # Rescale TP share allocations for a partial fill
        override = entry.get("exit_override", {})
        if fill_qty < pos.entry_shares:
            r_tp1 = override.get("exit_ratio_tp1", cfg.exit_ratio_tp1)
            r_tp2 = override.get("exit_ratio_tp2", cfg.exit_ratio_tp2)
            pos.tp1_shares = math.floor(fill_qty * r_tp1)
            pos.tp2_shares = math.floor(fill_qty * r_tp2)
            pos.tp3_shares = max(0, fill_qty - pos.tp1_shares - pos.tp2_shares)
            if self._log:
                self._log.warning(
                    "partial_entry_fill",
                    symbol=symbol,
                    requested=entry["shares"],
                    filled=fill_qty,
                    tp1_shares=pos.tp1_shares,
                    tp2_shares=pos.tp2_shares,
                    tp3_shares=pos.tp3_shares,
                )

        pos.actual_entry_price = fill_price if fill_price > 0 else entry["entry_price"]
        pos.entry_shares       = fill_qty
        pos.remaining          = fill_qty
        pos.status             = "open"
        self._positions[symbol] = pos
        self._pending_entries.pop(symbol, None)
        self._update_pos(pos)

        # Place OCA bracket (v1 TP1-only) or stop-only (multi-leg)
        exit_side   = "sell" if pos.direction == 1 else "buy"
        is_tp1_only = (pos.tp1_shares == fill_qty)
        place_failed = False
        try:
            if is_tp1_only and hasattr(self._broker, "submit_oca_pair"):
                oca = self._broker.submit_oca_pair(
                    symbol=symbol,
                    side=exit_side,
                    qty=fill_qty,
                    tp1_limit_price=pos.tp1_price,
                    stop_price=pos.stop_price,
                    timeout=5.0,
                )
                pos.tp1_order_id  = oca.get("tp1_order_id")
                pos.stop_order_id = oca.get("stop_order_id")
                if self._log:
                    self._log.info(
                        "oca_bracket_placed",
                        symbol=symbol,
                        tp1_order_id=pos.tp1_order_id,
                        stop_order_id=pos.stop_order_id,
                        tp1_price=pos.tp1_price,
                        stop_price=pos.stop_price,
                        qty=fill_qty,
                    )
            else:
                stop_order = self._broker.submit_stop_order(
                    symbol=symbol,
                    side=exit_side,
                    qty=fill_qty,
                    stop_price=pos.stop_price,
                    client_order_id=f"stop-{symbol}-{pos.session_date.isoformat()}",
                    timeout=5.0,
                )
                pos.stop_order_id = stop_order.get("id")
                if self._log:
                    self._log.info(
                        "stop_placed",
                        symbol=symbol,
                        stop_order_id=pos.stop_order_id,
                        stop_price=pos.stop_price,
                        qty=fill_qty,
                    )
            self._update_pos(pos)
        except Exception as exc:
            place_failed = True
            if self._log:
                self._log.critical(
                    "stop_placement_failed_flattening_position",
                    symbol=symbol,
                    stop_price=pos.stop_price,
                    error=str(exc),
                )

        if place_failed:
            try:
                self._broker.submit_market_order(symbol, exit_side, fill_qty)
            except Exception as flatten_exc:
                if self._log:
                    self._log.critical("emergency_flatten_failed",
                                       symbol=symbol, error=str(flatten_exc))
            pos.status      = "closed"
            pos.exit_reason = "STOP_PLACEMENT_FAILED"
            self._update_pos(pos)
            return None

        return pos

    def _check_pending_entry(self, symbol: str, ts: datetime) -> None:
        """Poll a pending entry order and act on fill / cancel / timeout."""
        pending = self._pending_entries.get(symbol)
        if not pending:
            return

        order_id = pending["order_id"]
        try:
            order_state   = self._broker.get_order(order_id)
            broker_status = order_state.get("status", "new")
            fill_qty      = int(float(order_state.get("filled_qty", 0) or 0))
            fill_price    = float(order_state.get("filled_avg_price", 0) or 0)
        except Exception as exc:
            if self._log:
                self._log.warning("pending_entry_poll_failed",
                                  symbol=symbol, exc=str(exc))
            return

        if broker_status in ("filled", "partially_filled") and fill_qty > 0:
            self._complete_entry_fill(
                symbol=symbol, fill_qty=fill_qty, fill_price=fill_price,
                pos=pending["pos"], entry=pending["entry"],
                side=pending["side"], session_date=pending["session_date"],
                attempt=pending["attempt"], order_id=order_id,
                submitted_at=pending["submitted_at"],
            )
            return

        if broker_status in ("cancelled", "expired", "rejected", "inactive"):
            if pending["attempt"] < self._config.entry_repeg_max_attempts:
                self._repeg_entry(symbol, pending)
            else:
                self._store.close_position(symbol)
                self._positions.pop(symbol, None)
                del self._pending_entries[symbol]
            return

        # Still working — check repeg timeout
        elapsed = _time.monotonic() - pending["submit_time"]
        if elapsed > self._config.entry_repeg_seconds:
            try:
                self._broker.cancel_order(order_id)
            except Exception:
                pass
            if pending["attempt"] < self._config.entry_repeg_max_attempts:
                self._repeg_entry(symbol, pending)
            else:
                self._store.close_position(symbol)
                self._positions.pop(symbol, None)
                del self._pending_entries[symbol]

    def _repeg_entry(self, symbol: str, pending: dict) -> None:
        """Cancel current entry order and resubmit at wider slippage BPS."""
        cfg     = self._config
        attempt = pending["attempt"] + 1
        side    = pending["side"]
        entry   = pending["entry"]

        try:
            limit_price = self._policy.compute_entry_limit(
                side, symbol, entry["entry_price"], bps=cfg.entry_slippage_max_bps
            )
            client_id = str(uuid.uuid4())
            order     = self._broker.submit_limit_order(
                symbol, side, entry["original_qty"], limit_price, client_id
            )
            order_id = order.get("id", client_id)
        except Exception as exc:
            if self._log:
                self._log.critical("entry_repeg_failed", symbol=symbol, exc=str(exc))
            self._store.close_position(symbol)
            self._positions.pop(symbol, None)
            del self._pending_entries[symbol]
            return

        pending["order_id"]    = order_id
        pending["attempt"]     = attempt
        pending["submit_time"] = _time.monotonic()
        pending["limit_price"] = limit_price

    # ── EOD sweep (runner calls this at 16:00:30 ET) ──────────────────────────

    def flatten_all(self, reason: str = "manual") -> None:
        """
        Emergency/EOD market exit for every open position.

        Called by the runner at 16:00:30 ET (production EOD mechanism),
        on session_kill, on SIGTERM/SIGINT, or manually.

        Also cancels pending entry orders and flattens any untracked broker
        positions in the universe (Fix 4 — safety net for undetected fills).
        """
        eod_ts = datetime.now(UTC)

        # 1. Exit tracked open positions
        for symbol in list(self._positions.keys()):
            pos = self._positions.get(symbol)
            if pos is None or pos.status != "open":
                continue
            exit_reason = ("TP2"      if pos.tp2_hit else
                           "TP1_ONLY" if pos.tp1_hit else "EOD")
            try:
                self._exit_all(pos, symbol, leg="eod", ref_price=None,
                               exit_reason=f"{exit_reason}_{reason}",
                               exit_time=eod_ts,
                               session_date=pos.session_date)
            except Exception as exc:
                if self._log:
                    self._log.critical("flatten_all_exception",
                                       symbol=symbol, reason=reason,
                                       exc=str(exc))

        # 2. Cancel any pending entry orders and clean up
        for sym in list(self._pending_entries.keys()):
            pending = self._pending_entries[sym]
            try:
                self._broker.cancel_order(pending["order_id"])
            except Exception:
                pass
            try:
                self._store.close_position(sym)
            except Exception:
                pass
            self._positions.pop(sym, None)
        self._pending_entries.clear()

        # 3. Flatten untracked broker positions in the universe (Fill 4 safety net)
        tracked = set(self._positions.keys())
        for sym in self._universe:
            if sym in tracked:
                continue
            try:
                bp = self._broker.get_position(sym)
                if not bp:
                    continue
                broker_qty = float(bp.get("qty", 0))
                if abs(broker_qty) == 0:
                    continue
                flat_side = "sell" if broker_qty > 0 else "buy"
                flat_qty  = int(abs(broker_qty))
                self._broker.submit_market_order(sym, flat_side, flat_qty)
                if self._log:
                    self._log.critical(
                        "flatten_untracked_broker_position",
                        symbol=sym, qty=flat_qty, reason=reason,
                    )
            except Exception as exc:
                if self._log:
                    self._log.critical(
                        "flatten_untracked_broker_error",
                        symbol=sym, reason=reason, error=str(exc),
                    )

    # ── Startup cleanup ────────────────────────────────────────────────────────

    def startup_reconcile(self) -> None:
        """Purge stale DB rows so they can't block today's entries.

        Must be called unconditionally at session start (not gated by --recover):
          1. Non-'open' rows (entering/unfilled) are abandoned attempts — delete.
          2. 'open' rows with no matching broker position are ghosts — delete.

        Does NOT load surviving positions into memory; use reconcile_from_broker()
        for crash-recovery (--recover path) which requires full position reload.
        """
        # Step 1: delete non-open rows
        purged = self._store.purge_stale_entering_rows()
        for sym in purged:
            if self._log:
                self._log.warning("startup_purged_stale_row", symbol=sym)

        # Step 2: delete 'open' rows that no longer have a broker position
        for sym in list(self._store.all_open_symbols()):
            try:
                broker_pos = self._broker.get_position(sym)
                broker_qty = abs(int(float((broker_pos or {}).get("qty", 0))))
            except Exception as exc:
                if self._log:
                    self._log.warning(
                        "startup_reconcile_broker_error", symbol=sym, exc=str(exc)
                    )
                continue
            if broker_qty == 0:
                self._store.close_position(sym)
                if self._log:
                    self._log.warning("startup_purged_orphan_position", symbol=sym)

    # ── Broker reconciliation ─────────────────────────────────────────────────

    def reconcile_from_broker(self) -> dict[str, str]:
        """
        Sync state_store open_positions against broker on startup
        or after detected network gap.

        Returns {symbol: action_taken} for every symbol examined.
        """
        results: dict[str, str] = {}
        for sym in self._store.all_open_symbols():
            db_pos = self._store.get_open_position(sym)
            if not db_pos or db_pos.get("status") != "open":
                results[sym] = "skipped_not_open"
                continue

            broker_pos = self._broker.get_position(sym)
            db_remaining = int(db_pos.get("remaining") or db_pos.get("qty", 0))

            if broker_pos is None:
                broker_qty = 0
            else:
                broker_qty = abs(int(float(broker_pos.get("qty", 0))))

            if broker_qty < db_remaining:
                delta = db_remaining - broker_qty
                self._store.update_open_position(
                    sym,
                    remaining=broker_qty,
                    status="open" if broker_qty > 0 else "closed",
                )
                self._store.log_alert(
                    level="WARN",
                    message=(
                        f"reconciliation_delta: {sym} db_remaining={db_remaining} "
                        f"broker_qty={broker_qty} delta={delta} "
                        "inferred_exit_reason=external_fill"
                    ),
                    category="reconciliation",
                    symbol=sym,
                )
                results[sym] = f"adjusted_down_{delta}"

            elif broker_qty > db_remaining:
                self._store.log_alert(
                    level="CRITICAL",
                    message=(
                        f"reconciliation_delta: {sym} broker_qty={broker_qty} "
                        f"> db_remaining={db_remaining} — SHOULD NOT HAPPEN"
                    ),
                    category="reconciliation",
                    symbol=sym,
                )
                if self._log:
                    self._log.critical("reconciliation_broker_more_than_db",
                                       symbol=sym, broker=broker_qty, db=db_remaining)
                self._gate.record_realized_pnl(0.0)   # trigger kill-check
                results[sym] = "critical_broker_overage"
            else:
                results[sym] = "ok"

        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _handle_stop_fired(
        self, pos: Position, symbol: str, stop_state: dict
    ) -> None:
        """Called when polling detects the exchange-resident stop has filled.

        IB already executed the stop. This method does NOT submit orders —
        it records what happened and archives the trade.

        exit_reason mirrors the (now-removed) bar-by-bar check:
          TRAIL    → TP1 hit and trail mode active
          TP1_ONLY → TP1 hit in breakeven mode
          STOP     → stopped out before any TP1
        """
        fill_qty   = int(float(stop_state.get("filled_qty",   0) or 0))
        fill_price = float(stop_state.get("filled_avg_price", 0) or 0)

        exit_reason = (
            "TRAIL"    if (pos.use_trail_atp1 and pos.tp1_hit) else
            "TP1_ONLY" if pos.tp1_hit else
            "STOP"
        )

        pos.status      = "closed"
        pos.exit_reason = exit_reason
        pos.exit_price  = fill_price if fill_price > 0 else pos.current_stop
        pos.exit_time   = datetime.now(UTC)
        pos.remaining   = max(0, pos.remaining - fill_qty)

        self._update_pos(pos)
        self._store.save_closed_trade(
            trade_date=pos.session_date,
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.actual_entry_price,
            exit_price=pos.exit_price,
            realized_exit_price=fill_price if fill_price > 0 else pos.current_stop,
            qty=pos.entry_shares,
            exit_reason=exit_reason,
            opened_at=datetime.now(UTC),
        )
        self._store.close_position(symbol)
        del self._positions[symbol]

        if self._log:
            self._log.info(
                "stop_fired_via_polling",
                symbol=symbol,
                stop_order_id=pos.stop_order_id,
                fill_qty=fill_qty,
                fill_price=fill_price,
                intended_stop=pos.current_stop,
                exit_reason=exit_reason,
            )

    def _recover_stop_after_modify_failure(
        self, pos: Position, symbol: str
    ) -> None:
        """Recovery when modify_stop_order fails after TP1.

        1. Cancel the now-incorrect stop (wrong qty/price)
        2. Place a fresh stop with correct post-TP1 params
        3. If both fail, flatten the remaining position via market exit
        """
        exit_side = "sell" if pos.direction == 1 else "buy"

        try:
            self._broker.cancel_order(pos.stop_order_id, timeout=5.0)
        except Exception as exc:
            if self._log:
                self._log.critical(
                    "stop_cancel_failed_during_recovery",
                    symbol=symbol,
                    error=str(exc),
                )

        try:
            fresh = self._broker.submit_stop_order(
                symbol=symbol,
                side=exit_side,
                qty=pos.remaining,
                stop_price=pos.current_stop,
                client_order_id=f"stop-recovered-{symbol}-{pos.session_date.isoformat()}",
                timeout=5.0,
            )
            pos.stop_order_id = fresh.get("id")
            self._update_pos(pos)
            if self._log:
                self._log.warning(
                    "stop_replaced_after_modify_failure",
                    symbol=symbol,
                    new_stop_order_id=pos.stop_order_id,
                )
            return
        except Exception as exc:
            if self._log:
                self._log.critical(
                    "stop_replace_failed_flattening_remaining",
                    symbol=symbol,
                    error=str(exc),
                )

        # Last resort: flatten the unprotected remaining position
        try:
            self._broker.submit_market_order(symbol, exit_side, pos.remaining)
        except Exception as flatten_exc:
            if self._log:
                self._log.critical(
                    "emergency_flatten_failed_after_modify_recovery",
                    symbol=symbol,
                    error=str(flatten_exc),
                )
        pos.status        = "closed"
        pos.exit_reason   = "STOP_RECOVERY_FAILED"
        pos.stop_order_id = None
        self._update_pos(pos)
        self._store.save_closed_trade(
            trade_date=pos.session_date,
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.actual_entry_price,
            exit_price=pos.current_stop,
            realized_exit_price=pos.current_stop,
            qty=pos.entry_shares,
            exit_reason="STOP_RECOVERY_FAILED",
            opened_at=datetime.now(UTC),
        )
        self._store.close_position(symbol)
        del self._positions[symbol]

    def _exit_partial(
        self,
        pos: Position,
        symbol: str,
        leg: str,
        qty: int,
        ref_price: float,
        session_date: date,
    ) -> "Fill":
        """Submit exit order for a partial lot (TP1/TP2).  Log on failure."""
        side = "sell" if pos.direction == 1 else "buy"
        try:
            fill = self._policy.sell(symbol, qty, leg, ref_price, session_date) \
                   if pos.direction == 1 else \
                   self._policy.buy(symbol, qty, leg, ref_price, session_date)
            pnl = (fill.avg_price - pos.actual_entry_price) * fill.qty * pos.direction
            self._gate.record_realized_pnl(pnl)
            return fill
        except Exception as exc:
            if self._log:
                self._log.critical("exit_partial_failed_market_fallback",
                                   symbol=symbol, leg=leg, exc=str(exc))
            # Market fallback
            fallback_side = "sell" if pos.direction == 1 else "buy"
            order = self._broker.submit_market_order(symbol, fallback_side, qty)
            order_id = order.get("id", "")
            from orb_live.execution.order_policy import Fill
            from datetime import timezone
            f = Fill(symbol=symbol, side=fallback_side, qty=qty,
                     avg_price=ref_price, order_id=order_id, leg=leg,
                     attempts=99, reason="market_fallback",
                     submitted_at=datetime.now(timezone.utc), filled_at=None)
            pnl = (f.avg_price - pos.actual_entry_price) * f.qty * pos.direction
            self._gate.record_realized_pnl(pnl)
            return f

    def _exit_all(
        self,
        pos: Position,
        symbol: str,
        leg: str,
        ref_price: Optional[float],
        exit_reason: str,
        exit_time: datetime,
        session_date: date,
    ) -> None:
        """Submit exit for all remaining shares and close the position."""
        if pos.remaining <= 0:
            return

        side = "sell" if pos.direction == 1 else "buy"
        try:
            if ref_price is None:
                # Market order (no price reference)
                order = self._broker.submit_market_order(
                    symbol, side, pos.remaining
                )
                order_id = order.get("id", "")
                from orb_live.execution.order_policy import Fill
                fill = Fill(symbol=symbol, side=side, qty=pos.remaining,
                            avg_price=0.0, order_id=order_id, leg=leg,
                            attempts=1, reason="market",
                            submitted_at=datetime.now(UTC), filled_at=None)
            else:
                fill = (self._policy.sell(symbol, pos.remaining, leg,
                                          ref_price, session_date)
                        if pos.direction == 1 else
                        self._policy.buy(symbol, pos.remaining, leg,
                                         ref_price, session_date))
        except Exception as exc:
            if self._log:
                self._log.critical("exit_all_failed_market_fallback",
                                   symbol=symbol, leg=leg, exc=str(exc))
            order = self._broker.submit_market_order(symbol, side, pos.remaining)
            from orb_live.execution.order_policy import Fill
            fill = Fill(symbol=symbol, side=side, qty=pos.remaining,
                        avg_price=ref_price or 0.0,
                        order_id=order.get("id", ""), leg=leg,
                        attempts=99, reason="market_fallback",
                        submitted_at=datetime.now(UTC), filled_at=None)

        pnl = (fill.avg_price - pos.actual_entry_price) * fill.qty * pos.direction
        self._gate.record_realized_pnl(pnl)

        pos.remaining    = 0
        pos.tp3_hit      = (leg == "tp3")
        pos.exit_reason  = exit_reason
        pos.exit_price   = fill.avg_price
        pos.exit_time    = exit_time
        pos.status       = "closed"

        # Persist closed state then archive
        self._update_pos(pos)
        self._store.save_closed_trade(
            trade_date=session_date,
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.actual_entry_price,
            exit_price=fill.avg_price,
            realized_exit_price=fill.avg_price,
            qty=pos.entry_shares,
            exit_reason=exit_reason,
            opened_at=datetime.now(UTC),
        )
        self._store.close_position(symbol)
        del self._positions[symbol]

    def _persist_new(self, pos: Position, session_date: date) -> None:
        """Write initial position row (status='entering') to state_store."""
        self._store.save_open_position(
            symbol=pos.symbol,
            trade_date=session_date,
            direction=pos.direction,
            status=pos.status,
            entry_price=pos.entry_price,
            actual_entry_price=pos.actual_entry_price,
            qty=float(pos.entry_shares),
            entry_shares=pos.entry_shares,
            remaining=pos.remaining,
            orb_range=pos.orb_range,
            stop_price=pos.stop_price,
            current_stop=pos.current_stop,
            tp1_price=pos.tp1_price,
            tp2_price=pos.tp2_price,
            tp1_shares=pos.tp1_shares,
            tp2_shares=pos.tp2_shares,
            tp3_shares=pos.tp3_shares,
            tp1_hit=pos.tp1_hit,
            tp2_hit=pos.tp2_hit,
            tp3_hit=pos.tp3_hit,
            use_trail_atp1=pos.use_trail_atp1,
            trail_atp1_dist=pos.trail_atp1_dist,
            max_fav=pos.max_fav,
            post_tp2_mfe=pos.post_tp2_mfe,
            decision_reason=pos.decision_reason,
            tp1_order_id=pos.tp1_order_id,
        )

    def _update_pos(self, pos: Position) -> None:
        """Update mutable fields of position row in state_store."""
        self._store.update_open_position(
            pos.symbol,
            status=pos.status,
            actual_entry_price=pos.actual_entry_price,
            entry_shares=pos.entry_shares,
            remaining=pos.remaining,
            current_stop=pos.current_stop,
            stop_order_id=pos.stop_order_id,
            tp1_order_id=pos.tp1_order_id,
            tp1_hit=pos.tp1_hit,
            tp2_hit=pos.tp2_hit,
            tp3_hit=pos.tp3_hit,
            trail_atp1_peak=pos.trail_atp1_peak,
            max_fav=pos.max_fav,
            post_tp2_mfe=pos.post_tp2_mfe,
            decision_reason=pos.decision_reason,
            exit_reason=pos.exit_reason,
            exit_price=pos.exit_price,
            exit_time=pos.exit_time,
            tp1_shares=pos.tp1_shares,
            tp2_shares=pos.tp2_shares,
            tp3_shares=pos.tp3_shares,
        )
