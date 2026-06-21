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

        entry — dict from compute_entry().  entry['shares'] is what we try to
                fill; actual fill may be partial or zero (slippage / thin book).

        Returns Position on open/partial fill.  Returns None if:
          - Risk gate rejects the entry.
          - Fill qty == 0 (unfilled).

        Position sizing uses CURRENT ACCOUNT EQUITY from broker.get_account(),
        not a per-symbol equity pool (intentional divergence from backtest).
        """
        cfg = self._config

        # Build open_positions notional for exposure check
        open_notional = {
            sym: (pos.actual_entry_price * pos.remaining)
            for sym, pos in self._positions.items()
            if pos.status == "open"
        }

        # Gate check
        current_equity  = float(self._broker.get_account().get("equity", 0))
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

        # Build position in 'entering' state and persist BEFORE order
        override = entry.get("exit_override", {})
        use_trail = (override.get("method") == "trail_after_tp1")
        trail_mult = float(override.get("mult", 1.0)) if use_trail else 0.0
        trail_dist = trail_mult * entry["orb_range"]

        pos = Position(
            symbol=symbol,
            session_date=session_date,
            direction=gap_direction,
            entry_price=entry["entry_price"],
            actual_entry_price=entry["entry_price"],  # placeholder until fill
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

        # Submit entry order
        side = "buy" if gap_direction == 1 else "sell"
        try:
            fill = self._policy.buy(symbol, entry["shares"], "entry",
                                    reference_price=entry["entry_price"],
                                    session_date=session_date) \
                   if gap_direction == 1 else \
                   self._policy.sell(symbol, entry["shares"], "entry",
                                     reference_price=entry["entry_price"],
                                     session_date=session_date)
        except Exception as exc:
            if self._log:
                self._log.critical("entry_order_exception", symbol=symbol,
                                   exc=str(exc))
            pos.status = "unfilled"
            pos.decision_reason = "entry_exception"
            self._update_pos(pos)
            return None

        if fill.qty == 0:
            pos.status = "unfilled"
            pos.decision_reason = "entry_unfilled"
            self._update_pos(pos)
            return None

        # Adjust share counts for partial fill
        if fill.qty < entry["shares"]:
            r_tp1 = override.get("exit_ratio_tp1", cfg.exit_ratio_tp1)
            r_tp2 = override.get("exit_ratio_tp2", cfg.exit_ratio_tp2)
            pos.tp1_shares = math.floor(fill.qty * r_tp1)
            pos.tp2_shares = math.floor(fill.qty * r_tp2)
            pos.tp3_shares = max(0, fill.qty - pos.tp1_shares - pos.tp2_shares)
            if self._log:
                self._log.warning(
                    "partial_entry_fill",
                    symbol=symbol,
                    requested=entry["shares"],
                    filled=fill.qty,
                    tp1_shares=pos.tp1_shares,
                    tp2_shares=pos.tp2_shares,
                    tp3_shares=pos.tp3_shares,
                )

        pos.actual_entry_price = fill.avg_price
        pos.entry_shares       = fill.qty
        pos.remaining          = fill.qty
        pos.status             = "open"
        self._positions[symbol] = pos
        self._update_pos(pos)

        # ── Place exchange-resident stop ──────────────────────────────────────
        # Stop-market at the strategy's computed stop_price. Sits at IB;
        # fires automatically if price crosses the level. Matches the
        # backtest's guaranteed-exit-at-stop assumption.
        exit_side = "sell" if pos.direction == 1 else "buy"
        try:
            stop_order = self._broker.submit_stop_order(
                symbol=symbol,
                side=exit_side,
                qty=fill.qty,
                stop_price=pos.stop_price,
                client_order_id=f"stop-{symbol}-{pos.session_date.isoformat()}",
                timeout=5.0,
            )
            pos.stop_order_id = stop_order.get("id")
            self._update_pos(pos)
            if self._log:
                self._log.info(
                    "stop_placed",
                    symbol=symbol,
                    stop_order_id=pos.stop_order_id,
                    stop_price=pos.stop_price,
                    qty=fill.qty,
                )
        except Exception as exc:
            # Stop placement failure is SERIOUS — position is live without
            # protection. Flatten immediately rather than continuing exposed.
            if self._log:
                self._log.critical(
                    "stop_placement_failed_flattening_position",
                    symbol=symbol,
                    stop_price=pos.stop_price,
                    error=str(exc),
                )
            try:
                self._broker.submit_market_order(symbol, exit_side, fill.qty)
            except Exception as flatten_exc:
                if self._log:
                    self._log.critical(
                        "emergency_flatten_failed",
                        symbol=symbol,
                        error=str(flatten_exc),
                    )
            pos.status      = "closed"
            pos.exit_reason = "STOP_PLACEMENT_FAILED"
            self._update_pos(pos)
            return None

        return pos

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
            tp1_hit = (
                (pos.direction == 1  and hi >= pos.tp1_price) or
                (pos.direction == -1 and lo <= pos.tp1_price)
            )
            if tp1_hit:
                self._exit_partial(pos, symbol, leg="tp1", qty=pos.tp1_shares,
                                   ref_price=pos.tp1_price,
                                   session_date=pos.session_date)
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

                # Propagate post-TP1 stop change to IB: qty drops to remaining,
                # price moves to breakeven (or trail init). Both changes atomic.
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

    # ── EOD sweep (runner calls this at 16:00:30 ET) ──────────────────────────

    def flatten_all(self, reason: str = "manual") -> None:
        """
        Emergency/EOD market exit for every open position.

        Called by the runner at 16:00:30 ET (production EOD mechanism),
        on session_kill, on SIGTERM/SIGINT, or manually.

        Uses exit_reason logic that mirrors the on_bar EOD branch so that
        the trade record is consistent whether closed by on_bar or by this
        method.
        """
        eod_ts = datetime.now(UTC)
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
            qty=pos.entry_shares,
            exit_reason=exit_reason,
            opened_at=datetime.now(UTC),  # opened_at not tracked here
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
