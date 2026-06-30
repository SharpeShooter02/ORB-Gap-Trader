"""
execution/order_policy.py — Marketable-limit order policy.

All entry and exit orders are submitted as limit orders priced to cross the
spread immediately ("marketable limit").  This prevents runaway market orders
while ensuring fills at a bounded price.  See MarketableLimitPolicy docstring
for the exact repeg and fallback rules.

Fill dataclass is the canonical record of every order outcome; it is persisted
to state_store.fills on completion before being returned to the caller.
"""

from __future__ import annotations

import json
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from orb_live.config.live_config import LiveConfig
    from orb_live.core.state_store import StateStore

UTC = timezone.utc

_POLL_INTERVAL = 0.5   # seconds between status checks (overridable in tests)


# ── Fill dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Fill:
    """Canonical record of a completed order leg."""
    symbol:            str
    side:              str           # 'buy' / 'sell'
    qty:               int           # actually filled shares (0 if unfilled)
    avg_price:         float         # average fill price (0.0 if unfilled)
    order_id:          str
    leg:               str           # entry / tp1 / tp2 / tp3 / stop / eod
    attempts:          int           # repeg attempts made (1 = first try)
    reason:            str           # filled / partial_unfilled / unfilled
    submitted_at:      datetime
    filled_at:         Optional[datetime]
    raw_response_json: str = field(default="{}")


# ── MarketableLimitPolicy ──────────────────────────────────────────────────────

class MarketableLimitPolicy:
    """
    Synchronous marketable-limit order policy.

    All methods return a Fill or raise.  The caller (LivePositionManager)
    must handle the raise for exit legs and fall back to market orders.

    Config consumed from live_cfg (LiveConfig):
        entry_slippage_bps      — bps above ask (buy) / below bid (sell)
        entry_repeg_seconds     — poll window per attempt
        entry_repeg_max_attempts— max repegs before accepting partial
        entry_slippage_max_bps  — bps for final repeg
        exit_slippage_bps       — bps for TP/EOD exit limit
        stop_order_type         — "market" or "stop_limit"

    Broker interface expected (duck-typed, MockBroker in tests):
        get_latest_quote(symbol)         → dict with 'bid', 'ask'
        submit_limit_order(symbol, side, qty, limit_price, client_order_id)
        submit_market_order(symbol, side, qty, client_order_id)
        get_order(order_id)              → dict with 'status', 'filled_qty', 'filled_avg_price'
        cancel_order(order_id)           → None
    """

    def __init__(
        self,
        broker,
        live_cfg: "LiveConfig",
        state_store: "StateStore",
        logger=None,
        _sleep: Callable = _time.sleep,   # injectable for tests
    ):
        self._broker  = broker
        self._cfg     = live_cfg
        self._store   = state_store
        self._log     = logger
        self._sleep   = _sleep

    # ── Public API ─────────────────────────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        qty: int,
        leg: str,
        reference_price: Optional[float] = None,
        session_date=None,
    ) -> Fill:
        """Submit a marketable-limit buy order and return Fill."""
        return self._submit("buy", symbol, qty, leg, reference_price, session_date)

    def sell(
        self,
        symbol: str,
        qty: int,
        leg: str,
        reference_price: Optional[float] = None,
        session_date=None,
    ) -> Fill:
        """Submit a marketable-limit sell order and return Fill."""
        return self._submit("sell", symbol, qty, leg, reference_price, session_date)

    def compute_entry_limit(
        self,
        side: str,
        symbol: str,
        reference_price: Optional[float] = None,
        bps: Optional[int] = None,
    ) -> float:
        """Return the marketable limit price for an entry order (no submission)."""
        _bps = bps if bps is not None else self._cfg.entry_slippage_bps
        return self._compute_limit(side, symbol, _bps, reference_price)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _submit(
        self,
        side: str,
        symbol: str,
        qty: int,
        leg: str,
        reference_price: Optional[float],
        session_date,
    ) -> Fill:
        if leg == "stop":
            return self._submit_stop(side, symbol, qty, leg, reference_price, session_date)
        if leg in ("entry",):
            return self._submit_entry(side, symbol, qty, leg, reference_price, session_date)
        return self._submit_exit(side, symbol, qty, leg, reference_price, session_date)

    def _submit_entry(self, side, symbol, qty, leg, reference_price, session_date) -> Fill:
        """
        Entry leg: poll for entry_repeg_seconds per attempt.
        Up to entry_repeg_max_attempts repegs at expanding bps.
        Never converts to market.
        """
        cfg       = self._cfg
        submitted = datetime.now(UTC)
        attempts  = 0
        filled_qty  = 0
        filled_price = 0.0
        order_id    = ""

        while attempts < cfg.entry_repeg_max_attempts:
            attempts += 1
            bps = cfg.entry_slippage_bps if attempts == 1 else cfg.entry_slippage_max_bps
            limit = self._compute_limit(side, symbol, bps, reference_price)
            client_id = str(uuid.uuid4())
            order = self._broker.submit_limit_order(symbol, side, qty - filled_qty, limit, client_id)
            order_id = order.get("id", client_id)

            # Poll until filled or timeout
            deadline = _time.monotonic() + cfg.entry_repeg_seconds
            while _time.monotonic() < deadline:
                self._sleep(_POLL_INTERVAL)
                status = self._broker.get_order(order_id)
                if status.get("status") in ("filled", "partially_filled", "cancelled", "expired"):
                    break

            fq = int(float(status.get("filled_qty", 0) or 0))
            fp = float(status.get("filled_avg_price", 0) or 0)

            if fq > 0:
                filled_qty   += fq
                filled_price  = (filled_price * (filled_qty - fq) + fp * fq) / filled_qty

            remaining = qty - filled_qty
            if remaining <= 0:
                break

            # Cancel unfilled remainder before repegging
            self._broker.cancel_order(order_id)

        if filled_qty == 0:
            reason = "unfilled"
        elif filled_qty < qty:
            reason = "partial_unfilled"
        else:
            reason = "filled"

        return self._make_fill(symbol, side, filled_qty, filled_price, order_id, leg,
                               attempts, reason, submitted, session_date)

    def _submit_exit(self, side, symbol, qty, leg, reference_price, session_date) -> Fill:
        """
        Exit leg (tp1/tp2/tp3/eod): up to 2 repegs at 1.5× and 2× exit_slippage_bps.
        If still unfilled after 2 repegs, raises RuntimeError (caller falls back to market).
        """
        cfg        = self._cfg
        submitted  = datetime.now(UTC)
        filled_qty  = 0
        filled_price = 0.0
        order_id    = ""

        for attempt in range(1, 4):   # attempts 1, 2, 3 → bps 1×, 1.5×, 2×
            bps_mult = 1.0 + 0.5 * (attempt - 1)
            bps = int(cfg.exit_slippage_bps * bps_mult)
            limit = self._compute_limit(side, symbol, bps, reference_price)
            client_id = str(uuid.uuid4())
            order = self._broker.submit_limit_order(symbol, side, qty, limit, client_id)
            order_id = order.get("id", client_id)

            deadline = _time.monotonic() + cfg.entry_repeg_seconds
            while _time.monotonic() < deadline:
                self._sleep(_POLL_INTERVAL)
                status = self._broker.get_order(order_id)
                if status.get("status") in ("filled", "partially_filled", "cancelled", "expired"):
                    break

            fq = int(float(status.get("filled_qty", 0) or 0))
            fp = float(status.get("filled_avg_price", 0) or 0)
            if fq >= qty:
                filled_qty   = fq
                filled_price = fp
                break
            if fq > 0:
                # Partial on exit — cancel and retry with remaining
                filled_qty   += fq
                filled_price = (filled_price * (filled_qty - fq) + fp * fq) / filled_qty
                qty -= fq
                self._broker.cancel_order(order_id)

        if filled_qty < qty and attempt >= 3:
            if self._log:
                self._log.critical("exit_order_failed_after_repegs",
                                   symbol=symbol, leg=leg, qty=qty)
            raise RuntimeError(
                f"Exit order for {symbol} leg={leg} failed after 3 attempts "
                "(caller must submit market fallback)"
            )

        return self._make_fill(symbol, side, filled_qty, filled_price, order_id, leg,
                               attempt, "filled", submitted, session_date)

    def _submit_stop(self, side, symbol, qty, leg, reference_price, session_date) -> Fill:
        """
        Stop leg: market order if stop_order_type=='market'; else limit with 2× bps,
        2 repegs, then ALWAYS falls back to market (stops never fail silently).
        """
        cfg       = self._cfg
        submitted = datetime.now(UTC)

        if cfg.stop_order_type == "market":
            client_id = str(uuid.uuid4())
            order = self._broker.submit_market_order(symbol, side, qty, client_id)
            order_id = order.get("id", client_id)
            # Wait for fill (market orders are typically immediate).
            # Break on partially_filled too: a partial stop exit is better
            # than an indefinite wait when liquidity is thin.
            deadline = _time.monotonic() + cfg.entry_repeg_seconds * 2
            while _time.monotonic() < deadline:
                self._sleep(_POLL_INTERVAL)
                status = self._broker.get_order(order_id)
                if status.get("status") in ("filled", "partially_filled", "cancelled", "expired"):
                    break
            fq = int(float(status.get("filled_qty", 0) or 0))
            fp = float(status.get("filled_avg_price", 0) or 0)
            return self._make_fill(symbol, side, fq, fp, order_id, leg,
                                   1, "filled", submitted, session_date)

        # stop_limit mode: try limit first, fall back to market
        try:
            fill = self._submit_exit(side, symbol, qty, leg, reference_price, session_date)
            return fill
        except RuntimeError:
            if self._log:
                self._log.critical("stop_limit_failed_falling_back_to_market",
                                   symbol=symbol, qty=qty)
            client_id = str(uuid.uuid4())
            order = self._broker.submit_market_order(symbol, side, qty, client_id)
            order_id = order.get("id", client_id)
            deadline = _time.monotonic() + cfg.entry_repeg_seconds * 2
            while _time.monotonic() < deadline:
                self._sleep(_POLL_INTERVAL)
                status = self._broker.get_order(order_id)
                if status.get("status") in ("filled", "partially_filled", "cancelled", "expired"):
                    break
            fq = int(float(status.get("filled_qty", 0) or 0))
            fp = float(status.get("filled_avg_price", 0) or 0)
            return self._make_fill(symbol, side, fq, fp, order_id, leg,
                                   3, "filled", submitted, session_date)

    def _compute_limit(self, side: str, symbol: str, bps: int,
                       reference_price: Optional[float]) -> float:
        """Compute a marketable limit price from NBBO or reference_price."""
        if reference_price is None:
            q = self._broker.get_latest_quote(symbol)
            reference_price = float(q.get("ask", 0)) if side == "buy" else float(q.get("bid", 0))
        factor = bps / 10_000.0
        if side == "buy":
            return reference_price * (1.0 + factor)
        return reference_price * (1.0 - factor)

    def _make_fill(
        self,
        symbol: str,
        side: str,
        qty: int,
        avg_price: float,
        order_id: str,
        leg: str,
        attempts: int,
        reason: str,
        submitted: datetime,
        session_date,
    ) -> Fill:
        filled_at = datetime.now(UTC) if qty > 0 else None
        f = Fill(
            symbol=symbol, side=side, qty=qty, avg_price=avg_price,
            order_id=order_id, leg=leg, attempts=attempts, reason=reason,
            submitted_at=submitted, filled_at=filled_at,
        )
        if session_date is not None and self._store is not None:
            self._store.save_fill(
                session_date=session_date,
                symbol=symbol,
                side=side,
                qty=qty,
                avg_price=avg_price,
                order_id=order_id,
                leg=leg,
                attempts=attempts,
                reason=reason,
                submitted_at=submitted,
                filled_at=filled_at,
                raw_response_json=json.dumps({}),
            )
        return f
