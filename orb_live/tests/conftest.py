"""
tests/conftest.py — Shared pytest fixtures.
"""

import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import pytest

import orb_live  # noqa: F401 — sys.path setup


@pytest.fixture(scope="session")
def live_cfg():
    from orb_live.config.live_config import load_live_config
    # use_rolling=False: frozen SIGMA seeds, no parquet access.
    # Keeps all tests independent of parquet presence on any machine.
    return load_live_config(use_rolling=False)


@pytest.fixture
def tmp_store(tmp_path):
    from orb_live.core.state_store import StateStore
    return StateStore(tmp_path / "test.db")


@pytest.fixture
def market_clock():
    from orb_live.core.clock import MarketClock
    return MarketClock()


@pytest.fixture
def today():
    return date.today()


# ── MockBroker ─────────────────────────────────────────────────────────────────

class MockBroker:
    """
    Deterministic broker mock for execution-layer tests.

    Default behavior: all orders fill immediately at the submitted limit price
    (fill_fraction=1.0).  Use set_fill_fraction() for all-orders override, or
    set_fill_sequence() to control successive submit_limit_order calls
    individually (values are popped from the front).

    Market orders always fill immediately at bid (sell) or ask (buy).

    Duck-typed to match the interface consumed by MarketableLimitPolicy and
    LivePositionManager:
        get_account()                   → {"equity": float}
        get_latest_quote(sym)           → {"bid": float, "ask": float}
        submit_limit_order(sym, side, qty, limit_price, client_order_id)
        submit_market_order(sym, side, qty, client_order_id=None)
        get_order(order_id)             → {"status", "filled_qty", "filled_avg_price"}
        cancel_order(order_id)
        get_position(sym)               → {"qty": int} | None
    """

    def __init__(self):
        self._equity: float = 100_000.0
        self._quote: dict = {"bid": 99.90, "ask": 100.10}
        self._orders: dict = {}
        self._positions: dict = {}
        self._fill_sequence: list = []
        self._default_fill_fraction: float = 1.0
        self._oca_siblings: dict = {}   # order_id → sibling_order_id (bidirectional)
        self._pending_fill_polls: int = 0  # >0 → next submit_limit_order fills async

    # ── Configuration helpers ─────────────────────────────────────────────────

    def set_fill_fraction(self, fraction: float) -> None:
        """Apply this fraction to every subsequent submit_limit_order call."""
        self._default_fill_fraction = fraction

    def set_fill_sequence(self, *fractions: float) -> None:
        """Per-order fill fractions, consumed left-to-right on each submit."""
        self._fill_sequence = list(fractions)

    def set_pending_fill(self, n_polls: int = 1) -> None:
        """Next submit_limit_order returns 'new' for n_polls get_order calls, then fills."""
        self._pending_fill_polls = n_polls

    def set_equity(self, equity: float) -> None:
        self._equity = equity

    def set_quote(self, bid: float, ask: float) -> None:
        self._quote = {"bid": bid, "ask": ask}

    def set_broker_position(self, symbol: str, qty: int) -> None:
        if qty:
            self._positions[symbol] = {"qty": qty}
        else:
            self._positions.pop(symbol, None)

    # ── Broker API ────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        return {
            "equity": self._equity,
            "buying_power": self._equity * 4,
            "available_funds": self._equity,
        }

    def check_margin(self, symbol: str, side: str, qty: float, price: float) -> dict:
        # Default mock: 50% initial margin (2x leveraged-ETF-like). Tests can
        # override self._margin_rate to simulate elevated (e.g. 3x) requirements.
        rate     = getattr(self, "_margin_rate", 0.5)
        notional = abs(qty) * price
        return {"ok": True, "init_margin": rate * notional, "maint_margin": rate * notional}

    def register_order_error_handler(self, callback) -> None:
        self._order_error_handlers = getattr(self, "_order_error_handlers", [])
        self._order_error_handlers.append(callback)

    def reconnect(self, max_attempts: int = 5, base_delay: float = 2.0) -> bool:
        self.reconnect_calls = getattr(self, "reconnect_calls", 0) + 1
        return getattr(self, "_reconnect_ok", True)

    def fire_order_error(self, order_id, code=201, message="insufficient margin", symbol=None):
        for cb in getattr(self, "_order_error_handlers", []):
            cb(str(order_id), code, message, symbol)

    def get_latest_quote(self, symbol: str) -> dict:
        return dict(self._quote)

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        client_order_id: str,
    ) -> dict:
        fraction = (
            self._fill_sequence.pop(0)
            if self._fill_sequence
            else self._default_fill_fraction
        )
        fill_qty = int(qty * fraction)
        if self._pending_fill_polls > 0:
            # Async fill: order starts as "new"; get_order transitions it after n polls
            self._orders[client_order_id] = {
                "id":                    client_order_id,
                "status":                "new",
                "filled_qty":            "0",
                "filled_avg_price":      "0",
                "_pending_polls_left":   self._pending_fill_polls,
                "_fill_qty":             str(fill_qty),
                "_fill_price":           str(limit_price),
            }
            self._pending_fill_polls = 0
        else:
            if fill_qty >= qty:
                status = "filled"
            elif fill_qty > 0:
                status = "partially_filled"
            else:
                status = "cancelled"
            self._orders[client_order_id] = {
                "id":               client_order_id,
                "status":           status,
                "filled_qty":       str(fill_qty),
                "filled_avg_price": str(limit_price),
            }
        return {"id": client_order_id}

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        client_order_id: Optional[str] = None,
    ) -> dict:
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())
        fill_price = self._quote["ask"] if side == "buy" else self._quote["bid"]
        self._orders[client_order_id] = {
            "id":               client_order_id,
            "status":           "filled",
            "filled_qty":       str(qty),
            "filled_avg_price": str(fill_price),
        }
        return {"id": client_order_id}

    def modify_stop_order(
        self,
        order_id: str,
        new_qty: Optional[float] = None,
        new_stop_price: Optional[float] = None,
        **kwargs,
    ) -> dict:
        if order_id in self._orders:
            if new_qty is not None:
                self._orders[order_id]["qty"] = new_qty
            if new_stop_price is not None:
                self._orders[order_id]["stop_price"] = new_stop_price
        return {"id": order_id, "status": "new"}

    def submit_stop_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        stop_price: float,
        client_order_id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())
        self._orders[client_order_id] = {
            "id":     client_order_id,
            "status": "new",
        }
        return {"id": client_order_id, "status": "new"}

    def submit_oca_pair(
        self,
        symbol: str,
        side: str,
        qty: int,
        tp1_limit_price: float,
        stop_price: float,
        **kwargs,
    ) -> dict:
        """Place a TP1 limit + protective stop as an OCA group.

        When one leg is set to 'filled', get_order will auto-cancel the sibling,
        simulating IB OCA (one-cancels-all) behavior.
        """
        tp1_id  = str(uuid.uuid4())
        stop_id = str(uuid.uuid4())
        self._orders[tp1_id] = {
            "id": tp1_id, "status": "new",
            "filled_qty": "0", "filled_avg_price": "0",
        }
        self._orders[stop_id] = {
            "id": stop_id, "status": "new",
            "filled_qty": "0", "filled_avg_price": "0",
            "stop_price": str(stop_price),
        }
        # Bidirectional OCA linkage
        self._oca_siblings[tp1_id]  = stop_id
        self._oca_siblings[stop_id] = tp1_id
        return {"tp1_order_id": tp1_id, "stop_order_id": stop_id, "oca_group": f"OCA-{symbol}"}

    def get_order(self, order_id: str) -> dict:
        raw = self._orders.get(
            order_id,
            {"status": "not_found", "filled_qty": "0", "filled_avg_price": "0"},
        )
        # Advance async-fill state machine
        if "_pending_polls_left" in raw:
            polls_left = raw["_pending_polls_left"]
            if polls_left > 0:
                raw["_pending_polls_left"] -= 1
            else:
                raw["status"]           = "filled"
                raw["filled_qty"]       = raw["_fill_qty"]
                raw["filled_avg_price"] = raw["_fill_price"]
        order = dict(raw)
        # OCA: auto-cancel the sibling when one leg is filled
        sibling_id = self._oca_siblings.get(order_id)
        if sibling_id and order.get("status") == "filled":
            sibling = self._orders.get(sibling_id)
            if sibling and sibling.get("status") not in ("filled", "cancelled"):
                sibling["status"] = "cancelled"
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            o = self._orders[order_id]
            if o["status"] not in ("filled",):
                o["status"] = "cancelled"
                return True
            # Already filled — cancel failed (race)
            return False
        return True   # not found → idempotent success

    def cancel_all_orders(self) -> int:
        cancelled = 0
        for o in self._orders.values():
            if o.get("status") not in ("filled", "cancelled"):
                o["status"] = "cancelled"
                cancelled += 1
        return cancelled

    def get_positions(self) -> list:
        return [{"symbol": sym, **p} for sym, p in self._positions.items()]

    def get_min_tick(self, symbol: str) -> float:
        return 0.01

    def submit_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        tp1_limit_price: float,
        tp1_qty: int,
        stop_price: float,
        **kwargs,
    ) -> dict:
        fraction = (
            self._fill_sequence.pop(0)
            if self._fill_sequence
            else self._default_fill_fraction
        )
        fill_qty = int(qty * fraction)
        if fill_qty >= qty:
            entry_status = "filled"
        elif fill_qty > 0:
            entry_status = "partially_filled"
        else:
            entry_status = "cancelled"

        entry_id = str(uuid.uuid4())
        tp1_id   = str(uuid.uuid4())
        stop_id  = str(uuid.uuid4())

        if self._pending_fill_polls > 0:
            self._orders[entry_id] = {
                "id": entry_id, "status": "new",
                "filled_qty": "0", "filled_avg_price": "0",
                "_pending_polls_left": self._pending_fill_polls,
                "_fill_qty": str(fill_qty),
                "_fill_price": str(entry_price),
            }
            self._pending_fill_polls = 0
        else:
            self._orders[entry_id] = {
                "id": entry_id, "status": entry_status,
                "filled_qty": str(fill_qty), "filled_avg_price": str(entry_price),
            }
        self._orders[tp1_id] = {
            "id": tp1_id, "status": "new",
            "filled_qty": "0", "filled_avg_price": "0",
        }
        self._orders[stop_id] = {
            "id": stop_id, "status": "new",
            "filled_qty": "0", "filled_avg_price": "0",
            "stop_price": str(stop_price),
        }
        self._oca_siblings[tp1_id]  = stop_id
        self._oca_siblings[stop_id] = tp1_id
        return {"entry_order_id": entry_id, "tp1_order_id": tp1_id, "stop_order_id": stop_id}

    def get_position(self, symbol: str) -> Optional[dict]:
        return self._positions.get(symbol)

    def register_fill_watcher(self, order_id: str, callback) -> None:
        if not hasattr(self, "_fill_watchers"):
            self._fill_watchers = {}
        self._fill_watchers[order_id] = callback

    def unregister_fill_watcher(self, order_id: str) -> None:
        if hasattr(self, "_fill_watchers"):
            self._fill_watchers.pop(order_id, None)

    def get_executions(self, trade_date=None) -> list:
        return [
            {
                "order_id": oid,
                "symbol": o.get("symbol", "?"),
                "qty": float(o.get("filled_qty", 0)),
                "price": float(o.get("filled_avg_price", 0)),
                "commission": 0.0,
            }
            for oid, o in self._orders.items()
            if o.get("status") == "filled" and float(o.get("filled_qty", 0)) > 0
        ]

    def submit_stop_limit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        stop_price: float,
        limit_price: float,
        client_order_id: Optional[str] = None,
    ) -> dict:
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())
        self._orders[client_order_id] = {
            "id":               client_order_id,
            "status":           "new",
            "filled_qty":       "0",
            "filled_avg_price": "0",
            "stop_price":       str(stop_price),
            "limit_price":      str(limit_price),
            "qty":              str(qty),
        }
        return {"id": client_order_id}

    def fire_fill_watcher(
        self,
        order_id: str,
        qty: float,
        price: float,
        total_qty: Optional[float] = None,
    ) -> None:
        """Test helper: simulate IB firing execDetailsEvent for order_id.

        total_qty — the order's totalQuantity; defaults to qty (full fill).
        Pass total_qty > qty to simulate a partial fill without triggering
        the watcher's cumQty < total guard.
        """
        if not hasattr(self, "_fill_watchers"):
            return
        watcher = self._fill_watchers.get(order_id)
        if not watcher:
            return
        from types import SimpleNamespace
        execution = SimpleNamespace(
            orderId=0,
            price=price, shares=qty, cumQty=qty, avgPrice=price,
            side="BOT", time="", execId="test-exec",
        )
        commission_report = SimpleNamespace(commission=0.0)
        fill  = SimpleNamespace(execution=execution, commissionReport=commission_report)
        order = SimpleNamespace(orderId=0, totalQuantity=total_qty if total_qty is not None else qty)
        trade = SimpleNamespace(order=order)
        watcher(order_id, trade, fill)


@pytest.fixture
def mock_broker():
    return MockBroker()
