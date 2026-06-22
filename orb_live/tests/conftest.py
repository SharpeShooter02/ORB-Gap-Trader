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

    # ── Configuration helpers ─────────────────────────────────────────────────

    def set_fill_fraction(self, fraction: float) -> None:
        """Apply this fraction to every subsequent submit_limit_order call."""
        self._default_fill_fraction = fraction

    def set_fill_sequence(self, *fractions: float) -> None:
        """Per-order fill fractions, consumed left-to-right on each submit."""
        self._fill_sequence = list(fractions)

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
        return {"equity": self._equity}

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
        order = dict(
            self._orders.get(
                order_id,
                {"status": "not_found", "filled_qty": "0", "filled_avg_price": "0"},
            )
        )
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

    def get_position(self, symbol: str) -> Optional[dict]:
        return self._positions.get(symbol)


@pytest.fixture
def mock_broker():
    return MockBroker()
