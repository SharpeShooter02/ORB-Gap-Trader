"""
runner/dry_run.py — Deterministic fill simulator for paper / dry-run sessions.

DryRunBroker wraps a real IBClient for market data but intercepts all
order calls with a deterministic slippage model:

    Limit buy  → fills at min(limit_price,  ask + 1¢)   (best-case for buyer)
    Limit sell → fills at max(limit_price,  bid - 1¢)   (best-case for seller)
    Market buy → fills at ask × (1 + 5 bps)             (adverse slippage)
    Market sell→ fills at bid × (1 - 5 bps)             (adverse slippage)

All orders are filled immediately at quantity == qty (no partial fills).
Equity is tracked from the starting balance; each fill updates the simulated
portfolio value so position_manager sees a realistic equity number.

Usage:
    real_client = build_client_from_env(paper=True)
    dry = DryRunBroker(real_client, starting_equity=100_000.0)
    # pass `dry` wherever a real IBClient would be used
"""

from __future__ import annotations

import uuid
from typing import Optional


_MARKET_ADVERSE_BPS = 5   # 5 bp adverse fill for market orders
_LIMIT_CENT_EDGE    = 0.01


class DryRunBroker:
    """
    Drop-in replacement for IBClient that simulates fills locally.

    Market-data methods (get_intraday_bars, get_latest_quote, get_daily_bars,
    get_clock, subscribe_bars) are delegated to the wrapped real client so
    that price discovery is accurate.
    """

    def __init__(self, real_client, starting_equity: float = 100_000.0):
        self._real           = real_client
        self._equity         = starting_equity
        self._orders: dict   = {}
        self._positions: dict = {}

    # ── Configuration ──────────────────────────────────────────────────────────

    def set_equity(self, equity: float) -> None:
        self._equity = equity

    # ── Order execution (simulated) ────────────────────────────────────────────

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        client_order_id: str,
    ) -> dict:
        quote = self._get_quote(symbol)
        bid, ask = quote["bid"], quote["ask"]

        if side == "buy":
            fill_price = min(limit_price, ask + _LIMIT_CENT_EDGE)
        else:
            fill_price = max(limit_price, bid - _LIMIT_CENT_EDGE)

        self._record_fill(client_order_id, symbol, side, qty, fill_price)
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
        quote = self._get_quote(symbol)
        bid, ask = quote["bid"], quote["ask"]
        adverse = _MARKET_ADVERSE_BPS / 10_000

        if side == "buy":
            fill_price = ask * (1.0 + adverse)
        else:
            fill_price = bid * (1.0 - adverse)

        self._record_fill(client_order_id, symbol, side, qty, fill_price)
        return {"id": client_order_id}

    def get_order(self, order_id: str) -> dict:
        return self._orders.get(
            order_id,
            {"status": "not_found", "filled_qty": "0", "filled_avg_price": "0"},
        )

    def cancel_order(self, order_id: str) -> None:
        pass  # dry-run orders fill immediately; nothing to cancel

    # ── Account ────────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        return {
            "equity": self._equity,
            "buying_power": self._equity * 4,
            "available_funds": self._equity,
        }

    def check_margin(self, symbol: str, side: str, qty: float, price: float) -> dict:
        real = getattr(self._real, "check_margin", None)
        if real is not None:
            try:
                return real(symbol, side, qty, price)
            except Exception:
                pass
        notional = abs(qty) * price
        return {"ok": True, "init_margin": notional, "maint_margin": notional}

    def register_order_error_handler(self, callback) -> None:
        return None

    def reconnect(self, max_attempts: int = 5, base_delay: float = 2.0) -> bool:
        real = getattr(self._real, "reconnect", None)
        if real is not None:
            return bool(real(max_attempts=max_attempts, base_delay=base_delay))
        return True

    def get_position(self, symbol: str) -> Optional[dict]:
        return self._positions.get(symbol)

    # ── Market data (delegated to real client) ─────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        return self._real.get_latest_quote(symbol)

    def get_intraday_bars(self, symbol, start_dt, end_dt, timeframe="1Min", feed="iex"):
        return self._real.get_intraday_bars(symbol, start_dt, end_dt, timeframe, feed)

    def get_daily_bars(self, symbol, lookback_days=10):
        return self._real.get_daily_bars(symbol, lookback_days)

    def get_clock(self):
        return self._real.get_clock()

    def get_asset(self, symbol: str) -> dict:
        return self._real.get_asset(symbol)

    def subscribe_bars(self, symbols, callback, entry_callback=None):
        return self._real.subscribe_bars(symbols, callback, entry_callback)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_quote(self, symbol: str) -> dict:
        try:
            return self._real.get_latest_quote(symbol)
        except Exception:
            return {"bid": 100.0, "ask": 100.0}

    def _record_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        fill_price: float,
    ) -> None:
        self._orders[order_id] = {
            "id":               order_id,
            "status":           "filled",
            "filled_qty":       str(qty),
            "filled_avg_price": str(round(fill_price, 4)),
        }

        # Update simulated positions
        if side == "buy":
            self._positions[symbol] = {"qty": qty}
            self._equity -= qty * fill_price
        else:
            prev = self._positions.get(symbol, {})
            prev_qty = int(prev.get("qty", qty))
            new_qty  = prev_qty - qty
            if new_qty <= 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = {"qty": new_qty}
            self._equity += qty * fill_price
