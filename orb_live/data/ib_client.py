"""
data/ib_client.py — Interactive Brokers client (Part 1: connection + account).

Implements BrokerClient for IB Gateway / TWS.  Thin synchronous wrapper
around ib_async.IB.  Parts 2-4 (market data, orders, streaming) are stubbed
as NotImplementedError and will be implemented later.

Config (read by build_client_from_env):
    IB_HOST         127.0.0.1 (default)
    IB_PORT         4002 paper / 4001 live (default)
    IB_CLIENT_ID    1 (default)
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import pandas as pd

from ib_async import IB
from orb_live.data.broker_client import BrokerClient


class IBClient(BrokerClient):
    """
    IB Gateway / TWS client.  Lazy connection: no network calls in __init__.

    Call connect() before using any data or account methods.
    """

    def __init__(self, paper: bool = True, **kwargs):
        self._paper     = paper
        self._host      = kwargs.get("host",      "127.0.0.1")
        self._port      = kwargs.get("port",      4002 if paper else 4001)
        self._client_id = kwargs.get("client_id", 1)
        self._log       = kwargs.get("logger")
        self._ib        = IB()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            self._ib.connect(
                self._host, self._port,
                clientId=self._client_id,
                timeout=10,
            )
        except Exception as exc:
            raise ConnectionError(
                f"IB Gateway not reachable at {self._host}:{self._port}"
            ) from exc
        if self._log:
            self._log.info(
                "ib_connected",
                host=self._host, port=self._port,
                client_id=self._client_id, paper=self._paper,
            )

    def disconnect(self) -> None:
        self._ib.disconnect()
        if self._log:
            self._log.info("ib_disconnected")

    def is_connected(self) -> bool:
        return self._ib.isConnected()

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        # accountValues() is populated after connect(); retry if not yet ready.
        vals = None
        for _ in range(3):
            vals = self._ib.accountValues()
            if vals:
                break
            time.sleep(1)

        if not vals:
            return {
                "equity": 0.0, "cash": 0.0, "buying_power": 0.0,
                "portfolio_value": 0.0, "daytrade_count": 0,
                "pattern_day_trader": False,
            }

        by_tag: dict[str, str] = {}
        for v in vals:
            if v.currency in ("USD", "BASE", ""):
                by_tag.setdefault(v.tag, v.value)

        def _float(tag: str, default: float = 0.0) -> float:
            try:
                return float(by_tag.get(tag, default))
            except (ValueError, TypeError):
                return default

        equity = _float("NetLiquidation")
        return {
            "equity":             equity,
            "cash":               _float("TotalCashValue"),
            "buying_power":       _float("BuyingPower"),
            "portfolio_value":    equity,
            "daytrade_count":     int(_float("DayTradesRemaining", 0)),
            "pattern_day_trader": False,
        }

    def get_equity(self) -> float:
        return self.get_account()["equity"]

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        result = []
        for item in self._ib.portfolio():
            qty = float(item.position)
            if qty == 0:
                continue
            result.append({
                "symbol":          item.contract.symbol,
                "qty":             qty,
                "market_value":    float(item.marketValue),
                "avg_entry_price": float(item.averageCost),
                "unrealized_pl":   float(item.unrealizedPNL),
                "side":            "long" if qty > 0 else "short",
            })
        return result

    def get_position(self, symbol: str) -> Optional[dict]:
        for pos in self.get_positions():
            if pos["symbol"] == symbol:
                return pos
        return None

    # ── Clock (not available from IB — use MarketClock without broker_client) ──

    def get_clock(self):
        raise NotImplementedError(
            "IBClient does not expose a market clock; "
            "use MarketClock(broker_client=None) for IB sessions"
        )

    def is_market_open(self) -> bool:
        raise NotImplementedError(
            "IBClient does not expose is_market_open; "
            "use MarketClock(broker_client=None) for IB sessions"
        )

    # ── Asset metadata (Part 2) ───────────────────────────────────────────────

    def get_asset(self, symbol: str) -> dict:
        raise NotImplementedError("IBClient.get_asset — Part 2")

    # ── Bars (Part 3) ─────────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        symbol: str,
        lookback_days: int = 20,
        feed: str = "iex",
    ) -> pd.DataFrame:
        raise NotImplementedError("IBClient.get_daily_bars — Part 3")

    def get_intraday_bars(
        self,
        symbol: str,
        start_dt,
        end_dt,
        timeframe: str = "1Min",
        feed: str = "iex",
    ) -> pd.DataFrame:
        raise NotImplementedError("IBClient.get_intraday_bars — Part 3")

    # ── Quote (Part 3) ────────────────────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        raise NotImplementedError("IBClient.get_latest_quote — Part 3")

    # ── Positions (additional) / Orders (Part 4) ──────────────────────────────

    def close_position(self, symbol: str) -> Optional[dict]:
        raise NotImplementedError("IBClient.close_position — Part 4")

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        client_order_id: Optional[str] = None,
    ):
        raise NotImplementedError("IBClient.submit_market_order — Part 4")

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        client_order_id: Optional[str] = None,
    ):
        raise NotImplementedError("IBClient.submit_limit_order — Part 4")

    def submit_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ):
        raise NotImplementedError("IBClient.submit_stop_order — Part 4")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("IBClient.cancel_order — Part 4")

    def cancel_all_orders(self) -> int:
        raise NotImplementedError("IBClient.cancel_all_orders — Part 4")

    def close_all_positions(self) -> None:
        raise NotImplementedError("IBClient.close_all_positions — Part 4")

    def get_order(self, order_id: str) -> Optional[dict]:
        raise NotImplementedError("IBClient.get_order — Part 4")

    def list_orders(self, status: str = "open") -> list[dict]:
        raise NotImplementedError("IBClient.list_orders — Part 4")

    # ── Streaming (Part 3) ────────────────────────────────────────────────────

    def subscribe_bars(self, symbols: list[str], callback: Callable) -> None:
        raise NotImplementedError("IBClient.subscribe_bars — Part 3")

    def stop_bars_stream(self) -> None:
        raise NotImplementedError("IBClient.stop_bars_stream — Part 3")


# ── Factory ───────────────────────────────────────────────────────────────────

def build_client_from_env(paper: bool = True) -> IBClient:
    """
    Construct IBClient from environment variables.

        IB_HOST        default 127.0.0.1
        IB_PORT        default 4002 (paper) / 4001 (live)
        IB_CLIENT_ID   default 1
    """
    host      = os.environ.get("IB_HOST", "127.0.0.1")
    port      = int(os.environ.get("IB_PORT", 4002 if paper else 4001))
    client_id = int(os.environ.get("IB_CLIENT_ID", "1"))
    return IBClient(paper=paper, host=host, port=port, client_id=client_id)
