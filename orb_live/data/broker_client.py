"""
data/broker_client.py — Abstract broker interface.

IBClient is the concrete implementation.  Concrete classes must implement
every abstractmethod.  list_positions() has a default implementation that
delegates to get_positions() so brokers that expose only one name don't
need to duplicate code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import pandas as pd


class BrokerClient(ABC):
    """Abstract base for all broker API clients."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    # ── Account ───────────────────────────────────────────────────────────────

    @abstractmethod
    def get_account(self) -> dict: ...

    @abstractmethod
    def get_equity(self) -> float: ...

    # ── Clock ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_clock(self): ...

    @abstractmethod
    def is_market_open(self) -> bool: ...

    # ── Asset metadata ────────────────────────────────────────────────────────

    @abstractmethod
    def get_asset(self, symbol: str) -> dict: ...

    # ── Bars ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        lookback_days: int = 20,
        feed: str = "iex",
    ) -> pd.DataFrame: ...

    @abstractmethod
    def get_intraday_bars(
        self,
        symbol: str,
        start_dt,
        end_dt,
        timeframe: str = "1Min",
        feed: str = "iex",
    ) -> pd.DataFrame: ...

    # ── Quote ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> dict: ...

    # ── Positions ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_positions(self) -> list[dict]: ...

    def list_positions(self) -> list[dict]:
        return self.get_positions()

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[dict]: ...

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[dict]: ...

    # ── Orders ────────────────────────────────────────────────────────────────

    @abstractmethod
    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        client_order_id: Optional[str] = None,
    ): ...

    @abstractmethod
    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        client_order_id: Optional[str] = None,
    ): ...

    @abstractmethod
    def submit_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ): ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def cancel_all_orders(self) -> int: ...

    @abstractmethod
    def close_all_positions(self) -> None: ...

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[dict]: ...

    @abstractmethod
    def list_orders(self, status: str = "open") -> list[dict]: ...

    # ── Streaming ─────────────────────────────────────────────────────────────

    @abstractmethod
    def subscribe_bars(self, symbols: list[str], callback: Callable) -> None: ...

    @abstractmethod
    def stop_bars_stream(self) -> None: ...
