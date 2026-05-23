"""
data/ib_client.py — Interactive Brokers client (Parts 1-2: connection, account,
asset metadata, and market clock).

Implements BrokerClient for IB Gateway / TWS.  Thin synchronous wrapper
around ib_async.IB.  Parts 3-4 (market data, orders, streaming) are stubbed
as NotImplementedError and will be implemented later.

Config (read by build_client_from_env):
    IB_HOST         127.0.0.1 (default)
    IB_PORT         4002 paper / 4001 live (default)
    IB_CLIENT_ID    1 (default)
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, time as dtime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from ib_async import IB, Stock
from orb_live.data.broker_client import BrokerClient

_ET = ZoneInfo("America/New_York")


class IBClient(BrokerClient):
    """
    IB Gateway / TWS client.  Lazy connection: no network calls in __init__.

    Call connect() before using any data or account methods.
    """

    # Hard-coded NYSE holidays for 2025-2027.
    # Used when pandas_market_calendars is not installed.
    _US_HOLIDAYS: frozenset[date] = frozenset({
        # 2025
        date(2025, 1, 1),  date(2025, 1, 20), date(2025, 2, 17),
        date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
        date(2025, 7, 4),  date(2025, 9, 1),  date(2025, 11, 27),
        date(2025, 12, 25),
        # 2026
        date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16),
        date(2026, 4, 3),  date(2026, 5, 25), date(2026, 6, 19),
        date(2026, 7, 3),  date(2026, 9, 7),  date(2026, 11, 26),
        date(2026, 12, 25),
        # 2027
        date(2027, 1, 1),  date(2027, 1, 18), date(2027, 2, 15),
        date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
        date(2027, 7, 5),  date(2027, 9, 6),  date(2027, 11, 25),
        date(2027, 12, 24),
    })

    def __init__(self, paper: bool = True, **kwargs):
        self._paper       = paper
        self._host        = kwargs.get("host",      "127.0.0.1")
        self._port        = kwargs.get("port",      4002 if paper else 4001)
        self._client_id   = kwargs.get("client_id", 1)
        self._log         = kwargs.get("logger")
        self._ib          = IB()
        self._asset_cache: dict[str, dict] = {}

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

    # ── Clock ────────────────────────────────────────────────────────────────
    # Computed locally from system time + hard-coded NYSE schedule.
    # IB does not expose a REST clock endpoint; we do not call the broker here.

    def get_clock(self) -> dict:
        """
        Return a market-clock snapshot computed from local time.

        Keys: is_open (bool), next_open (datetime ET), next_close (datetime ET),
              timestamp (datetime ET, current moment).

        NOTE: Pass MarketClock(broker_client=None) for session-level timing;
        this method exists so IBClient satisfies the BrokerClient interface.
        Half-day detection is not supported here (Part 4).
        """
        now_et = self._now_et()
        today  = now_et.date()

        market_open_today  = datetime.combine(today, dtime(9, 30), tzinfo=_ET)
        market_close_today = datetime.combine(today, dtime(16, 0), tzinfo=_ET)

        if self._is_market_day(today):
            if now_et < market_open_today:
                is_open    = False
                next_open  = market_open_today
                next_close = market_close_today
            elif now_et < market_close_today:
                is_open = True
                next_day  = self._next_market_day(today + timedelta(days=1))
                next_open  = datetime.combine(next_day, dtime(9, 30), tzinfo=_ET)
                next_close = market_close_today
            else:
                is_open   = False
                next_day  = self._next_market_day(today + timedelta(days=1))
                next_open  = datetime.combine(next_day, dtime(9, 30), tzinfo=_ET)
                next_close = datetime.combine(next_day, dtime(16, 0), tzinfo=_ET)
        else:
            is_open  = False
            next_day = self._next_market_day(today)
            next_open  = datetime.combine(next_day, dtime(9, 30), tzinfo=_ET)
            next_close = datetime.combine(next_day, dtime(16, 0), tzinfo=_ET)

        return {
            "is_open":    is_open,
            "next_open":  next_open,
            "next_close": next_close,
            "timestamp":  now_et,
        }

    def is_market_open(self) -> bool:
        return self.get_clock()["is_open"]

    # ── Clock helpers ─────────────────────────────────────────────────────────

    def _now_et(self) -> datetime:
        return datetime.now(tz=_ET)

    def _is_holiday(self, d: date) -> bool:
        """True if d is a US market holiday (weekday check is caller's responsibility)."""
        try:
            import pandas_market_calendars as mcal  # optional dependency
            nyse     = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(
                start_date=d.strftime("%Y-%m-%d"),
                end_date=d.strftime("%Y-%m-%d"),
            )
            return bool(schedule.empty)
        except ImportError:
            pass
        return d in self._US_HOLIDAYS

    def _is_market_day(self, d: date) -> bool:
        return d.weekday() < 5 and not self._is_holiday(d)

    def _next_market_day(self, from_date: date) -> date:
        """First market day on or after from_date."""
        d = from_date
        for _ in range(14):  # safety: longest US holiday stretch < 7 calendar days
            if self._is_market_day(d):
                return d
            d += timedelta(days=1)
        return d

    # ── Asset metadata ────────────────────────────────────────────────────────

    def get_asset(self, symbol: str) -> dict:
        """
        Return asset metadata for symbol, qualifying the contract via IB.

        Results are cached in self._asset_cache for the lifetime of the client.
        """
        if symbol in self._asset_cache:
            return self._asset_cache[symbol]

        contract = Stock(symbol, "SMART", "USD")
        try:
            qualified = self._ib.qualifyContracts(contract)
            if qualified:
                c = qualified[0]
                result = {
                    "symbol":           symbol,
                    "tradable":         True,
                    # TODO: implement real shortability check via reqMktData(genericTickList='236')
                    # when market data subscription is available. For now, assume shortable.
                    "shortable":        True,
                    "status":           "active",
                    "primary_exchange": c.primaryExchange,
                    "conId":            c.conId,
                }
            else:
                result = {
                    "symbol":           symbol,
                    "tradable":         False,
                    "shortable":        False,
                    "status":           "not_found",
                    "primary_exchange": None,
                    "conId":            None,
                }
        except Exception:
            result = {
                "symbol":           symbol,
                "tradable":         False,
                "shortable":        False,
                "status":           "not_found",
                "primary_exchange": None,
                "conId":            None,
            }

        self._asset_cache[symbol] = result
        return result

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
