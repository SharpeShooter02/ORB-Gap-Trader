"""
data/ib_client.py — Interactive Brokers client (Parts 1-4: connection, account,
asset metadata, market clock, order operations, and market data).

Implements BrokerClient for IB Gateway / TWS.  Thin synchronous wrapper
around ib_async.IB.

Config (read by build_client_from_env):
    IB_HOST         127.0.0.1 (default)
    IB_PORT         4002 paper / 4001 live (default)
    IB_CLIENT_ID    1 (default)

Order design — event-driven cache with polling interface:
    IB sends order updates via callbacks (orderStatusEvent, execDetailsEvent).
    The strategy layer polls via get_order(id).  The cache bridges the two:
    event handlers update self._order_cache; get_order() reads from it.

Streaming design — 5-second bar aggregation:
    IB only streams 5-second real-time bars.  BarAggregator accumulates
    them and emits 1-minute bars to the user callback when the minute
    boundary is crossed.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, time as dtime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from ib_async import IB, LimitOrder, MarketOrder, Stock
from orb_live.data.broker_client import BrokerClient

_ET = ZoneInfo("America/New_York")

# timeframe string → IB barSizeSetting
_BAR_SIZE_MAP: dict[str, str] = {
    "1min":  "1 min",
    "1m":    "1 min",
    "5min":  "5 mins",
    "5m":    "5 mins",
    "15min": "15 mins",
    "15m":   "15 mins",
    "1h":    "1 hour",
    "1hour": "1 hour",
}


class BarAggregator:
    """Aggregates 5-second IB real-time bars into 1-minute bars.

    Call add_5sec_bar() on each incoming bar.  When a minute boundary is
    crossed the method returns the completed 1-min bar dict; otherwise None.
    Call finalize() to flush any pending partial bar at shutdown.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol          = symbol
        self.current_minute: Optional[datetime] = None
        self.open:           Optional[float]    = None
        self.high:           float              = float("-inf")
        self.low:            float              = float("inf")
        self.close:          Optional[float]    = None
        self.volume:         float              = 0.0

    def add_5sec_bar(self, bar) -> Optional[dict]:
        """Add a 5-sec bar.  Returns completed 1-min bar dict or None."""
        t = bar.time
        if t.tzinfo is None:
            t = t.replace(tzinfo=_ET)
        bar_minute = t.replace(second=0, microsecond=0)

        if self.current_minute is None:
            self.current_minute = bar_minute
            self.open  = float(bar.open)
            self.high  = float(bar.high)
            self.low   = float(bar.low)
            self.close = float(bar.close)
            self.volume = float(bar.volume)
            return None

        if bar_minute > self.current_minute:
            completed = self.finalize()
            self.current_minute = bar_minute
            self.open  = float(bar.open)
            self.high  = float(bar.high)
            self.low   = float(bar.low)
            self.close = float(bar.close)
            self.volume = float(bar.volume)
            return completed

        # Same minute — update running aggregate
        self.high   = max(self.high,  float(bar.high))
        self.low    = min(self.low,   float(bar.low))
        self.close  = float(bar.close)
        self.volume += float(bar.volume)
        return None

    def finalize(self) -> Optional[dict]:
        """Emit current running state as a 1-min bar dict, or None if empty."""
        if self.current_minute is None:
            return None
        return {
            "symbol":    self.symbol,
            "timestamp": self.current_minute,
            "open":      self.open,
            "high":      self.high,
            "low":       self.low,
            "close":     self.close,
            "volume":    self.volume,
        }


class IBClient(BrokerClient):
    """
    IB Gateway / TWS client.  Lazy connection: no network calls in __init__.

    Call connect() before using any data or account methods.
    """

    # Holiday and market-day logic lives in orb_live.core.calendar so
    # MarketClock and IBClient share a single authoritative source.

    _IB_TO_ALPACA_STATUS: dict[str, str] = {
        "PendingSubmit":   "new",
        "PreSubmitted":    "new",
        "Submitted":       "new",
        "Filled":          "filled",
        "PartiallyFilled": "partially_filled",
        "Cancelled":       "canceled",
        "ApiCancelled":    "canceled",
        "Inactive":        "rejected",
    }

    def __init__(self, paper: bool = True, **kwargs):
        self._paper          = paper
        self._host           = kwargs.get("host",      "127.0.0.1")
        self._port           = kwargs.get("port",      4002 if paper else 4001)
        self._client_id      = kwargs.get("client_id", 1)
        self._log            = kwargs.get("logger")
        self._ib             = IB()
        self._asset_cache:    dict[str, dict]         = {}
        self._contract_cache: dict[str, object]       = {}
        self._order_cache:    dict[int, dict]         = {}
        self._streams:        dict[str, object]       = {}
        self._bar_aggregators: dict[str, BarAggregator] = {}
        self._bar_callback:   Optional[Callable]      = None

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
        # Register order-event handlers so the cache stays current.
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_exec_details
        if self._log:
            self._log.info(
                "ib_connected",
                host=self._host, port=self._port,
                client_id=self._client_id, paper=self._paper,
            )

    def disconnect(self) -> None:
        if self._streams:
            self.stop_bars_stream()
        try:
            self._ib.orderStatusEvent -= self._on_order_status
            self._ib.execDetailsEvent -= self._on_exec_details
        except Exception:
            pass
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
                is_open  = True
                next_day = self._next_market_day(today + timedelta(days=1))
                next_open  = datetime.combine(next_day, dtime(9, 30), tzinfo=_ET)
                next_close = market_close_today
            else:
                is_open  = False
                next_day = self._next_market_day(today + timedelta(days=1))
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
        from orb_live.core.calendar import is_trading_day
        return d.weekday() < 5 and not is_trading_day(d)

    def _is_market_day(self, d: date) -> bool:
        from orb_live.core.calendar import is_trading_day
        return is_trading_day(d)

    def _next_market_day(self, from_date: date) -> date:
        """First market day on or after from_date."""
        from orb_live.core.calendar import next_trading_day
        return next_trading_day(from_date)

    # ── Asset metadata ────────────────────────────────────────────────────────

    def get_asset(self, symbol: str) -> dict:
        """
        Return asset metadata for symbol, qualifying the contract via IB.

        Results are cached in self._asset_cache for the lifetime of the client.
        The qualified Contract object is cached in self._contract_cache and used
        by order-submission methods.
        """
        if symbol in self._asset_cache:
            return self._asset_cache[symbol]

        contract = Stock(symbol, "SMART", "USD")
        try:
            qualified = self._ib.qualifyContracts(contract)
            if qualified:
                c = qualified[0]
                self._contract_cache[symbol] = c
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
                self._contract_cache[symbol] = None
                result = {
                    "symbol":           symbol,
                    "tradable":         False,
                    "shortable":        False,
                    "status":           "not_found",
                    "primary_exchange": None,
                    "conId":            None,
                }
        except Exception:
            self._contract_cache[symbol] = None
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

    # ── Orders ────────────────────────────────────────────────────────────────

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
        tif: str = "day",
    ) -> dict:
        if not self.is_connected():
            raise ConnectionError("IBClient is not connected")

        asset = self.get_asset(symbol)
        if not asset["tradable"]:
            raise ValueError(f"{symbol} is not tradable on IB")

        contract = self._contract_cache[symbol]
        action   = "BUY" if side.lower() == "buy" else "SELL"
        order    = LimitOrder(action, qty, limit_price)
        order.tif        = tif.upper()
        order.outsideRth = extended_hours
        if client_order_id:
            order.orderRef = client_order_id

        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.5)

        order_dict = self._trade_to_dict(trade)
        self._order_cache[trade.order.orderId] = order_dict
        return order_dict

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        client_order_id: Optional[str] = None,
    ) -> dict:
        if not self.is_connected():
            raise ConnectionError("IBClient is not connected")

        asset = self.get_asset(symbol)
        if not asset["tradable"]:
            raise ValueError(f"{symbol} is not tradable on IB")

        contract = self._contract_cache[symbol]
        action   = "BUY" if side.lower() == "buy" else "SELL"
        order    = MarketOrder(action, qty)
        if client_order_id:
            order.orderRef = client_order_id

        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.5)

        order_dict = self._trade_to_dict(trade)
        self._order_cache[trade.order.orderId] = order_dict
        return order_dict

    def get_order(self, order_id: str) -> dict:
        key = int(order_id)
        if key not in self._order_cache:
            raise KeyError(f"Order {order_id} not found in cache")
        return self._order_cache[key]

    def cancel_order(self, order_id: str) -> bool:
        order_int = int(order_id)
        matching  = [t for t in self._ib.trades() if t.order.orderId == order_int]
        if not matching:
            return True  # already done — idempotent success
        self._ib.cancelOrder(matching[0].order)
        self._ib.sleep(0.5)
        return True

    def submit_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ):
        raise NotImplementedError("IBClient.submit_stop_order — Part 4")

    def cancel_all_orders(self) -> int:
        raise NotImplementedError("IBClient.cancel_all_orders — Part 4")

    def close_position(self, symbol: str) -> Optional[dict]:
        raise NotImplementedError("IBClient.close_position — Part 4")

    def close_all_positions(self) -> None:
        raise NotImplementedError("IBClient.close_all_positions — Part 4")

    def list_orders(self, status: str = "open") -> list[dict]:
        raise NotImplementedError("IBClient.list_orders — Part 4")

    # ── Order event handlers (private) ────────────────────────────────────────

    def _on_order_status(self, trade) -> None:
        """Update the order cache whenever IB pushes a status change."""
        order_id = trade.order.orderId
        self._order_cache[order_id] = self._trade_to_dict(trade)

        status = trade.orderStatus.status
        if status in ("Filled", "Cancelled", "Inactive"):
            if self._log:
                self._log.info(
                    "order_status_change",
                    order_id=order_id,
                    symbol=trade.contract.symbol,
                    status=status,
                    filled=trade.orderStatus.filled,
                )

    def _on_exec_details(self, trade, fill) -> None:
        """Capture fill price/qty from each execution report."""
        order_id = trade.order.orderId
        if order_id in self._order_cache:
            self._order_cache[order_id]["last_fill_price"] = fill.execution.price
            self._order_cache[order_id]["last_fill_qty"]   = fill.execution.shares

    def _normalize_status(self, ib_status: str) -> str:
        return self._IB_TO_ALPACA_STATUS.get(ib_status, ib_status.lower())

    def _trade_to_dict(self, trade) -> dict:
        """Normalise an ib_async Trade object to our standard order dict."""
        oid        = trade.order.orderId
        lmt        = (float(trade.order.lmtPrice)
                      if trade.order.orderType == "LMT" else None)
        price      = trade.orderStatus.avgFillPrice
        status_raw = trade.orderStatus.status
        return {
            "id":               str(oid),
            "alpaca_id":        str(oid),   # compatibility alias for order_policy.py
            "symbol":           trade.contract.symbol,
            "side":             "buy" if trade.order.action == "BUY" else "sell",
            "qty":              float(trade.order.totalQuantity),
            "limit_price":      lmt,
            "status":           self._normalize_status(status_raw),
            "status_raw":       status_raw,
            "filled_qty":       float(trade.orderStatus.filled),
            "avg_fill_price":   float(price) if price else None,
            "filled_avg_price": float(price) if price else None,  # alias
            "remaining":        float(trade.orderStatus.remaining),
            "client_order_id":  trade.order.orderRef or None,
        }

    # ── Bars (Part 4) ─────────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        symbol: str,
        lookback_days: int = 20,
        feed: str = "iex",
    ) -> pd.DataFrame:
        """
        Return daily OHLCV bars as a DataFrame with columns
        [date, open, high, low, close, volume].

        date is a tz-naive pd.Timestamp normalized to midnight, sorted
        ascending.  Shape matches AlpacaClient._normalise_bar_df output
        so pre_market.py works without modification.
        """
        _empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        if not self.is_connected():
            raise ConnectionError("IBClient is not connected")

        asset = self.get_asset(symbol)
        if not asset["tradable"]:
            return _empty

        contract = self._contract_cache[symbol]
        duration = f"{lookback_days * 2} D"

        try:
            raw = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                timeout=15,
            )
        except Exception as exc:
            if self._log:
                self._log.warning(
                    "get_daily_bars_failed",
                    symbol=symbol, duration=duration, error=str(exc),
                )
            return _empty

        if not raw:
            return _empty

        rows = []
        for bar in raw:
            try:
                # formatDate=1 → bar.date is already a datetime (tz-aware or naive)
                ts = pd.Timestamp(bar.date).tz_localize(None).normalize()
                rows.append({
                    "date":   ts,
                    "open":   float(bar.open),
                    "high":   float(bar.high),
                    "low":    float(bar.low),
                    "close":  float(bar.close),
                    "volume": float(bar.volume),
                })
            except (ValueError, TypeError):
                continue

        if not rows:
            return _empty

        df = (
            pd.DataFrame(rows)
            .sort_values("date")
            .tail(lookback_days)
            .reset_index(drop=True)
        )
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        return df

    def get_intraday_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
        feed: str = "iex",
    ) -> pd.DataFrame:
        """
        Fetch historical intraday bars from IB for [start, end].

        Returns a DataFrame indexed by ET-aware datetime with columns
        open, high, low, close, volume.  Returns an empty DataFrame if
        the symbol is not tradable or IB returns no data.
        """
        _empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if not self.is_connected():
            raise ConnectionError("IBClient is not connected")

        asset = self.get_asset(symbol)
        if not asset["tradable"]:
            return _empty

        contract = self._contract_cache[symbol]

        # Normalise naive datetimes to ET
        if start.tzinfo is None:
            start = start.replace(tzinfo=_ET)
        if end.tzinfo is None:
            end = end.replace(tzinfo=_ET)

        # Build IB durationStr from the window size
        delta_s = int((end - start).total_seconds())
        delta_m = max(1, delta_s // 60)
        delta_h = max(1, delta_s // 3600)
        delta_d = max(1, (end.date() - start.date()).days + 1)

        if delta_s <= 60:
            duration_str = f"{delta_s} S"
        elif delta_m <= 30:
            duration_str = f"{delta_m} M"
        elif delta_h <= 6:
            duration_str = f"{delta_h} H"
        else:
            duration_str = f"{delta_d} D"

        bar_size = _BAR_SIZE_MAP.get(timeframe.lower(), "1 min")

        from datetime import timezone as _tz
        _now_utc = datetime.now(_tz.utc)
        if abs((end.astimezone(_tz.utc) - _now_utc).total_seconds()) < 60:
            end_dt_str = ""
        else:
            end_dt_str = end.strftime("%Y%m%d %H:%M:%S US/Eastern")

        try:
            raw = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_dt_str,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,
                timeout=15,
            )
        except Exception as exc:
            if self._log:
                self._log.warning(
                    "reqHistoricalData_failed",
                    symbol=symbol,
                    duration=duration_str,
                    error=str(exc),
                )
            return _empty

        if not raw:
            return _empty

        rows = []
        for bar in raw:
            try:
                ts = datetime.fromtimestamp(float(str(bar.date)), tz=_ET)
            except (ValueError, TypeError):
                continue
            rows.append({
                "timestamp": ts,
                "open":   float(bar.open),
                "high":   float(bar.high),
                "low":    float(bar.low),
                "close":  float(bar.close),
                "volume": float(bar.volume),
            })

        if not rows:
            return _empty

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df.loc[(df.index >= start) & (df.index <= end)]

    # ── Quote (Part 5) ────────────────────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        """
        Return the latest NBBO quote: {bid, ask, bid_size, ask_size, ts}.

        Shape matches AlpacaClient.get_latest_quote() so pre_market.py
        works without modification.  Falls back to last/close if bid/ask
        are both zero (pre-market or after-hours snapshot).
        """
        _zero = {"bid": 0.0, "ask": 0.0, "bid_size": 0, "ask_size": 0, "ts": None}

        if not self.is_connected():
            raise ConnectionError("IBClient is not connected")

        asset = self.get_asset(symbol)
        if not asset["tradable"]:
            return _zero

        contract = self._contract_cache[symbol]

        try:
            ticker = self._ib.reqMktData(
                contract, "", snapshot=True, regulatorySnapshot=False,
            )
            self._ib.sleep(2)

            bid      = float(ticker.bid)      if ticker.bid      and ticker.bid      > 0 else 0.0
            ask      = float(ticker.ask)      if ticker.ask      and ticker.ask      > 0 else 0.0
            bid_size = int(ticker.bidSize)    if ticker.bidSize  else 0
            ask_size = int(ticker.askSize)    if ticker.askSize  else 0
            ts       = ticker.time            if ticker.time     else None

            # Pre-market / after-hours: bid/ask may both be 0 — fall back to
            # last trade price, then to prior close.
            if bid == 0.0 and ask == 0.0:
                fallback = 0.0
                if ticker.last and float(ticker.last) > 0:
                    fallback = float(ticker.last)
                elif ticker.close and float(ticker.close) > 0:
                    fallback = float(ticker.close)
                if fallback > 0:
                    bid = ask = fallback

            return {
                "bid":      bid,
                "ask":      ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "ts":       ts,
            }
        except Exception as exc:
            if self._log:
                self._log.warning(
                    "get_latest_quote_failed",
                    symbol=symbol, error=str(exc),
                )
            return _zero
        finally:
            try:
                self._ib.cancelMktData(contract)
            except Exception:
                pass

    # ── Streaming (Part 4) ────────────────────────────────────────────────────

    def subscribe_bars(self, symbols: list[str], callback: Callable) -> None:
        """
        Stream 1-minute bars for the given symbols.

        Internally subscribes to IB's 5-second real-time bars (the only
        streaming granularity IB supports) and uses BarAggregator to emit
        a completed 1-min bar dict whenever a minute boundary is crossed.

        callback signature: fn(bar: dict) where bar has keys:
            symbol, timestamp, open, high, low, close, volume
        """
        if not self.is_connected():
            self.connect()

        self._bar_callback = callback

        for sym in symbols:
            if sym in self._streams:
                continue  # already streaming

            asset = self.get_asset(sym)
            if not asset["tradable"]:
                if self._log:
                    self._log.warning("subscribe_bars_skip_untradable", symbol=sym)
                continue

            contract = self._contract_cache[sym]

            stream = self._ib.reqRealTimeBars(
                contract,
                barSize=5,
                whatToShow="TRADES",
                useRTH=False,
            )

            self._streams[sym]          = stream
            self._bar_aggregators[sym]  = BarAggregator(sym)

            def _make_listener(symbol: str):
                def _on_new_bar(bars, hasNewBar: bool) -> None:
                    if not (hasNewBar and bars):
                        return
                    completed = self._bar_aggregators[symbol].add_5sec_bar(bars[-1])
                    if completed and self._bar_callback:
                        try:
                            self._bar_callback(completed)
                        except Exception as exc:
                            if self._log:
                                self._log.error(
                                    "bar_callback_error",
                                    symbol=symbol, error=str(exc),
                                )
                return _on_new_bar

            stream.updateEvent += _make_listener(sym)

            if self._log:
                self._log.info("subscribed_bars", symbol=sym)

    def stop_bars_stream(self) -> None:
        """Unsubscribe from all bar streams, flushing any pending partial bars."""
        for sym in list(self._streams.keys()):
            try:
                pending = self._bar_aggregators[sym].finalize()
                if pending and self._bar_callback:
                    try:
                        self._bar_callback(pending)
                    except Exception as exc:
                        if self._log:
                            self._log.error(
                                "final_bar_callback_error",
                                symbol=sym, error=str(exc),
                            )
                self._ib.cancelRealTimeBars(self._streams[sym])
                if self._log:
                    self._log.info("unsubscribed_bars", symbol=sym)
            except Exception as exc:
                if self._log:
                    self._log.warning("stop_bars_stream_error",
                                      symbol=sym, error=str(exc))

        self._streams.clear()
        self._bar_aggregators.clear()
        self._bar_callback = None


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
