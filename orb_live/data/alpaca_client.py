"""
data/alpaca_client.py — Alpaca brokerage interface.

Wraps alpaca-py (TradingClient, StockHistoricalDataClient,
StockDataStream, CryptoHistoricalDataClient, CryptoDataStream).

All methods return plain Python dicts / DataFrames so the rest of the
system never imports alpaca SDK objects directly.

Retry policy: 3 attempts with exponential back-off (1s, 2s, 4s) on
transient HTTP errors (429, 5xx, timeouts).  Auth errors (401, 403)
raise immediately without retry.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable

import pandas as pd

from orb_live.data.broker_client import BrokerClient

logger = logging.getLogger(__name__)

# ── Retry helper ──────────────────────────────────────────────────────────────

_AUTH_CODES = {401, 403}
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0


def _retry(fn: Callable, *args, **kwargs):
    """Call fn(*args, **kwargs) with exponential back-off on transient errors."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            # Auth errors: re-raise immediately.
            code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if code in _AUTH_CODES:
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                sleep = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "alpaca_retry", attempt=attempt + 1, error=str(exc),
                    sleep_s=sleep,
                )
                time.sleep(sleep)
    raise last_exc


# ── OrderResult ───────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    client_order_id: str
    alpaca_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    status: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_avg_price: Optional[float] = None
    filled_qty: Optional[float] = None


# ── AlpacaClient ──────────────────────────────────────────────────────────────

class AlpacaClient(BrokerClient):
    """
    Unified wrapper around the alpaca-py SDK.

    Instantiate once per session; reuse across all API calls.

    Current universe is all-equity ETFs.  Crypto data methods are
    implemented for future-proofing and for underlying daily cross-checks.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient

        self._trading = TradingClient(
            api_key=api_key, secret_key=secret_key, paper=paper
        )
        self._stock_data = StockHistoricalDataClient(
            api_key=api_key, secret_key=secret_key
        )
        self._crypto_data = CryptoHistoricalDataClient(
            api_key=api_key, secret_key=secret_key
        )
        self._paper = paper
        self._asset_cache: dict[str, dict] = {}
        self._active_stream = None  # live StockDataStream; set during subscribe_bars

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        pass  # connection established at construction time via alpaca-py SDK

    def disconnect(self) -> None:
        self.stop_bars_stream()

    def is_connected(self) -> bool:
        return True

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        acct = _retry(self._trading.get_account)
        return {
            "equity":             float(acct.equity),
            "cash":               float(acct.cash),
            "buying_power":       float(acct.buying_power),
            "portfolio_value":    float(acct.portfolio_value),
            "daytrade_count":     int(acct.daytrade_count),
            "pattern_day_trader": acct.pattern_day_trader,
        }

    def get_equity(self) -> float:
        return float(_retry(self._trading.get_account).equity)

    # ── Clock ─────────────────────────────────────────────────────────────────

    def get_clock(self):
        return _retry(self._trading.get_clock)

    def is_market_open(self) -> bool:
        return _retry(self._trading.get_clock).is_open

    # ── Asset metadata ────────────────────────────────────────────────────────

    def get_asset(self, symbol: str) -> dict:
        """
        Return asset metadata.  Cached per session.

        Returns dict with keys:
            tradable, shortable, easy_to_borrow, marginable, fractionable,
            status, exchange, asset_class ('us_equity' or 'crypto')
        """
        if symbol in self._asset_cache:
            return self._asset_cache[symbol]
        asset = _retry(self._trading.get_asset, symbol)
        result = {
            "tradable":       asset.tradable,
            "shortable":      asset.shortable,
            "easy_to_borrow": asset.easy_to_borrow,
            "marginable":     asset.marginable,
            "fractionable":   asset.fractionable,
            "status":         str(asset.status),
            "exchange":       str(asset.exchange) if asset.exchange else "",
            "asset_class":    str(asset.asset_class),
        }
        self._asset_cache[symbol] = result
        return result

    # ── Daily bars ────────────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        symbol: str,
        lookback_days: int = 20,
        feed: str = "iex",
    ) -> pd.DataFrame:
        """
        Return daily OHLCV bars as a DataFrame with columns
        [date, open, high, low, close, volume].

        Routes to crypto or equity endpoint based on cached asset_class.
        """
        asset = self.get_asset(symbol)
        if asset["asset_class"] == "crypto":
            return self._get_crypto_daily_bars(symbol, lookback_days)
        return self._get_stock_daily_bars(symbol, lookback_days, feed)

    def _get_stock_daily_bars(
        self, symbol: str, lookback_days: int, feed: str
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        import datetime

        end   = pd.Timestamp.now(tz="America/New_York").normalize()
        start = end - pd.Timedelta(days=lookback_days * 2)  # buffer for non-trading days
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            feed=feed,
            adjustment="split",
        )
        bars = _retry(self._stock_data.get_stock_bars, req)
        df = bars.df.reset_index()
        return self._normalise_bar_df(df, symbol, lookback_days)

    def _get_crypto_daily_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end   = pd.Timestamp.now(tz="UTC").normalize()
        start = end - pd.Timedelta(days=lookback_days * 2)
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
        )
        bars = _retry(self._crypto_data.get_crypto_bars, req)
        df = bars.df.reset_index()
        return self._normalise_bar_df(df, symbol, lookback_days)

    @staticmethod
    def _normalise_bar_df(df: pd.DataFrame, symbol: str,
                          lookback_days: int) -> pd.DataFrame:
        # Alpaca returns multi-index (symbol, timestamp) or just timestamp.
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df["date"] = pd.to_datetime(df[ts_col]).dt.tz_localize(None).dt.normalize()
        needed = ["date"] + [c for c in ["open", "high", "low", "close", "volume"]
                             if c in df.columns]
        df = df[needed].sort_values("date").tail(lookback_days).reset_index(drop=True)
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df

    # ── Intraday bars ─────────────────────────────────────────────────────────

    def get_intraday_bars(
        self,
        symbol: str,
        start_dt,
        end_dt,
        timeframe: str = "1Min",
        feed: str = "iex",
    ) -> pd.DataFrame:
        """Return 1-min bars between start_dt and end_dt (tz-aware datetimes)."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        }
        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))

        asset = self.get_asset(symbol)
        if asset["asset_class"] == "crypto":
            from alpaca.data.requests import CryptoBarsRequest
            req = CryptoBarsRequest(
                symbol_or_symbols=symbol, timeframe=tf,
                start=start_dt, end=end_dt,
            )
            bars = _retry(self._crypto_data.get_crypto_bars, req)
        else:
            req = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=tf,
                start=start_dt, end=end_dt, feed=feed,
            )
            bars = _retry(self._stock_data.get_stock_bars, req)

        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df["timestamp"] = pd.to_datetime(df[ts_col])
        needed = ["timestamp"] + [c for c in ["open", "high", "low", "close", "volume"]
                                  if c in df.columns]
        return df[needed].sort_values("timestamp").reset_index(drop=True)

    # ── Quote ─────────────────────────────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        """Return latest NBBO quote: {bid, ask, bid_size, ask_size, ts}."""
        from alpaca.data.requests import StockLatestQuoteRequest
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = _retry(self._stock_data.get_stock_latest_quote, req)
        q = quotes[symbol]
        return {
            "bid":      float(q.bid_price),
            "ask":      float(q.ask_price),
            "bid_size": int(q.bid_size),
            "ask_size": int(q.ask_size),
            "ts":       q.timestamp,
        }

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        positions = _retry(self._trading.get_all_positions)
        return [
            {
                "symbol":            p.symbol,
                "qty":               float(p.qty),
                "market_value":      float(p.market_value),
                "avg_entry_price":   float(p.avg_entry_price),
                "unrealized_pl":     float(p.unrealized_pl),
                "side":              str(p.side),
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            p = _retry(self._trading.get_open_position, symbol)
            return {
                "symbol":            p.symbol,
                "qty":               float(p.qty),
                "market_value":      float(p.market_value),
                "avg_entry_price":   float(p.avg_entry_price),
                "unrealized_pl":     float(p.unrealized_pl),
                "side":              str(p.side),
            }
        except Exception:
            return None

    def close_position(self, symbol: str) -> Optional[dict]:
        try:
            order = _retry(self._trading.close_position, symbol)
            return {"alpaca_id": str(order.id), "status": str(order.status)}
        except Exception:
            return None

    # ── Orders ────────────────────────────────────────────────────────────────

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = _retry(self._trading.submit_order, req)
        return OrderResult(
            client_order_id=order.client_order_id,
            alpaca_id=str(order.id),
            symbol=order.symbol,
            side=str(order.side),
            qty=float(order.qty),
            order_type="market",
            status=str(order.status),
        )

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        order = _retry(self._trading.submit_order, req)
        return OrderResult(
            client_order_id=order.client_order_id,
            alpaca_id=str(order.id),
            symbol=order.symbol,
            side=str(order.side),
            qty=float(order.qty),
            order_type="limit",
            status=str(order.status),
            limit_price=limit_price,
        )

    def submit_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )
        order = _retry(self._trading.submit_order, req)
        return OrderResult(
            client_order_id=order.client_order_id,
            alpaca_id=str(order.id),
            symbol=order.symbol,
            side=str(order.side),
            qty=float(order.qty),
            order_type="stop",
            status=str(order.status),
            stop_price=stop_price,
        )

    def cancel_order(self, alpaca_id: str) -> bool:
        try:
            _retry(self._trading.cancel_order_by_id, alpaca_id)
            return True
        except Exception:
            return False

    def cancel_all_orders(self) -> int:
        results = _retry(self._trading.cancel_orders)
        return len(results) if results else 0

    def close_all_positions(self) -> None:
        _retry(self._trading.close_all_positions, cancel_orders=True)

    def get_order(self, alpaca_id: str) -> Optional[dict]:
        try:
            o = _retry(self._trading.get_order_by_id, alpaca_id)
            return {
                "id":               str(o.id),
                "client_order_id":  o.client_order_id,
                "symbol":           o.symbol,
                "status":           str(o.status),
                "filled_qty":       float(o.filled_qty) if o.filled_qty else 0.0,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            }
        except Exception:
            return None

    def list_orders(self, status: str = "open") -> list[dict]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        status_map = {
            "open":   QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all":    QueryOrderStatus.ALL,
        }
        req = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.OPEN))
        orders = _retry(self._trading.get_orders, req)
        return [
            {
                "id":               str(o.id),
                "client_order_id":  o.client_order_id,
                "symbol":           o.symbol,
                "side":             str(o.side),
                "qty":              float(o.qty),
                "filled_qty":       float(o.filled_qty) if o.filled_qty else 0.0,
                "status":           str(o.status),
                "order_type":       str(o.order_type),
            }
            for o in orders
        ]

    # ── Data stream subscription ───────────────────────────────────────────────

    def subscribe_bars(self, symbols: list[str], callback: Callable) -> None:
        """Subscribe to 1-min bar stream for equity symbols.  Blocks until stopped."""
        from alpaca.data.live.stock import StockDataStream
        stream = StockDataStream(self._trading._api_key, self._trading._secret_key)
        self._active_stream = stream

        # alpaca-py requires async coroutine handlers; wrap the sync callback.
        async def _async_callback(bar):
            callback(bar)

        try:
            stream.subscribe_bars(_async_callback, *symbols)
            stream.run()
        finally:
            self._active_stream = None

    def stop_bars_stream(self) -> None:
        """Signal the active WebSocket stream to stop (triggers clean exit)."""
        if self._active_stream is not None:
            try:
                self._active_stream.stop()
            except Exception as exc:
                logger.warning("stop_bars_stream_error error=%s", exc)


def build_client_from_env(paper: bool = True) -> AlpacaClient:
    """Convenience constructor — reads ALPACA_API_KEY / ALPACA_SECRET_KEY."""
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    return AlpacaClient(api_key=api_key, secret_key=secret_key, paper=paper)
