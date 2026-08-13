"""
tests/test_ib_client.py — Unit tests for IBClient (Parts 1-4).

All IB Gateway network calls are mocked so these run without a live connection.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from orb_live.data.ib_client import BarAggregator, IBClient, build_client_from_env

_ET = ZoneInfo("America/New_York")


def _et(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_ET)


def _make_account_value(tag: str, value: str, currency: str = "USD") -> SimpleNamespace:
    return SimpleNamespace(tag=tag, value=value, currency=currency)


def _make_portfolio_item(
    symbol: str,
    position: float,
    market_value: float,
    avg_cost: float,
    unrealized_pnl: float,
) -> SimpleNamespace:
    contract = SimpleNamespace(symbol=symbol)
    return SimpleNamespace(
        contract=contract,
        position=position,
        marketValue=market_value,
        averageCost=avg_cost,
        unrealizedPNL=unrealized_pnl,
    )


# ── Lifecycle ─────────────────────────────────────────────────────────────────

class TestConnectLifecycle:
    def test_connect_lifecycle(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            MockIB.return_value = mock_ib

            client = IBClient(paper=True, host="127.0.0.1", port=4002, client_id=3)
            client._allow_delayed_data = True  # skip SPY validation in this plumbing test
            client.connect()

            mock_ib.connect.assert_called_once_with(
                "127.0.0.1", 4002, clientId=3, timeout=10
            )
            assert client.is_connected() == mock_ib.isConnected()

    def test_connect_raises_connection_error_on_failure(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.connect.side_effect = OSError("refused")
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            with pytest.raises(ConnectionError, match="IB Gateway not reachable"):
                client.connect()


class TestDisconnectLifecycle:
    def test_disconnect_lifecycle(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            client.disconnect()

            mock_ib.disconnect.assert_called_once()

    def test_is_connected_delegates_to_ib(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.isConnected.return_value = False
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            assert client.is_connected() is False


# ── Account ───────────────────────────────────────────────────────────────────

class TestGetAccount:
    def test_get_account_parses_correctly(self):
        account_values = [
            _make_account_value("NetLiquidation",  "125000.50"),
            _make_account_value("TotalCashValue",  "50000.00"),
            _make_account_value("BuyingPower",     "200000.00"),
            _make_account_value("DayTradesRemaining", "3"),
            _make_account_value("UnrelatedTag",    "999"),
        ]

        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.accountValues.return_value = account_values
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            acct = client.get_account()

        assert acct["equity"]          == pytest.approx(125000.50)
        assert acct["cash"]            == pytest.approx(50000.00)
        assert acct["buying_power"]    == pytest.approx(200000.00)
        assert acct["portfolio_value"] == pytest.approx(125000.50)
        assert acct["daytrade_count"]  == 3
        assert acct["pattern_day_trader"] is False

    def test_get_account_returns_zeros_when_empty(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.accountValues.return_value = []
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            # patch time.sleep to avoid delay
            with patch("orb_live.data.ib_client.time") as mock_time:
                mock_time.sleep = MagicMock()
                acct = client.get_account()

        assert acct["equity"] == 0.0
        assert acct["cash"]   == 0.0

    def test_await_account_data_polls_until_ready(self):
        """_await_account_data must pump the loop (ib.sleep) until accountValues
        populates — the connect-time guard that prevents the 0.0-equity race."""
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.accountValues.side_effect = [[], [], ["ready"]]
            MockIB.return_value = mock_ib
            client = IBClient(paper=True)

            assert client._await_account_data(timeout=5.0) is True
            assert mock_ib.sleep.call_count == 2   # slept twice before data arrived

    def test_await_account_data_times_out(self):
        """If account data never arrives it returns False (does not hang)."""
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.accountValues.return_value = []
            MockIB.return_value = mock_ib
            client = IBClient(paper=True)

            assert client._await_account_data(timeout=0.5) is False   # 2 iterations


# ── Positions ─────────────────────────────────────────────────────────────────

class TestGetPosition:
    def test_get_position_not_held(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.portfolio.return_value = []
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            assert client.get_position("TQQQ") is None

    def test_get_position_held(self):
        items = [
            _make_portfolio_item("TQQQ",  100.0, 5500.0, 54.50, 50.0),
            _make_portfolio_item("SQQQ",  -50.0, 2000.0, 41.00, -20.0),
        ]

        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.portfolio.return_value = items
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            pos = client.get_position("TQQQ")

        assert pos is not None
        assert pos["symbol"]          == "TQQQ"
        assert pos["qty"]             == pytest.approx(100.0)
        assert pos["market_value"]    == pytest.approx(5500.0)
        assert pos["avg_entry_price"] == pytest.approx(54.50)
        assert pos["unrealized_pl"]   == pytest.approx(50.0)
        assert pos["side"]            == "long"


class TestListPositions:
    def test_list_positions_empty(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.portfolio.return_value = []
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            assert client.list_positions() == []

    def test_list_positions_with_positions(self):
        items = [
            _make_portfolio_item("TQQQ", 200.0, 11000.0, 54.00, 200.0),
            _make_portfolio_item("SQQQ",   0.0,     0.0, 40.00,   0.0),  # zero qty, excluded
            _make_portfolio_item("UVXY",  -30.0, 1800.0, 61.00, -60.0),
        ]

        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.portfolio.return_value = items
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            positions = client.list_positions()

        assert len(positions) == 2
        symbols = {p["symbol"] for p in positions}
        assert symbols == {"TQQQ", "UVXY"}

        tqqq = next(p for p in positions if p["symbol"] == "TQQQ")
        assert tqqq["side"] == "long"

        uvxy = next(p for p in positions if p["symbol"] == "UVXY")
        assert uvxy["side"] == "short"
        assert uvxy["qty"] == pytest.approx(-30.0)


# ── Factory ───────────────────────────────────────────────────────────────────

class TestBuildClientFromEnv:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("IB_HOST",      raising=False)
        monkeypatch.delenv("IB_PORT",      raising=False)
        monkeypatch.delenv("IB_CLIENT_ID", raising=False)

        with patch("orb_live.data.ib_client.IB"):
            client = build_client_from_env(paper=True)

        assert client._host      == "127.0.0.1"
        assert client._port      == 4002
        assert client._client_id == 1
        assert client._paper     is True

    def test_live_port_default(self, monkeypatch):
        monkeypatch.delenv("IB_PORT", raising=False)

        with patch("orb_live.data.ib_client.IB"):
            client = build_client_from_env(paper=False)

        assert client._port == 4001

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("IB_HOST",      "192.168.1.100")
        monkeypatch.setenv("IB_PORT",      "7497")
        monkeypatch.setenv("IB_CLIENT_ID", "5")

        with patch("orb_live.data.ib_client.IB"):
            client = build_client_from_env(paper=True)

        assert client._host      == "192.168.1.100"
        assert client._port      == 7497
        assert client._client_id == 5


# ── Asset metadata ────────────────────────────────────────────────────────────

def _make_qualified_contract(symbol: str, con_id: int, exchange: str) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, conId=con_id, primaryExchange=exchange)


class TestGetAsset:
    def test_get_asset_caches(self):
        """qualifyContracts must be called exactly once across two identical lookups."""
        qualified = [_make_qualified_contract("SOXL", 99001, "ARCA")]

        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.qualifyContracts.return_value = qualified
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            client.get_asset("SOXL")
            client.get_asset("SOXL")

        mock_ib.qualifyContracts.assert_called_once()

    def test_get_asset_qualified(self):
        qualified = [_make_qualified_contract("SOXL", 12345, "ARCA")]

        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.qualifyContracts.return_value = qualified
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            asset = client.get_asset("SOXL")

        assert asset["symbol"]           == "SOXL"
        assert asset["tradable"]         is True
        assert asset["shortable"]        is True
        assert asset["status"]           == "active"
        assert asset["primary_exchange"] == "ARCA"
        assert asset["conId"]            == 12345

    def test_get_asset_not_found(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.qualifyContracts.return_value = []
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            asset = client.get_asset("BADTICKER")

        assert asset["symbol"]   == "BADTICKER"
        assert asset["tradable"] is False
        assert asset["status"]   == "not_found"
        assert asset["conId"]    is None

    def test_get_asset_exception_returns_not_found(self):
        with patch("orb_live.data.ib_client.IB") as MockIB:
            mock_ib = MagicMock()
            mock_ib.qualifyContracts.side_effect = RuntimeError("IB error")
            MockIB.return_value = mock_ib

            client = IBClient(paper=True)
            asset = client.get_asset("FAIL")

        assert asset["tradable"] is False
        assert asset["status"]   == "not_found"


# ── Market clock ──────────────────────────────────────────────────────────────

class TestGetClock:
    def _client(self) -> IBClient:
        with patch("orb_live.data.ib_client.IB"):
            return IBClient(paper=True)

    def test_get_clock_during_session(self):
        # 2026-05-19 is Tuesday — regular trading day
        client = self._client()
        with patch.object(client, "_now_et", return_value=_et(2026, 5, 19, 11)):
            clock = client.get_clock()

        assert clock["is_open"] is True
        assert clock["next_close"].date() == date(2026, 5, 19)
        assert clock["next_close"].hour   == 16
        assert clock["next_close"].minute == 0
        assert clock["next_open"].date()  > date(2026, 5, 19)

    def test_get_clock_premarket(self):
        client = self._client()
        with patch.object(client, "_now_et", return_value=_et(2026, 5, 19, 8)):
            clock = client.get_clock()

        assert clock["is_open"] is False
        assert clock["next_open"].date()   == date(2026, 5, 19)
        assert clock["next_open"].hour     == 9
        assert clock["next_open"].minute   == 30
        assert clock["next_close"].date()  == date(2026, 5, 19)

    def test_get_clock_after_hours(self):
        # 2026-05-19 (Tue) at 17:00 — next open is Wed May 20
        client = self._client()
        with patch.object(client, "_now_et", return_value=_et(2026, 5, 19, 17)):
            clock = client.get_clock()

        assert clock["is_open"] is False
        assert clock["next_open"].date() == date(2026, 5, 20)
        assert clock["next_open"].hour   == 9

    def test_get_clock_weekend(self):
        # 2026-05-30 (Sat) at noon — next open is Mon June 1 (no holiday)
        client = self._client()
        with patch.object(client, "_now_et", return_value=_et(2026, 5, 30, 12)):
            clock = client.get_clock()

        assert clock["is_open"] is False
        assert clock["next_open"].date() == date(2026, 6, 1)
        assert clock["next_open"].hour   == 9

    def test_get_clock_friday_evening(self):
        # 2026-05-29 (Fri) at 17:00 — next open is Mon June 1 (no holiday)
        client = self._client()
        with patch.object(client, "_now_et", return_value=_et(2026, 5, 29, 17)):
            clock = client.get_clock()

        assert clock["is_open"] is False
        assert clock["next_open"].date() == date(2026, 6, 1)

    def test_get_clock_before_holiday(self):
        # 2026-05-22 (Fri) at 17:00 — next open skips Sat/Sun/Memorial Day Mon 5/25
        # → Tuesday 2026-05-26
        client = self._client()
        with patch.object(client, "_now_et", return_value=_et(2026, 5, 22, 17)):
            clock = client.get_clock()

        assert clock["is_open"] is False
        assert clock["next_open"].date() == date(2026, 5, 26)

    def test_get_clock_timestamp_matches_now(self):
        fixed = _et(2026, 5, 19, 11, 15)
        client = self._client()
        with patch.object(client, "_now_et", return_value=fixed):
            clock = client.get_clock()

        assert clock["timestamp"] == fixed

    def test_is_market_open_delegates_to_get_clock(self):
        client = self._client()
        with patch.object(client, "get_clock", return_value={"is_open": True}):
            assert client.is_market_open() is True
        with patch.object(client, "get_clock", return_value={"is_open": False}):
            assert client.is_market_open() is False


class TestIsHoliday:
    def _client(self) -> IBClient:
        with patch("orb_live.data.ib_client.IB"):
            return IBClient(paper=True)

    def test_is_holiday_known_dates(self):
        client = self._client()
        assert client._is_holiday(date(2026, 5, 25)) is True   # Memorial Day
        assert client._is_holiday(date(2026, 7, 3))  is True   # Independence Day (observed)
        assert client._is_holiday(date(2026, 12, 25)) is True  # Christmas
        assert client._is_holiday(date(2026, 1, 19)) is True   # MLK Day
        assert client._is_holiday(date(2026, 11, 26)) is True  # Thanksgiving

    def test_regular_days_are_not_holidays(self):
        client = self._client()
        assert client._is_holiday(date(2026, 5, 26)) is False  # day after Memorial Day
        assert client._is_holiday(date(2026, 7, 6))  is False  # Mon after July 4 observed
        assert client._is_holiday(date(2026, 12, 24)) is False # Christmas Eve (not a holiday)

    def test_is_market_day_false_for_weekend(self):
        client = self._client()
        assert client._is_market_day(date(2026, 5, 23)) is False  # Saturday
        assert client._is_market_day(date(2026, 5, 24)) is False  # Sunday

    def test_is_market_day_false_for_holiday(self):
        client = self._client()
        assert client._is_market_day(date(2026, 5, 25)) is False  # Memorial Day

    def test_next_market_day_skips_weekend_and_holiday(self):
        client = self._client()
        # From Sat May 23 → skips Sun + Mon Memorial Day → Tue May 26
        assert client._next_market_day(date(2026, 5, 23)) == date(2026, 5, 26)


# ── Orders ────────────────────────────────────────────────────────────────────

def _make_trade(
    order_id: int = 1,
    symbol: str = "TQQQ",
    action: str = "BUY",
    order_type: str = "LMT",
    total_qty: float = 100.0,
    lmt_price: float = 50.0,
    aux_price: float = 0.0,
    status: str = "Submitted",
    filled: float = 0.0,
    avg_fill: float = 0.0,
    remaining: float = 100.0,
    order_ref: str | None = None,
) -> SimpleNamespace:
    order = SimpleNamespace(
        orderId=order_id,
        action=action,
        orderType=order_type,
        totalQuantity=total_qty,
        lmtPrice=lmt_price,
        auxPrice=aux_price,
        orderRef=order_ref,
    )
    order_status = SimpleNamespace(
        status=status,
        filled=filled,
        avgFillPrice=avg_fill,
        remaining=remaining,
    )
    contract = SimpleNamespace(symbol=symbol)
    return SimpleNamespace(order=order, orderStatus=order_status, contract=contract)


def _client_with_asset(
    symbol: str = "TQQQ",
    tradable: bool = True,
    min_tick: float = 0.01,
) -> IBClient:
    """Return an IBClient whose asset + contract caches are pre-populated."""
    with patch("orb_live.data.ib_client.IB") as MockIB:
        mock_ib = MagicMock()
        MockIB.return_value = mock_ib
        client = IBClient(paper=True)

    client._asset_cache[symbol] = {
        "symbol": symbol, "tradable": tradable, "shortable": True,
        "status": "active" if tradable else "not_found",
        "primary_exchange": "ARCA", "conId": 12345,
    }
    fake_contract = SimpleNamespace(symbol=symbol, conId=12345, primaryExchange="ARCA")
    client._contract_cache[symbol] = fake_contract if tradable else None
    client._min_tick_cache[symbol] = min_tick
    return client


class TestSubmitLimitOrder:
    def test_submit_limit_order_returns_dict(self):
        client = _client_with_asset("TQQQ")
        # The mock trade's orderRef must match what submit_limit_order will set,
        # because _trade_to_dict reads from the trade object returned by placeOrder.
        trade  = _make_trade(order_id=42, symbol="TQQQ", action="BUY",
                              order_type="LMT", lmt_price=50.0, status="Submitted",
                              order_ref="uuid-abc")

        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = trade

        result = client.submit_limit_order("TQQQ", "buy", 100, 50.0,
                                           client_order_id="uuid-abc")

        assert result["id"]            == "42"
        assert result["symbol"]        == "TQQQ"
        assert result["side"]          == "buy"
        assert result["qty"]           == pytest.approx(100.0)
        assert result["limit_price"]   == pytest.approx(50.0)
        assert result["status"]        == "new"
        assert result["filled_qty"]    == pytest.approx(0.0)
        assert result["client_order_id"] == "uuid-abc"

    def test_submit_limit_order_caches(self):
        client = _client_with_asset("TQQQ")
        trade  = _make_trade(order_id=7)

        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = trade

        client.submit_limit_order("TQQQ", "buy", 10, 50.0)
        assert 7 in client._order_cache

    def test_submit_limit_order_rejects_non_tradable(self):
        client = _client_with_asset("BADTICKER", tradable=False)
        client._ib.isConnected.return_value = True

        with pytest.raises(ValueError, match="not tradable"):
            client.submit_limit_order("BADTICKER", "buy", 10, 0.01)

    def test_submit_limit_order_raises_when_disconnected(self):
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = False

        with pytest.raises(ConnectionError):
            client.submit_limit_order("TQQQ", "buy", 10, 50.0)

    def test_submit_limit_order_returns_initial_state_immediately(self):
        client = _client_with_asset("TQQQ")
        pending = _make_trade(order_id=50, status="PendingSubmit")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = pending

        result = client.submit_limit_order("TQQQ", "buy", 10, 50.0)
        assert result["status"]     == "new"
        assert result["status_raw"] == "PendingSubmit"
        # Fire-and-forget: no polling; trades() must never be called during submit
        client._ib.trades.assert_not_called()

    def test_submit_limit_order_reflects_immediate_inactive(self):
        client = _client_with_asset("TQQQ")
        # If IB returns the trade already marked Inactive (synchronous rejection),
        # the cached result must reflect that.
        inactive = _make_trade(order_id=51, status="Inactive")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = inactive

        result = client.submit_limit_order("TQQQ", "buy", 10, 50.0)
        assert result["status"]     == "rejected"
        assert result["status_raw"] == "Inactive"

    def test_submit_limit_order_no_blocking_regardless_of_timeout(self):
        client = _client_with_asset("TQQQ")
        pending = _make_trade(order_id=52, status="PendingSubmit")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = pending

        # timeout param is accepted but ignored; must return immediately with
        # initial state and must NOT call ib.sleep (which would re-enter the loop)
        result = client.submit_limit_order("TQQQ", "buy", 10, 50.0, timeout=0.1)
        assert result["status"] == "new"
        client._ib.sleep.assert_not_called()


class TestSubmitMarketOrder:
    def test_submit_market_order_uses_marketorder(self):
        """placeOrder must be called with a MarketOrder (not LimitOrder)."""
        client = _client_with_asset("SOXL")
        # action="SELL" so the mock trade reflects what submit_market_order passes
        trade  = _make_trade(order_id=99, action="SELL", order_type="MKT", lmt_price=0.0)

        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = trade

        result = client.submit_market_order("SOXL", "sell", 50)

        client._ib.placeOrder.assert_called_once()
        _, placed_order = client._ib.placeOrder.call_args[0]
        assert placed_order.__class__.__name__ == "MarketOrder"
        assert result["side"] == "sell"

    def test_submit_market_order_rejects_non_tradable(self):
        client = _client_with_asset("BADTICKER", tradable=False)
        client._ib.isConnected.return_value = True

        with pytest.raises(ValueError, match="not tradable"):
            client.submit_market_order("BADTICKER", "buy", 10)


class TestSubmitStopOrder:
    def test_submit_stop_order_builds_stp_order(self):
        """placeOrder must receive a StopOrder with correct fields."""
        from ib_async import StopOrder
        client = _client_with_asset("SOXL")
        trade  = _make_trade(order_id=70, action="SELL", order_type="STP",
                              lmt_price=0.0, status="Submitted")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = trade

        client.submit_stop_order("SOXL", "sell", 100, 185.50)

        client._ib.placeOrder.assert_called_once()
        _, placed_order = client._ib.placeOrder.call_args[0]
        assert isinstance(placed_order, StopOrder)
        assert placed_order.orderType  == "STP"
        assert placed_order.auxPrice   == pytest.approx(185.50)
        assert float(placed_order.totalQuantity) == pytest.approx(100.0)
        assert placed_order.action     == "SELL"
        assert placed_order.lmtPrice   != pytest.approx(185.50)  # no limit price on stop-market

    def test_submit_stop_order_returns_initial_state_immediately(self):
        client = _client_with_asset("SOXL")
        pending = _make_trade(order_id=71, status="PendingSubmit")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = pending

        result = client.submit_stop_order("SOXL", "sell", 1, 185.50)
        assert result["status"]     == "new"
        assert result["status_raw"] == "PendingSubmit"
        client._ib.trades.assert_not_called()

    def test_submit_stop_order_raises_if_disconnected(self):
        client = _client_with_asset("SOXL")
        client._ib.isConnected.return_value = False

        with pytest.raises(ConnectionError):
            client.submit_stop_order("SOXL", "sell", 1, 185.50)

    def test_submit_stop_order_raises_if_not_tradable(self):
        client = _client_with_asset("BADTICKER", tradable=False)
        client._ib.isConnected.return_value = True

        with pytest.raises(ValueError, match="not tradable"):
            client.submit_stop_order("BADTICKER", "sell", 1, 50.0)

    def test_submit_stop_order_buy_side(self):
        from ib_async import StopOrder
        client = _client_with_asset("SOXL")
        trade  = _make_trade(order_id=72, action="BUY", order_type="STP",
                              lmt_price=0.0, status="Submitted")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = trade

        client.submit_stop_order("SOXL", "buy", 50, 10.0)

        _, placed_order = client._ib.placeOrder.call_args[0]
        assert placed_order.action == "BUY"

    def test_submit_stop_order_caches_order(self):
        client = _client_with_asset("SOXL")
        trade  = _make_trade(order_id=73, order_type="STP", lmt_price=0.0,
                              status="Submitted")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = trade

        client.submit_stop_order("SOXL", "sell", 1, 185.50)
        assert 73 in client._order_cache

    def test_submit_stop_order_includes_client_order_id(self):
        from ib_async import StopOrder
        client = _client_with_asset("SOXL")
        trade  = _make_trade(order_id=74, order_type="STP", lmt_price=0.0,
                              order_ref="test-123", status="Submitted")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = trade

        client.submit_stop_order("SOXL", "sell", 1, 185.50,
                                  client_order_id="test-123")

        _, placed_order = client._ib.placeOrder.call_args[0]
        assert placed_order.orderRef == "test-123"


class TestModifyStopOrder:
    def _client(self) -> IBClient:
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)
        client._ib.isConnected.return_value = True
        return client

    def _stop_trade(self, order_id=80, qty=100.0, aux_price=95.0, status="Submitted"):
        return _make_trade(
            order_id=order_id, order_type="STP",
            total_qty=qty, aux_price=aux_price, lmt_price=0.0, status=status,
        )

    def test_modify_stop_order_updates_quantity(self):
        client = self._client()
        trade = self._stop_trade(order_id=80, qty=100.0, aux_price=95.0)
        client._ib.trades.return_value = [trade]
        client._ib.placeOrder.return_value = trade

        client.modify_stop_order("80", new_qty=20)

        assert float(trade.order.totalQuantity) == pytest.approx(20.0)
        assert float(trade.order.auxPrice) == pytest.approx(95.0)  # unchanged
        client._ib.placeOrder.assert_called_once_with(trade.contract, trade.order)

    def test_modify_stop_order_updates_stop_price(self):
        client = self._client()
        trade = self._stop_trade(order_id=81, qty=100.0, aux_price=95.0)
        client._ib.trades.return_value = [trade]
        client._ib.placeOrder.return_value = trade

        client.modify_stop_order("81", new_stop_price=100.10)

        assert float(trade.order.auxPrice) == pytest.approx(100.10)
        assert float(trade.order.totalQuantity) == pytest.approx(100.0)  # unchanged

    def test_modify_stop_order_updates_both(self):
        client = self._client()
        trade = self._stop_trade(order_id=82, qty=100.0, aux_price=95.0)
        client._ib.trades.return_value = [trade]
        client._ib.placeOrder.return_value = trade

        client.modify_stop_order("82", new_qty=65, new_stop_price=100.10)

        assert float(trade.order.totalQuantity) == pytest.approx(65.0)
        assert float(trade.order.auxPrice) == pytest.approx(100.10)
        assert client._ib.placeOrder.call_count == 1  # single atomic call

    def test_modify_stop_order_raises_if_disconnected(self):
        client = self._client()
        client._ib.isConnected.return_value = False

        with pytest.raises(ConnectionError):
            client.modify_stop_order("80", new_qty=20)

    def test_modify_stop_order_raises_if_no_changes_requested(self):
        client = self._client()
        with pytest.raises(ValueError, match="requires new_qty or new_stop_price"):
            client.modify_stop_order("80")

    def test_modify_stop_order_raises_if_order_not_found(self):
        client = self._client()
        client._ib.trades.return_value = []

        with pytest.raises(KeyError, match="9999"):
            client.modify_stop_order("9999", new_qty=10)

    def test_modify_stop_order_raises_if_terminal(self):
        client = self._client()
        for terminal_status in ("Filled", "Cancelled"):
            trade = self._stop_trade(order_id=83, status=terminal_status)
            client._ib.trades.return_value = [trade]
            with pytest.raises(ValueError, match="terminal state"):
                client.modify_stop_order("83", new_qty=10)

    def test_modify_stop_order_raises_if_not_stop_order(self):
        client = self._client()
        trade = _make_trade(order_id=84, order_type="LMT", status="Submitted")
        client._ib.trades.return_value = [trade]

        with pytest.raises(ValueError, match="non-stop order"):
            client.modify_stop_order("84", new_stop_price=100.0)

    def test_modify_stop_order_caches_updated_state(self):
        client = self._client()
        trade = self._stop_trade(order_id=85, qty=100.0, aux_price=95.0)
        client._ib.trades.return_value = [trade]
        client._ib.placeOrder.return_value = trade

        client.modify_stop_order("85", new_qty=20, new_stop_price=100.10)

        assert 85 in client._order_cache


class TestGetOrder:
    def test_get_order_from_cache(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        expected = {"id": "5", "status": "Filled"}
        client._order_cache[5] = expected

        assert client.get_order("5") is expected

    def test_get_order_unknown_raises(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        with pytest.raises(KeyError, match="999"):
            client.get_order("999")


class TestCancelOrder:
    def test_cancel_order_calls_ib(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        submitted = _make_trade(order_id=10, status="Submitted")
        cancelled = _make_trade(order_id=10, status="Cancelled")
        client._ib.trades.side_effect = [[submitted], [cancelled]]

        result = client.cancel_order("10")

        client._ib.cancelOrder.assert_called_once_with(submitted.order)
        assert result is True

    def test_cancel_unknown_silent(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        client._ib.trades.return_value = []

        result = client.cancel_order("9999")  # no exception
        assert result is True

    def test_cancel_blocks_until_terminal(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        submitted = _make_trade(order_id=10, status="Submitted")
        cancelled = _make_trade(order_id=10, status="Cancelled")
        # Stays Submitted for one extra poll, then transitions.
        client._ib.trades.side_effect = [[submitted], [submitted], [cancelled]]

        result = client.cancel_order("10")
        assert result is True
        client._ib.cancelOrder.assert_called_once()

    def test_cancel_when_already_filled_returns_false(self):
        """Order already Filled before cancel is called → returns False + logs warning."""
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)
        mock_log = MagicMock()
        client._log = mock_log
        filled = _make_trade(order_id=11, status="Filled")
        client._ib.trades.return_value = [filled]

        result = client.cancel_order("11")
        assert result is False
        client._ib.cancelOrder.assert_not_called()
        mock_log.warning.assert_called_once()

    def test_cancel_fire_and_forget_returns_true_when_pending(self):
        """Active order: cancelOrder called once, True returned immediately (no wait)."""
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)
        submitted = _make_trade(order_id=12, status="Submitted")
        client._ib.trades.return_value = [submitted]

        result = client.cancel_order("12", timeout=0.1)
        assert result is True
        client._ib.cancelOrder.assert_called_once()
        client._ib.sleep.assert_not_called()

    def test_pending_cancel_not_terminal(self):
        """PendingCancel must not cause cancel_order to return early."""
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        pending   = _make_trade(order_id=13, status="PendingCancel")
        cancelled = _make_trade(order_id=13, status="Cancelled")
        client._ib.trades.side_effect = [[pending], [pending], [cancelled]]

        result = client.cancel_order("13")
        assert result is True

    def test_nan_bid_ask_sizes(self):
        """NaN bid_size / ask_size must be coerced to 0, not raise ValueError."""
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = True
        client._ib.reqMktData.return_value = _make_ticker(
            bid=10.0, ask=10.5, bid_size=float("nan"), ask_size=float("nan"),
        )
        result = client.get_latest_quote("TQQQ")
        assert result["bid_size"] == 0
        assert result["ask_size"] == 0


class TestEmergencyMethods:
    """cancel_all_orders, close_position, close_all_positions, list_orders."""

    def _client(self) -> IBClient:
        with patch("orb_live.data.ib_client.IB"):
            return IBClient(paper=True)

    # ── cancel_all_orders ────────────────────────────────────────────────────

    def test_cancel_all_orders_cancels_each_open_trade(self):
        client = self._client()
        t1 = _make_trade(order_id=1, status="Submitted")
        t2 = _make_trade(order_id=2, status="Submitted")
        client._ib.openTrades.return_value = [t1, t2]

        count = client.cancel_all_orders()

        assert count == 2
        assert client._ib.cancelOrder.call_count == 2

    def test_cancel_all_orders_returns_zero_when_no_open_orders(self):
        client = self._client()
        client._ib.openTrades.return_value = []
        assert client.cancel_all_orders() == 0

    def test_cancel_all_orders_skips_failed_cancel(self):
        client = self._client()
        mock_log = MagicMock()
        client._log = mock_log
        t1 = _make_trade(order_id=1, status="Submitted")
        t2 = _make_trade(order_id=2, status="Submitted")
        client._ib.openTrades.return_value = [t1, t2]
        client._ib.cancelOrder.side_effect = [Exception("timeout"), None]

        count = client.cancel_all_orders()

        assert count == 1   # t2 succeeded; t1 failed
        mock_log.warning.assert_called_once()

    # ── close_position ───────────────────────────────────────────────────────

    def test_close_position_submits_sell_for_long(self):
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = True
        item = SimpleNamespace(
            contract=SimpleNamespace(symbol="TQQQ"),
            position=50.0,
            marketValue=5000.0,
            averageCost=100.0,
            unrealizedPNL=0.0,
        )
        client._ib.portfolio.return_value = [item]

        result = client.close_position("TQQQ")

        assert result is not None
        placed = client._ib.placeOrder.call_args
        assert placed.args[1].action == "SELL"
        assert placed.args[1].totalQuantity == 50.0

    def test_close_position_submits_buy_for_short(self):
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = True
        item = SimpleNamespace(
            contract=SimpleNamespace(symbol="TQQQ"),
            position=-30.0,
            marketValue=-3000.0,
            averageCost=100.0,
            unrealizedPNL=0.0,
        )
        client._ib.portfolio.return_value = [item]

        result = client.close_position("TQQQ")

        assert result is not None
        placed = client._ib.placeOrder.call_args
        assert placed.args[1].action == "BUY"
        assert placed.args[1].totalQuantity == 30.0

    def test_close_position_returns_none_when_flat(self):
        client = _client_with_asset("TQQQ")
        client._ib.portfolio.return_value = []

        result = client.close_position("TQQQ")
        assert result is None
        client._ib.placeOrder.assert_not_called()

    # ── close_all_positions ──────────────────────────────────────────────────

    def test_close_all_positions_closes_each(self):
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = True
        item = SimpleNamespace(
            contract=SimpleNamespace(symbol="TQQQ"),
            position=100.0,
            marketValue=10000.0,
            averageCost=100.0,
            unrealizedPNL=0.0,
        )
        client._ib.portfolio.return_value = [item]

        client.close_all_positions()

        client._ib.placeOrder.assert_called_once()
        placed = client._ib.placeOrder.call_args
        assert placed.args[1].action == "SELL"

    def test_close_all_positions_logs_failure(self):
        client = _client_with_asset("TQQQ")
        mock_log = MagicMock()
        client._log = mock_log
        client._ib.isConnected.return_value = True
        item = SimpleNamespace(
            contract=SimpleNamespace(symbol="TQQQ"),
            position=10.0,
            marketValue=1000.0,
            averageCost=100.0,
            unrealizedPNL=0.0,
        )
        client._ib.portfolio.return_value = [item]
        client._ib.placeOrder.side_effect = Exception("IB disconnect")

        client.close_all_positions()   # must not raise

        mock_log.critical.assert_called_once()

    # ── list_orders ──────────────────────────────────────────────────────────

    def test_list_orders_open_returns_open_trades(self):
        client = self._client()
        t1 = _make_trade(order_id=1, status="Submitted")
        t2 = _make_trade(order_id=2, status="Submitted")
        client._ib.openTrades.return_value = [t1, t2]

        orders = client.list_orders(status="open")

        assert len(orders) == 2
        assert all(o["status"] == "new" for o in orders)

    def test_list_orders_filled_filters_trades(self):
        client = self._client()
        submitted = _make_trade(order_id=1, status="Submitted")
        filled    = _make_trade(order_id=2, status="Filled", filled=100.0, remaining=0.0)
        client._ib.trades.return_value = [submitted, filled]
        client._ib.openTrades.return_value = [submitted]

        orders = client.list_orders(status="filled")

        assert len(orders) == 1
        assert orders[0]["id"] == "2"

    def test_list_orders_empty_when_none(self):
        client = self._client()
        client._ib.openTrades.return_value = []
        assert client.list_orders() == []


class TestRoundToTick:
    """BUG 7 — price rounding to minTick."""

    def _r(self, price, tick, side):
        return IBClient._round_to_tick(price, tick, side)

    def test_buy_rounds_up(self):
        assert self._r(26.603369999999998, 0.01, "buy") == pytest.approx(26.61)

    def test_sell_rounds_down(self):
        assert self._r(26.603369999999998, 0.01, "sell") == pytest.approx(26.60)

    def test_already_on_tick_unchanged(self):
        assert self._r(26.60, 0.01, "buy")  == pytest.approx(26.60)
        assert self._r(26.60, 0.01, "sell") == pytest.approx(26.60)

    def test_zero_tick_passthrough(self):
        assert self._r(26.603369999999998, 0.0, "buy") == pytest.approx(26.603369999999998)

    def test_subpenny_tick_clamped_to_penny_above_1(self):
        """IB reports minTick 0.0001 for some ETFs (e.g. ETHU), but orders ≥ $1
        must be penny-conforming or IB rejects with Error 110. Rounding must
        clamp to $0.01 — not produce a sub-penny like 15.4445."""
        sell = self._r(15.44454, 0.0001, "sell")
        buy  = self._r(15.80001, 0.0001, "buy")
        assert sell == pytest.approx(15.44)
        assert buy  == pytest.approx(15.81)
        # both must be exact multiples of a penny
        for p in (sell, buy):
            assert round(p * 100) == pytest.approx(p * 100), f"{p} is sub-penny"

    def test_subdollar_keeps_fine_tick(self):
        """Below $1, sub-penny pricing IS allowed, so the fine tick is kept."""
        assert self._r(0.44454, 0.0001, "sell") == pytest.approx(0.4445)

    def test_submitted_price_is_multiple_of_mintick(self):
        """submit_limit_order must round price so IB never sees a sub-tick value."""
        client = _client_with_asset("TQQQ", min_tick=0.01)
        pending = _make_trade(order_id=60, status="PendingSubmit")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = pending

        # sub-penny price that caused IB Error 110 in production
        client.submit_limit_order("TQQQ", "buy", 10, 26.603369999999998)

        _, placed_order = client._ib.placeOrder.call_args[0]
        price = placed_order.lmtPrice
        tick  = client.get_min_tick("TQQQ")
        remainder = round(price / tick) * tick
        assert price == pytest.approx(remainder), f"price {price} not on tick {tick}"

    def test_sell_stop_price_rounds_down(self):
        """submit_stop_order sell must round stop price DOWN to stay marketable."""
        client = _client_with_asset("SOXL", min_tick=0.01)
        pending = _make_trade(order_id=61, status="PendingSubmit")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = pending

        client.submit_stop_order("SOXL", "sell", 1, 185.507)

        _, placed_order = client._ib.placeOrder.call_args[0]
        assert placed_order.auxPrice == pytest.approx(185.50)


class TestNonBlockingSubmit:
    """BUG 0e — submit methods must not call ib.sleep (safe in ib_async callbacks)."""

    def test_submit_limit_order_does_not_call_ib_sleep(self):
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = _make_trade(order_id=70, status="PendingSubmit")
        client.submit_limit_order("TQQQ", "buy", 5, 50.0)
        client._ib.sleep.assert_not_called()

    def test_submit_stop_order_does_not_call_ib_sleep(self):
        client = _client_with_asset("SOXL")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = _make_trade(order_id=71, status="PendingSubmit")
        client.submit_stop_order("SOXL", "sell", 1, 185.0)
        client._ib.sleep.assert_not_called()

    def test_submit_market_order_does_not_call_ib_sleep(self):
        client = _client_with_asset("SOXL")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = _make_trade(order_id=72, status="PendingSubmit")
        client.submit_market_order("SOXL", "sell", 1)
        client._ib.sleep.assert_not_called()

    def test_submit_from_running_asyncio_loop_no_reentrance(self):
        """
        Calling submit_limit_order from inside a running asyncio event loop must
        not raise 'This event loop is already running'.
        """
        import asyncio

        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = True
        client._ib.placeOrder.return_value  = _make_trade(order_id=73, status="PendingSubmit")

        errors = []

        async def _run():
            try:
                # Simulate being called from inside an ib_async event callback:
                # the asyncio loop is already running at this point.
                client.submit_limit_order("TQQQ", "buy", 10, 50.0)
            except RuntimeError as exc:
                errors.append(str(exc))

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()
        assert errors == [], f"submit raised from async context: {errors}"


class TestOrderEventHandlers:
    def test_event_handler_updates_cache(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        trade = _make_trade(order_id=20, status="Submitted")
        client._on_order_status(trade)

        assert 20 in client._order_cache
        assert client._order_cache[20]["status"] == "new"

    def test_filled_status_triggers_log(self):
        mock_log = MagicMock()
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True, logger=mock_log)

        trade = _make_trade(order_id=21, status="Filled", filled=100.0)
        client._on_order_status(trade)

        mock_log.info.assert_called_once()
        call_kwargs = mock_log.info.call_args
        assert "order_status_change" in call_kwargs[0] or "Filled" in str(call_kwargs)

    def test_exec_details_updates_fill_fields(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        trade = _make_trade(order_id=30)
        client._order_cache[30] = {"id": "30", "status": "Submitted"}

        fill = SimpleNamespace(execution=SimpleNamespace(price=54.75, shares=50))
        client._on_exec_details(trade, fill)

        assert client._order_cache[30]["last_fill_price"] == pytest.approx(54.75)
        assert client._order_cache[30]["last_fill_qty"]   == pytest.approx(50)

    def test_exec_details_ignores_unknown_order(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        trade = _make_trade(order_id=9999)
        fill  = SimpleNamespace(execution=SimpleNamespace(price=1.0, shares=1))
        client._on_exec_details(trade, fill)  # must not raise
        assert 9999 not in client._order_cache


class TestTradeToDict:
    def _client(self):
        with patch("orb_live.data.ib_client.IB"):
            return IBClient(paper=True)

    def test_trade_to_dict_buy_side(self):
        client = self._client()
        trade  = _make_trade(action="BUY", order_type="LMT", lmt_price=55.0)
        d = client._trade_to_dict(trade)
        assert d["side"]        == "buy"
        assert d["limit_price"] == pytest.approx(55.0)

    def test_trade_to_dict_sell_side(self):
        client = self._client()
        trade  = _make_trade(action="SELL", order_type="LMT", lmt_price=60.0)
        d = client._trade_to_dict(trade)
        assert d["side"] == "sell"

    def test_trade_to_dict_market_order_limit_price_is_none(self):
        client = self._client()
        trade  = _make_trade(order_type="MKT", lmt_price=0.0)
        d = client._trade_to_dict(trade)
        assert d["limit_price"] is None

    def test_trade_to_dict_has_id_key(self):
        client = self._client()
        trade  = _make_trade(order_id=77)
        d = client._trade_to_dict(trade)
        assert d["id"] == "77"
        assert "alpaca_id" not in d

    def test_trade_to_dict_status_is_normalized(self):
        client = self._client()
        trade  = _make_trade(status="Submitted")
        d = client._trade_to_dict(trade)
        assert d["status"] == "new"

    def test_trade_to_dict_status_raw_preserved(self):
        client = self._client()
        trade  = _make_trade(status="Filled")
        d = client._trade_to_dict(trade)
        assert d["status"]     == "filled"
        assert d["status_raw"] == "Filled"

    def test_trade_to_dict_unknown_status_lowercased(self):
        client = self._client()
        trade  = _make_trade(status="SomeWeirdStatus")
        d = client._trade_to_dict(trade)
        assert d["status"] == "someweirdstatus"


class TestNormalizeStatus:
    def _client(self):
        with patch("orb_live.data.ib_client.IB"):
            return IBClient(paper=True)

    @pytest.mark.parametrize("ib_status,expected", [
        ("PendingSubmit",   "new"),
        ("PreSubmitted",    "new"),
        ("Submitted",       "new"),
        ("Filled",          "filled"),
        ("PartiallyFilled", "partially_filled"),
        ("Cancelled",       "canceled"),
        ("ApiCancelled",    "canceled"),
        ("Inactive",        "rejected"),
        ("Unknown",         "unknown"),
    ])
    def test_known_and_fallback_statuses(self, ib_status, expected):
        client = self._client()
        assert client._normalize_status(ib_status) == expected


# ── BarAggregator ─────────────────────────────────────────────────────────────

def _make_5sec_bar(
    dt: datetime,
    open: float = 100.0,
    high: float = 101.0,
    low:  float = 99.0,
    close: float = 100.5,
    volume: float = 1000.0,
) -> SimpleNamespace:
    return SimpleNamespace(time=dt, open_=open, high=high, low=low,
                           close=close, volume=volume)


def _min(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_ET)


class TestBarAggregator:
    def test_aggregator_first_bar_returns_none(self):
        agg = BarAggregator("TQQQ")
        bar = _make_5sec_bar(_min(2026, 5, 19, 10, 0))
        assert agg.add_5sec_bar(bar) is None

    def test_aggregator_same_minute_updates_high(self):
        agg = BarAggregator("TQQQ")
        b1 = _make_5sec_bar(_min(2026, 5, 19, 10, 0), high=101.0)
        b2 = _make_5sec_bar(_min(2026, 5, 19, 10, 0).replace(second=5), high=103.0)
        agg.add_5sec_bar(b1)
        assert agg.add_5sec_bar(b2) is None  # still same minute
        assert agg.high == pytest.approx(103.0)

    def test_aggregator_new_minute_emits_completed(self):
        agg = BarAggregator("TQQQ")
        b1 = _make_5sec_bar(_min(2026, 5, 19, 10, 0),
                             open=100.0, high=102.0, low=99.0, close=101.0, volume=500)
        b2 = _make_5sec_bar(_min(2026, 5, 19, 10, 1),  # new minute
                             open=101.0, high=103.0, low=100.0, close=102.0, volume=200)
        assert agg.add_5sec_bar(b1) is None
        completed = agg.add_5sec_bar(b2)

        assert completed is not None
        assert completed["symbol"]    == "TQQQ"
        assert completed["timestamp"] == _min(2026, 5, 19, 10, 0)
        assert completed["open"]      == pytest.approx(100.0)
        assert completed["high"]      == pytest.approx(102.0)
        assert completed["low"]       == pytest.approx(99.0)
        assert completed["close"]     == pytest.approx(101.0)
        assert completed["volume"]    == pytest.approx(500.0)

    def test_aggregator_volume_accumulates(self):
        agg = BarAggregator("SOXL")
        for i in range(12):
            t = datetime(2026, 5, 19, 10, 0, i * 5, tzinfo=_ET)
            agg.add_5sec_bar(_make_5sec_bar(t, volume=100.0))
        assert agg.volume == pytest.approx(1200.0)

    def test_aggregator_finalize_returns_current_state(self):
        agg = BarAggregator("UVXY")
        agg.add_5sec_bar(_make_5sec_bar(_min(2026, 5, 19, 10, 0),
                                        open=50.0, high=52.0, low=49.0,
                                        close=51.0, volume=300))
        completed = agg.finalize()
        assert completed is not None
        assert completed["open"]  == pytest.approx(50.0)
        assert completed["close"] == pytest.approx(51.0)

    def test_aggregator_finalize_empty_returns_none(self):
        agg = BarAggregator("TQQQ")
        assert agg.finalize() is None

    def test_aggregator_naive_datetime_handled(self):
        agg = BarAggregator("TQQQ")
        naive = datetime(2026, 5, 19, 10, 0, 0)   # no tzinfo
        assert agg.add_5sec_bar(_make_5sec_bar(naive)) is None
        assert agg.current_minute is not None


# ── get_intraday_bars ─────────────────────────────────────────────────────────

def _make_hist_bar(dt: datetime, open=100.0, high=101.0, low=99.0,
                   close=100.5, volume=500.0) -> SimpleNamespace:
    # formatDate=1: bar.date is a naive datetime in local exchange time (ET).
    naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
    return SimpleNamespace(date=naive, open=open, high=high,
                           low=low, close=close, volume=volume)


class TestGetIntradayBars:
    def _client(self, tradable=True, symbol="TQQQ") -> IBClient:
        client = _client_with_asset(symbol, tradable=tradable)
        client._ib.isConnected.return_value = True
        return client

    def test_get_intraday_bars_empty_for_non_tradable(self):
        client = self._client(tradable=False, symbol="BADTICKER")
        start  = datetime(2026, 5, 19, 9, 30, tzinfo=_ET)
        end    = datetime(2026, 5, 19, 10, 0, tzinfo=_ET)

        df = client.get_intraday_bars("BADTICKER", start, end)

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_intraday_bars_returns_empty_when_no_data(self):
        client = self._client()
        client._ib.reqHistoricalData.return_value = []

        start = datetime(2026, 5, 19, 9, 30, tzinfo=_ET)
        end   = datetime(2026, 5, 19, 10, 0, tzinfo=_ET)
        df    = client.get_intraday_bars("TQQQ", start, end)

        assert df.empty

    def test_get_intraday_bars_filters_to_window(self):
        start = datetime(2026, 5, 19, 9, 30, tzinfo=_ET)
        end   = datetime(2026, 5, 19, 10, 0, tzinfo=_ET)

        dt_before = datetime(2026, 5, 19, 9, 15)   # naive ET — before window
        dt_in     = datetime(2026, 5, 19, 9, 35)   # naive ET — inside window
        dt_after  = datetime(2026, 5, 19, 10, 15)  # naive ET — after window

        client = self._client()
        client._ib.reqHistoricalData.return_value = [
            _make_hist_bar(dt_before),
            _make_hist_bar(dt_in),
            _make_hist_bar(dt_after),
        ]

        df = client.get_intraday_bars("TQQQ", start, end)
        assert len(df) == 1

    def test_get_intraday_bars_dataframe_shape(self):
        start  = datetime(2026, 5, 19, 9, 30, tzinfo=_ET)
        end    = datetime(2026, 5, 19, 10, 0, tzinfo=_ET)
        dt_in  = datetime(2026, 5, 19, 9, 35)  # naive ET — inside window

        client = self._client()
        client._ib.reqHistoricalData.return_value = [_make_hist_bar(dt_in)]

        df = client.get_intraday_bars("TQQQ", start, end)

        assert not df.empty
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns
        assert df.index.tzinfo is not None

    def test_get_intraday_bars_raises_if_disconnected(self):
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = False

        start = datetime(2026, 5, 19, 9, 30, tzinfo=_ET)
        end   = datetime(2026, 5, 19, 10, 0, tzinfo=_ET)
        with pytest.raises(ConnectionError):
            client.get_intraday_bars("TQQQ", start, end)

    def test_get_intraday_bars_uses_empty_endtime_for_now(self):
        from datetime import timezone
        client = self._client()
        client._ib.reqHistoricalData.return_value = []

        now   = datetime.now(timezone.utc)
        start = now - timedelta(minutes=30)
        client.get_intraday_bars("TQQQ", start, now)

        call_kwargs = client._ib.reqHistoricalData.call_args.kwargs
        assert call_kwargs.get("endDateTime") == ""

    def test_get_intraday_bars_uses_explicit_endtime_for_past(self):
        from datetime import timezone
        client = self._client()
        client._ib.reqHistoricalData.return_value = []

        past_end = datetime(2026, 5, 22, 14, 30, tzinfo=timezone.utc)
        client.get_intraday_bars("TQQQ", past_end - timedelta(minutes=30), past_end)

        call_kwargs = client._ib.reqHistoricalData.call_args.kwargs
        end_dt = call_kwargs.get("endDateTime")
        # A datetime object is passed directly (ib_async formats it correctly).
        assert isinstance(end_dt, datetime)
        assert end_dt != ""

    def test_get_intraday_bars_returns_empty_on_timeout(self):
        client = self._client()
        client._ib.reqHistoricalData.side_effect = Exception("Timed out")

        start = datetime(2026, 5, 22, 9, 30, tzinfo=_ET)
        end   = datetime(2026, 5, 22, 10, 0, tzinfo=_ET)
        df = client.get_intraday_bars("TQQQ", start, end)

        assert df.empty

    def test_get_intraday_bars_passes_timeout_kwarg(self):
        client = self._client()
        client._ib.reqHistoricalData.return_value = []

        start = datetime(2026, 5, 22, 9, 30, tzinfo=_ET)
        end   = datetime(2026, 5, 22, 10, 0, tzinfo=_ET)
        client.get_intraday_bars("TQQQ", start, end)

        call_kwargs = client._ib.reqHistoricalData.call_args.kwargs
        assert call_kwargs.get("timeout") == 15


# ── get_daily_bars ─────────────────────────────────────────────────────────────

def _make_daily_bar(dt: datetime, open=100.0, high=102.0, low=99.0,
                    close=101.0, volume=1_000_000.0) -> SimpleNamespace:
    """Daily bar where bar.date is a tz-aware datetime (formatDate=1 behaviour)."""
    return SimpleNamespace(date=dt, open=open, high=high,
                           low=low, close=close, volume=volume)


class TestGetDailyBars:
    def _client(self, tradable=True, symbol="TQQQ") -> IBClient:
        client = _client_with_asset(symbol, tradable=tradable)
        client._ib.isConnected.return_value = True
        return client

    def _bars(self, n: int = 5):
        """Return n daily bar stubs ending 2026-05-20."""
        base = datetime(2026, 5, 14, 16, 0, tzinfo=_ET)
        return [_make_daily_bar(base + timedelta(days=i)) for i in range(n)]

    def test_get_daily_bars_shape(self):
        client = self._client()
        client._ib.reqHistoricalData.return_value = self._bars(5)
        df = client.get_daily_bars("TQQQ", lookback_days=10)
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(df) == 5

    def test_get_daily_bars_date_is_naive(self):
        client = self._client()
        client._ib.reqHistoricalData.return_value = self._bars(3)
        df = client.get_daily_bars("TQQQ")
        assert df["date"].dt.tz is None

    def test_get_daily_bars_date_is_normalized(self):
        client = self._client()
        client._ib.reqHistoricalData.return_value = self._bars(2)
        df = client.get_daily_bars("TQQQ")
        for ts in df["date"]:
            assert ts == ts.normalize()

    def test_get_daily_bars_sorted_ascending(self):
        client = self._client()
        client._ib.reqHistoricalData.return_value = self._bars(5)
        df = client.get_daily_bars("TQQQ")
        assert list(df["date"]) == sorted(df["date"].tolist())

    def test_get_daily_bars_lookback_limit(self):
        client = self._client()
        client._ib.reqHistoricalData.return_value = self._bars(10)
        df = client.get_daily_bars("TQQQ", lookback_days=5)
        assert len(df) == 5

    def test_get_daily_bars_empty_on_no_data(self):
        client = self._client()
        client._ib.reqHistoricalData.return_value = []
        df = client.get_daily_bars("TQQQ")
        assert df.empty
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_get_daily_bars_non_tradable_returns_empty(self):
        client = self._client(tradable=False)
        df = client.get_daily_bars("TQQQ")
        assert df.empty

    def test_get_daily_bars_disconnected_raises(self):
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = False
        with pytest.raises(ConnectionError):
            client.get_daily_bars("TQQQ")

    def test_get_daily_bars_exception_returns_empty(self):
        client = self._client()
        client._ib.reqHistoricalData.side_effect = Exception("timeout")
        df = client.get_daily_bars("TQQQ")
        assert df.empty


# ── get_latest_quote ──────────────────────────────────────────────────────────

def _make_ticker(bid=10.0, ask=10.5, bid_size=100, ask_size=200,
                 last=0.0, close=0.0, time=None) -> SimpleNamespace:
    return SimpleNamespace(
        bid=bid, ask=ask, bidSize=bid_size, askSize=ask_size,
        last=last, close=close, time=time,
    )


class TestGetLatestQuote:
    def _client(self, tradable=True) -> IBClient:
        client = _client_with_asset("TQQQ", tradable=tradable)
        client._ib.isConnected.return_value = True
        return client

    def test_get_latest_quote_returns_dict_shape(self):
        client = self._client()
        client._ib.reqMktData.return_value = _make_ticker(bid=10.0, ask=10.5)
        result = client.get_latest_quote("TQQQ")
        for key in ("bid", "ask", "bid_size", "ask_size", "ts"):
            assert key in result

    def test_get_latest_quote_values(self):
        client = self._client()
        client._ib.reqMktData.return_value = _make_ticker(
            bid=10.0, ask=10.5, bid_size=100, ask_size=200,
        )
        result = client.get_latest_quote("TQQQ")
        assert result["bid"]      == pytest.approx(10.0)
        assert result["ask"]      == pytest.approx(10.5)
        assert result["bid_size"] == 100
        assert result["ask_size"] == 200

    def test_get_latest_quote_falls_back_to_last(self):
        client = self._client()
        client._ib.reqMktData.return_value = _make_ticker(
            bid=0.0, ask=0.0, last=10.5,
        )
        result = client.get_latest_quote("TQQQ")
        assert result["bid"] == pytest.approx(10.5)
        assert result["ask"] == pytest.approx(10.5)

    def test_get_latest_quote_falls_back_to_close_if_no_last(self):
        client = self._client()
        client._ib.reqMktData.return_value = _make_ticker(
            bid=0.0, ask=0.0, last=0.0, close=9.0,
        )
        result = client.get_latest_quote("TQQQ")
        assert result["bid"] == pytest.approx(9.0)
        assert result["ask"] == pytest.approx(9.0)

    def test_get_latest_quote_all_zero_raises(self):
        client = self._client()
        client._ib.reqMktData.return_value = _make_ticker(
            bid=0.0, ask=0.0, last=0.0, close=0.0,
        )
        with pytest.raises(RuntimeError, match="all-zero"):
            client.get_latest_quote("TQQQ")

    def test_get_latest_quote_cancels_market_data(self):
        client = self._client()
        client._ib.reqMktData.return_value = _make_ticker()
        client.get_latest_quote("TQQQ")
        client._ib.cancelMktData.assert_called_once()

    def test_get_latest_quote_non_tradable_returns_zero(self):
        client = self._client(tradable=False)
        result = client.get_latest_quote("TQQQ")
        assert result == {"bid": 0.0, "ask": 0.0, "bid_size": 0, "ask_size": 0, "ts": None}

    def test_get_latest_quote_disconnected_raises(self):
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = False
        with pytest.raises(ConnectionError):
            client.get_latest_quote("TQQQ")

    def test_get_latest_quote_exception_raises(self):
        client = self._client()
        client._ib.reqMktData.side_effect = Exception("network error")
        with pytest.raises(Exception, match="network error"):
            client.get_latest_quote("TQQQ")

    def test_get_latest_quote_cancels_even_on_exception(self):
        client = self._client()
        client._ib.reqMktData.side_effect = Exception("boom")
        with pytest.raises(Exception):
            client.get_latest_quote("TQQQ")
        client._ib.cancelMktData.assert_called_once()


# ── Market data guards ────────────────────────────────────────────────────────

class TestMarketDataGuards:
    def _client(self, tradable: bool = True) -> IBClient:
        client = _client_with_asset("TQQQ", tradable=tradable)
        client._ib.isConnected.return_value = True
        return client

    def test_on_ib_error_10089_sets_degraded(self):
        client = self._client()
        assert not client._market_data_degraded
        client._on_ib_error(1, 10089, "Market data farm connection is OK:ddfarm1", None)
        assert client._market_data_degraded

    def test_on_ib_error_other_code_does_not_set_degraded(self):
        client = self._client()
        client._on_ib_error(1, 200, "No security definition has been found", None)
        assert not client._market_data_degraded

    def test_get_latest_quote_raises_when_degraded(self):
        client = self._client()
        client._market_data_degraded = True
        with pytest.raises(RuntimeError, match="10089"):
            client.get_latest_quote("TQQQ")

    def test_get_latest_quote_raises_on_zero_quote(self):
        client = self._client()
        client._ib.reqMktData.return_value = _make_ticker(
            bid=0.0, ask=0.0, last=0.0, close=0.0,
        )
        with pytest.raises(RuntimeError, match="all-zero"):
            client.get_latest_quote("TQQQ")

    def test_allow_delayed_overrides_zero_check(self):
        client = self._client()
        client._allow_delayed_data = True
        client._ib.reqMktData.return_value = _make_ticker(
            bid=0.0, ask=0.0, last=0.0, close=0.0,
        )
        result = client.get_latest_quote("TQQQ")
        assert result["bid"] == 0.0
        assert result["ask"] == 0.0

    def test_allow_delayed_overrides_degraded_check(self):
        client = self._client()
        client._allow_delayed_data = True
        client._market_data_degraded = True
        client._ib.reqMktData.return_value = _make_ticker(bid=10.0, ask=10.5)
        result = client.get_latest_quote("TQQQ")
        assert result["bid"] == pytest.approx(10.0)

    def test_validate_subscription_raises_connection_error_on_zero(self):
        """_validate_subscription translates RuntimeError → ConnectionError."""
        client = _client_with_asset("SPY")
        client._ib.isConnected.return_value = True
        client._ib.reqMktData.return_value = _make_ticker(
            bid=0.0, ask=0.0, last=0.0, close=0.0,
        )
        with pytest.raises(ConnectionError, match="validation"):
            client._validate_subscription()

    def test_validate_subscription_passes_on_good_quote(self):
        """_validate_subscription does not raise when SPY returns a real quote."""
        client = _client_with_asset("SPY")
        client._ib.isConnected.return_value = True
        client._ib.reqMktData.return_value = _make_ticker(bid=500.0, ask=500.1)
        client._validate_subscription()  # must not raise


# ── subscribe_bars / stop_bars_stream ─────────────────────────────────────────

class TestSubscribeBars:
    def _client(self) -> IBClient:
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = True
        return client

    def test_subscribe_bars_calls_reqRealTimeBars(self):
        client = self._client()
        mock_stream = MagicMock()
        client._ib.reqRealTimeBars.return_value = mock_stream

        client.subscribe_bars(["TQQQ"], callback=lambda bar: None)

        client._ib.reqRealTimeBars.assert_called_once()
        assert "TQQQ" in client._streams

    def test_subscribe_bars_skips_untradable(self):
        client = _client_with_asset("BADTICKER", tradable=False)
        client._ib.isConnected.return_value = True

        client.subscribe_bars(["BADTICKER"], callback=lambda bar: None)

        client._ib.reqRealTimeBars.assert_not_called()
        assert "BADTICKER" not in client._streams

    def test_subscribe_bars_idempotent(self):
        client = self._client()
        mock_stream = MagicMock()
        client._ib.reqRealTimeBars.return_value = mock_stream

        client.subscribe_bars(["TQQQ"], callback=lambda bar: None)
        client.subscribe_bars(["TQQQ"], callback=lambda bar: None)  # second call

        assert client._ib.reqRealTimeBars.call_count == 1

    def test_subscribe_bars_attaches_listener(self):
        client = self._client()
        mock_stream = MagicMock()
        client._ib.reqRealTimeBars.return_value = mock_stream

        client.subscribe_bars(["TQQQ"], callback=lambda bar: None)

        # Aggregator must be created so the listener can service incoming bars
        assert "TQQQ" in client._bar_aggregators
        assert isinstance(client._bar_aggregators["TQQQ"], BarAggregator)

    def test_entry_callback_fires_on_every_5sec_bar(self):
        """The entry_callback must receive a bar dict on EVERY 5-sec bar, while
        the 1-min callback only fires when a minute completes."""
        class _FakeEvent:
            def __init__(self): self.handlers = []
            def __iadd__(self, fn): self.handlers.append(fn); return self

        client = self._client()
        mock_stream = MagicMock()
        mock_stream.updateEvent = _FakeEvent()
        client._ib.reqRealTimeBars.return_value = mock_stream

        minute_bars: list = []
        entry_bars:  list = []
        client.subscribe_bars(
            ["TQQQ"], callback=minute_bars.append,
            entry_callback=entry_bars.append,
        )
        listener = mock_stream.updateEvent.handlers[0]

        # Two 5-sec bars in the same minute → 2 entry callbacks, 0 minute bars
        listener([_make_5sec_bar(_min(2026, 5, 19, 10, 0), high=101.0)], True)
        listener([_make_5sec_bar(
            _min(2026, 5, 19, 10, 0).replace(second=5), high=103.0)], True)
        assert len(entry_bars) == 2
        assert entry_bars[0]["symbol"] == "TQQQ"
        assert entry_bars[1]["high"] == 103.0
        assert minute_bars == []

        # A bar in the next minute completes the first → 1 minute bar emitted
        listener([_make_5sec_bar(_min(2026, 5, 19, 10, 1))], True)
        assert len(entry_bars) == 3
        assert len(minute_bars) == 1


class TestStopBarsStream:
    def _subscribed_client(self, symbol: str = "TQQQ"):
        client = _client_with_asset(symbol)
        client._ib.isConnected.return_value = True

        mock_stream = MagicMock()
        client._ib.reqRealTimeBars.return_value = mock_stream
        client.subscribe_bars([symbol], callback=lambda bar: None)
        return client, mock_stream

    def test_stop_bars_stream_clears_streams(self):
        client, _ = self._subscribed_client()
        client.stop_bars_stream()
        assert client._streams == {}
        assert client._bar_aggregators == {}
        assert client._bar_callback is None

    def test_stop_bars_stream_cancels_each(self):
        client, mock_stream = self._subscribed_client()
        client.stop_bars_stream()
        client._ib.cancelRealTimeBars.assert_called_once_with(mock_stream)

    def test_stop_bars_stream_finalizes_pending(self):
        emitted = []
        client = _client_with_asset("TQQQ")
        client._ib.isConnected.return_value = True
        mock_stream = MagicMock()
        client._ib.reqRealTimeBars.return_value = mock_stream
        client.subscribe_bars(["TQQQ"], callback=emitted.append)

        # Feed one 5-sec bar (partial minute)
        bar = _make_5sec_bar(_min(2026, 5, 19, 10, 0))
        client._bar_aggregators["TQQQ"].add_5sec_bar(bar)

        client.stop_bars_stream()

        assert len(emitted) == 1
        assert emitted[0]["symbol"] == "TQQQ"

    def test_stop_bars_stream_no_op_when_empty(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)
        client.stop_bars_stream()  # must not raise


# ── Heartbeat ─────────────────────────────────────────────────────────────────

class TestHeartbeat:
    def _client(self) -> IBClient:
        with patch("orb_live.data.ib_client.IB"):
            return IBClient(paper=True)

    def test_heartbeat_ok_when_no_event_yet(self):
        client = self._client()
        result = client.check_heartbeat()
        assert result["ok"] is True
        assert result["seconds_since_event"] is None

    def test_heartbeat_updates_on_order_status_event(self):
        client = self._client()
        trade = _make_trade(order_id=1, status="Submitted")
        client._on_order_status(trade)
        result = client.check_heartbeat()
        assert result["ok"] is True
        assert result["seconds_since_event"] is not None
        assert result["seconds_since_event"] < 1.0

    def test_heartbeat_detects_stale_connection(self):
        client = self._client()
        client._last_event_time    = time.time() - 120
        client._heartbeat_timeout  = 60
        result = client.check_heartbeat()
        assert result["ok"] is False
        assert result["seconds_since_event"] >= 60

    def test_heartbeat_resets_after_disconnect(self):
        client = self._client()
        client._last_event_time = time.time()
        client.disconnect()
        result = client.check_heartbeat()
        assert result["ok"] is True
        assert result["seconds_since_event"] is None


# ── Interface parity ──────────────────────────────────────────────────────────

def test_broker_interface_parity():
    """IBClient must implement every BrokerClient abstractmethod.
    Every non-helper public method on MockBroker must also exist on IBClient —
    catches the case where a method is added to the mock for tests but never
    implemented on the real client (e.g. submit_bracket_order was the motivating
    bug for this test)."""
    from orb_live.tests.conftest import MockBroker

    # IBClient subclasses BrokerClient: Python populates __abstractmethods__ with
    # any unimplemented methods at class-definition time — no instantiation needed.
    missing_ib = IBClient.__abstractmethods__
    assert not missing_ib, f"IBClient missing BrokerClient abstractmethods: {missing_ib}"

    # MockBroker is a duck-type test double; it doesn't subclass BrokerClient.
    # Check that every broker-operation method on MockBroker is also on IBClient.
    # Exclude test-only helpers that have no real-client counterpart.
    _MOCK_HELPERS = frozenset({
        "set_fill_fraction", "set_fill_sequence", "set_pending_fill",
        "set_equity", "set_quote", "set_broker_position", "fire_fill_watcher",
        "fire_order_error",
    })
    mock_broker_methods = {
        name for name in vars(MockBroker)   # vars() → only class-defined; skips object builtins
        if not name.startswith("_")
        and callable(getattr(MockBroker, name))
        and name not in _MOCK_HELPERS
    }
    missing_from_ib = mock_broker_methods - set(dir(IBClient))
    assert not missing_from_ib, (
        f"IBClient missing broker methods present on MockBroker: {missing_from_ib}\n"
        "Add them to BrokerClient ABC and implement in IBClient."
    )

