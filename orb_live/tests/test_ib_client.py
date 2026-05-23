"""
tests/test_ib_client.py — Unit tests for IBClient (Parts 1-2).

All IB Gateway network calls are mocked so these run without a live connection.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from orb_live.data.ib_client import IBClient, build_client_from_env

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


def _client_with_asset(symbol: str = "TQQQ", tradable: bool = True) -> IBClient:
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
        assert result["status"]        == "Submitted"
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

        trade = _make_trade(order_id=10)
        client._ib.trades.return_value = [trade]

        result = client.cancel_order("10")

        client._ib.cancelOrder.assert_called_once_with(trade.order)
        assert result is True

    def test_cancel_unknown_silent(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        client._ib.trades.return_value = []

        result = client.cancel_order("9999")  # no exception
        assert result is True


class TestOrderEventHandlers:
    def test_event_handler_updates_cache(self):
        with patch("orb_live.data.ib_client.IB"):
            client = IBClient(paper=True)

        trade = _make_trade(order_id=20, status="Submitted")
        client._on_order_status(trade)

        assert 20 in client._order_cache
        assert client._order_cache[20]["status"] == "Submitted"

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

    def test_trade_to_dict_has_alpaca_id_alias(self):
        client = self._client()
        trade  = _make_trade(order_id=77)
        d = client._trade_to_dict(trade)
        assert d["alpaca_id"] == d["id"] == "77"
