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
