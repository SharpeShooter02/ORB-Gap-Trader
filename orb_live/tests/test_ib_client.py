"""
tests/test_ib_client.py — Unit tests for IBClient (Part 1).

All IB Gateway network calls are mocked so these run without a live connection.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from orb_live.data.ib_client import IBClient, build_client_from_env


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
