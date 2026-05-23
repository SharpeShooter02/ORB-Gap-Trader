"""
tests/integration/test_ib_client_live.py — Live IB Gateway integration tests.

These tests connect to a real IB Gateway / TWS instance and are SKIPPED unless
the environment variable RUN_IB_INTEGRATION=1 is set.

    RUN_IB_INTEGRATION=1 pytest tests/integration/test_ib_client_live.py -v

Prerequisites:
    - IB Gateway or TWS running and logged in
    - IB_HOST, IB_PORT, IB_CLIENT_ID set (or defaults: 127.0.0.1:4002, clientId=1)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pytest

RUN_IB = os.environ.get("RUN_IB_INTEGRATION") == "1"
skip_unless_ib = pytest.mark.skipif(not RUN_IB, reason="RUN_IB_INTEGRATION=1 not set")


@skip_unless_ib
class TestIBClientLive:
    @pytest.fixture(scope="class")
    def client(self):
        from orb_live.data.ib_client import build_client_from_env
        c = build_client_from_env(paper=True)
        c.connect()
        yield c
        c.disconnect()

    def test_is_connected(self, client):
        assert client.is_connected() is True

    def test_get_account_returns_equity(self, client):
        acct = client.get_account()
        assert "equity" in acct
        assert isinstance(acct["equity"], float)
        assert acct["equity"] > 0, "Expected non-zero equity in a funded paper account"

    def test_get_account_has_required_keys(self, client):
        acct = client.get_account()
        for key in ("equity", "cash", "buying_power", "portfolio_value",
                    "daytrade_count", "pattern_day_trader"):
            assert key in acct, f"Missing key: {key}"

    def test_list_positions_returns_list(self, client):
        positions = client.list_positions()
        assert isinstance(positions, list)
        for pos in positions:
            assert "symbol"          in pos
            assert "qty"             in pos
            assert "avg_entry_price" in pos
            assert "side"            in pos
            assert pos["side"] in ("long", "short")

    def test_get_position_unknown_symbol_returns_none(self, client):
        result = client.get_position("ZZZNOTREAL")
        assert result is None

    def test_disconnect_then_is_connected_false(self, client):
        client.disconnect()
        assert client.is_connected() is False
        # reconnect so later fixtures still work
        client.connect()


@skip_unless_ib
class TestIBClientAssetLive:
    @pytest.fixture(scope="class")
    def client(self):
        from orb_live.data.ib_client import build_client_from_env
        c = build_client_from_env(paper=True)
        c.connect()
        yield c
        c.disconnect()

    def test_live_get_asset_soxl(self, client):
        asset = client.get_asset("SOXL")
        assert asset["symbol"]   == "SOXL"
        assert asset["tradable"] is True
        assert asset["status"]   == "active"
        assert asset["conId"]    is not None and asset["conId"] > 0
        assert asset["primary_exchange"] in ("ARCA", "NASDAQ", "NYSE", "BATS")

    def test_live_get_asset_invalid(self, client):
        asset = client.get_asset("INVALIDXYZ")
        assert asset["tradable"] is False
        assert asset["status"]   == "not_found"

    def test_live_get_asset_cache(self, client):
        # First call (may be cache-warm from previous test); do an uncached symbol
        t0 = time.perf_counter()
        asset1 = client.get_asset("TQQQ")
        t1 = time.perf_counter()
        # Second call must come from cache (< 10ms)
        asset2 = client.get_asset("TQQQ")
        t2 = time.perf_counter()

        assert asset1 == asset2
        second_call_ms = (t2 - t1) * 1000
        assert second_call_ms < 10, f"Cache miss: second call took {second_call_ms:.1f}ms"


@skip_unless_ib
class TestIBClientClockLive:
    @pytest.fixture(scope="class")
    def client(self):
        from orb_live.data.ib_client import build_client_from_env
        # clock methods don't need a live connection, but use the connected
        # client for consistency with other live tests
        c = build_client_from_env(paper=True)
        c.connect()
        yield c
        c.disconnect()

    def test_live_get_clock_consistent(self, client):
        before = datetime.now(tz=timezone.utc)
        clock  = client.get_clock()
        after  = datetime.now(tz=timezone.utc)

        ts_utc = clock["timestamp"].astimezone(timezone.utc)
        assert before <= ts_utc <= after, "clock timestamp is not within the expected range"

    def test_live_get_clock_sensible(self, client):
        clock = client.get_clock()
        assert isinstance(clock["is_open"], bool)
        assert clock["next_open"]  < clock["next_close"] or not clock["is_open"]
        assert clock["next_open"].tzinfo  is not None
        assert clock["next_close"].tzinfo is not None
