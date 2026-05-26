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


@skip_unless_ib
class TestIBClientOrdersLive:
    """
    Live order tests.  All use prices far from market to avoid accidental fills.
    Every submitted order is cancelled in a try/finally block.
    """

    @pytest.fixture(scope="class")
    def client(self):
        from orb_live.data.ib_client import build_client_from_env
        c = build_client_from_env(paper=True)
        c.connect()
        yield c
        c.disconnect()

    def test_live_submit_and_cancel_limit(self, client):
        order = None
        try:
            order = client.submit_limit_order("SOXL", "buy", 1, 0.01,
                                              client_order_id="inttest-buy-001")
            assert "id"     in order
            assert "status" in order
            assert order["symbol"] == "SOXL"
        finally:
            if order:
                client.cancel_order(order["id"])

    def test_live_short_limit_accepted(self, client):
        order = None
        try:
            order = client.submit_limit_order("SBIT", "sell", 1, 9999.99,
                                              client_order_id="inttest-sell-001")
            assert order["side"] == "sell"
        finally:
            if order:
                client.cancel_order(order["id"])

    def test_live_get_order_after_submit(self, client):
        order = None
        try:
            order = client.submit_limit_order("SOXL", "buy", 1, 0.01)
            fetched = client.get_order(order["id"])
            assert fetched["id"] == order["id"]
        finally:
            if order:
                client.cancel_order(order["id"])

    def test_live_event_handler_fires(self, client):
        """Order cache must be populated within 2 seconds of submission."""
        order = None
        try:
            order = client.submit_limit_order("SOXL", "buy", 1, 0.01)
            order_int = int(order["id"])
            assert order_int in client._order_cache, (
                f"Event handler did not populate _order_cache for order {order_int}"
            )
        finally:
            if order:
                client.cancel_order(order["id"])

    def test_live_cancel_order_changes_status(self, client):  # noqa: keep at end
        order = None
        try:
            order = client.submit_limit_order("SOXL", "buy", 1, 0.01)
            client.cancel_order(order["id"])
            time.sleep(2)  # allow cancel confirmation to arrive
            fetched = client.get_order(order["id"])
            assert fetched["status"] in ("canceled", "rejected"), (
                f"Unexpected normalized status: {fetched['status']}"
            )
            assert fetched["status_raw"] in ("Cancelled", "ApiCancelled", "Inactive"), (
                f"Unexpected raw status: {fetched['status_raw']}"
            )
        finally:
            pass  # already cancelled above


@skip_unless_ib
class TestIBClientMarketDataLive:
    @pytest.fixture(scope="class")
    def client(self):
        from orb_live.data.ib_client import build_client_from_env
        c = build_client_from_env(paper=True)
        c.connect()
        yield c
        c.disconnect()

    def test_live_get_intraday_bars_recent(self, client):
        from datetime import timedelta
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        end   = datetime.now(tz=_ET)
        start = end - timedelta(minutes=30)

        clock = client.get_clock()
        if not clock["is_open"]:
            pytest.skip("Market closed; historical bars may be empty")

        df = client.get_intraday_bars("SOXL", start, end)
        assert not df.empty, "Expected at least 1 bar in the last 30 minutes"

    def test_live_get_intraday_bars_columns(self, client):
        from datetime import timedelta
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        end   = datetime.now(tz=_ET)
        start = end - timedelta(minutes=30)
        df    = client.get_intraday_bars("SOXL", start, end)

        if df.empty:
            pytest.skip("No bars returned (market closed?)")

        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"Missing column: {col}"
        assert df.index.tzinfo is not None, "Index must be timezone-aware"

    def test_live_subscribe_brief(self, client):
        """Subscribe for 15 seconds during market hours and expect at least 1 bar."""
        clock = client.get_clock()
        if not clock["is_open"]:
            pytest.skip("Market closed; cannot test live streaming")

        received = []
        client.subscribe_bars(["SPY"], callback=received.append)
        time.sleep(15)
        client.stop_bars_stream()

        assert len(received) >= 1, (
            "Expected at least 1 bar within 15 seconds during market hours"
        )
        bar = received[0]
        assert "symbol"    in bar
        assert "timestamp" in bar
        assert "close"     in bar

    def test_live_stop_bars_stream_clean(self, client):
        """Subscribe and stop without errors; streams dict must be empty after."""
        clock = client.get_clock()
        if not clock["is_open"]:
            pytest.skip("Market closed; IB may reject real-time bar request")

        client.subscribe_bars(["SOXL"], callback=lambda bar: None)
        time.sleep(2)
        client.stop_bars_stream()

        assert client._streams == {}


@skip_unless_ib
class TestIBClientQuoteLive:
    @pytest.fixture(scope="class")
    def client(self):
        from orb_live.data.ib_client import build_client_from_env
        c = build_client_from_env(paper=True)
        c.connect()
        yield c
        c.disconnect()

    def test_live_get_latest_quote_soxl(self, client):
        clock = client.get_clock()
        if not clock["is_open"]:
            pytest.skip("Market closed; live bid/ask may be zero")
        q = client.get_latest_quote("SOXL")
        assert q["bid"] > 0, f"Expected positive bid, got {q['bid']}"
        assert q["ask"] > 0, f"Expected positive ask, got {q['ask']}"
        assert q["ask"] >= q["bid"]

    def test_live_get_latest_quote_shape(self, client):
        q = client.get_latest_quote("SOXL")
        for key in ("bid", "ask", "bid_size", "ask_size", "ts"):
            assert key in q, f"Missing key: {key}"
        assert isinstance(q["bid"], float)
        assert isinstance(q["ask"], float)


@skip_unless_ib
class TestIBClientDailyBarsLive:
    @pytest.fixture(scope="class")
    def client(self):
        from orb_live.data.ib_client import build_client_from_env
        c = build_client_from_env(paper=True)
        c.connect()
        yield c
        c.disconnect()

    def test_live_get_daily_bars_soxl(self, client):
        df = client.get_daily_bars("SOXL", lookback_days=10)
        assert not df.empty, "Expected at least 1 daily bar"
        assert len(df) >= 5, f"Expected >=5 bars, got {len(df)}"
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_live_get_daily_bars_dates_ascending(self, client):
        df = client.get_daily_bars("SOXL", lookback_days=10)
        if df.empty:
            pytest.skip("No bars returned")
        assert list(df["date"]) == sorted(df["date"].tolist())

    def test_live_get_daily_bars_date_tz_naive(self, client):
        df = client.get_daily_bars("SOXL", lookback_days=5)
        if df.empty:
            pytest.skip("No bars returned")
        assert df["date"].dt.tz is None, "date column must be tz-naive"

    def test_live_get_daily_bars_lookback_respected(self, client):
        df = client.get_daily_bars("SOXL", lookback_days=5)
        assert len(df) <= 5


@skip_unless_ib
class TestIBClientEndToEnd:
    """
    End-to-end: exercises the full lifecycle via build_broker_from_env()
    exactly as the runner does, including connect-at-factory-time and
    status normalisation.
    """

    def test_full_lifecycle_via_factory(self):
        import os
        os.environ["BROKER"] = "ib"
        try:
            from orb_live.runner.main import build_broker_from_env
            broker = build_broker_from_env(paper=True)

            # Account check
            acc = broker.get_account()
            assert acc["equity"] > 0

            # Asset metadata
            asset = broker.get_asset("SOXL")
            assert asset["tradable"] is True

            # Order lifecycle — status must be normalised to lowercase Alpaca form
            order = broker.submit_limit_order("SOXL", "buy", 1, 0.01,
                                              client_order_id="e2e-test-001")
            assert order["status"] in ("new", "filled", "canceled", "rejected")
            assert order["status_raw"] is not None  # raw IB string preserved

            broker.cancel_order(order["id"])
            time.sleep(1)

            broker.disconnect()
        finally:
            os.environ.pop("BROKER", None)
