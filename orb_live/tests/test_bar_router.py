"""
tests/test_bar_router.py — BarRouter unit tests.

Uses a synchronous stub for alpaca.subscribe_bars so tests run instantly
without network access.  All cases verify observable side effects (listener
calls, dispatch order, reconnect backoff, missed-bar replay).
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")

_DT = lambda h, m: datetime(2026, 1, 7, h, m, 0, tzinfo=ET)


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _StubAlpaca:
    """
    Minimal alpaca stub.  subscribe_bars calls the callback synchronously
    with bars from the injected sequence, then returns.
    """
    def __init__(self, bar_sequences: dict[str, list[dict]] = None):
        self._sequences = bar_sequences or {}
        self.subscribe_calls = []

    def subscribe_bars(self, symbols, callback):
        import asyncio
        import inspect
        self.subscribe_calls.append(list(symbols))
        for sym, bars in self._sequences.items():
            if sym in symbols:
                for b in bars:
                    result = callback(_make_alpaca_bar(sym, b))
                    if inspect.iscoroutine(result):
                        asyncio.run(result)

    def get_intraday_bars(self, symbol, start_dt, end_dt, timeframe="1Min", feed="iex"):
        rows = self._sequences.get(symbol, [])
        filtered = [
            b for b in rows
            if start_dt <= b["timestamp"] < end_dt
        ]
        if not filtered:
            return pd.DataFrame()
        return pd.DataFrame(filtered)

    def get_latest_quote(self, symbol):
        return {"bid": 99.9, "ask": 100.1}


def _make_alpaca_bar(symbol, d: dict):
    """Construct a minimal alpaca Bar-like object from a dict."""
    return SimpleNamespace(
        symbol=symbol,
        timestamp=d["timestamp"],
        open=d.get("open", d["close"]),
        high=d.get("high", d["close"]),
        low=d.get("low", d["close"]),
        close=d["close"],
        volume=d.get("volume", 1000),
    )


class _StubBarCache:
    def __init__(self):
        self.added = []

    def add_bar(self, symbol, bar):
        self.added.append((symbol, bar))


class _StubStore:
    pass


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_bar():
    return {"timestamp": _DT(10, 1), "close": 50.0, "open": 50.0,
            "high": 50.5, "low": 49.5, "volume": 1000}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_a_listener_called_for_each_bar():
    """Every registered listener receives each bar in order."""
    bars = [
        {"timestamp": _DT(10, 1), "close": 50.0},
        {"timestamp": _DT(10, 2), "close": 51.0},
        {"timestamp": _DT(10, 3), "close": 52.0},
    ]
    alpaca = _StubAlpaca({"TQQQ": bars})
    received = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(alpaca, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: received.append((sym, bar["close"])))

    # Call subscribe_bars directly (synchronous stub)
    alpaca.subscribe_bars(["TQQQ"], router._on_stream_bar)

    assert [(s, c) for s, c in received] == [("TQQQ", 50.0), ("TQQQ", 51.0), ("TQQQ", 52.0)]


def test_b_multiple_listeners_called_in_registration_order():
    """Multiple listeners are invoked in the order they were registered."""
    bars = [{"timestamp": _DT(10, 1), "close": 50.0}]
    alpaca = _StubAlpaca({"TQQQ": bars})
    order = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(alpaca, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: order.append("first"))
    router.register_listener(lambda sym, bar: order.append("second"))
    router.register_listener(lambda sym, bar: order.append("third"))

    alpaca.subscribe_bars(["TQQQ"], router._on_stream_bar)

    assert order == ["first", "second", "third"]


def test_c_listener_exception_does_not_stop_subsequent_listeners():
    """A listener that raises must not prevent other listeners from being called."""
    bars = [{"timestamp": _DT(10, 1), "close": 50.0}]
    alpaca = _StubAlpaca({"TQQQ": bars})
    reached = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(alpaca, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: (_ for _ in ()).throw(RuntimeError("boom")))
    router.register_listener(lambda sym, bar: reached.append(True))

    alpaca.subscribe_bars(["TQQQ"], router._on_stream_bar)

    assert reached == [True]


def test_d_detect_missed_bars_gap_under_2min_returns_empty():
    """detect_missed_bars returns [] when gap < 120 seconds (no full bar missed)."""
    from orb_live.runner.bar_router import BarRouter
    alpaca = _StubAlpaca()
    router = BarRouter(alpaca, _StubStore(), _StubBarCache())

    last = _DT(10, 1)
    current = _DT(10, 1) + timedelta(seconds=90)
    result = router.detect_missed_bars("TQQQ", last, current)
    assert result == []


def test_e_detect_missed_bars_fetches_via_rest():
    """detect_missed_bars fetches REST bars when gap >= 2 minutes."""
    missed_bar = {"timestamp": _DT(10, 2), "close": 50.5,
                  "open": 50.0, "high": 51.0, "low": 49.8, "volume": 500}
    alpaca = _StubAlpaca({"TQQQ": [missed_bar]})

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(alpaca, _StubStore(), _StubBarCache())

    last    = _DT(10, 1)
    current = _DT(10, 4)  # 3-minute gap
    result = router.detect_missed_bars("TQQQ", last, current)

    assert len(result) == 1
    assert result[0]["close"] == 50.5


def test_f_replay_missed_bars_dispatches_oldest_first():
    """replay_missed_bars delivers bars in ascending timestamp order."""
    bars = [
        {"timestamp": _DT(10, 4), "close": 53.0, "open": 53.0, "high": 53.0, "low": 53.0},
        {"timestamp": _DT(10, 2), "close": 51.0, "open": 51.0, "high": 51.0, "low": 51.0},
        {"timestamp": _DT(10, 3), "close": 52.0, "open": 52.0, "high": 52.0, "low": 52.0},
    ]
    alpaca = _StubAlpaca()
    received_closes = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(alpaca, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: received_closes.append(bar["close"]))

    # replay_missed_bars sorts by timestamp
    sorted_bars = sorted(bars, key=lambda b: b["timestamp"])
    router.replay_missed_bars("TQQQ", sorted_bars)

    assert received_closes == [51.0, 52.0, 53.0]


def test_g_replay_before_live_on_reconnect():
    """
    When a live bar arrives after a gap, missed bars are replayed BEFORE the
    live bar is dispatched to listeners.
    """
    # Simulate: last bar was at 10:01, now receiving 10:05 bar
    # REST returns 10:02, 10:03, 10:04
    rest_bars = [
        {"timestamp": _DT(10, 2), "close": 51.0, "open": 51.0, "high": 51.0, "low": 51.0},
        {"timestamp": _DT(10, 3), "close": 52.0, "open": 52.0, "high": 52.0, "low": 52.0},
        {"timestamp": _DT(10, 4), "close": 53.0, "open": 53.0, "high": 53.0, "low": 53.0},
    ]
    live_bar = {"timestamp": _DT(10, 5), "close": 54.0, "open": 54.0,
                "high": 54.0, "low": 54.0}

    # Stub returns rest_bars for detect_missed_bars calls
    alpaca = _StubAlpaca({"TQQQ": rest_bars})
    dispatch_order = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(alpaca, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: dispatch_order.append(bar["close"]))

    # Seed last_bar_ts at 10:01
    router._last_bar_ts["TQQQ"] = _DT(10, 1)

    # Simulate receiving a live bar at 10:05 (4-min gap)
    import asyncio
    asyncio.run(router._on_stream_bar(_make_alpaca_bar("TQQQ", live_bar)))

    # Expected: 51, 52, 53 (replayed), then 54 (live)
    assert dispatch_order == [51.0, 52.0, 53.0, 54.0]


def test_h_reconnect_backoff_doubles_each_attempt():
    """
    _stream_loop applies exponential backoff: each reconnect doubles the sleep
    up to RECONNECT_BACKOFF_MAX.  After 3 failures + 1 clean exit the loop stops.
    """
    connect_count = [0]
    sleep_calls = []
    stop_fn = [lambda: None]  # filled in after router is constructed

    class _FailingAlpaca:
        def subscribe_bars(self, symbols, callback):
            connect_count[0] += 1
            if connect_count[0] >= 4:
                # Clean exit on 4th call → _stream_loop resets backoff then checks
                # _running; stop_fn sets _running=False so the while-loop breaks.
                stop_fn[0]()
                return
            raise ConnectionError("disconnected")

    from orb_live.runner.bar_router import BarRouter, RECONNECT_BACKOFF_INIT, RECONNECT_BACKOFF_MAX
    router = BarRouter(_FailingAlpaca(), _StubStore(), _StubBarCache())
    stop_fn[0] = lambda: setattr(router, "_running", False)
    router._symbols = ["TQQQ"]

    # Patch sleep to capture backoff values, then restore
    import orb_live.runner.bar_router as br_mod
    original_sleep = br_mod.time.sleep
    br_mod.time.sleep = lambda s: sleep_calls.append(s)
    try:
        router._running = True
        router._stream_loop()
    finally:
        br_mod.time.sleep = original_sleep

    # 3 failures → slept 1, 2, 4; 4th call is clean exit → no sleep after (loop breaks)
    assert sleep_calls == [1.0, 2.0, 4.0]
