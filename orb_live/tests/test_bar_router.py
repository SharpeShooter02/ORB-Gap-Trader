"""
tests/test_bar_router.py — BarRouter unit tests.

Uses a synchronous stub for broker.subscribe_bars so tests run instantly
without network access.  All cases verify observable side effects (listener
calls, dispatch order, missed-bar replay, subscribe/unsubscribe lifecycle).

WS-era reconnect / backoff / degraded-mode tests removed in BUG 0c fix:
BarRouter no longer spawns a daemon thread — subscribe_bars is called directly
on the main thread so ib_async callbacks are serviced on the correct loop thread.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")

_DT = lambda h, m: datetime(2026, 1, 7, h, m, 0, tzinfo=ET)


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _StubBroker:
    """
    Minimal broker stub.  subscribe_bars calls the callback synchronously
    with bars from the injected sequence, then returns.
    """
    def __init__(self, bar_sequences: dict = None):
        self._sequences = bar_sequences or {}
        self.subscribe_calls: list[list[str]] = []
        self.stop_calls: int = 0

    def subscribe_bars(self, symbols, callback):
        self.subscribe_calls.append(list(symbols))
        for sym, bars in self._sequences.items():
            if sym in symbols:
                for b in bars:
                    callback(_make_broker_bar(sym, b))

    def stop_bars_stream(self):
        self.stop_calls += 1

    def get_intraday_bars(self, symbol, start_dt, end_dt, timeframe="1Min", feed="iex"):
        rows = self._sequences.get(symbol, [])
        filtered = [b for b in rows if start_dt <= b["timestamp"] < end_dt]
        if not filtered:
            return pd.DataFrame()
        return pd.DataFrame(filtered)


def _make_broker_bar(symbol, d: dict):
    """Construct a bar dict matching IBClient's BarAggregator.finalize() output."""
    return {
        "symbol":    symbol,
        "timestamp": d["timestamp"],
        "open":      d.get("open",   d["close"]),
        "high":      d.get("high",   d["close"]),
        "low":       d.get("low",    d["close"]),
        "close":     d["close"],
        "volume":    d.get("volume", 1000),
    }


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


# ── Tests: listener fan-out ───────────────────────────────────────────────────

def test_a_listener_called_for_each_bar():
    """Every registered listener receives each bar in order."""
    bars = [
        {"timestamp": _DT(10, 1), "close": 50.0},
        {"timestamp": _DT(10, 2), "close": 51.0},
        {"timestamp": _DT(10, 3), "close": 52.0},
    ]
    broker = _StubBroker({"TQQQ": bars})
    received = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(broker, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: received.append((sym, bar["close"])))
    router.subscribe(["TQQQ"])

    assert [(s, c) for s, c in received] == [("TQQQ", 50.0), ("TQQQ", 51.0), ("TQQQ", 52.0)]


def test_b_multiple_listeners_called_in_registration_order():
    """Multiple listeners are invoked in the order they were registered."""
    bars = [{"timestamp": _DT(10, 1), "close": 50.0}]
    broker = _StubBroker({"TQQQ": bars})
    order = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(broker, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: order.append("first"))
    router.register_listener(lambda sym, bar: order.append("second"))
    router.register_listener(lambda sym, bar: order.append("third"))
    router.subscribe(["TQQQ"])

    assert order == ["first", "second", "third"]


def test_c_listener_exception_does_not_stop_subsequent_listeners():
    """A listener that raises must not prevent other listeners from being called."""
    bars = [{"timestamp": _DT(10, 1), "close": 50.0}]
    broker = _StubBroker({"TQQQ": bars})
    reached = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(broker, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: (_ for _ in ()).throw(RuntimeError("boom")))
    router.register_listener(lambda sym, bar: reached.append(True))
    router.subscribe(["TQQQ"])

    assert reached == [True]


# ── Tests: missed-bar detection ───────────────────────────────────────────────

def test_d_detect_missed_bars_gap_under_2min_returns_empty():
    """detect_missed_bars returns [] when gap < 120 seconds (no full bar missed)."""
    from orb_live.runner.bar_router import BarRouter
    broker = _StubBroker()
    router = BarRouter(broker, _StubStore(), _StubBarCache())

    last    = _DT(10, 1)
    current = _DT(10, 1) + timedelta(seconds=90)
    assert router.detect_missed_bars("TQQQ", last, current) == []


def test_e_detect_missed_bars_fetches_via_rest():
    """detect_missed_bars fetches REST bars when gap >= 2 minutes."""
    missed_bar = {"timestamp": _DT(10, 2), "close": 50.5,
                  "open": 50.0, "high": 51.0, "low": 49.8, "volume": 500}
    broker = _StubBroker({"TQQQ": [missed_bar]})

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(broker, _StubStore(), _StubBarCache())

    last    = _DT(10, 1)
    current = _DT(10, 4)   # 3-minute gap
    result  = router.detect_missed_bars("TQQQ", last, current)

    assert len(result) == 1
    assert result[0]["close"] == 50.5


def test_f_replay_missed_bars_dispatches_oldest_first():
    """replay_missed_bars delivers bars in ascending timestamp order."""
    bars = [
        {"timestamp": _DT(10, 4), "close": 53.0, "open": 53.0, "high": 53.0, "low": 53.0},
        {"timestamp": _DT(10, 2), "close": 51.0, "open": 51.0, "high": 51.0, "low": 51.0},
        {"timestamp": _DT(10, 3), "close": 52.0, "open": 52.0, "high": 52.0, "low": 52.0},
    ]
    broker = _StubBroker()
    received_closes = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(broker, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: received_closes.append(bar["close"]))

    sorted_bars = sorted(bars, key=lambda b: b["timestamp"])
    router.replay_missed_bars("TQQQ", sorted_bars)

    assert received_closes == [51.0, 52.0, 53.0]


def test_g_replay_before_live_on_reconnect():
    """
    When a live bar arrives after a gap, missed bars are replayed BEFORE the
    live bar is dispatched to listeners.
    """
    rest_bars = [
        {"timestamp": _DT(10, 2), "close": 51.0, "open": 51.0, "high": 51.0, "low": 51.0},
        {"timestamp": _DT(10, 3), "close": 52.0, "open": 52.0, "high": 52.0, "low": 52.0},
        {"timestamp": _DT(10, 4), "close": 53.0, "open": 53.0, "high": 53.0, "low": 53.0},
    ]
    live_bar = {"timestamp": _DT(10, 5), "close": 54.0, "open": 54.0,
                "high": 54.0, "low": 54.0}

    broker = _StubBroker({"TQQQ": rest_bars})
    dispatch_order = []

    from orb_live.runner.bar_router import BarRouter
    router = BarRouter(broker, _StubStore(), _StubBarCache())
    router.register_listener(lambda sym, bar: dispatch_order.append(bar["close"]))

    # Seed last_bar_ts at 10:01 to simulate a 4-minute gap
    router._last_bar_ts["TQQQ"] = _DT(10, 1)

    router._on_stream_bar(_make_broker_bar("TQQQ", live_bar))

    # Expected: 51, 52, 53 (replayed), then 54 (live)
    assert dispatch_order == [51.0, 52.0, 53.0, 54.0]


# ── Tests: subscribe/unsubscribe lifecycle ────────────────────────────────────

def test_h_subscribe_calls_broker_subscribe_bars():
    """subscribe() must call broker.subscribe_bars on the main thread (no daemon thread)."""
    from orb_live.runner.bar_router import BarRouter
    broker = _StubBroker()
    router = BarRouter(broker, _StubStore(), _StubBarCache())

    router.subscribe(["TQQQ", "SQQQ"])

    assert len(broker.subscribe_calls) == 1
    assert set(broker.subscribe_calls[0]) == {"TQQQ", "SQQQ"}
    assert router._subscribed
    assert set(router._symbols) == {"TQQQ", "SQQQ"}


def test_i_unsubscribe_calls_stop_bars_stream():
    """unsubscribe() must call stop_bars_stream on the broker and reset state."""
    from orb_live.runner.bar_router import BarRouter
    broker = _StubBroker()
    router = BarRouter(broker, _StubStore(), _StubBarCache())

    router.subscribe(["TQQQ"])
    router.unsubscribe()

    assert broker.stop_calls == 1
    assert not router._subscribed
    assert router._symbols == []


def test_j_subscribe_idempotent_same_symbols():
    """subscribe() with the same symbols must not call subscribe_bars again."""
    from orb_live.runner.bar_router import BarRouter
    broker = _StubBroker()
    router = BarRouter(broker, _StubStore(), _StubBarCache())

    router.subscribe(["TQQQ", "SQQQ"])
    router.subscribe(["TQQQ", "SQQQ"])   # same symbols — no-op

    assert len(broker.subscribe_calls) == 1
    assert broker.stop_calls == 0


def test_k_subscribe_different_symbols_stops_then_resubscribes():
    """subscribe() with different symbols stops the old subscription first."""
    from orb_live.runner.bar_router import BarRouter
    broker = _StubBroker()
    router = BarRouter(broker, _StubStore(), _StubBarCache())

    router.subscribe(["TQQQ", "SQQQ"])
    router.subscribe(["SOXL", "SOXS"])   # different symbols

    assert broker.stop_calls == 1
    assert len(broker.subscribe_calls) == 2
    assert set(router._symbols) == {"SOXL", "SOXS"}
    assert router._subscribed


def test_l_unsubscribe_then_resubscribe_lifecycle():
    """Full lifecycle: subscribe A → unsubscribe → subscribe B."""
    from orb_live.runner.bar_router import BarRouter
    broker = _StubBroker()
    router = BarRouter(broker, _StubStore(), _StubBarCache())

    router.subscribe(["TQQQ"])
    router.unsubscribe()
    router.subscribe(["SOXL"])

    assert len(broker.subscribe_calls) == 2
    assert set(router._symbols) == {"SOXL"}
    assert router._subscribed
