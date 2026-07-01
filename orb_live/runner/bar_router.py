"""
runner/bar_router.py — Synchronous bar subscription with missed-bar replay.

For IB/ib_async: broker.subscribe_bars() registers reqRealTimeBars callbacks
and returns immediately.  Callbacks fire on ib_async's own event loop, which is
pumped by ib.sleep() in the main thread (SessionRunner's wait phases).

CRITICAL: subscribe() must be called from the MAIN THREAD.  ib_async is not
thread-safe — a reqRealTimeBars subscription registered from a background thread
never gets its updateEvent callbacks serviced.  No daemon thread is spawned here.

Listener signature: fn(symbol: str, bar: dict) -> None
bar dict keys: timestamp (datetime, ET-aware), open, high, low, close, volume
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class BarRouter:
    """
    Synchronous bar subscription with missed-bar replay.

    Usage:
        router = BarRouter(broker, state_store, bar_cache)
        router.register_listener(on_bar)
        router.subscribe(["TQQQ", "SQQQ"])   # main thread only
        # SessionRunner's ib.sleep() pumps the loop; bars arrive via callbacks
        router.unsubscribe()

    Missed-bar detection fires on every incoming bar: if the gap between the
    last-seen bar for a symbol and the current bar is >= 2 minutes, REST bars
    are fetched oldest-first and replayed before the live bar is dispatched.
    """

    def __init__(self, broker, state_store, bar_cache, logger=None):
        self._broker  = broker
        self._store   = state_store
        self._cache   = bar_cache
        self._log     = logger

        self._listeners:   list[Callable[[str, dict], None]] = []
        self._symbols:     list[str] = []
        self._subscribed:  bool = False

        # Last delivered bar timestamp per symbol (for missed-bar detection)
        self._last_bar_ts: dict[str, Optional[datetime]] = {}
        self._ts_lock = threading.Lock()

        # Per-symbol bar counters (for zero-bars watchdog)
        self._bars_received: dict[str, int] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_listener(self, fn: Callable[[str, dict], None]) -> None:
        """Register a callback invoked on every bar (live or replayed)."""
        self._listeners.append(fn)

    def subscribe(self, symbols: list[str]) -> None:
        """
        Subscribe to real-time bars for the given symbols.

        MUST be called from the MAIN THREAD.  ib_async's reqRealTimeBars
        updateEvent callbacks are serviced by the event loop; a subscription
        made from a daemon thread strands the callbacks and delivers no bars.

        broker.subscribe_bars() returns immediately for IB — bars arrive
        asynchronously as the main thread's ib.sleep() pumps the loop.

        Idempotent:
          - Same symbols already subscribed → no-op.
          - Different symbols → stops the current subscription first.
        """
        if self._subscribed and set(symbols) == set(self._symbols):
            return

        if self._subscribed:
            self.unsubscribe()

        self._symbols = list(symbols)
        with self._ts_lock:
            self._last_bar_ts = {s: None for s in symbols}
        self._broker.subscribe_bars(symbols, self._on_stream_bar)
        self._subscribed = True

    def bars_received(self, symbol: str) -> int:
        """Return the number of bars delivered for symbol since last subscribe()."""
        return self._bars_received.get(symbol, 0)

    def unsubscribe(self) -> None:
        """Cancel all real-time bar subscriptions."""
        stop_fn = getattr(self._broker, "stop_bars_stream", None)
        if stop_fn is not None:
            try:
                stop_fn()
            except Exception:
                pass
        self._subscribed = False
        self._symbols    = []

    def detect_missed_bars(
        self,
        symbol: str,
        last_seen_ts: Optional[datetime],
        current_ts: datetime,
    ) -> list[dict]:
        """
        Return bars that arrived between last_seen_ts and current_ts via REST.

        Returns [] if the gap is < 2 minutes (no full 1-min bar could have
        been missed), or if the REST fetch fails.

        Bars are returned sorted oldest-first.
        """
        if last_seen_ts is None:
            return []

        gap_seconds = (current_ts - last_seen_ts).total_seconds()
        if gap_seconds < 120:
            return []

        fetch_start = last_seen_ts + timedelta(minutes=1)
        fetch_end   = current_ts

        try:
            df = self._broker.get_intraday_bars(symbol, fetch_start, fetch_end)
        except Exception as exc:
            if self._log:
                self._log.warning("missed_bar_fetch_failed", symbol=symbol, exc=str(exc))
            return []

        if df is None or df.empty:
            return []

        bars = []
        for idx, row in df.iterrows():
            close = float(row["close"])
            # IBClient returns a timestamp-indexed DataFrame; stubs may use a
            # plain column.  Prefer the column; fall back to the index value.
            ts = row["timestamp"] if "timestamp" in row else idx
            bar = {
                "timestamp": ts,
                "open":      float(row.get("open",   close)),
                "high":      float(row.get("high",   close)),
                "low":       float(row.get("low",    close)),
                "close":     close,
                "volume":    float(row.get("volume", 0.0)),
            }
            bars.append(bar)

        return sorted(bars, key=lambda b: b["timestamp"])

    def replay_missed_bars(self, symbol: str, missed: list[dict]) -> None:
        """Dispatch missed bars (oldest-first) to all registered listeners."""
        for bar in missed:
            self._dispatch(symbol, bar)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _dispatch(self, symbol: str, bar: dict) -> None:
        """Deliver one bar to every registered listener in registration order."""
        with self._ts_lock:
            self._last_bar_ts[symbol] = bar.get("timestamp")
            self._bars_received[symbol] = self._bars_received.get(symbol, 0) + 1

        for fn in self._listeners:
            try:
                fn(symbol, bar)
            except Exception as exc:
                if self._log:
                    self._log.error(
                        "listener_exception", symbol=symbol,
                        fn=getattr(fn, "__name__", "?"), exc=str(exc),
                    )

    def _on_stream_bar(self, bar: dict) -> None:
        """
        Callback invoked by the broker on each incoming bar.
        IBClient passes a plain dict from BarAggregator.finalize(); detect
        missed bars, replay them, then dispatch the live bar.
        """
        symbol = str(bar["symbol"])
        ts     = bar["timestamp"]
        if hasattr(ts, "astimezone"):
            ts = ts.astimezone(ET)

        bar_dict = {
            "timestamp": ts,
            "open":      float(bar["open"]),
            "high":      float(bar["high"]),
            "low":       float(bar["low"]),
            "close":     float(bar["close"]),
            "volume":    float(bar["volume"]),
        }

        with self._ts_lock:
            last = self._last_bar_ts.get(symbol)

        missed = self.detect_missed_bars(symbol, last, ts)
        if missed:
            if self._log:
                self._log.warning(
                    "replaying_missed_bars", symbol=symbol, count=len(missed)
                )
            self.replay_missed_bars(symbol, missed)

        self._dispatch(symbol, bar_dict)
