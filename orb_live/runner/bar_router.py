"""
runner/bar_router.py — WebSocket bar subscription with missed-bar replay.

Wraps broker.subscribe_bars in a background thread with exponential-backoff
reconnect.  On each reconnect, REST-fetches any missed bars and replays them
(oldest-first) before the live stream resumes.

Listener signature: fn(symbol: str, bar: dict) -> None
bar dict keys: timestamp (datetime, ET-aware), open, high, low, close, volume
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

RECONNECT_BACKOFF_INIT = 1.0
RECONNECT_BACKOFF_MAX  = 30.0

WS_TOKEN_REFRESH_HOURS = 12.0        # force reconnect to refresh auth token
_DEGRADED_THRESHOLD_S  = 300.0       # enter REST-poll fallback after 5min of failures
_DEGRADED_RETRY_S      = 300.0       # retry WebSocket every 5min from degraded mode


class BarRouter:
    """
    WebSocket bar subscription with reconnect and missed-bar replay.

    Usage:
        router = BarRouter(broker, state_store, bar_cache)
        router.register_listener(on_bar)
        router.subscribe(["TQQQ", "SQQQ"])
        # ... session runs ...
        router.unsubscribe()

    Missed-bar detection fires on every bar delivery: if the gap between the
    last-seen bar for a symbol and the current bar is >= 2 minutes, REST bars
    are fetched to fill the gap and replayed oldest-first before the live bar
    is dispatched.
    """

    def __init__(self, broker, state_store, bar_cache, logger=None):
        self._broker    = broker
        self._store     = state_store
        self._cache     = bar_cache
        self._log       = logger  # None OK; structlog when provided

        self._listeners:   list[Callable[[str, dict], None]] = []
        self._symbols:     list[str] = []
        self._running:     bool = False
        self._thread:      Optional[threading.Thread] = None

        # Last delivered bar timestamp per symbol (for missed-bar detection)
        self._last_bar_ts: dict[str, Optional[datetime]] = {}
        self._ts_lock = threading.Lock()

        # Per-symbol bar counters (for /metrics endpoint)
        self._bars_received: dict[str, int] = {}

        # WebSocket / reconnect telemetry
        self._ws_connected_at: Optional[datetime] = None
        self._last_reconnect_ts: Optional[datetime] = None
        self._last_reconnect_duration_s: Optional[float] = None
        self._ws_reconnect_count: int = 0

        # Degraded-mode REST polling fallback
        self._degraded: bool = False
        self._refresh_thread: Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_listener(self, fn: Callable[[str, dict], None]) -> None:
        """Register a callback invoked on every bar (live or replayed)."""
        self._listeners.append(fn)

    def subscribe(self, symbols: list[str]) -> None:
        """
        Start background WebSocket streaming for the given symbols.
        Non-blocking: spawns a daemon thread.

        Idempotent:
          - Same symbols already streaming → no-op.
          - Different symbols (or no stream running) → tears down any existing
            stream cleanly before starting the new one.
        """
        if self._running and set(symbols) == set(self._symbols):
            return  # already streaming these exact symbols

        if self._running:
            self.unsubscribe()  # clean teardown before restarting

        self._symbols = list(symbols)
        with self._ts_lock:
            self._last_bar_ts = {s: None for s in symbols}
        self._running  = True
        self._degraded = False
        self._thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="BarRouter"
        )
        self._thread.start()
        # Token-refresh watcher — forces reconnect every WS_TOKEN_REFRESH_HOURS
        self._refresh_thread = threading.Thread(
            target=self._token_refresh_watcher,
            daemon=True, name="BarRouter-TokenRefresh",
        )
        self._refresh_thread.start()

    @property
    def ws_token_age_minutes(self) -> Optional[float]:
        """Minutes since the current WebSocket connection was established."""
        if self._ws_connected_at is None:
            return None
        return (datetime.now(ET) - self._ws_connected_at).total_seconds() / 60.0

    def bars_received(self, symbol: str) -> int:
        """Return the number of bars delivered for symbol since last subscribe()."""
        return self._bars_received.get(symbol, 0)

    def enter_degraded_mode(self) -> None:
        """Force immediate REST-poll fallback (e.g. triggered by zero-bars watchdog)."""
        self._enter_degraded_mode()

    def unsubscribe(self) -> None:
        """
        Stop streaming: close the WebSocket connection, wait for the streaming
        thread to fully exit, then reset subscription state.

        Blocks until the thread is dead (or times out after 10 s).  This
        ensures the broker WebSocket connection is fully released before the
        next subscribe() call, preventing connection limit errors.
        """
        self._running = False
        # Close the WebSocket so subscribe_bars() returns promptly
        stop_fn = getattr(self._broker, "stop_bars_stream", None)
        if stop_fn is not None:
            try:
                stop_fn()
            except Exception:
                pass
        # Wait for threads to fully exit before returning
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5.0)
        # Reset state so the next subscribe() starts with a clean slate
        self._thread         = None
        self._refresh_thread = None
        self._symbols        = []
        self._ws_connected_at = None
        self._degraded        = False

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
            self._log.warning("missed_bar_fetch_failed", symbol=symbol, exc=str(exc))
            return []

        if df.empty:
            return []

        bars = []
        for _, row in df.iterrows():
            close = float(row["close"])
            bar = {
                "timestamp": row["timestamp"],
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

    def _on_stream_bar(self, bar) -> None:
        """
        Callback invoked by the broker on each incoming WebSocket bar.
        Normalises the bar object to a plain dict, detects missed bars,
        replays them, then dispatches the live bar.
        """
        symbol = str(bar.symbol)
        ts     = bar.timestamp
        if hasattr(ts, "astimezone"):
            ts = ts.astimezone(ET)

        bar_dict = {
            "timestamp": ts,
            "open":      float(bar.open),
            "high":      float(bar.high),
            "low":       float(bar.low),
            "close":     float(bar.close),
            "volume":    float(bar.volume),
        }

        # Replay any missed bars before the live bar
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

    def _stream_loop(self) -> None:
        """
        Blocking stream loop.  Reconnects with exponential backoff (1s→30s max).
        On each reconnect, missed bars from the gap are replayed before the
        live stream resumes (via _on_stream_bar's detect_missed_bars call).
        """
        backoff = RECONNECT_BACKOFF_INIT
        _fail_start: Optional[datetime] = None
        while self._running:
            connect_start = datetime.now(ET)
            self._ws_connected_at = connect_start
            try:
                # subscribe_bars blocks until the stream terminates
                self._broker.subscribe_bars(self._symbols, self._on_stream_bar)
                backoff = RECONNECT_BACKOFF_INIT  # clean exit → reset backoff
                _fail_start = None
            except Exception as exc:
                if not self._running:
                    break
                if self._log:
                    self._log.error("stream_disconnected", exc=str(exc), next_retry=backoff)

            if not self._running:
                break

            # Record reconnect telemetry
            now = datetime.now(ET)
            self._ws_connected_at = None
            self._last_reconnect_ts = now
            self._last_reconnect_duration_s = (now - connect_start).total_seconds()
            self._ws_reconnect_count += 1

            # Degraded-mode guard: enter REST polling after persistent failures
            if _fail_start is None:
                _fail_start = now
            elif (now - _fail_start).total_seconds() > _DEGRADED_THRESHOLD_S:
                self._enter_degraded_mode()
                _fail_start = None

            time.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)

    def _token_refresh_watcher(self) -> None:
        """Background thread: force WS reconnect every WS_TOKEN_REFRESH_HOURS."""
        refresh_interval_s = WS_TOKEN_REFRESH_HOURS * 3600.0
        while self._running:
            time.sleep(60)
            if not self._running:
                break
            if self._ws_connected_at is None:
                continue
            age_s = (datetime.now(ET) - self._ws_connected_at).total_seconds()
            if age_s >= refresh_interval_s:
                self._do_token_refresh(age_s)

    def _do_token_refresh(self, age_s: float) -> None:
        """Trigger WS token refresh by stopping the current stream (causes reconnect)."""
        if self._log:
            self._log.info("ws_token_refresh",
                           age_hours=round(age_s / 3600, 2),
                           ts=datetime.now(ET).isoformat())
        stop_fn = getattr(self._broker, "stop_bars_stream", None)
        if stop_fn is not None:
            try:
                stop_fn()
            except Exception as exc:
                if self._log:
                    self._log.critical("ws_token_refresh_failed", exc=str(exc))
                self._enter_degraded_mode()
        # If stop_fn is None (e.g. DryRunBroker), nothing to refresh — silently skip

    def _enter_degraded_mode(self) -> None:
        """Switch to REST polling fallback when WebSocket is unavailable."""
        if self._degraded:
            return
        self._degraded = True
        if self._log:
            self._log.critical(
                "ws_degraded_mode_entered",
                msg="WebSocket unavailable; falling back to 1-min REST polling",
            )
        t = threading.Thread(
            target=self._rest_poll_loop, daemon=True, name="BarRouter-REST"
        )
        t.start()

    def _rest_poll_loop(self) -> None:
        """Poll REST API at 1-min cadence when WebSocket is down."""
        last_retry = datetime.now(ET)
        while self._running and self._degraded:
            try:
                now = datetime.now(ET)
                for symbol in self._symbols:
                    start = now - timedelta(minutes=2)
                    df = self._broker.get_intraday_bars(symbol, start, now)
                    if df is not None and not df.empty:
                        row = df.iloc[-1]
                        close = float(row["close"])
                        bar = {
                            "timestamp": row["timestamp"],
                            "open":   float(row.get("open",   close)),
                            "high":   float(row.get("high",   close)),
                            "low":    float(row.get("low",    close)),
                            "close":  close,
                            "volume": float(row.get("volume", 0.0)),
                        }
                        self._dispatch(symbol, bar)
            except Exception as exc:
                if self._log:
                    self._log.error("rest_poll_failed", exc=str(exc))

            # Retry WebSocket every _DEGRADED_RETRY_S seconds
            if (datetime.now(ET) - last_retry).total_seconds() >= _DEGRADED_RETRY_S:
                if self._log:
                    self._log.info("ws_reconnect_retry_from_degraded")
                self._degraded = False
                # Stream loop is still running; it will reconnect naturally
                return

            time.sleep(60)
