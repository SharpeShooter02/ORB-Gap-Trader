"""
tests/test_daemon_loop.py — Tests for the daemon loop pre-market sleep logic.

All tests use fixed clocks and injected sleep/shutdown so no real time elapses.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


# ── Helpers ────────────────────────────────────────────────────────────────────

class _FakeClock:
    """Minimal MarketClock stub that returns a fixed 'now' with no Alpaca client."""

    def __init__(self, fixed_now: datetime):
        self._fixed_now = fixed_now
        self._client = None  # no Alpaca — uses weekday fallback

    def now_et(self) -> datetime:
        return self._fixed_now

    def next_market_day(self) -> date:
        from orb_live.core.clock import MarketClock
        # Delegate to the real implementation via a throw-away instance
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: self._fixed_now  # type: ignore[method-assign]
        return mc.next_market_day()

    def next_premarket_start(self) -> datetime:
        from orb_live.core.clock import MarketClock
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: self._fixed_now  # type: ignore[method-assign]
        return mc.next_premarket_start()


# ── MarketClock unit tests ─────────────────────────────────────────────────────

def test_next_market_day_before_premarket_is_today():
    """Before 08:30 on a weekday, next market day is today."""
    from orb_live.core.clock import MarketClock, PREMARKET_START

    fixed = datetime(2026, 5, 18, 3, 0, 0, tzinfo=ET)  # Monday 03:00
    mc = MarketClock(broker_client=None)
    mc.now_et = lambda: fixed  # type: ignore[method-assign]

    assert mc.next_market_day() == date(2026, 5, 18)
    pm = mc.next_premarket_start()
    assert pm.date() == date(2026, 5, 18)
    assert pm.hour == PREMARKET_START.hour
    assert pm.minute == PREMARKET_START.minute


def test_next_market_day_after_close_friday_is_monday():
    """After Friday close, next market day is Monday."""
    from orb_live.core.clock import MarketClock, PREMARKET_START

    fixed = datetime(2026, 5, 22, 16, 30, 0, tzinfo=ET)  # Friday 16:30
    mc = MarketClock(broker_client=None)
    mc.now_et = lambda: fixed  # type: ignore[method-assign]

    assert mc.next_market_day() == date(2026, 5, 25)
    pm = mc.next_premarket_start()
    assert pm.date() == date(2026, 5, 25)
    assert pm.hour == PREMARKET_START.hour
    assert pm.minute == PREMARKET_START.minute


def test_next_market_day_after_premarket_but_before_open_is_today():
    """Between 08:30 and 09:30 (pre-open), pre-market is past → same session."""
    from orb_live.core.clock import MarketClock

    fixed = datetime(2026, 5, 18, 8, 45, 0, tzinfo=ET)  # Monday 08:45
    mc = MarketClock(broker_client=None)
    mc.now_et = lambda: fixed  # type: ignore[method-assign]

    # next_market_day returns today (pre-market passed this session)
    assert mc.next_market_day() == date(2026, 5, 18)
    # next_premarket_start is 08:30 today — already in the past
    pm = mc.next_premarket_start()
    assert pm < fixed  # confirms it's in the past → daemon runs immediately


# ── Daemon loop tests ─────────────────────────────────────────────────────────

def test_daemon_sleeps_until_premarket_then_runs():
    """
    Started at 03:00 ET Monday: daemon sleeps the exact number of seconds
    until 08:30 ET, then runs the session for that date.
    """
    from orb_live.runner.main import _run_daemon

    fixed_now = datetime(2026, 5, 18, 3, 0, 0, tzinfo=ET)  # Monday 03:00
    clock = _FakeClock(fixed_now)
    expected_pm   = clock.next_premarket_start()
    expected_secs = (expected_pm - fixed_now).total_seconds()

    sessions_run = []
    shutdown = [False]

    class _FakeRunner:
        def run_session(self, d):
            sessions_run.append(d)
            shutdown[0] = True  # stop after first session

    slept = []

    def _fake_sleep(s):
        slept.append(s)

    # Large interval → sleep happens in a single chunk
    _run_daemon(
        _FakeRunner(), clock,
        _sleep=_fake_sleep, _shutdown=shutdown,
        _sleep_interval=expected_secs + 1,
    )

    assert abs(sum(slept) - expected_secs) < 1.0
    assert sessions_run == [date(2026, 5, 18)]


def test_daemon_skips_weekend_to_monday():
    """
    Started at Friday 16:30 ET: daemon sleeps ~64 hours to Monday 08:30,
    then runs the Monday session.
    """
    from orb_live.runner.main import _run_daemon

    fixed_now = datetime(2026, 5, 22, 16, 30, 0, tzinfo=ET)  # Friday 16:30
    clock = _FakeClock(fixed_now)
    expected_pm   = clock.next_premarket_start()
    expected_secs = (expected_pm - fixed_now).total_seconds()

    assert expected_pm.date() == date(2026, 5, 25)  # Monday
    assert abs(expected_secs - 64 * 3600) < 1.0    # 64 hours exactly

    sessions_run = []
    shutdown = [False]

    class _FakeRunner:
        def run_session(self, d):
            sessions_run.append(d)
            shutdown[0] = True

    slept = []

    def _fake_sleep(s):
        slept.append(s)

    _run_daemon(
        _FakeRunner(), clock,
        _sleep=_fake_sleep, _shutdown=shutdown,
        _sleep_interval=expected_secs + 1,
    )

    assert abs(sum(slept) - expected_secs) < 1.0
    assert sessions_run == [date(2026, 5, 25)]


def test_daemon_runs_immediately_if_premarket_already_passed():
    """
    Started mid-session (10:00 ET): next_premarket_start is in the past
    (08:30 today), so the daemon runs the session without sleeping.
    """
    from orb_live.runner.main import _run_daemon

    fixed_now = datetime(2026, 5, 18, 10, 0, 0, tzinfo=ET)  # Monday 10:00
    clock = _FakeClock(fixed_now)

    sessions_run = []
    shutdown = [False]

    class _FakeRunner:
        def run_session(self, d):
            sessions_run.append(d)
            shutdown[0] = True

    slept = []

    def _fake_sleep(s):
        slept.append(s)

    _run_daemon(
        _FakeRunner(), clock,
        _sleep=_fake_sleep, _shutdown=shutdown,
        _sleep_interval=60.0,
    )

    assert slept == []               # no sleep
    assert sessions_run == [date(2026, 5, 18)]


def test_daemon_shutdown_during_sleep_skips_session():
    """Shutdown flag set during the first sleep chunk aborts before the session runs."""
    from orb_live.runner.main import _run_daemon

    fixed_now = datetime(2026, 5, 18, 3, 0, 0, tzinfo=ET)
    clock = _FakeClock(fixed_now)
    expected_secs = (clock.next_premarket_start() - fixed_now).total_seconds()

    sessions_run = []
    shutdown = [False]

    class _FakeRunner:
        def run_session(self, d):
            sessions_run.append(d)  # pragma: no cover

    def _fake_sleep(s):
        shutdown[0] = True  # signal during first sleep chunk

    _run_daemon(
        _FakeRunner(), clock,
        _sleep=_fake_sleep, _shutdown=shutdown,
        _sleep_interval=60.0,
    )

    assert sessions_run == []
