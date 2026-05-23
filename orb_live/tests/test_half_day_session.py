"""
tests/test_half_day_session.py — Half-day schedule correctness tests.

Verifies that SessionRunner._wait_until_eod respects MarketClock.effective_close()
rather than the config's fixed eod_exit_hour so positions are not left
unmanaged for hours on early-close days.
"""

from datetime import date, datetime, time as dtime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import orb_live  # noqa: F401 — path setup

ET    = ZoneInfo("America/New_York")
TDATE = date(2026, 7, 3)   # Independence Day eve — typical half day


# ── Component stubs ────────────────────────────────────────────────────────────

class _StubRouter:
    def register_listener(self, fn): pass
    def subscribe(self, symbols): pass
    def unsubscribe(self): pass


# ── Clock stubs ────────────────────────────────────────────────────────────────

def _et(h: int, m: int) -> datetime:
    return datetime(TDATE.year, TDATE.month, TDATE.day, h, m, 0, tzinfo=ET)


class _HalfDayClock:
    """Simulates a half-day session: market close 13:00 ET."""
    def now_et(self): return _et(10, 5)
    def orb_end_et(self, orb_minutes=30): return _et(10, 0)
    def eod_exit_et(self, h=16, m=0): return _et(13, 0)
    def is_half_day(self): return True
    def effective_close(self): return dtime(13, 0)
    def is_rth_open(self): return False


class _FullDayClock:
    """Simulates a normal full-day session: market close 16:00 ET."""
    def now_et(self): return _et(10, 5)
    def orb_end_et(self, orb_minutes=30): return _et(10, 0)
    def eod_exit_et(self, h=16, m=0): return _et(16, 0)
    def is_half_day(self): return False
    def effective_close(self): return dtime(16, 0)
    def is_rth_open(self): return False


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_1_half_day_sleeps_to_1300_not_1600(mock_alpaca, tmp_store):
    """
    _wait_until_eod on a half-day should sleep until 13:00 ET, not 16:00.

    now_et() returns 10:05 ET.  Sleep seconds = (13:00 - 10:05) = 2h55m = 10500s.
    A full-day would produce (16:00 - 10:05) = 5h55m = 21300s.
    """
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.config.live_config import load_live_config

    cfg = load_live_config()
    slept: list[float] = []

    runner = SessionRunner(
        config=cfg,
        broker=mock_alpaca,
        state_store=tmp_store,
        bar_cache=SimpleNamespace(),
        bar_router=_StubRouter(),
        pre_market_job=SimpleNamespace(),
        strategy_engine=SimpleNamespace(),
        position_manager=SimpleNamespace(),
        risk_gate=SimpleNamespace(),
        indicators_store={},
        underlying_store=SimpleNamespace(),
        clock=_HalfDayClock(),
        _sleep=lambda s: slept.append(s),
    )
    runner._wait_until_eod(TDATE)

    assert len(slept) == 1
    slept_s = slept[0]
    # 13:00 - 10:05 = 2h55m = 10500 seconds
    assert 10490 < slept_s < 10510, (
        f"Expected ~10500s sleep for half-day (13:00 close), got {slept_s:.1f}s"
    )


def test_2_full_day_sleeps_to_1600(mock_alpaca, tmp_store):
    """
    _wait_until_eod on a normal day should sleep until 16:00 ET.

    now_et() returns 10:05 ET.  Sleep seconds = (16:00 - 10:05) = 5h55m = 21300s.
    """
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.config.live_config import load_live_config

    cfg = load_live_config()
    slept: list[float] = []

    runner = SessionRunner(
        config=cfg,
        broker=mock_alpaca,
        state_store=tmp_store,
        bar_cache=SimpleNamespace(),
        bar_router=_StubRouter(),
        pre_market_job=SimpleNamespace(),
        strategy_engine=SimpleNamespace(),
        position_manager=SimpleNamespace(),
        risk_gate=SimpleNamespace(),
        indicators_store={},
        underlying_store=SimpleNamespace(),
        clock=_FullDayClock(),
        _sleep=lambda s: slept.append(s),
    )
    runner._wait_until_eod(TDATE)

    assert len(slept) == 1
    slept_s = slept[0]
    # 16:00 - 10:05 = 5h55m = 21300 seconds
    assert 21290 < slept_s < 21310, (
        f"Expected ~21300s sleep for full day (16:00 close), got {slept_s:.1f}s"
    )


def test_3_half_day_warning_fires(mock_alpaca, tmp_store):
    """
    _wait_until_eod must log 'half_day_detected' when is_half_day() is True.
    """
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.config.live_config import load_live_config

    cfg = load_live_config()
    logged: list[str] = []

    class _CapturingLog:
        def warning(self, event, **kw):
            logged.append(event)
        def info(self, event, **kw):
            pass

    runner = SessionRunner(
        config=cfg,
        broker=mock_alpaca,
        state_store=tmp_store,
        bar_cache=SimpleNamespace(),
        bar_router=_StubRouter(),
        pre_market_job=SimpleNamespace(),
        strategy_engine=SimpleNamespace(),
        position_manager=SimpleNamespace(),
        risk_gate=SimpleNamespace(),
        indicators_store={},
        underlying_store=SimpleNamespace(),
        clock=_HalfDayClock(),
        _sleep=lambda _: None,
    )
    runner._log = _CapturingLog()
    runner._wait_until_eod(TDATE)

    assert "half_day_detected" in logged, (
        f"Expected 'half_day_detected' log event on half-day, got: {logged}"
    )


def test_4_full_day_no_half_day_warning(mock_alpaca, tmp_store):
    """
    _wait_until_eod must NOT log 'half_day_detected' on a normal trading day.
    """
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.config.live_config import load_live_config

    cfg = load_live_config()
    logged: list[str] = []

    class _CapturingLog:
        def warning(self, event, **kw):
            logged.append(event)
        def info(self, event, **kw):
            pass

    runner = SessionRunner(
        config=cfg,
        broker=mock_alpaca,
        state_store=tmp_store,
        bar_cache=SimpleNamespace(),
        bar_router=_StubRouter(),
        pre_market_job=SimpleNamespace(),
        strategy_engine=SimpleNamespace(),
        position_manager=SimpleNamespace(),
        risk_gate=SimpleNamespace(),
        indicators_store={},
        underlying_store=SimpleNamespace(),
        clock=_FullDayClock(),
        _sleep=lambda _: None,
    )
    runner._log = _CapturingLog()
    runner._wait_until_eod(TDATE)

    assert "half_day_detected" not in logged, (
        f"Got unexpected 'half_day_detected' on full trading day: {logged}"
    )
