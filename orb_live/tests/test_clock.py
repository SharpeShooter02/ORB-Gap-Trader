"""
tests/test_clock.py — MarketClock unit tests.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def test_now_et_is_tz_aware(market_clock):
    now = market_clock.now_et()
    assert now.tzinfo is not None
    assert str(now.tzinfo) == "America/New_York"


def test_market_open_et_correct_time(market_clock):
    open_dt = market_clock.market_open_et()
    assert open_dt.hour == 9
    assert open_dt.minute == 30
    assert open_dt.second == 0


def test_orb_end_default_30min(market_clock):
    from datetime import timedelta
    orb_end = market_clock.orb_end_et(30)
    expected = market_clock.market_open_et() + timedelta(minutes=30)
    assert orb_end == expected


def test_current_phase_returns_valid_string(market_clock):
    phase = market_clock.current_phase()
    assert phase in ("pre_market", "orb_window", "trade_active", "closed")


def test_seconds_until_open_numeric(market_clock):
    secs = market_clock.seconds_until_open()
    assert isinstance(secs, float)


def test_eod_exit_correct(market_clock):
    eod = market_clock.eod_exit_et(16, 0)
    assert eod.hour == 16
    assert eod.minute == 0
