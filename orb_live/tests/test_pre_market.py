"""
tests/test_pre_market.py — Unit tests for Phase 1 gap reference price and
open-eval clock trigger.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")
TDATE = date(2026, 1, 7)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bar_df(close: float, ts_hour: int = 9, ts_minute: int = 30) -> pd.DataFrame:
    ts = datetime(TDATE.year, TDATE.month, TDATE.day, ts_hour, ts_minute, tzinfo=ET)
    return pd.DataFrame(
        {"open": [close - 0.5], "high": [close + 0.5],
         "low": [close - 0.5], "close": [close], "volume": [10_000]},
        index=[ts],
    )


def _make_job(client) -> object:
    """Construct a PreMarketJob with only _client populated (no other deps needed)."""
    from orb_live.signals.pre_market import PreMarketJob
    job = PreMarketJob.__new__(PreMarketJob)
    job._client = client
    return job


# ── _get_ref_price ─────────────────────────────────────────────────────────────

class TestGetRefPrice:
    def test_uses_930_bar_close(self):
        """_get_ref_price must return the close of the first RTH bar (09:30)."""
        client = MagicMock()
        client.get_intraday_bars.return_value = _make_bar_df(185.50)

        job = _make_job(client)
        ref = job._get_ref_price("TQQQ", None, TDATE)

        assert ref == pytest.approx(185.50)

    def test_fetch_starts_at_0930(self):
        """The fetch must start at 09:30 ET to guarantee the first bar is the RTH open bar."""
        client = MagicMock()
        client.get_intraday_bars.return_value = _make_bar_df(200.0)

        job = _make_job(client)
        job._get_ref_price("TQQQ", None, TDATE)

        args, kwargs = client.get_intraday_bars.call_args
        # Second positional arg is `start`
        start = args[1]
        assert start.hour == 9 and start.minute == 30, (
            f"Expected start=09:30 (RTH filter), got {start.hour}:{start.minute:02d}"
        )

    def test_none_when_bar_unavailable(self):
        """Empty bars → None so the symbol is recorded as no_ref_price."""
        client = MagicMock()
        client.get_intraday_bars.return_value = pd.DataFrame()

        job = _make_job(client)
        assert job._get_ref_price("TQQQ", None, TDATE) is None

    def test_none_on_broker_exception(self):
        """Broker exception → None (fail gracefully, not with an unhandled error)."""
        client = MagicMock()
        client.get_intraday_bars.side_effect = RuntimeError("timeout")

        job = _make_job(client)
        assert job._get_ref_price("TQQQ", None, TDATE) is None

    def test_injected_ref_price_bypasses_broker(self):
        """Pre-injected ref_prices dict takes priority over the broker fetch."""
        client = MagicMock()

        job = _make_job(client)
        ref = job._get_ref_price("TQQQ", {"TQQQ": 99.99}, TDATE)

        assert ref == pytest.approx(99.99)
        client.get_intraday_bars.assert_not_called()


# ── Gap formula parity with backtest ─────────────────────────────────────────

class TestGapFormulaBacktestParity:
    def test_phase1_gap_matches_backtest_formula(self):
        """
        The live gap calculation must produce the same result as the backtest:
            gap = (bar_close - prior_close) / prior_close
        where bar_close is the close of the 9:30 bar.
        """
        from orb_live.signals.strategy_signals import compute_gap

        prior_close = 100.0
        bar_close   = 107.0

        daily_df = pd.DataFrame({
            "date":  [pd.Timestamp("2026-01-06")],
            "close": [prior_close],
        })

        gap_result = compute_gap(TDATE, daily_df, bar_close)
        assert gap_result is not None

        gap_abs, direction, pc = gap_result
        expected_gap = (bar_close - prior_close) / prior_close

        assert gap_abs == pytest.approx(abs(expected_gap))
        assert direction == 1    # gap up
        assert pc == pytest.approx(prior_close)

    def test_gap_down_direction(self):
        prior_close = 100.0
        bar_close   = 93.0

        from orb_live.signals.strategy_signals import compute_gap
        daily_df = pd.DataFrame({
            "date":  [pd.Timestamp("2026-01-06")],
            "close": [prior_close],
        })

        gap_result = compute_gap(TDATE, daily_df, bar_close)
        assert gap_result is not None
        gap_abs, direction, _ = gap_result
        assert direction == -1
        assert gap_abs == pytest.approx((prior_close - bar_close) / prior_close)


# ── Open-eval clock trigger ───────────────────────────────────────────────────

class TestOpenEvalTrigger:
    def test_open_eval_trigger_is_0931(self):
        """next_open_eval_start() must return 09:31:00 ET on the current trading day
        when called before 09:31."""
        from orb_live.core.clock import MarketClock, OPEN_EVAL_START

        fixed = datetime(2026, 5, 19, 8, 45, 0, tzinfo=ET)  # Monday 08:45
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: fixed  # type: ignore[method-assign]

        trigger = mc.next_open_eval_start()

        assert trigger.date() == date(2026, 5, 19)
        assert trigger.hour   == OPEN_EVAL_START.hour   == 9
        assert trigger.minute == OPEN_EVAL_START.minute == 31
        assert trigger.second == 0
        assert trigger.tzinfo is not None

    def test_open_eval_skips_holidays(self):
        """next_open_eval_start() must skip Memorial Day (2026-05-25) to 2026-05-26."""
        from orb_live.core.clock import MarketClock

        # Friday 2026-05-22 evening — session closed; next trading day skips holiday
        fixed = datetime(2026, 5, 22, 18, 0, 0, tzinfo=ET)
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: fixed  # type: ignore[method-assign]

        trigger = mc.next_open_eval_start()

        # May 25 = Memorial Day; expect May 26 (Tuesday)
        assert trigger.date() == date(2026, 5, 26)
        assert trigger.hour   == 9
        assert trigger.minute == 31

    def test_open_eval_after_trigger_time_returns_past(self):
        """After 09:31 on a trading day next_open_eval_start returns today at 09:31
        (in the past), so the runner proceeds without sleeping."""
        from orb_live.core.clock import MarketClock

        fixed = datetime(2026, 5, 19, 10, 5, 0, tzinfo=ET)  # 10:05 — already past
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: fixed  # type: ignore[method-assign]

        trigger = mc.next_open_eval_start()

        assert trigger.date() == date(2026, 5, 19)
        assert trigger < fixed  # confirms it's in the past → no sleep
