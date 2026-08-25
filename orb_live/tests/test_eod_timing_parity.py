"""Live and the backtest must exit at the same clock time.

For four sessions they did not, and the reason was that this file previously
asserted the wrong object. LiveConfig.eod_exit_hour is 16:00 and really is a
never-fires sentinel -- but LivePositionManager is not given LiveConfig. It is
given cfg.strategy_config (runner/main.py), and StrategyConfig defaulted to
15:55. So the bar-driven exit in position_manager fired on the 15:55 bar,
delivered ~15:56:05, while the scheduled flatten arrived at 15:59 and found
nothing left to close. Confirmed in live.db: EOD trade rows land at 19:56:0X
UTC on every session, day_state closes at 19:59:00.2.

The old test passed throughout, because it checked the field that wasn't wired
to anything. A parity test has to assert on the value the running code reads.

Two exit paths exist and their ORDER matters:

  bar-driven   position_manager, `bar_ts.time() >= eod_time`. Primary. Fires
               when the 15:58 bar is DELIVERED, ~15:59:05.
  flatten      session_runner._wait_until_eod, at close - eod_flatten_lead_secs.
               Safety net. Must land AFTER the bar exit, or it pre-empts it and
               becomes the de facto exit path at a different price.

The backtester uses the identical predicate (orb_backtester.py:2158) against
the identical field name, so matching the two configs is sufficient for parity.

Timing value, measured full-sample:
    15:59 +19.172   15:58 +18.880 (-1.5%)   15:57 +18.837   15:55 +18.115 (-5.5%)
"""
from __future__ import annotations

from datetime import time

from orb_live.config.live_config import LiveConfig
from orb_live.config.strategy_config import StrategyConfig


#: The single source of truth for this project's exit time.
EOD_EXIT_ET = time(15, 58)
MARKET_CLOSE_ET = time(16, 0)


def _secs(t: time) -> int:
    return t.hour * 3600 + t.minute * 60


class TestLiveExitsAt1558:
    def test_strategy_config_drives_the_real_exit(self):
        """This is the field position_manager reads. It is the exit time."""
        cfg = StrategyConfig()
        assert (cfg.eod_exit_hour, cfg.eod_exit_minute) == (
            EOD_EXIT_ET.hour, EOD_EXIT_ET.minute)

    def test_position_manager_is_handed_the_strategy_config(self):
        """The wiring that made the old test vacuous. If main.py ever passes
        LiveConfig here instead, the exit silently jumps to 16:00."""
        import inspect
        from orb_live.runner import main
        src = inspect.getsource(main)
        assert "config=cfg.strategy_config" in src, (
            "LivePositionManager is no longer constructed with strategy_config; "
            "the exit time this test guards is read from somewhere else now")

    def test_live_config_sentinel_is_not_the_exit(self):
        """LiveConfig.eod_exit_* is genuinely unreachable by an RTH bar. It is
        kept only so nothing reads it and gets a plausible-looking answer."""
        cfg = LiveConfig()
        assert (cfg.eod_exit_hour, cfg.eod_exit_minute) == (16, 0)
        assert _secs(time(cfg.eod_exit_hour, cfg.eod_exit_minute)) >= _secs(
            MARKET_CLOSE_ET)


class TestSafetyNetDoesNotPreemptTheBarExit:
    def test_flatten_lands_after_the_bar_exit_is_delivered(self):
        """The 15:58 bar arrives ~15:59:05. A flatten scheduled at or before
        that hijacks the exit -- positions close at the flatten's market price
        instead of the bar the backtest models."""
        cfg = LiveConfig()
        flatten = _secs(MARKET_CLOSE_ET) - cfg.eod_flatten_lead_secs
        delivery = _secs(EOD_EXIT_ET) + 65   # bar close + delivery lag
        assert flatten > delivery, (
            f"flatten at {flatten//3600:02d}:{(flatten%3600)//60:02d}:"
            f"{flatten%60:02d} pre-empts the bar exit delivered at "
            f"{delivery//3600:02d}:{(delivery%3600)//60:02d}:{delivery%60:02d}")

    def test_flatten_leaves_room_before_the_bell(self):
        """Market exits sent after 16:00 are rejected by IB on leveraged ETFs."""
        cfg = LiveConfig()
        assert cfg.eod_flatten_lead_secs >= 20


class TestBacktestMatches:
    def test_run_v1_defaults_match_live(self):
        import inspect
        import sys
        sys.path.insert(0, r"c:/Users/buttn/Documents/Projects/BacktestingGaps")
        from scripts.optimize_v1_3class import run_v1_at_k

        sig = inspect.signature(run_v1_at_k)
        h = sig.parameters["eod_exit_hour"].default
        m = sig.parameters["eod_exit_minute"].default
        assert (h, m) == (EOD_EXIT_ET.hour, EOD_EXIT_ET.minute)

    def test_backtester_default_matches_live(self):
        """run_v1_at_k is one caller. The backtester's own default is what any
        other script inherits, and it drifted to 15:55 independently."""
        import sys
        sys.path.insert(0, r"c:/Users/buttn/Documents/Projects/BacktestingGaps")
        from orb_backtester import StrategyConfig as BTConfig

        c = BTConfig()
        assert (c.eod_exit_hour, c.eod_exit_minute) == (
            EOD_EXIT_ET.hour, EOD_EXIT_ET.minute)
