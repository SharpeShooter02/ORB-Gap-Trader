"""Live and the backtest must flatten at the same time.

They did not. run_v1_at_k overrode eod_exit to 16:00, which no delivered RTH
bar ever reaches (the last is 15:59), so the backtest fell through to its
end-of-loop fallback and booked the 15:59 close. Live schedules its flatten off
effective_close() minus eod_flatten_lead_secs, which at 60s meant ~15:59 --
close in intent, but the two were arriving there by unrelated mechanisms and
neither referenced the other.

Measured across the full sample, exit timing is worth real money and the value
is front-loaded rather than spread evenly:

    15:59  +19.172      15:58  +18.880 (-1.5%)
    15:57  +18.837 (-1.7%)      15:55  +18.115 (-5.5%, negative in all 7 years)

15:58 is the chosen setting: it captures 72% of the gap from 15:55 while
leaving two minutes of margin before the bell. The final minute is worth
another 1.5% but is also where the backtest is least trustworthy -- it books
the 15:59 bar's close, effectively the 16:00 print, which assumes a fill at the
closing price during the auction.
"""
from __future__ import annotations

from datetime import time

from orb_live.config.live_config import LiveConfig


#: The single source of truth for this project's flatten time.
EOD_FLATTEN_ET = time(15, 58)
MARKET_CLOSE_ET = time(16, 0)


class TestLiveFlattensAt1558:
    def test_lead_puts_the_flatten_at_1558(self):
        cfg = LiveConfig()
        close_secs = MARKET_CLOSE_ET.hour * 3600 + MARKET_CLOSE_ET.minute * 60
        flatten_secs = close_secs - cfg.eod_flatten_lead_secs
        want = EOD_FLATTEN_ET.hour * 3600 + EOD_FLATTEN_ET.minute * 60
        assert flatten_secs == want, (
            f"flatten lands at {flatten_secs//3600:02d}:"
            f"{(flatten_secs%3600)//60:02d}, want {EOD_FLATTEN_ET}")

    def test_lead_leaves_margin_before_the_close(self):
        """Today's flatten took ~66s end to end. Anything under that is a
        position carried overnight, which this strategy has no model for."""
        assert LiveConfig().eod_flatten_lead_secs >= 90

    def test_eod_exit_hour_is_the_never_fires_sentinel(self):
        """16:00 is deliberate: no RTH bar reaches it, so the bar-driven exit
        never pre-empts the scheduled flatten. Changing it to a reachable time
        would create a second, earlier exit path."""
        cfg = LiveConfig()
        assert (cfg.eod_exit_hour, cfg.eod_exit_minute) == (16, 0)


class TestBacktestMatches:
    def test_run_v1_defaults_match_live(self):
        """The backtest's default exit must be the same clock time live uses,
        or every backtest number carries an exit-timing bias."""
        import inspect
        import sys
        sys.path.insert(0, r"c:/Users/buttn/Documents/Projects/BacktestingGaps")
        from scripts.optimize_v1_3class import run_v1_at_k

        sig = inspect.signature(run_v1_at_k)
        h = sig.parameters["eod_exit_hour"].default
        m = sig.parameters["eod_exit_minute"].default
        assert (h, m) == (EOD_FLATTEN_ET.hour, EOD_FLATTEN_ET.minute), (
            f"backtest exits {h:02d}:{m:02d}, live flattens at "
            f"{EOD_FLATTEN_ET}")
