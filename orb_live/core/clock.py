"""
core/clock.py — Market timing helpers.

All times are Eastern.  Uses zoneinfo (Python 3.9+) for DST-safe conversions.
"""

from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
HALF_DAY_CLOSE = dtime(13, 0)

# Alpaca is imported lazily so that imports don't fail without credentials.
_alpaca_client = None


class MarketClock:
    """
    Wraps market timing logic.  Optionally backed by an Alpaca trading client
    for authoritative half-day / holiday awareness.
    """

    def __init__(self, alpaca_client=None):
        self._client = alpaca_client

    # ── Time helpers ──────────────────────────────────────────────────────────

    def now_et(self) -> datetime:
        return datetime.now(tz=ET)

    def today_et(self):
        return self.now_et().date()

    # ── Session state ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """True if the regular session is currently open."""
        if self._client is not None:
            try:
                clock = self._client.get_clock()
                return clock.is_open
            except Exception:
                pass
        now = self.now_et().time()
        return MARKET_OPEN <= now < MARKET_CLOSE

    def is_rth_open(self) -> bool:
        """Alias for is_market_open(); named for readability at call sites."""
        return self.is_market_open()

    def is_half_day(self) -> bool:
        """True if today is a half-day (early close at 13:00 ET)."""
        if self._client is not None:
            try:
                clock = self._client.get_clock()
                close_et = clock.next_close.astimezone(ET)
                return close_et.time() < MARKET_CLOSE
            except Exception:
                pass
        return False

    def effective_close(self) -> dtime:
        return HALF_DAY_CLOSE if self.is_half_day() else MARKET_CLOSE

    # ── Session boundary datetimes ────────────────────────────────────────────

    def market_open_et(self) -> datetime:
        """Return today's 9:30 ET as an aware datetime."""
        now = self.now_et()
        return now.replace(hour=9, minute=30, second=0, microsecond=0)

    def orb_end_et(self, orb_minutes: int = 30) -> datetime:
        """Return the end of the ORB window (default 10:00 ET)."""
        return self.market_open_et() + timedelta(minutes=orb_minutes)

    def eod_exit_et(self, hour: int = 16, minute: int = 0) -> datetime:
        now = self.now_et()
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # ── Countdown helpers ─────────────────────────────────────────────────────

    def seconds_until_open(self) -> float:
        """Seconds until 9:30 ET today.  Negative if past open."""
        return (self.market_open_et() - self.now_et()).total_seconds()

    def seconds_until_orb_end(self, orb_minutes: int = 30) -> float:
        return (self.orb_end_et(orb_minutes) - self.now_et()).total_seconds()

    def seconds_until_eod(self, hour: int = 16, minute: int = 0) -> float:
        return (self.eod_exit_et(hour, minute) - self.now_et()).total_seconds()

    # ── Phase detection ───────────────────────────────────────────────────────

    def current_phase(self, orb_minutes: int = 30) -> str:
        """
        Return a string indicating the current trading phase:
          'pre_market'  — before 9:30
          'orb_window'  — 9:30..10:00 (formation)
          'trade_active' — 10:00..eod
          'closed'       — after eod or non-trading day
        """
        now_t = self.now_et().time()
        if now_t < MARKET_OPEN:
            return "pre_market"
        orb_end = (datetime.combine(self.today_et(), MARKET_OPEN)
                   + timedelta(minutes=orb_minutes)).time()
        if now_t < orb_end:
            return "orb_window"
        effective = self.effective_close()
        if now_t < effective:
            return "trade_active"
        return "closed"
