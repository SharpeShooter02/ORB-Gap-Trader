"""
core/clock.py — Market timing helpers.

All times are Eastern.  Uses zoneinfo (Python 3.9+) for DST-safe conversions.
"""

from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

MARKET_OPEN     = dtime(9, 30)
MARKET_CLOSE    = dtime(16, 0)
HALF_DAY_CLOSE  = dtime(13, 0)
PREMARKET_START = dtime(8, 30)
OPEN_EVAL_START = dtime(9, 31)  # earliest safe time to read the 9:30 bar close


class MarketClock:
    """
    Wraps market timing logic.  Optionally backed by a broker client for
    authoritative half-day / holiday awareness.
    """

    def __init__(self, broker_client=None):
        self._client = broker_client

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

    # ── Next-session helpers ──────────────────────────────────────────────────

    def next_market_day(self) -> date:
        """
        Return the date of the next market session whose pre-market window
        (08:30 ET) has not yet started.  If the current time is before 08:30
        today on a weekday, returns today.  Uses the broker client for holiday
        awareness when one is available; falls back to the shared calendar
        module otherwise.

        The broker API call is bounded to 10 seconds so a slow or unreachable
        endpoint degrades gracefully to the calendar fallback rather than
        blocking the daemon loop indefinitely.
        """
        import threading as _threading
        from orb_live.core.calendar import is_trading_day, next_trading_day

        now   = self.now_et()
        today = now.date()
        now_t = now.time()

        if self._client is not None:
            _clock_result: list = [None]

            def _fetch():
                try:
                    _clock_result[0] = self._client.get_clock()
                except Exception:
                    pass

            _t = _threading.Thread(target=_fetch, daemon=True)
            _t.start()
            _t.join(timeout=10.0)

            clk = _clock_result[0]
            if clk is not None:
                try:
                    # get_clock() returns a dict for IBClient; some clients return
                    # an object with attribute access — handle both.
                    is_open   = clk["is_open"]   if isinstance(clk, dict) else clk.is_open
                    next_open = clk["next_open"]  if isinstance(clk, dict) else clk.next_open
                    if is_open:
                        return today
                    next_open_date = next_open.astimezone(ET).date()
                    if next_open_date == today and now_t < PREMARKET_START:
                        return today
                    return next_open_date
                except Exception:
                    pass

        # Fallback: use shared calendar (holiday-aware).
        # Return today if today is a trading day and the session hasn't closed,
        # so mid-session startups trigger an immediate run rather than sleeping.
        today_eod = now.replace(
            hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute,
            second=0, microsecond=0,
        )
        if is_trading_day(today) and now < today_eod:
            return today
        return next_trading_day(today + timedelta(days=1))

    def next_premarket_start(self) -> datetime:
        """Return the next 08:30 ET on a market day as an aware datetime."""
        next_day = self.next_market_day()
        return datetime.combine(next_day, PREMARKET_START).replace(tzinfo=ET)

    def next_open_eval_start(self) -> datetime:
        """Return the next 09:31 ET on a market day as an aware datetime.

        09:31 is the earliest safe trigger for Phase 1 gap evaluation: the
        9:30 bar has closed and its close price is available from the broker's
        historical data API.
        """
        next_day = self.next_market_day()
        return datetime.combine(next_day, OPEN_EVAL_START).replace(tzinfo=ET)

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
