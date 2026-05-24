"""
core/calendar.py — Shared NYSE trading calendar helpers.

Provides a single authoritative holiday set and market-day helpers used by
both IBClient and MarketClock so the two never diverge.
"""

from __future__ import annotations

from datetime import date, timedelta

# Hard-coded NYSE holidays for 2025-2027.
# Upgrade path: install pandas_market_calendars and is_trading_day() will
# use it automatically; the frozenset is the offline fallback.
NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2025
    date(2025, 1, 1),  date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4),  date(2025, 9, 1),  date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3),  date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3),  date(2026, 9, 7),  date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1),  date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5),  date(2027, 9, 6),  date(2027, 11, 25),
    date(2027, 12, 24),
})


def is_trading_day(d: date) -> bool:
    """True if d is a NYSE trading day (weekday and not a US market holiday)."""
    try:
        import pandas_market_calendars as mcal  # optional dependency
        nyse     = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=d.strftime("%Y-%m-%d"),
            end_date=d.strftime("%Y-%m-%d"),
        )
        return not schedule.empty
    except ImportError:
        pass
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def next_trading_day(from_date: date) -> date:
    """First NYSE trading day on or after from_date."""
    d = from_date
    for _ in range(14):  # longest US holiday stretch < 7 calendar days
        if is_trading_day(d):
            return d
        d += timedelta(days=1)
    return d
