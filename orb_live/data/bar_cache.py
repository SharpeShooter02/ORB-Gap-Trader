"""
data/bar_cache.py — In-session 1-minute bar accumulator.

Collects bars from the broker's data stream during the ORB window and provides
a clean DataFrame interface for ORB high/low computation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd


class BarCache:
    """
    In-memory store for intraday 1-min bars accumulated during the session.

    Keyed by (symbol, bar_timestamp).  After the ORB window closes, call
    `get_orb_window()` to retrieve the canonical high/low/close.
    """

    def __init__(self):
        self._bars: dict[str, list[dict]] = {}

    def add_bar(self, symbol: str, bar: dict) -> None:
        """
        Append one 1-minute bar.

        bar must contain: timestamp (datetime), open, high, low, close, volume.
        """
        if symbol not in self._bars:
            self._bars[symbol] = []
        self._bars[symbol].append(bar)

    def get_bars(self, symbol: str) -> pd.DataFrame:
        """Return all bars for symbol as a DataFrame, sorted by timestamp."""
        rows = self._bars.get(symbol, [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_orb_window(
        self,
        symbol: str,
        orb_start: datetime,
        orb_end: datetime,
    ) -> Optional[dict]:
        """
        Compute ORB high, low, close from bars within [orb_start, orb_end).

        Returns dict with keys: orb_high, orb_low, orb_close, orb_range_pct.
        Returns None if no bars fall within the window.
        """
        df = self.get_bars(symbol)
        if df.empty:
            return None
        mask = (df["timestamp"] >= pd.Timestamp(orb_start)) & \
               (df["timestamp"] < pd.Timestamp(orb_end))
        window = df.loc[mask]
        if window.empty:
            return None
        orb_high  = float(window["high"].max())
        orb_low   = float(window["low"].min())
        orb_close = float(window["close"].iloc[-1])
        orb_range_pct = (orb_high - orb_low) / orb_low if orb_low > 0 else 0.0
        return {
            "orb_high":      orb_high,
            "orb_low":       orb_low,
            "orb_close":     orb_close,
            "orb_range_pct": orb_range_pct,
        }

    def symbols_with_bars(self) -> list[str]:
        return [s for s, rows in self._bars.items() if rows]

    def clear(self, symbol: Optional[str] = None) -> None:
        if symbol:
            self._bars.pop(symbol, None)
        else:
            self._bars.clear()
