"""
data/underlying_data.py — yfinance-backed prior-session underlying data.

Provides:
  1. UnderlyingDataStore — persistent parquet-backed store for daily bars.
  2. Legacy standalone fetch functions (kept for smoke_test.py compatibility).

TICKER MAPPING
--------------
Crypto underlyings use yfinance's Yahoo Finance convention (BTC-USD etc.).
VIX uses the Yahoo Finance caret prefix (^VIX).
All equity tickers map to themselves.

HOLIDAY / WEEKEND ALIGNMENT
----------------------------
The backtest filters with:
    ul_df[ul_df["date"] < pd.Timestamp(date_)].tail(2)
and takes the LAST TWO ROWS.  This is replicated exactly — no calendar
alignment is attempted.  Crypto has weekend bars; equity does not.  The
filter naturally uses the most recent 2 closes available for each
underlying, regardless of whether those dates coincide with the ETF's
trading calendar.

YFINANCE NOTE
-------------
yfinance and AlphaVantage may differ on exact close values by small
amounts due to different aggregation cutoffs (especially for crypto).
This is acceptable: the PS filter's k=1.25 sigma multiplier provides
sufficient buffer to absorb these minor data-source differences.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Ticker mapping ────────────────────────────────────────────────────────────
# Key = underlying symbol as used in SIGMA / PS_FILTERS.
# Value = yfinance ticker string.
# Equity tickers map to themselves; raise on first access if not present.

YF_TICKER_MAP: dict[str, str] = {
    # ── Crypto ────────────────────────────────────────────────────────────
    "BTC":  "BTC-USD",
    "ETH":  "ETH-USD",
    "SOL":  "SOL-USD",
    "XRP":  "XRP-USD",
    "LINK": "LINK-USD",
    "ADA":  "ADA-USD",
    "XLM":  "XLM-USD",
    # ── Volatility index ─────────────────────────────────────────────────
    "VIX":  "^VIX",
    # ── All equity underlyings (identity mapping) ─────────────────────────
    "QQQ":  "QQQ",  "SPY":  "SPY",  "IWM":  "IWM",  "DIA":  "DIA",
    "XLF":  "XLF",  "XLE":  "XLE",  "IBB":  "IBB",  "IHE":  "IHE",
    "GDX":  "GDX",  "GDXJ": "GDXJ", "SOXX": "SOXX",
    "EEM":  "EEM",  "VWO":  "VWO",  "FXI":  "FXI",  "EWY":  "EWY",
    "EWW":  "EWW",  "ASHR": "ASHR", "INDY": "INDY", "KWEB": "KWEB",
    "XLK":  "XLK",  "XLC":  "XLC",  "XLV":  "XLV",
    "KRE":  "KRE",  "XRT":  "XRT",  "ITB":  "ITB",
    "ITA":  "ITA",  "IYR":  "IYR",  "XOP":  "XOP",
    "GLD":  "GLD",  "SLV":  "SLV",  "USO":  "USO",  "UNG":  "UNG",
    "TLT":  "TLT",  "IEF":  "IEF",
    "MSTR": "MSTR", "COIN": "COIN", "TSM":  "TSM",
    "NVDA": "NVDA", "TSLA": "TSLA", "SMCI": "SMCI", "AMD":  "AMD",
    "MDY":  "MDY",  "FEZ":  "FEZ",  "VGK":  "VGK",  "EWJ":  "EWJ",
    "EWZ":  "EWZ",  "XLY":  "XLY",  "XLU":  "XLU",  "XLB":  "XLB",
    "XLP":  "XLP",  "XLI":  "XLI",  "KBE":  "KBE",
}


def _yf_ticker(underlying: str) -> str:
    """Resolve underlying symbol → yfinance ticker.  Raises if unknown."""
    if underlying in YF_TICKER_MAP:
        return YF_TICKER_MAP[underlying]
    raise KeyError(
        f"Unknown underlying '{underlying}' — add it to YF_TICKER_MAP "
        f"in data/underlying_data.py before using it."
    )


# ── UnderlyingDataStore ───────────────────────────────────────────────────────

class UnderlyingDataStore:
    """
    Persistent parquet-backed store for underlying daily OHLCV bars.

    Storage layout: <data_dir>/<UNDERLYING>.parquet
    Columns: [date (datetime64[ns]), open, high, low, close, volume]
    Sorted ascending by date; deduplicated on date.

    All reads return a view — do not mutate the returned DataFrames.
    """

    def __init__(self, data_dir: Path, logger_=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._log = logger_ or logger

    def _path(self, ul_sym: str) -> Path:
        return self.data_dir / f"{ul_sym}.parquet"

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, ul_sym: str) -> pd.DataFrame:
        """
        Return full daily bar history as a DataFrame with columns
        [date (datetime64[ns]), open, high, low, close, volume].

        Returns empty DataFrame if no parquet file exists yet.
        Raises KeyError if ul_sym is not in YF_TICKER_MAP (fail fast).
        """
        _yf_ticker(ul_sym)   # validate; raises if unknown
        path = self._path(ul_sym)
        if not path.exists():
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def warn_if_stale(
        self,
        ul_sym: str,
        today: date,
        max_age_days: int = 3,
    ) -> Optional[str]:
        """
        Return a warning string if the most recent bar is older than expected.
        Returns None if data is fresh.

        Uses calendar days — not trading days — to keep it simple.
        For crypto, max_age_days=2 is a sensible tighter bound since
        crypto trades 7 days a week.
        """
        df = self.get(ul_sym)
        if df.empty:
            return f"{ul_sym}: no data on disk"
        latest = df["date"].iloc[-1].date()
        age = (today - latest).days
        if age > max_age_days:
            return (
                f"{ul_sym}: latest bar is {latest} ({age} days ago, "
                f"max_age={max_age_days})"
            )
        return None

    # ── Write ─────────────────────────────────────────────────────────────────

    def update_one(self, ul_sym: str, lookback_days: int = 10) -> int:
        """
        Fetch the most recent `lookback_days` of daily bars from yfinance
        and append to the parquet file (deduplicating on date).

        Returns the number of NEW rows written.
        """
        import yfinance as yf

        yf_sym = _yf_ticker(ul_sym)
        end    = pd.Timestamp.today().normalize()
        start  = end - pd.Timedelta(days=max(lookback_days * 2, 30))

        raw = yf.download(
            yf_sym,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if raw.empty:
            self._log.warning(f"yfinance returned no data for {ul_sym} ({yf_sym})")
            return 0

        new_df = _normalise_yf(raw, ul_sym)

        # Merge with existing
        existing = self.get(ul_sym)
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates("date").sort_values("date")
            new_rows = len(combined) - len(existing)
        else:
            combined = new_df
            new_rows = len(combined)

        combined.to_parquet(self._path(ul_sym), index=False)
        return max(new_rows, 0)

    def update_all(self, lookback_days: int = 10) -> dict[str, int]:
        """
        Call update_one() for every underlying referenced in the live
        prior_session_filters.  Returns dict of symbol -> new rows written.

        Imports live config lazily to avoid circular import.
        """
        from orb_live.config.live_config import load_live_config
        cfg = load_live_config()
        underlyings = {spec[0] for spec in cfg.prior_session_filters.values()
                       if spec is not None}

        results: dict[str, int] = {}
        for ul in sorted(underlyings):
            try:
                n = self.update_one(ul, lookback_days)
                results[ul] = n
                self._log.info(f"underlying_update {ul}: +{n} rows")
            except Exception as exc:
                self._log.error(f"underlying_update_failed {ul}: {exc}")
                results[ul] = -1
        return results

    def bulk_fetch(self, ul_syms: list[str], years: float = 5.0) -> dict[str, int]:
        """
        Fetch `years` of history for each ul_sym in one batch yfinance call.
        Used by the backfill script. Returns dict of symbol -> rows written.
        """
        import yfinance as yf

        end   = pd.Timestamp.today().normalize()
        start = end - pd.DateOffset(years=years)
        tickers = [_yf_ticker(s) for s in ul_syms]
        yfmap   = {_yf_ticker(s): s for s in ul_syms}

        raw = yf.download(
            " ".join(tickers),
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True, group_by="ticker",
        )

        results: dict[str, int] = {}

        # yfinance groups multi-ticker output differently depending on N.
        if len(tickers) == 1:
            ul = yfmap[tickers[0]]
            new_df = _normalise_yf(raw, ul)
            new_df.to_parquet(self._path(ul), index=False)
            results[ul] = len(new_df)
            return results

        for yf_sym in tickers:
            ul = yfmap[yf_sym]
            try:
                df_sym = raw[yf_sym] if yf_sym in raw.columns.get_level_values(0) else pd.DataFrame()
                if df_sym.empty:
                    results[ul] = 0
                    continue
                new_df = _normalise_yf(df_sym, ul)
                new_df.to_parquet(self._path(ul), index=False)
                results[ul] = len(new_df)
            except Exception as exc:
                self._log.error(f"bulk_fetch_failed {ul}: {exc}")
                results[ul] = -1

        return results


# ── Normalisation helper ──────────────────────────────────────────────────────

def _normalise_yf(raw: pd.DataFrame, ul_sym: str) -> pd.DataFrame:
    """Convert yfinance download output to standard [date, open, high, low, close, volume]."""
    df = raw.copy()
    # yfinance may nest columns (multi-level) or be flat.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Reset DatetimeIndex → 'date' column.
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    # Standardise column names (yfinance capitalises).
    df.columns = [c.lower() if c != "date" else c for c in df.columns]
    needed = ["date"] + [c for c in ["open", "high", "low", "close", "volume"]
                         if c in df.columns]
    df = df[needed].dropna(subset=["close"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


# ── Legacy standalone functions (kept for smoke_test.py) ─────────────────────

def fetch_prev_close(underlying: str, as_of: "Optional[date]" = None) -> "Optional[float]":
    """Return the previous trading session close for `underlying`."""
    import yfinance as yf

    yf_sym = _yf_ticker(underlying)
    end    = pd.Timestamp(as_of or date.today())
    start  = end - pd.Timedelta(days=10)
    df = yf.download(yf_sym,
                     start=start.strftime("%Y-%m-%d"),
                     end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True)
    if df.empty:
        return None
    df = df[df.index.normalize() < end] if as_of else df
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


def fetch_prev_two_closes(underlying: str) -> "tuple[Optional[float], Optional[float]]":
    import yfinance as yf

    yf_sym = _yf_ticker(underlying)
    end    = pd.Timestamp.today().normalize()
    start  = end - pd.Timedelta(days=15)
    df = yf.download(yf_sym,
                     start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True)
    if len(df) < 2:
        return None, None
    closes = df["Close"].values
    return float(closes[-1]), float(closes[-2])


def compute_prev_session_move(underlying: str) -> "Optional[float]":
    c1, c0 = fetch_prev_two_closes(underlying)
    if c1 is None or c0 is None or c0 == 0:
        return None
    return abs(c1 / c0 - 1.0)


def batch_prev_session_moves(underlyings: "list[str]") -> "dict[str, Optional[float]]":
    import yfinance as yf

    end    = pd.Timestamp.today().normalize()
    start  = end - pd.Timedelta(days=15)
    yf_map = {_yf_ticker(u): u for u in underlyings}
    tickers = list(yf_map.keys())
    df = yf.download(" ".join(tickers),
                     start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True)

    results: dict[str, "Optional[float]"] = {}
    if df.empty:
        return {u: None for u in underlyings}

    close_df = df["Close"] if "Close" in df.columns else df.xs("Close", axis=1, level=0)
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=tickers[0])

    for yf_sym, ul in yf_map.items():
        col = yf_sym if yf_sym in close_df.columns else None
        if col is None:
            results[ul] = None
            continue
        series = close_df[col].dropna()
        if len(series) < 2:
            results[ul] = None
            continue
        results[ul] = abs(float(series.iloc[-1]) / float(series.iloc[-2]) - 1.0)

    return results
