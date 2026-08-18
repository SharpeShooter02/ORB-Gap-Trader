"""
data/underlying_data.py — prior-session underlying daily data.

Daily-bar source is hybrid: for equity underlyings IB (via the injected
broker's get_daily_bars) is primary — authoritative, subscription-backed, and
doesn't lag the prior close the way the free yfinance feed can — with yfinance
as the fallback. Crypto underlyings use yfinance only (IB spot crypto needs a
separate Paxos subscription/contract handling). When no broker is injected the
store is yfinance-only (backfill script, tests).

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
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# yfinance is prone to transient failures (rate-limits, empty responses) at
# pre-market. Retry with exponential backoff so a single hiccup doesn't leave
# an underlying stale and abort the whole session. Delays: 2s, 4s, 8s.
_YF_MAX_ATTEMPTS   = 4
_YF_BACKOFF_BASE_S = 2.0

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
    "MSOS": "MSOS",  # AdvisorShares Pure US Cannabis — underlying for MSOX
}


# Underlyings whose parquets contain weekend/holiday rows (trade 7 days/week).
# Derived from YF_TICKER_MAP entries that use the *-USD yfinance convention.
CRYPTO_UNDERLYINGS: frozenset[str] = frozenset(
    k for k, v in YF_TICKER_MAP.items() if v.endswith("-USD")
)


def _yf_ticker(underlying: str) -> str:
    """Resolve underlying symbol → yfinance ticker.

    Special cases (crypto → ``*-USD``, VIX → ``^VIX``) come from YF_TICKER_MAP.
    Unknown symbols default to identity — equity ETF tickers are their own
    yfinance symbol — so a missing map entry degrades to a normal fetch (and,
    if that returns nothing, a seed-sigma fallback) instead of crashing the
    entire backfill on the first unmapped name.
    """
    return YF_TICKER_MAP.get(underlying, underlying)


# ── UnderlyingDataStore ───────────────────────────────────────────────────────

class UnderlyingDataStore:
    """
    Persistent parquet-backed store for underlying daily OHLCV bars.

    Storage layout: <data_dir>/<UNDERLYING>.parquet
    Columns: [date (datetime64[ns]), open, high, low, close, volume]
    Sorted ascending by date; deduplicated on date.

    All reads return a view — do not mutate the returned DataFrames.
    """

    def __init__(self, data_dir: Path, broker=None, logger_=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._log = logger_ or logger
        # Optional broker (IBClient) used as the PRIMARY daily-bar source for
        # equity underlyings — authoritative, subscription-backed, and doesn't
        # lag the prior session's close the way the free yfinance feed can.
        # yfinance remains the fallback (and the sole source for crypto).
        self._broker = broker

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

    # ── Freshness gate ────────────────────────────────────────────────────────

    def check_freshness(self, today: date, underlyings: set[str]) -> list[str]:
        """Return a list of human-readable stale-data messages (empty = all fresh).

        Freshness rules:
          equity ULs  — latest bar must be >= prev_trading_day(today)
          crypto ULs  — latest bar must be >= today - 1 calendar day
                        (crypto trades 7 days/week so Saturday and Sunday bars
                         must be present on a Monday)
        """
        from orb_live.core.calendar import prev_trading_day
        issues: list[str] = []
        for ul in sorted(underlyings):
            df = self.get(ul)
            if df.empty:
                issues.append(f"{ul}: no data on disk")
                continue
            latest = df["date"].iloc[-1].date()
            if ul in CRYPTO_UNDERLYINGS:
                required = today - timedelta(days=1)
            else:
                required = prev_trading_day(today)
            if latest < required:
                issues.append(
                    f"{ul}: stale — latest={latest}, required>={required}"
                )
        return issues

    def refresh_and_assert_fresh(self, today: date, underlyings: set[str]) -> None:
        """Refresh all UL data then abort loudly if any remain stale.

        Call once at session startup, before pre-market.  Raises RuntimeError
        so the session runner can catch it and emit a CRITICAL alert before
        propagating the abort.
        """
        self.update_all()
        issues = self.check_freshness(today, underlyings)
        if issues:
            msg = "STALE UNDERLYING DATA — session aborted:\n  " + "\n  ".join(issues)
            # f-string (not kwargs) so this is safe on a stdlib logger too.
            self._log.critical(f"stale_underlying_data: {issues}")
            raise RuntimeError(msg)

    # ── Write ─────────────────────────────────────────────────────────────────

    def _yf_download_retry(self, ul_sym: str, yf_sym: str,
                           start: "pd.Timestamp", end: "pd.Timestamp"):
        """Fetch daily bars from yfinance with exponential-backoff retry.

        Treats BOTH exceptions and empty results as transient (yfinance signals
        rate-limiting either way). Returns the raw DataFrame, or None if every
        attempt failed — the caller logs and returns 0 new rows.
        """
        import yfinance as yf

        for attempt in range(1, _YF_MAX_ATTEMPTS + 1):
            try:
                raw = yf.download(
                    yf_sym,
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=True,
                )
                if raw is not None and not raw.empty:
                    return raw
                reason = "empty response"
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"

            if attempt < _YF_MAX_ATTEMPTS:
                delay = _YF_BACKOFF_BASE_S * (2 ** (attempt - 1))
                self._log.warning(
                    f"yf_download_retry {ul_sym} ({yf_sym}) attempt "
                    f"{attempt}/{_YF_MAX_ATTEMPTS}: {reason} — retry in {delay:.0f}s"
                )
                time.sleep(delay)
            else:
                self._log.error(
                    f"yf_download_exhausted {ul_sym} ({yf_sym}) after "
                    f"{_YF_MAX_ATTEMPTS} attempts: {reason}"
                )
        return None

    def _fetch_ib_daily(self, ul_sym: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Recent daily bars from IB via get_daily_bars, or None on any failure.

        IB already returns the parquet schema ([date, open, high, low, close,
        volume] with a tz-naive normalized date), so no yfinance-style
        normalisation is needed. Any error (not connected, no subscription,
        empty) returns None so the caller falls back to yfinance.
        """
        if self._broker is None:
            return None
        try:
            df = self._broker.get_daily_bars(ul_sym, lookback_days=max(lookback_days, 20))
        except Exception as exc:
            self._log.warning(f"ib_daily_fetch_failed {ul_sym}: {exc}")
            return None
        if df is None or df.empty:
            return None
        cols = ["date", "open", "high", "low", "close", "volume"]
        if not set(cols).issubset(df.columns):
            return None
        out = df[cols].copy()
        out["date"] = pd.to_datetime(out["date"])
        return out

    def _fetch_daily(self, ul_sym: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Normalized daily bars for `ul_sym`, or None if every source failed.

        Equities: IB primary, yfinance fallback. Crypto: yfinance only (IB spot
        crypto needs a separate Paxos subscription/contract handling)."""
        if ul_sym not in CRYPTO_UNDERLYINGS:
            ib_df = self._fetch_ib_daily(ul_sym, lookback_days)
            if ib_df is not None and not ib_df.empty:
                return ib_df

        yf_sym = _yf_ticker(ul_sym)
        end    = pd.Timestamp.today().normalize()
        start  = end - pd.Timedelta(days=max(lookback_days * 2, 30))
        raw = self._yf_download_retry(ul_sym, yf_sym, start, end)
        if raw is None or raw.empty:
            return None
        return _normalise_yf(raw, ul_sym)

    def update_one(self, ul_sym: str, lookback_days: int = 10) -> int:
        """
        Fetch the most recent `lookback_days` of daily bars — IB primary for
        equities, yfinance fallback (and sole crypto source) — and append to the
        parquet file (deduplicating on date, keeping existing rows on overlap).

        Returns the number of NEW rows written.
        """
        new_df = self._fetch_daily(ul_sym, lookback_days)
        if new_df is None or new_df.empty:
            self._log.warning(f"no daily data for {ul_sym} (IB + yfinance both empty)")
            return 0

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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    closes = df["Close"]
    return float(closes.iloc[-1]), float(closes.iloc[-2])


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
