"""Refresh the daily underlying cache used by the backtester.

Pulls daily OHLCV for the underlyings referenced by the live trade window
and merges into orb_event_study/cache/daily/<UL>.parquet.

Equity underlyings come from IB TWS (must be running). Crypto (BTC, ETH, SOL, XRP)
comes from yfinance since IB's daily historical for crypto cash tickers is unreliable.

Idempotent — reads existing parquet, appends new rows only, dedups on date.

Usage:
    python scripts/refresh_daily_ul_cache.py [--lookback 200]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

import orb_live  # noqa: F401 — sys.path setup
from orb_live.data.ib_client import build_client_from_env

DAILY_CACHE = Path("c:/Users/buttn/Documents/Projects/BacktestingGaps/orb_event_study/cache/daily")

MASTER = Path("c:/Users/buttn/Documents/Projects/BacktestingGaps/master_universe.csv")

CRYPTO_YF = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
    "ADA": "ADA-USD", "LINK": "LINK-USD", "XLM": "XLM-USD",
}
CRYPTO_ULS = sorted(CRYPTO_YF)

#: Several `underlying` cells hold an index NAME rather than a ticker
#: ("Dow Jones U.S. Utilities Index", "Uranium"). They have no quotable
#: symbol, which is why they show up in sigma_unresolved.csv -- see
#: OPEN_ISSUES D6. Ticker shape is the cheapest reliable filter.
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")


def equity_underlyings() -> list[str]:
    """Every non-crypto underlying the universe actually references.

    This was a hardcoded list of 14 while master_universe.csv referenced 71,
    so 41 underlyings silently stopped updating in Apr/May 2026 and the
    backtest's candidate set quietly shrank. Deriving it means adding an ETF
    to the universe is enough to get its underlying refreshed.
    """
    uls = pd.read_csv(MASTER)["underlying"].dropna().astype(str).str.strip()
    return sorted({u for u in uls if u not in CRYPTO_YF and _TICKER.match(u)})


def load_existing(ul: str) -> pd.DataFrame:
    p = DAILY_CACHE / f"{ul}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def merge_and_write(ul: str, existing: pd.DataFrame, new: pd.DataFrame) -> int:
    """Return count of NEW rows appended."""
    if new.empty:
        return 0
    new = new.copy()
    new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None).dt.normalize()

    # Preserve adj_close if existing has it; for new rows use close as fallback
    if "adj_close" not in new.columns:
        new["adj_close"] = new["close"]

    # Union, dedup keeping existing (existing may be split-adjusted;
    # we don't want to overwrite historical rows)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="first")
    combined = combined.sort_values("date").reset_index(drop=True)

    n_before = len(existing)
    n_after  = len(combined)
    added    = n_after - n_before

    if added > 0:
        # Keep only the columns of the existing schema (or standard 7 if new file)
        cols = ["date","open","high","low","close","adj_close","volume"]
        for c in cols:
            if c not in combined.columns:
                combined[c] = None
        combined = combined[cols]
        DAILY_CACHE.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(DAILY_CACHE / f"{ul}.parquet")
    return added


def refresh_equity(client, ul: str, lookback: int, sleep: float) -> tuple[int, int]:
    """Returns (n_pulled, n_added)."""
    try:
        df = client.get_daily_bars(ul, lookback_days=lookback)
    except Exception as exc:
        print(f"  {ul:<6}  IB FAIL: {exc}")
        return 0, 0

    if df.empty:
        print(f"  {ul:<6}  IB returned 0 rows")
        return 0, 0

    existing = load_existing(ul)
    added = merge_and_write(ul, existing, df)
    last  = pd.to_datetime(df["date"]).max().date()
    print(f"  {ul:<6}  pulled {len(df)} rows through {last}  |  added {added} new")
    time.sleep(sleep)
    return len(df), added


def refresh_crypto(ul: str) -> tuple[int, int]:
    """Returns (n_pulled, n_added). Uses yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print(f"  {ul:<6}  SKIP: yfinance not installed (pip install yfinance)")
        return 0, 0

    ticker = CRYPTO_YF[ul]
    try:
        # 2 years of history, daily
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    except Exception as exc:
        print(f"  {ul:<6}  yfinance FAIL: {exc}")
        return 0, 0

    if df.empty:
        print(f"  {ul:<6}  yfinance returned 0 rows")
        return 0, 0

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })

    existing = load_existing(ul)
    added = merge_and_write(ul, existing, df)
    last  = pd.to_datetime(df["date"]).max().date()
    print(f"  {ul:<6}  pulled {len(df)} rows through {last}  |  added {added} new")
    return len(df), added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=200, help="IB daily lookback days")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between IB calls")
    args = ap.parse_args()

    print(f"Refreshing daily cache at {DAILY_CACHE}")
    equity_uls = equity_underlyings()
    print(f"Equity underlyings ({len(equity_uls)}): {', '.join(equity_uls)}")
    print(f"Crypto underlyings ({len(CRYPTO_ULS)}): {', '.join(CRYPTO_ULS)}")
    print()

    # ── Equities via IB ──────────────────────────────────────────────────────
    print("== IB (equities) ==")
    client = build_client_from_env(paper=True)
    print(f"Connecting to IB on {client._host}:{client._port} ...")
    client.connect()

    eq_pulled = eq_added = 0
    try:
        for ul in equity_uls:
            n, a = refresh_equity(client, ul, args.lookback, args.sleep)
            eq_pulled += n; eq_added += a
    finally:
        try: client.disconnect()
        except Exception: pass

    # ── Crypto via yfinance ──────────────────────────────────────────────────
    print()
    print("== yfinance (crypto) ==")
    cr_pulled = cr_added = 0
    for ul in CRYPTO_ULS:
        n, a = refresh_crypto(ul)
        cr_pulled += n; cr_added += a

    print()
    print(f"Equities: pulled {eq_pulled} bars, added {eq_added} new rows")
    print(f"Crypto  : pulled {cr_pulled} bars, added {cr_added} new rows")


if __name__ == "__main__":
    main()
