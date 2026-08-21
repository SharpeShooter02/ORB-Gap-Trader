"""Pull IB TWS 1-minute bars for the (symbol, session) universe of the live
trade window and store them in the BacktestingGaps cache format.

Runs via ib_async (paper account, IBClient wrapper). One request per
(symbol, session_date) chunk, aggregated per (symbol, YYYY-MM) parquet.
Skips existing monthly parquets so re-runs are idempotent.

Usage:
    python scripts/pull_bars_for_backtest_comparison.py \
        --db  path/to/live.db \
        --cache path/to/BacktestingGaps/cache/intraday \
        [--force]  # overwrite existing monthly files
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import orb_live  # noqa: F401 — sys.path setup
from orb_live.data.ib_client import build_client_from_env

_ET = ZoneInfo("America/New_York")
_RTH_OPEN  = dtime(9, 30)
_RTH_CLOSE = dtime(16, 0)


def load_pull_targets(db_path: Path) -> dict[tuple[str, str], list[pd.Timestamp]]:
    """Return {(symbol, YYYY-MM): [session_dates]} from the live candidates table."""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql(
        "SELECT DISTINCT session_date, symbol FROM candidates",
        conn,
    )
    conn.close()
    df["session_date"] = pd.to_datetime(df["session_date"])
    df["month"] = df["session_date"].dt.strftime("%Y-%m")

    targets: dict[tuple[str, str], list[pd.Timestamp]] = defaultdict(list)
    for _, r in df.iterrows():
        targets[(r["symbol"], r["month"])].append(r["session_date"])
    return {k: sorted(v) for k, v in targets.items()}


def write_month_parquet(cache_root: Path, symbol: str, month: str,
                        frames: list[pd.DataFrame]) -> int:
    """Concatenate session frames and write to cache/intraday/<SYM>/<YYYY-MM>.parquet.

    Returns number of bars written.
    """
    if not frames:
        return 0
    df = pd.concat(frames, axis=0)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    out_df = pd.DataFrame({
        "timestamp": df.index,
        "open":  df["open"].astype(float),
        "high":  df["high"].astype(float),
        "low":   df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype("int64"),
    }).reset_index(drop=True)

    out_dir = cache_root / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{month}.parquet"
    out_df.to_parquet(out_path)
    return len(out_df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        default=r"c:/Users/buttn/Documents/Projects/orb-live-trading/orb_live/state/live.db",
    )
    ap.add_argument(
        "--cache",
        default=r"c:/Users/buttn/Documents/Projects/BacktestingGaps/cache/intraday",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing monthly parquets")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between IB requests")
    args = ap.parse_args()

    db_path = Path(args.db)
    cache_root = Path(args.cache)
    cache_root.mkdir(parents=True, exist_ok=True)

    targets = load_pull_targets(db_path)
    print(f"Pull targets: {len(targets)} (symbol, month) chunks", flush=True)

    # Skip existing
    to_pull: dict[tuple[str, str], list[pd.Timestamp]] = {}
    skipped = 0
    for key, dates in targets.items():
        symbol, month = key
        parquet_path = cache_root / symbol / f"{month}.parquet"
        if parquet_path.exists() and not args.force:
            skipped += 1
            continue
        to_pull[key] = dates
    print(f"Skipping {skipped} existing; will pull {len(to_pull)}", flush=True)

    if not to_pull:
        print("Nothing to pull.")
        return

    total_sessions = sum(len(v) for v in to_pull.values())
    print(f"Total session-level IB requests: {total_sessions}", flush=True)
    print(f"Estimated elapsed at {args.sleep}s pacing: "
          f"{total_sessions * args.sleep / 60:.1f} min", flush=True)

    client = build_client_from_env(paper=True)
    print(f"Connecting to IB Gateway on {client._host}:{client._port} …", flush=True)
    client.connect()
    print("Connected.\n", flush=True)

    ok = fail = empty = 0
    written_files = written_bars = 0
    try:
        for i, ((symbol, month), dates) in enumerate(sorted(to_pull.items()), 1):
            frames = []
            for d in dates:
                start = datetime.combine(d.date(), _RTH_OPEN, tzinfo=_ET)
                end   = datetime.combine(d.date(), _RTH_CLOSE, tzinfo=_ET)
                try:
                    df = client.get_intraday_bars(symbol, start, end, "1Min")
                except Exception as exc:
                    print(f"  [{i}/{len(to_pull)}] {symbol} {d.date()} FAIL: {exc}",
                          flush=True)
                    fail += 1
                    continue
                if df.empty:
                    empty += 1
                else:
                    frames.append(df)
                    ok += 1
                time.sleep(args.sleep)

            if frames:
                n = write_month_parquet(cache_root, symbol, month, frames)
                written_files += 1
                written_bars += n
                print(f"  [{i:>3}/{len(to_pull)}] {symbol} {month}  "
                      f"{len(dates)} sessions -> {n} bars written",
                      flush=True)
            else:
                print(f"  [{i:>3}/{len(to_pull)}] {symbol} {month}  "
                      f"{len(dates)} sessions -> ALL EMPTY, no file written",
                      flush=True)
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    print()
    print(f"Done. Session fetches: ok={ok} empty={empty} fail={fail}")
    print(f"Wrote {written_files} parquet files ({written_bars} bars total)")


if __name__ == "__main__":
    main()
