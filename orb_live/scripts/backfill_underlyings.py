#!/usr/bin/env python3
"""
scripts/backfill_underlyings.py — Fetch 5-year underlying daily bars into
parquet storage for PS filter warm-up.

Uses UnderlyingDataStore.bulk_fetch() which downloads all underlyings in a
single yfinance batch call.  Run once before the first live session.

Usage:
    python -m orb_live.scripts.backfill_underlyings [--data-dir PATH] [--years N]
"""

import argparse
from pathlib import Path

import orb_live  # noqa: F401 — sys.path setup
from orb_live.config.live_config import load_live_config
from orb_live.data.underlying_data import UnderlyingDataStore


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical underlying daily bars for PS filter."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory to store parquet files (default: live_cfg.data_dir)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=5.0,
        help="Years of history to fetch (default: 5.0)",
    )
    args = parser.parse_args()

    cfg = load_live_config()
    data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir

    store = UnderlyingDataStore(data_dir)

    # Collect all underlyings referenced by PS filters.
    underlyings = sorted({
        spec[0]
        for spec in cfg.prior_session_filters.values()
        if spec is not None
    })

    if not underlyings:
        print("No underlyings in PS filters — nothing to backfill.")
        return

    print(f"Backfilling {len(underlyings)} underlyings ({args.years}y) → {data_dir}")
    results = store.bulk_fetch(underlyings, years=args.years)

    for ul, n in sorted(results.items()):
        status = f"+{n} rows" if n >= 0 else "FAILED"
        print(f"  {ul:<12} {status}")

    total = sum(n for n in results.values() if n >= 0)
    failed = sum(1 for n in results.values() if n < 0)
    print(f"\n  Done: {total} rows written, {failed} failures.")


if __name__ == "__main__":
    main()
