#!/usr/bin/env python3
"""
scripts/refresh_universe.py — Sync ACTIVE universe into the state DB.

Run after any change to _production_run.ACTIVE to ensure the universe
table reflects the current instrument list.

Usage:
    python -m orb_live.scripts.refresh_universe [--db path/to/live.db]
"""

import argparse
from datetime import datetime
from pathlib import Path

import orb_live  # noqa: F401
from reference._production_run import UNIVERSE, _is_class_a
from orb_live.core.state_store import StateStore, universe as universe_table


def refresh(db_path: Path) -> None:
    store = StateStore(db_path)
    now = datetime.utcnow()

    with store.conn() as conn:
        # Truncate and repopulate.
        conn.execute(universe_table.delete())
        rows = [
            {
                "symbol":     sym,
                "underlying": info["underlying"],
                "inverse":    info["inverse"],
                "leverage":   float(info["leverage"]),
                "is_class_a": _is_class_a(sym, info["underlying"]),
                "updated_at": now,
            }
            for sym, info in UNIVERSE.items()
        ]
        conn.execute(universe_table.insert(), rows)
        conn.commit()

    print(f"  Refreshed {len(rows)} symbols in universe table → {db_path}")
    for row in rows:
        flag = " [A]" if row["is_class_a"] else ""
        print(f"    {row['symbol']:<8}  {row['underlying']:<8}  {row['leverage']}x  "
              f"{'inv' if row['inverse'] else 'fwd'}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="orb_live/state/live.db",
                        help="Path to live state SQLite DB")
    args = parser.parse_args()
    refresh(Path(args.db))


if __name__ == "__main__":
    main()
