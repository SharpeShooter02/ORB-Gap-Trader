#!/usr/bin/env python3
"""
scripts/recalibrate_sigmas.py — Recompute sigma estimates and optionally
write sigma_override.yaml for the live prior-session filter.

Usage:
    # Dry-run report only (no file written):
    python -m orb_live.scripts.recalibrate_sigmas --years 3.0

    # Write sigma_override.yaml after reviewing report:
    python -m orb_live.scripts.recalibrate_sigmas --years 3.0 --confirm

    # Also persist to state DB for audit trail:
    python -m orb_live.scripts.recalibrate_sigmas --years 3.0 --confirm --db PATH

After writing, restart the live session or reload live_config to pick up
the updated thresholds.  Delete sigma_override.yaml to revert to reference.
"""

import argparse
from pathlib import Path

import orb_live  # noqa: F401 — sys.path setup
from orb_live.ops.calibration import recalibrate_sigmas


def main():
    parser = argparse.ArgumentParser(
        description="Recalibrate sigma thresholds for the prior-session filter."
    )
    parser.add_argument(
        "--years", type=float, default=3.0,
        help="Lookback period in years (default: 3.0)",
    )
    parser.add_argument(
        "--k", type=float, default=1.25,
        help="Sigma multiplier for threshold (default: 1.25, matches production)",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Write sigma_override.yaml (default: dry-run only)",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to live.db for sigma_history audit log (optional)",
    )
    args = parser.parse_args()

    state_store = None
    if args.db:
        from orb_live.core.state_store import StateStore
        state_store = StateStore(Path(args.db))

    recalibrate_sigmas(
        lookback_years=args.years,
        k=args.k,
        confirm=args.confirm,
        state_store=state_store,
    )


if __name__ == "__main__":
    main()
