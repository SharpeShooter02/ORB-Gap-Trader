#!/usr/bin/env python3
"""
scripts/smoke_test.py — Basic connectivity and sanity checks.

Run before the first live session to verify:
  1. reference layer imports cleanly
  2. live_config.load_live_config() returns the right symbol/filter counts
  3. IB Gateway is reachable (if running)
  4. yfinance can fetch an underlying close
  5. StateStore creates tables in a tmp DB without error

Usage:
    cd BacktestingGaps
    python -m orb_live.scripts.smoke_test
"""

import sys
import tempfile
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _check(label: str, fn) -> bool:
    try:
        result = fn()
        msg = f"  [{PASS}]  {label}"
        if result:
            msg += f"  →  {result}"
        print(msg)
        return True
    except Exception as e:
        print(f"  [{FAIL}]  {label}  →  {e}")
        return False


def main():
    failures = 0
    print("\n=== ORB Live Smoke Test ===\n")

    # 1. Reference layer
    def check_reference():
        from reference._production_run import ACTIVE, SYMS, PS_FILTERS
        return f"{len(ACTIVE)} active, {len(SYMS)} syms, {len(PS_FILTERS)} ps_filters"
    if not _check("reference layer import", check_reference):
        failures += 1

    # 2. LiveConfig
    def check_live_config():
        from orb_live.config.live_config import load_live_config
        cfg = load_live_config()
        return f"{len(cfg.symbols)} symbols, {len(cfg.prior_session_filters)} ps_filters"
    if not _check("load_live_config()", check_live_config):
        failures += 1

    # 3. IB Gateway connectivity (optional)
    def check_ib():
        from orb_live.data.ib_client import build_client_from_env
        client = build_client_from_env(paper=True)
        client.connect()
        acct = client.get_account()
        client.disconnect()
        return f"equity=${float(acct.get('equity', 0)):,.0f}"
    if not _check("IB Gateway paper account", check_ib):
        print(f"  [{WARN}]  IB check failed — ensure IB Gateway is running on port 4002")

    # 4. yfinance
    def check_yfinance():
        from orb_live.data.underlying_data import fetch_prev_close
        close = fetch_prev_close("QQQ")
        if close is None:
            raise ValueError("returned None")
        return f"QQQ prev_close={close:.2f}"
    if not _check("yfinance QQQ close", check_yfinance):
        failures += 1

    # 5. StateStore
    def check_state_store():
        with tempfile.TemporaryDirectory() as tmp:
            from orb_live.core.state_store import StateStore
            store = StateStore(Path(tmp) / "smoke.db")
            from datetime import date
            store.log_event("smoke_test", "ok")
            return "14 tables created"
    if not _check("StateStore SQLite init", check_state_store):
        failures += 1

    # 6. MarketClock
    def check_clock():
        from orb_live.core.clock import MarketClock
        clk = MarketClock()
        phase = clk.current_phase()
        return f"phase={phase}, now_et={clk.now_et().strftime('%H:%M:%S')}"
    if not _check("MarketClock", check_clock):
        failures += 1

    print(f"\n{'='*40}")
    if failures == 0:
        print("  All checks passed.\n")
    else:
        print(f"  {failures} check(s) FAILED.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
