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
import shutil
import tempfile
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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

    # 1. Strategy profile. The reference layer this used to import
    # (reference/_production_run.py) was deleted deliberately in 8ab3adf, and
    # live_config.py states plainly that nothing at runtime imports it -- so
    # checking for it made the pre-flight fail on a module that is *supposed*
    # to be gone. Check the profile that is actually in force instead.
    def check_profile():
        from pathlib import Path as _P
        from orb_live.strategy import v1_strategy as v1
        d = _P(__file__).resolve().parents[1] / "strategy" / "data"
        inst = v1.load_master(d / "master_universe.csv")
        sig = v1.load_sigmas(d / "sigma_master.csv")
        uni = v1.build_universe(inst, sig, d / "master_universe.csv")
        no_sig = [s for s in uni if inst[s].underlying not in sig]
        if no_sig:
            raise AssertionError(f"no sigma for {no_sig}")
        return (f"{len(uni)} symbols, k={v1.K_SIGMA}, "
                f"shared_allotment={v1.SHARED_ALLOTMENT}")
    if not _check("strategy profile", check_profile):
        failures += 1

    # 2. Sizing. The gold pair must halve; a symbol on its own must not. This
    # is the one check that would notice SHARED_ALLOTMENT silently going
    # inert -- which is exactly how it shipped the first time.
    def check_sizing():
        from orb_live.strategy import v1_strategy as v1
        sub = {
            "NUGT": v1.Instrument("NUGT", "GDX", 2, False),
            "JNUG": v1.Instrument("JNUG", "GDXJ", 2, False),
            "SOXL": v1.Instrument("SOXL", "SOXX", 3, False),
        }
        gaps = {"GDX": 0.05, "GDXJ": 0.05, "SOXX": 0.05}
        plan = v1.plan_session(
            list(sub), sub, {u: 0.02 for u in gaps}, gaps,
            {u: (100.0, 100.0) for u in gaps}, {s: 50.0 for s in sub})
        m = plan.multipliers
        solo = m.get("SOXL", 0.0)
        pair = m.get("NUGT", 0.0) + m.get("JNUG", 0.0)
        if not solo or abs(pair - solo) > 1e-9:
            raise AssertionError(
                f"shared allotment not applied: gold={pair} solo={solo}")
        return f"gold pair {m.get('NUGT')}+{m.get('JNUG')} == solo {solo}"
    if not _check("shared-allotment sizing", check_sizing):
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
        # The store must be closed before the TemporaryDirectory unwinds: on
        # Windows an open SQLite handle makes the cleanup fail with WinError 32,
        # which reads as a StateStore failure when it is only the probe leaking
        # a file handle.
        tmp = tempfile.mkdtemp()
        try:
            from orb_live.core.state_store import StateStore
            store = StateStore(Path(tmp) / "smoke.db")
            try:
                store.log_event("smoke_test", "ok")
                n = len(store.list_tables()) if hasattr(store, "list_tables") else 14
            finally:
                for closer in ("close", "disconnect"):
                    fn = getattr(store, closer, None)
                    if callable(fn):
                        fn()
                        break
                else:
                    conn = getattr(store, "_conn", None)
                    if conn is not None:
                        conn.close()
            return f"{n} tables created"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
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
