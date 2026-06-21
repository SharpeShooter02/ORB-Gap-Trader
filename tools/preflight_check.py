"""
tools/preflight_check.py — Pre-trading-day gate check.

Run each morning before starting the session runner (e.g., at 07:30 ET):

    python tools/preflight_check.py

Checks (in order):
  1. IB Gateway reachable and connected
  2. Account equity > 0
  3. Market data subscription active (SPY quote)
  4. All underlying parquets present and fresh (< 3 days old)
  5. Key BrokerClient read-methods callable (no NotImplementedError)

Exits 0 if all pass, exits 1 if any fail.  Runs in ~30 seconds.
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

_PASS = "  [PASS]"
_FAIL = "  [FAIL]"
_SKIP = "  [SKIP]"


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_gateway(client) -> str:
    assert client.is_connected(), "is_connected() returned False after connect()"
    return "connected"


def _check_equity(client) -> str:
    acct   = client.get_account()
    equity = float(acct.get("equity", 0.0))
    assert equity > 0.0, f"equity={equity} — is the paper account funded?"
    return f"equity=${equity:,.2f}"


def _check_subscription(client) -> str:
    """
    Explicitly test that SPY returns a non-zero mid-price.

    The client is connected with _allow_delayed_data=True so connect()
    itself does not gate on subscription.  This check performs the
    subscription test manually so we can report a clear PASS/FAIL.
    """
    # Temporarily tighten to strict mode so get_latest_quote raises on zeros.
    orig = client._allow_delayed_data
    client._allow_delayed_data = False
    try:
        quote = client.get_latest_quote("SPY")
        bid   = float(quote.get("bid", 0.0))
        ask   = float(quote.get("ask", 0.0))
        mid   = (bid + ask) / 2.0 if (bid > 0 or ask > 0) else 0.0
        assert mid > 0.0, f"SPY mid=${mid:.2f} — quote is zero (check subscription)"
        return f"SPY mid=${mid:.2f}"
    finally:
        client._allow_delayed_data = orig


def _check_underlyings(cfg) -> str:
    """
    Verify that every underlying referenced by prior_session_filters has a
    parquet file in cfg.data_dir and that its latest bar is ≤ 3 days old.
    """
    import pandas as pd

    underlyings = sorted({
        spec[0]
        for spec in cfg.prior_session_filters.values()
        if spec is not None
    })
    if not underlyings:
        return "no underlyings required"

    today   = date.today()
    missing: list[str] = []
    stale:   list[str] = []

    for ul in underlyings:
        path = cfg.data_dir / f"{ul}.parquet"
        if not path.exists():
            missing.append(ul)
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty:
                stale.append(f"{ul}(empty)")
                continue
            date_col = "date" if "date" in df.columns else df.columns[0]
            latest   = pd.to_datetime(df[date_col]).max().date()
            age      = (today - latest).days
            if age > 3:
                stale.append(f"{ul}(latest={latest}, age={age}d)")
        except Exception as exc:
            stale.append(f"{ul}(read_error: {exc})")

    if missing:
        raise AssertionError(
            f"Missing underlying parquets: {missing}\n"
            f"  Run: python -m orb_live.data.underlying_data"
        )
    if stale:
        raise AssertionError(
            f"Stale underlying parquets (>3 days old): {stale}\n"
            f"  Run: python -m orb_live.data.underlying_data"
        )
    return f"{len(underlyings)} underlyings present and fresh"


def _check_contract_methods(client) -> str:
    """Verify that non-mutating BrokerClient methods are callable."""
    import pandas as pd

    client.is_connected()
    client.is_market_open()
    client.get_clock()

    acct = client.get_account()
    assert isinstance(acct, dict), "get_account must return dict"

    eq = client.get_equity()
    assert isinstance(eq, float), "get_equity must return float"

    positions = client.get_positions()
    assert isinstance(positions, list), "get_positions must return list"

    asset = client.get_asset("SPY")
    assert "tradable" in asset, "get_asset must include 'tradable' key"

    daily = client.get_daily_bars("SPY", lookback_days=3)
    assert isinstance(daily, pd.DataFrame), "get_daily_bars must return DataFrame"

    return "all read-methods callable"


# ── Runner ────────────────────────────────────────────────────────────────────

def _run(label: str, fn: Callable) -> bool:
    """Execute fn(), print PASS/FAIL with timing, return success bool."""
    import time
    t0 = time.monotonic()
    try:
        msg = fn()
        elapsed = time.monotonic() - t0
        suffix  = f": {msg}" if msg else ""
        print(f"{_PASS} {label}{suffix}  ({elapsed:.1f}s)")
        return True
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"{_FAIL} {label}: {exc}  ({elapsed:.1f}s)")
        return False


def run_preflight() -> int:
    """Run all pre-flight checks.  Returns 0 if all pass, 1 otherwise."""
    from dotenv import load_dotenv
    load_dotenv()

    import orb_live  # noqa: F401 — triggers path setup

    from orb_live.config.live_config import load_live_config
    from orb_live.data.ib_client import build_client_from_env

    now = datetime.now(tz=ET)
    print(f"\nPre-flight check — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 56)

    # Load config (needed for underlyings check and data_dir).
    print("Loading config ...")
    try:
        cfg = load_live_config()
    except Exception as exc:
        print(f"{_FAIL} load_live_config: {exc}")
        return 1

    # Build client in delayed mode so connect() doesn't auto-gate on subscription.
    client = build_client_from_env(paper=True)
    client._allow_delayed_data = True

    results: list[bool] = []

    # ── 1. Gateway ────────────────────────────────────────────────────────────
    try:
        client.connect()
    except Exception as exc:
        print(f"{_FAIL} IB Gateway reachable: {exc}")
        print("\nPreflight FAILED — cannot connect to IB Gateway.")
        print("Start IB Gateway / TWS and try again.")
        return 1
    results.append(_run("IB Gateway reachable", lambda: _check_gateway(client)))

    # ── 2. Account ────────────────────────────────────────────────────────────
    results.append(_run("Account equity > 0", lambda: _check_equity(client)))

    # ── 3. Subscription ───────────────────────────────────────────────────────
    results.append(_run("Market data subscription (SPY)",
                        lambda: _check_subscription(client)))

    # ── 4. Underlyings ────────────────────────────────────────────────────────
    results.append(_run("Underlying parquets", lambda: _check_underlyings(cfg)))

    # ── 5. Contract methods ───────────────────────────────────────────────────
    results.append(_run("BrokerClient contract methods",
                        lambda: _check_contract_methods(client)))

    client.disconnect()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 56)
    n_pass = sum(results)
    n_fail = len(results) - n_pass

    if n_fail == 0:
        print(f"Preflight PASS — {n_pass}/{len(results)} checks passed.")
        print("Safe to start the session runner.")
        return 0
    else:
        print(f"Preflight FAIL — {n_fail}/{len(results)} check(s) failed.")
        print("Fix the failures above before starting the session runner.")
        return 1


if __name__ == "__main__":
    sys.exit(run_preflight())
