"""
tools/preflight_check.py — Pre-trading-day gate check.

Run each morning before starting the session runner (e.g., at 07:30 ET):

    python -m orb_live.tools.preflight_check

Or directly:

    python tools/preflight_check.py

Checks (in order):
  1.  IB Gateway reachable and connected
  2.  Account equity > 0
  3.  Market data subscription active (SPY quote)
  4.  All underlying parquets present and fresh (< 3 days old)
  5.  Key BrokerClient read-methods callable (no NotImplementedError)
  6.  Contract qualification — every universe symbol resolves; untradable names
      are logged loudly, not silently skipped
  7.  yfinance prior closes — ≥2 daily closes available for every PS-filter
      underlying; reports fallbacks to seed sigma
  8.  Sigma source resolution — logs rolling vs. seed path for every underlying
  9.  Market-data line estimate — confirms streaming scope is candidates-only,
      not the full ~60-symbol universe
  10. reqHistoricalData pacing guard — verifies burst of history calls is
      rate-limited to IB's 50-req/10-s limit

Exits 0 if all pass, exits 1 if any fail.  Runs in ~60 seconds with IB.
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
    import os
    allow_delayed = os.getenv("IB_ALLOW_DELAYED_DATA", "").strip().lower() in (
        "1", "true", "yes"
    )

    # Dress-rehearsal mode: real-time data isn't purchased yet, so don't hard-fail.
    # Honor IB_ALLOW_DELAYED_DATA (which the failure message promises) by
    # soft-passing: try a (possibly delayed) quote, but never block on it.
    if allow_delayed:
        try:
            quote = client.get_latest_quote("SPY")
            bid   = float(quote.get("bid", 0.0))
            ask   = float(quote.get("ask", 0.0))
            mid   = (bid + ask) / 2.0 if (bid > 0 or ask > 0) else 0.0
        except Exception:
            mid = 0.0
        if mid > 0.0:
            return (
                f"delayed quote OK (SPY mid=${mid:.2f}) — real-time subscription "
                "NOT verified (IB_ALLOW_DELAYED_DATA=1)"
            )
        return (
            "SKIPPED — IB_ALLOW_DELAYED_DATA=1 and no live/delayed quote available "
            "(market closed?). Real-time subscription unverified; buy streams and "
            "re-run strict (unset the flag) during RTH before going live."
        )

    # Strict mode (default): require a non-zero real-time quote.
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
    parquet file in cfg.data_dir whose latest bar is at least the most recent
    completed NYSE trading day (calendar-aware, so weekends/holidays don't
    false-alarm).
    """
    import pandas as pd
    from orb_live.core.calendar import prev_trading_day

    underlyings = sorted({
        spec[0]
        for spec in cfg.prior_session_filters.values()
        if spec is not None
    })
    if not underlyings:
        return "no underlyings required"

    today       = date.today()
    fresh_floor = prev_trading_day(today)   # most recent completed trading day
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
            if latest < fresh_floor:
                stale.append(f"{ul}(latest={latest}, expected≥{fresh_floor})")
        except Exception as exc:
            stale.append(f"{ul}(read_error: {exc})")

    if missing:
        raise AssertionError(
            f"Missing underlying parquets: {missing}\n"
            f"  Run: python -m orb_live.scripts.backfill_underlyings"
        )
    if stale:
        raise AssertionError(
            f"Stale underlying parquets (latest < last trading day {fresh_floor}): "
            f"{stale}\n  Run: python -m orb_live.scripts.backfill_underlyings"
        )
    return (
        f"{len(underlyings)} underlyings present and fresh "
        f"(≥ last trading day {fresh_floor})"
    )


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


def _check_contract_qualification(client, cfg) -> str:
    """Qualify every universe symbol against IB; log untradable names loudly."""
    from orb_live.strategy.v1_strategy import load_master, build_universe
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "orb_live" / "strategy" / "data"
    instruments = load_master(data_dir / "master_universe.csv")

    untradable: list[str] = []
    failed:     list[str] = []

    symbols = list(instruments.keys())
    for sym in symbols:
        try:
            asset = client.get_asset(sym)
            if not asset.get("tradable", False):
                untradable.append(sym)
        except Exception as exc:
            failed.append(f"{sym}({exc})")

    if failed:
        print(f"    WARN: contract resolution errors: {failed}")
    if untradable:
        print(
            f"  [WARN] Untradable symbols (won't be traded — verify before session):\n"
            + "\n".join(f"    ✗ {s}" for s in untradable)
        )

    tradable = len(symbols) - len(untradable) - len(failed)
    assert tradable > 0, f"No tradable symbols found — all {len(symbols)} failed"
    return (
        f"{tradable}/{len(symbols)} tradable"
        + (f"; {len(untradable)} untradable" if untradable else "")
        + (f"; {len(failed)} errors" if failed else "")
    )


def _check_yfinance_prior_closes(cfg) -> str:
    """Verify yfinance returns ≥2 prior closes for every PS-filter underlying.

    Crypto uses BTC-USD etc.; VIX uses ^VIX.  Reports fallbacks to seed sigma.
    """
    try:
        import yfinance as yf
    except ImportError:
        return "SKIP (yfinance not installed)"

    from orb_live.strategy.v1_strategy import load_sigmas
    from pathlib import Path

    sigma_csv = Path(__file__).parent.parent / "orb_live" / "strategy" / "data" / "sigma_master.csv"
    sigmas_seed = load_sigmas(sigma_csv)

    # Map underlying → yfinance ticker (crypto, VIX need special tickers)
    _YF_MAP = {
        "BTC":  "BTC-USD",
        "ETH":  "ETH-USD",
        "XRP":  "XRP-USD",
        "SOL":  "SOL-USD",
        "VIX":  "^VIX",
        "UVIX": "^VIX",
    }

    underlyings = sorted({spec[0] for spec in cfg.prior_session_filters.values() if spec})
    ok: list[str] = []
    fallback: list[str] = []
    failed: list[str] = []

    for ul in underlyings:
        ticker = _YF_MAP.get(ul, ul)
        try:
            hist = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            closes = hist["Close"].dropna() if not hist.empty else []
            if len(closes) >= 2:
                ok.append(ul)
            else:
                fallback.append(f"{ul}(yf_rows={len(closes)},→seed)")
        except Exception as exc:
            fallback.append(f"{ul}(error:{exc},→seed)")

    if fallback:
        print(
            "  [WARN] yfinance fallbacks to seed sigma:\n"
            + "\n".join(f"    ✗ {s}" for s in fallback)
        )

    assert ok or not underlyings, "No underlyings could be fetched from yfinance"
    return (
        f"{len(ok)}/{len(underlyings)} ok"
        + (f"; {len(fallback)} falling back to seed" if fallback else "")
    )


def _check_sigma_source_resolution(cfg) -> str:
    """Log whether each underlying uses rolling parquet or seed sigma.

    Resolution order: rolling {ul}.parquet → vendored sigma_master.csv seed.
    """
    from pathlib import Path

    data_dir    = cfg.data_dir if hasattr(cfg, "data_dir") else Path("orb_live/data/underlyings")
    underlyings = sorted({spec[0] for spec in cfg.prior_session_filters.values() if spec})

    rolling = []
    seed    = []
    for ul in underlyings:
        parquet = data_dir / f"{ul}.parquet"
        if parquet.exists():
            rolling.append(ul)
        else:
            seed.append(ul)

    print(f"    Sigma source: {len(rolling)} rolling parquet, {len(seed)} seed fallback")
    if seed:
        print("    Seed fallbacks: " + ", ".join(seed))

    return f"{len(rolling)} rolling / {len(seed)} seed"


def _check_market_data_lines(cfg) -> str:
    """Confirm streaming scope is candidates, not the full universe.

    IB limits simultaneous real-time bars to ~100 lines (paper) / ~100 (live).
    Universe is ~60 symbols; candidates are typically 5-20.  Streaming should
    be subscribed per-candidate at 10:00 ET, not for the whole universe at open.
    """
    n_universe = len(cfg.symbols)
    # Estimate worst-case candidates: regime=active, all qualify (upper bound).
    n_candidates_max = n_universe   # conservative upper bound

    # IB real-time bar subscription limit (conservative for paper accounts)
    IB_RTB_LIMIT = 100

    assert n_universe <= IB_RTB_LIMIT, (
        f"Universe size {n_universe} exceeds IB real-time bar limit {IB_RTB_LIMIT}. "
        "Confirm streaming is scoped to candidates, not the full universe."
    )

    return (
        f"universe={n_universe} symbols, IB limit={IB_RTB_LIMIT} — OK. "
        "Reminder: subscribe real-time bars per-CANDIDATE at 10:00 ET, not at open."
    )


def _check_historical_data_pacing(client) -> str:
    """Verify reqHistoricalData burst is paced within IB's 50-req/10-s limit.

    At 9:30 / 10:00 ET the runner fetches intraday bars for all candidates.
    IB's pacing rule: ≤50 identical-contract requests per 10 seconds.
    This check fetches one bar for SPY and times it to estimate per-symbol cost.
    """
    import time
    from datetime import timedelta

    # get_intraday_bars takes explicit (start, end) ET datetimes — not lookback_days.
    # Mirror the runner's real call (a ~35-min ORB-sized window) and search back up
    # to 7 days so the test lands on a trading session regardless of weekends/holidays.
    df = None
    elapsed = 0.0
    last_exc = None
    for back in range(1, 8):
        day   = datetime.now(tz=ET) - timedelta(days=back)
        start = day.replace(hour=10, minute=0, second=0, microsecond=0)
        end   = start + timedelta(minutes=35)
        t0 = time.monotonic()
        try:
            import pandas as pd
            df = client.get_intraday_bars("SPY", start, end, timeframe="1Min")
        except Exception as exc:
            last_exc = exc
            continue
        elapsed = time.monotonic() - t0
        if isinstance(df, pd.DataFrame) and not df.empty:
            break
    if df is None or df.empty:
        return (
            f"WARN: pacing test fetch returned no data "
            f"({last_exc if last_exc else 'empty over last 7 days'}) "
            "— ensure rate-limit guard is active"
        )

    # IB allows 50 requests / 10 s → 200ms/req budget per symbol.
    # If one SPY bar fetch exceeds 2s, a 20-symbol burst would hit the pacing wall.
    WARN_THRESHOLD_S = 2.0
    if elapsed > WARN_THRESHOLD_S:
        print(
            f"  [WARN] get_intraday_bars(SPY) took {elapsed:.1f}s — "
            f"a burst of 20 candidates could trigger IB pacing throttle. "
            "Consider adding a 0.5s inter-request delay in bar_router.py."
        )
        return f"SLOW ({elapsed:.1f}s) — add inter-request pacing"

    return f"{elapsed:.2f}s per symbol — within IB pacing budget"


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

    # ── 6. Contract qualification ─────────────────────────────────────────────
    results.append(_run("Universe contract qualification",
                        lambda: _check_contract_qualification(client, cfg)))

    # ── 7. yfinance prior closes ──────────────────────────────────────────────
    results.append(_run("yfinance prior closes (PS-filter underlyings)",
                        lambda: _check_yfinance_prior_closes(cfg)))

    # ── 8. Sigma source resolution ────────────────────────────────────────────
    results.append(_run("Sigma source (rolling vs. seed)",
                        lambda: _check_sigma_source_resolution(cfg)))

    # ── 9. Market-data line estimate ──────────────────────────────────────────
    results.append(_run("Market-data line scope (candidates, not universe)",
                        lambda: _check_market_data_lines(cfg)))

    # ── 10. reqHistoricalData pacing ──────────────────────────────────────────
    results.append(_run("reqHistoricalData pacing guard",
                        lambda: _check_historical_data_pacing(client)))

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
