"""Measure IB's real initial-margin rate for every tradable symbol, both sides,
and write the result into master_universe.csv.

Why this exists: the backtest's margin model assumed FINRA 4210 statutory
minima (long 25%, short 30%, scaled by leverage). Spot checks against IB show
that is right for conventional leveraged ETFs and badly wrong for crypto-linked
ones:

    TQQQ 3x long    IB 0.79 init / 0.75 maint     model 0.75   ok
    SQQQ 3x short   IB 0.95 / 0.90                model 0.90   ok
    SOLT 2x long    IB 1.01 / 1.01                model 0.50   2x understated
    XRPT 2x long    IB 1.06 / 1.06                model 0.50   2x understated
    ETHD 2x short   IB 1.11 / 1.01                model 0.60   2x understated

IB charges essentially full notional on crypto ETFs, so they consume buying
power at twice the modelled rate. That inverts the ranking: C1 is crypto, has
the highest P(fire), and is the *most* expensive use of margin rather than the
cheapest.

Rates are stored per symbol AND per side, because they differ: buying SQQQ
costs 0.79 while selling it costs 0.95. The strategy trades either side
depending on gap direction, so one number per symbol would be wrong half the
time.

READ-ONLY against the broker: whatIfOrder is a pre-trade preview and never
reaches the market. The only thing written is the CSV.

    python -m orb_live.scripts.measure_universe_margin [--dry-run]

Re-run periodically. House requirements change, and a stale rate is worse than
an honest default because it looks authoritative.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

CLIENT_ID = int(os.environ.get("IB_MARGIN_CLIENT_ID", 77))

DATA_DIR = Path(__file__).resolve().parents[1] / "strategy" / "data"
MASTER_CSV = DATA_DIR / "master_universe.csv"

#: Mirror target — the backtest repo keeps a copy of the profile CSV.
BACKTEST_COPY = Path(r"C:\Users\buttn\Documents\Projects\BacktestingGaps\master_universe.csv")

#: Probe notional. The rate is init_margin/notional and so is scale-invariant;
#: this only needs to be large enough to avoid rounding on a single share.
PROBE_NOTIONAL = 3_000.0

COLUMNS = ["margin_init_long", "margin_maint_long",
           "margin_init_short", "margin_maint_short", "margin_measured"]


def _probe(ib, Stock, MarketOrder, symbol: str, price: float) -> dict:
    """{side: (init_rate, maint_rate)} for one symbol, or {} on failure."""
    c = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(c)
    qty = max(1, int(PROBE_NOTIONAL / price))
    notional = qty * price
    out = {}
    for side, action in (("long", "BUY"), ("short", "SELL")):
        o = MarketOrder(action, qty)
        o.whatIf = True
        o.tif = "DAY"          # otherwise IB warns and applies a preset
        st = ib.whatIfOrder(c, o)
        if isinstance(st, list):
            st = st[0] if st else None
        if st is None:
            continue
        init = float(st.initMarginChange or 0)
        maint = float(st.maintMarginChange or 0)
        out[side] = (init / notional, maint / notional)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and print, but do not write the CSV")
    args = ap.parse_args()

    from ib_async import IB, MarketOrder, Stock

    from orb_live.strategy.v1_strategy import load_master, load_sigmas, build_universe

    instruments = load_master(MASTER_CSV)
    sigmas = load_sigmas(DATA_DIR / "sigma_master.csv")
    universe = build_universe(instruments, sigmas, MASTER_CSV)
    print(f"{len(universe)} tradable symbols")

    ib = IB()
    host = os.environ.get("IB_HOST", "127.0.0.1")
    port = int(os.environ.get("IB_PORT", 4002))
    print(f"connecting {host}:{port} clientId={CLIENT_ID} ...", flush=True)
    ib.connect(host, port, clientId=CLIENT_ID, timeout=15)

    rates: dict[str, dict] = {}
    try:
        # One batched snapshot rather than 58 historical requests, which would
        # sit right on IB's pacing limit.
        contracts = []
        for sym in universe:
            try:
                c = Stock(sym, "SMART", "USD")
                ib.qualifyContracts(c)
                contracts.append((sym, c))
            except Exception as exc:
                print(f"  {sym}: qualify failed — {exc}")
        tickers = ib.reqTickers(*[c for _, c in contracts])
        prices = {}
        for (sym, _), t in zip(contracts, tickers):
            p = next((x for x in (t.marketPrice(), t.close, t.last)
                      if x and x == x and x > 0), None)
            if p:
                prices[sym] = float(p)

        print(f"\n{'symbol':<8}{'price':>9}{'long i/m':>14}{'short i/m':>14}")
        for sym, _c in contracts:
            price = prices.get(sym)
            if not price:
                bars = ib.reqHistoricalData(
                    Stock(sym, "SMART", "USD"), "", "5 D", "1 day",
                    "TRADES", True, 1, False)
                price = float(bars[-1].close) if bars else None
            if not price:
                print(f"{sym:<8}{'no price':>9}")
                continue
            try:
                r = _probe(ib, Stock, MarketOrder, sym, price)
            except Exception as exc:
                print(f"{sym:<8}  probe failed — {exc}")
                continue
            if not r:
                continue
            rates[sym] = r
            lo = r.get("long", (float("nan"),) * 2)
            sh = r.get("short", (float("nan"),) * 2)
            print(f"{sym:<8}{price:>9.2f}{lo[0]:>8.2f}/{lo[1]:<5.2f}"
                  f"{sh[0]:>8.2f}/{sh[1]:<5.2f}")
    finally:
        ib.disconnect()
        print("\ndisconnected")

    if not rates:
        print("no rates measured — nothing written")
        return 1

    df = pd.read_csv(MASTER_CSV)
    today = date.today().isoformat()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    hit = 0
    for i, row in df.iterrows():
        r = rates.get(str(row["etf"]).strip())
        if not r:
            continue
        hit += 1
        if "long" in r:
            df.at[i, "margin_init_long"] = round(r["long"][0], 4)
            df.at[i, "margin_maint_long"] = round(r["long"][1], 4)
        if "short" in r:
            df.at[i, "margin_init_short"] = round(r["short"][0], 4)
            df.at[i, "margin_maint_short"] = round(r["short"][1], 4)
        df.at[i, "margin_measured"] = today

    missing = sorted(set(rates) - set(df["etf"].astype(str).str.strip()))
    if missing:
        print(f"\nmeasured but not rows in master_universe.csv: {missing}")

    print(f"\n{hit} of {len(df)} rows updated ({len(rates)} symbols measured)")
    if args.dry_run:
        print("--dry-run: not written")
        return 0

    df.to_csv(MASTER_CSV, index=False)
    print(f"wrote {MASTER_CSV}")
    if BACKTEST_COPY.exists():
        df.to_csv(BACKTEST_COPY, index=False)
        print(f"wrote {BACKTEST_COPY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
