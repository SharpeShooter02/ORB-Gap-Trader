"""L4: is IB's intraday requirement lower than whatIfOrder reports?

MUST RUN DURING REGULAR TRADING HOURS (09:30-16:00 ET). Outside RTH IB returns
an empty OrderState and ib_client.check_margin falls back to full notional, so
every symbol reads 100% and the measurement is meaningless. Live is unaffected:
_prewarm_candidate_margin runs at ORB close, inside RTH.

What to look for:
  init% well below the ~0.79 recorded for TQQQ  -> a real intraday allowance
                                                   exists; capacity rises and
                                                   the margin analysis needs
                                                   redoing at a looser bound.
  init% == the recorded overnight rate          -> no hidden capacity. The 4x
                                                   BuyingPower is just 1/0.25
                                                   and does not apply to
                                                   instruments charged ~79%.

    python scripts/probe_ib_margin.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:/Users/buttn/Documents/Projects/orb-live-trading")

from orb_live.data.ib_client import IBClient

ET = ZoneInfo("America/New_York")
SYMS = ["TQQQ", "SOXL", "SOXS", "NUGT", "JNUG", "BITX", "KOLD", "AMDL", "SPY"]

#: Rates measured previously and stored in the live profile, for comparison.
KNOWN = {"TQQQ": 0.782, "SOXL": 0.771, "SOXS": 0.803, "NUGT": 0.515,
         "JNUG": 0.516, "BITX": 0.953, "KOLD": 0.521, "AMDL": 0.789}


def main() -> None:
    now = datetime.now(ET)
    rth = now.replace(hour=9, minute=30) <= now <= now.replace(hour=16, minute=0)
    print(f"now {now:%Y-%m-%d %H:%M} ET  |  RTH: {rth}")
    if not rth:
        print("\nNOT IN RTH -- whatIfOrder will return nothing and every symbol\n"
              "will read 100%. Run this between 09:30 and 16:00 ET.\n")

    c = IBClient(host=os.environ.get("IB_HOST", "127.0.0.1"),
                 port=int(os.environ.get("IB_PORT", "4002")),
                 client_id=int(os.environ.get("IB_CLIENT_ID", "95")))
    c.connect()

    vals = {}
    for v in c._ib.accountValues():
        if v.currency in ("USD", "BASE", ""):
            vals.setdefault(v.tag, v.value)

    def f(t):
        try:
            return float(vals.get(t, 0) or 0)
        except ValueError:
            return 0.0

    nl, af, bp = f("NetLiquidation"), f("AvailableFunds"), f("BuyingPower")
    print(f"\nNetLiquidation {nl:>12,.2f}")
    print(f"AvailableFunds {af:>12,.2f}")
    print(f"BuyingPower    {bp:>12,.2f}   = {bp/nl:.2f}x NetLiq"
          f"  {bp/af:.2f}x AvailFunds" if nl and af else "")
    print(f"ExcessLiquidity {f('ExcessLiquidity'):>11,.2f}   SMA {f('SMA'):,.2f}")
    for t in ["InitMarginReq", "MaintMarginReq", "FullInitMarginReq",
              "FullMaintMarginReq", "LookAheadInitMarginReq"]:
        print(f"  {t:<24} {vals.get(t, '-')}")

    print(f"\n{'sym':<6}{'px':>9}{'notional':>10}{'init':>10}{'maint':>10}"
          f"{'init%':>8}{'maint%':>8}{'known':>8}{'delta':>8}")
    for sym in SYMS:
        try:
            d = c.get_daily_bars(sym, lookback_days=5)
            if d is None or len(d) == 0:
                print(f"{sym:<6} no bars")
                continue
            px = float(d["close"].iloc[-1])
            qty = max(1, int(3000 / px))
            r = c.check_margin(sym, "buy", qty, px)
            n = qty * px
            ir, mr = r["init_margin"] / n, r["maint_margin"] / n
            k = KNOWN.get(sym)
            print(f"{sym:<6}{px:>9.2f}{n:>10,.0f}{r['init_margin']:>10,.0f}"
                  f"{r['maint_margin']:>10,.0f}{ir:>8.3f}{mr:>8.3f}"
                  f"{(f'{k:.3f}' if k else '-'):>8}"
                  f"{(f'{ir-k:+.3f}' if k else '-'):>8}")
        except Exception as exc:
            print(f"{sym:<6} ERR {type(exc).__name__}: {exc}")

    print("\ninit% == 1.000 across the board means the preview failed, not that\n"
          "IB requires 100% -- see the module docstring.")


if __name__ == "__main__":
    main()
