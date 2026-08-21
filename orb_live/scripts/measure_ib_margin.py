"""Measure IBKR's real margin terms. Read-only.

The margin model in the backtest assumes FINRA 4210 statutory minima (long 25%,
short 30%, scaled by the fund's leverage factor) against a 200% gross cap. Two
things suggest that is wrong for this account:

  - Live saturated at 3-4 positions on consecutive sessions, ~90% of equity
    gross, nowhere near 200%.
  - A rejection logged buying_power=5296.73 after ~$27.3k of positions on a
    ~$30k account, which is what you would see if BuyingPower tracked
    AvailableFunds with no day-trading multiple at all.

This measures rather than infers. It places NO orders: whatIfOrder is IB's
pre-trade preview and never reaches the market. Run it with IB Gateway up.

    python -m orb_live.scripts.measure_ib_margin

Uses a distinct client id so it cannot collide with a running daemon.
"""
from __future__ import annotations

import os
import sys

#: Deliberately not 1 — the live runner uses that.
CLIENT_ID = int(os.environ.get("IB_MARGIN_CLIENT_ID", 77))

#: Representative of each margin bucket the model distinguishes, plus the two
#: symbols involved in the observed saturation.
PROBES = [
    ("TQQQ", "buy",  "3x long"),
    ("SQQQ", "sell", "3x short (inverse fund, sold)"),
    ("SQQQ", "buy",  "3x inverse, bought"),
    ("GDXU", "buy",  "3x long (was mislabelled 2x)"),
    ("SOLT", "buy",  "2x long — rejected 2026-08-20"),
    ("XRPT", "buy",  "2x long — filled 2026-08-20"),
    ("ETHD", "sell", "2x inverse, sold"),
    ("SPY",  "buy",  "1x reference"),
]

ACCOUNT_TAGS = [
    "NetLiquidation", "TotalCashValue", "BuyingPower", "AvailableFunds",
    "FullAvailableFunds", "ExcessLiquidity", "FullInitMarginReq",
    "FullMaintMarginReq", "InitMarginReq", "MaintMarginReq",
    "GrossPositionValue", "DayTradesRemaining", "Leverage-S",
]


def main() -> int:
    from ib_async import IB, MarketOrder, Stock

    host = os.environ.get("IB_HOST", "127.0.0.1")
    port = int(os.environ.get("IB_PORT", 4002))

    ib = IB()
    print(f"connecting {host}:{port} clientId={CLIENT_ID} ...", flush=True)
    ib.connect(host, port, clientId=CLIENT_ID, timeout=15)
    try:
        vals = ib.accountValues()
        by_tag = {}
        for v in vals:
            if v.currency in ("USD", "BASE", ""):
                by_tag.setdefault(v.tag, v.value)

        print("\n=== ACCOUNT ===")
        for tag in ACCOUNT_TAGS:
            if tag in by_tag:
                print(f"  {tag:<22} {by_tag[tag]}")

        def f(tag, d=0.0):
            try:
                return float(by_tag.get(tag, d))
            except (TypeError, ValueError):
                return d

        equity = f("NetLiquidation")
        bp, af = f("BuyingPower"), f("AvailableFunds")
        print("\n=== THE QUESTION: is day-trading leverage available? ===")
        if equity > 0:
            print(f"  BuyingPower / equity          = {bp/equity:.2f}x")
            print(f"  AvailableFunds / equity       = {af/equity:.2f}x")
            if af > 0:
                print(f"  BuyingPower / AvailableFunds  = {bp/af:.2f}x"
                      "   (~4 = PDT day-trading margin, ~1 = Reg-T overnight only)")

        print("\n=== PER-SYMBOL INITIAL MARGIN (whatIf — no order is placed) ===")
        print(f"{'symbol':<8}{'side':<6}{'price':>9}{'notional':>11}"
              f"{'init_margin':>13}{'init':>8}{'maint':>8}  note")
        for sym, side, note in PROBES:
            try:
                c = Stock(sym, "SMART", "USD")
                ib.qualifyContracts(c)
                # Historical close rather than a snapshot: this runs after the
                # close, and the rate is init_margin/notional, which is
                # scale-invariant anyway. A pending snapshot subscription also
                # makes whatIfOrder return an empty result.
                bars = ib.reqHistoricalData(
                    c, "", "5 D", "1 day", "TRADES", True, 1, False)
                price = float(bars[-1].close) if bars else 0.0
                if price <= 0:
                    print(f"{sym:<8}{side:<6}{'no price':>9}")
                    continue

                qty = max(1, int(3000 / price))     # ~1 unit at 10% of a $30k account
                o = MarketOrder("BUY" if side == "buy" else "SELL", qty)
                o.whatIf = True
                o.tif = "DAY"                      # else IB warns and preset-overrides
                st = ib.whatIfOrder(c, o)
                if isinstance(st, list):
                    st = st[0] if st else None
                if st is None:
                    print(f"{sym:<8}{side:<6}  no margin preview returned")
                    continue
                init = float(st.initMarginChange or 0)
                maint = float(st.maintMarginChange or 0)
                notional = qty * price
                rate = init / notional if notional else float("nan")
                print(f"{sym:<8}{side:<6}{price:>9.2f}{notional:>11,.0f}"
                      f"{init:>13,.0f}{rate:>8.2f}{maint/notional:>8.2f}  {note}")
            except Exception as exc:
                print(f"{sym:<8}{side:<6}  FAILED: {exc}")
        return 0
    finally:
        ib.disconnect()
        print("\ndisconnected")


if __name__ == "__main__":
    sys.exit(main())
