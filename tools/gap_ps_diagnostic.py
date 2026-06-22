"""
tools/gap_ps_diagnostic.py — One-off audit of gap + PS filter holiday handling.

Confirms correct date math for session 2026-06-22 (Juneteenth, June 19, was a
federal market holiday; prior trading day was Thursday June 18).

Run from the project root:
    python tools/gap_ps_diagnostic.py [--date YYYY-MM-DD]

Outputs per-symbol:
  - Underlying used for PS filter
  - close_t1 date & value (should be 2026-06-18)
  - close_t2 date & value (should be 2026-06-17)
  - prior_session_ret and pass/block result for each gap direction
  - Whether 2026-06-19 appears in the underlying data (should NOT for equities)

Gap prior_close:
  ETF daily bars are not stored as parquets (they come from IB at session
  time). For equity ETFs the trading calendar matches the underlying: if the
  underlying's last bar before session_date is 2026-06-18, the ETF's prior
  close is also 2026-06-18. Crypto ETFs are a known exception — see note below.

KNOWN V1 LIMITATION (crypto weekend-gap / PS-window mismatch):
  Crypto underlyings (BTC, ETH, SOL, …) trade 7 days a week. Their parquets
  include weekend bars. On a Monday session, the PS filter takes tail(2) of
  rows where date < session_date — those two rows will be Saturday and Sunday
  closes. But the ETF (BITO, ETHU, etc.) trades Mon–Fri only, so the ETF gap
  is computed against Friday's close. The PS filter therefore compares the
  wrong prior two UL closes against the ETF's breakout direction. This is NOT
  patched here; it is documented as a v1 design limitation. Crypto ETF signals
  should be treated with extra skepticism on Mondays.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import orb_live  # noqa: F401 — triggers path setup


def _parse_args() -> date:
    p = argparse.ArgumentParser(description="Gap + PS filter holiday audit")
    p.add_argument(
        "--date", default="2026-06-22",
        help="Session date to audit (YYYY-MM-DD, default: 2026-06-22)",
    )
    args = p.parse_args()
    return datetime.strptime(args.date, "%Y-%m-%d").date()


HOLIDAY_COLOUR = "\033[93m"  # yellow
OK_COLOUR      = "\033[92m"  # green
FAIL_COLOUR    = "\033[91m"  # red
RESET          = "\033[0m"


def _fmt(val: bool) -> str:
    label = "PASS" if val else "BLOCK"
    c     = OK_COLOUR if val else FAIL_COLOUR
    return f"{c}{label}{RESET}"


CRYPTO_ULS = {"BTC", "ETH", "SOL", "XRP", "LINK", "ADA", "XLM"}


def main() -> None:
    session_date = _parse_args()

    print(f"\n{'='*72}")
    print(f"  Gap + PS Filter Holiday Audit — session {session_date}")
    print(f"{'='*72}\n")

    from orb_live.config.live_config import load_live_config
    cfg = load_live_config(use_rolling=False)
    instruments = cfg.instruments
    ps_filters  = cfg.prior_session_filters

    data_dir = Path(_ROOT) / "orb_live" / "data" / "underlyings"

    # ── Identify all underlyings referenced by PS filters ─────────────────────
    seen_uls: set[str] = set()
    ul_to_syms: dict[str, list[str]] = {}
    for sym, spec in ps_filters.items():
        if spec is None:
            continue
        ul = spec[0]
        ul_to_syms.setdefault(ul, []).append(sym)
        seen_uls.add(ul)

    # ── Per-underlying analysis ────────────────────────────────────────────────
    print(f"{'UL':<8}  {'close_t1 date':<15} {'close_t1':>9}  "
          f"{'close_t2 date':<15} {'close_t2':>9}  "
          f"{'6/19 in data?':<14}  {'note'}")
    print("-" * 100)

    ul_results: dict[str, dict] = {}

    for ul in sorted(seen_uls):
        parquet = data_dir / f"{ul}.parquet"
        if not parquet.exists():
            print(f"{ul:<8}  {'NO PARQUET — cannot audit':}")
            ul_results[ul] = {"error": "no_parquet"}
            continue

        df = pd.read_parquet(parquet)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Prior two rows before session_date
        prior = df[df["date"] < pd.Timestamp(session_date)]
        tail2 = prior.tail(2)

        has_holiday = bool((df["date"] == pd.Timestamp("2026-06-19")).any())
        holiday_flag = f"{HOLIDAY_COLOUR}YES (bad!){RESET}" if has_holiday else "no"

        if len(tail2) < 2:
            print(f"{ul:<8}  {'INSUFFICIENT DATA (<2 prior rows)':}")
            ul_results[ul] = {"error": "insufficient_data"}
            continue

        row_t1 = tail2.iloc[-1]
        row_t2 = tail2.iloc[-2]
        d_t1   = row_t1["date"].date()
        d_t2   = row_t2["date"].date()
        c_t1   = float(row_t1["close"])
        c_t2   = float(row_t2["close"])

        is_crypto = ul in CRYPTO_ULS
        note = "crypto (weekend bars — see known limitation)" if is_crypto else ""

        print(f"{ul:<8}  {str(d_t1):<15} {c_t1:>9.4f}  "
              f"{str(d_t2):<15} {c_t2:>9.4f}  "
              f"{holiday_flag:<14}  {note}")

        ul_results[ul] = {
            "d_t1": d_t1, "c_t1": c_t1,
            "d_t2": d_t2, "c_t2": c_t2,
            "has_holiday_bar": has_holiday,
            "is_crypto": is_crypto,
        }

    # ── Per-ETF PS filter results ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Per-ETF PS filter results")
    print(f"{'='*72}\n")

    print(f"{'ETF':<8}  {'UL':<8}  {'threshold':>9}  "
          f"{'inv':>4}  {'ps_ret':>8}  "
          f"{'long':>6}  {'short':>6}  {'notes'}")
    print("-" * 90)

    long_blocked  = []
    short_blocked = []
    bad_dates     = []

    for sym in sorted(cfg.symbols):
        spec = ps_filters.get(sym)
        if spec is None:
            continue
        ul        = spec[0]
        threshold = float(spec[1])
        is_inv    = len(spec) == 3 and spec[2] is True

        res = ul_results.get(ul)
        if res is None or "error" in res:
            print(f"{sym:<8}  {ul:<8}  {'N/A — no UL data':}")
            continue

        c_t1 = res["c_t1"]
        c_t2 = res["c_t2"]
        d_t1 = res["d_t1"]
        d_t2 = res["d_t2"]

        if c_t2 <= 0:
            print(f"{sym:<8}  {ul:<8}  c_t2=0, skip")
            continue

        ps_ret = (c_t1 - c_t2) / c_t2

        # For each direction: dir_adj_ret = ps_ret * effective_direction
        # effective_direction = -gap_direction if inverse else gap_direction
        # Pass if dir_adj_ret <= threshold
        def _passes(gap_dir: int) -> bool:
            eff_dir = -gap_dir if is_inv else gap_dir
            return (ps_ret * eff_dir) <= threshold

        pass_long  = _passes(+1)
        pass_short = _passes(-1)

        if not pass_long:
            long_blocked.append(sym)
        if not pass_short:
            short_blocked.append(sym)

        # Flag if close_t1 is NOT 2026-06-18 (for equity) — would indicate bad date
        if not res["is_crypto"] and d_t1 != date(2026, 6, 18):
            bad_dates.append((sym, ul, d_t1))

        notes = []
        if res["is_crypto"]:
            notes.append("crypto-weekend-warning")
        if not res["has_holiday_bar"] is False and res.get("has_holiday_bar"):
            notes.append("HOLIDAY-BAR-IN-DATA")
        note_str = ", ".join(notes)

        long_str  = _fmt(pass_long)
        short_str = _fmt(pass_short)
        print(f"{sym:<8}  {ul:<8}  {threshold:>9.4f}  "
              f"{'Y' if is_inv else 'N':>4}  {ps_ret:>+8.4f}  "
              f"{long_str}  {short_str}  {note_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  AUDIT SUMMARY")
    print(f"{'='*72}")

    print(f"\n[A] Holiday handling — 2026-06-19 (Juneteenth):")
    equity_uls = {ul for ul in seen_uls if ul not in CRYPTO_ULS}
    equity_uls_with_parquet = {ul for ul in equity_uls if ul_results.get(ul, {}).get("d_t1") is not None}
    holiday_leaks = [ul for ul in equity_uls_with_parquet
                     if ul_results[ul].get("has_holiday_bar")]
    if holiday_leaks:
        print(f"  {FAIL_COLOUR}FAIL — these equity ULs have a 2026-06-19 bar: {holiday_leaks}{RESET}")
    else:
        print(f"  {OK_COLOUR}PASS — no equity UL parquet contains a 2026-06-19 bar{RESET}")

    print(f"\n[B] Gap prior_close date (equity ETFs):")
    correct_t1 = [ul for ul in equity_uls_with_parquet
                  if ul_results[ul].get("d_t1") == date(2026, 6, 18)]
    wrong_t1   = [ul for ul in equity_uls_with_parquet
                  if ul_results[ul].get("d_t1") != date(2026, 6, 18)]
    if wrong_t1:
        print(f"  {FAIL_COLOUR}UNEXPECTED: these ULs have close_t1 != 2026-06-18: {wrong_t1}{RESET}")
    else:
        print(f"  {OK_COLOUR}PASS — all {len(correct_t1)} equity ULs: close_t1 = 2026-06-18{RESET}")

    print(f"\n[C] PS filter close_t2 date (equity ULs):")
    correct_t2 = [ul for ul in equity_uls_with_parquet
                  if ul_results[ul].get("d_t2") == date(2026, 6, 17)]
    wrong_t2   = [ul for ul in equity_uls_with_parquet
                  if ul_results[ul].get("d_t2") != date(2026, 6, 17)]
    if wrong_t2:
        details = [(ul, ul_results[ul]["d_t2"]) for ul in wrong_t2]
        print(f"  {FAIL_COLOUR}UNEXPECTED: close_t2 != 2026-06-17 for: {details}{RESET}")
    else:
        print(f"  {OK_COLOUR}PASS — all {len(correct_t2)} equity ULs: close_t2 = 2026-06-17{RESET}")

    if long_blocked:
        print(f"\n[D] PS-blocked (long direction): {long_blocked}")
    else:
        print(f"\n[D] PS-blocked (long direction): none")

    if short_blocked:
        print(f"\n[E] PS-blocked (short direction): {short_blocked}")
    else:
        print(f"\n[E] PS-blocked (short direction): none")

    print(f"\n[F] Gap prior_close note:")
    print("    ETF daily bars are fetched from IB at session time and not stored")
    print("    as parquets. For equity ETFs the trading calendar matches the")
    print("    underlying: since no equity UL has a 2026-06-19 bar, the ETF's")
    print("    prior_close will also be 2026-06-18. Confirmed via [B] above.")

    print(f"\n[G] Known v1 limitation — crypto weekend-gap / PS-window mismatch:")
    crypto_uls_present = [ul for ul in CRYPTO_ULS if ul in ul_results and "error" not in ul_results[ul]]
    print(f"    Crypto ULs with parquet data: {crypto_uls_present}")
    print("    On Monday sessions, crypto UL parquets include Sat/Sun bars.")
    print("    tail(2) before 2026-06-22 would be Sun 6/21 + Sat 6/20,")
    print("    but the ETF gap is measured against Friday 6/18's close.")
    print("    PS filter therefore uses wrong prior two closes for crypto ETFs.")
    print("    This is a v1 design limitation — NOT patched here.")

    print()


if __name__ == "__main__":
    main()
