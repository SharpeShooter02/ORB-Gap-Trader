"""
ops/calibration.py — Sigma recalibration for the prior-session filter.

Computes new sigma estimates from yfinance daily data and compares them to
the reference SIGMA dict.  Optionally writes sigma_override.yaml so the
live config picks up updated thresholds on next startup.

IMPORTANT: This module never modifies _production_run.py.  The override
file is the only persistence mechanism.  Delete sigma_override.yaml to
revert to reference values.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import orb_live  # noqa: F401 — sys.path setup
from reference._production_run import SIGMA
from orb_live.config.live_config import _SIGMA_OVERRIDE_FILE


def compute_sigma(
    underlying: str,
    lookback_years: float = 3.0,
    end_date: Optional[date] = None,
) -> tuple[float, int]:
    """
    Compute sigma (std of daily absolute returns) for `underlying`.

    Returns (sigma, n_obs).  Uses yfinance auto-adjusted closes.
    Raises ValueError if fewer than 30 observations are available.
    """
    import yfinance as yf

    end   = pd.Timestamp(end_date or date.today())
    start = end - pd.DateOffset(years=lookback_years)
    df = yf.download(
        underlying,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False, auto_adjust=True,
    )
    if df.empty or len(df) < 30:
        raise ValueError(f"Insufficient data for {underlying}: {len(df)} rows")
    returns = df["Close"].pct_change().dropna().abs()
    return float(returns.std()), len(returns)


def recalibrate_all(
    underlyings: Optional[list[str]] = None,
    lookback_years: float = 3.0,
) -> pd.DataFrame:
    """
    Recalibrate sigma for all (or specified) underlyings.

    Returns a DataFrame with columns:
        underlying, ref_sigma, new_sigma, n_obs, delta_pct, flag
    """
    targets = underlyings or list(SIGMA.keys())
    rows = []
    for ul in targets:
        try:
            new_sig, n_obs = compute_sigma(ul, lookback_years)
            ref_sig = SIGMA.get(ul)
            delta = (new_sig / ref_sig - 1.0) * 100 if ref_sig else None
            flag = ""
            if delta is not None:
                if abs(delta) > 20:
                    flag = "LARGE_CHANGE"
                elif abs(delta) > 10:
                    flag = "MODERATE_CHANGE"
            rows.append({
                "underlying": ul,
                "ref_sigma":  ref_sig,
                "new_sigma":  round(new_sig, 6),
                "n_obs":      n_obs,
                "delta_pct":  round(delta, 2) if delta is not None else None,
                "flag":       flag,
            })
        except Exception as exc:
            rows.append({
                "underlying": ul,
                "ref_sigma":  SIGMA.get(ul),
                "new_sigma":  None,
                "n_obs":      0,
                "delta_pct":  None,
                "flag":       f"ERROR: {exc}",
            })
    return pd.DataFrame(rows).sort_values("flag", ascending=False)


def print_calibration_report(df: pd.DataFrame) -> None:
    print("\n=== Sigma Recalibration Report ===")
    print(f"{'Underlying':<12} {'Ref σ':>8} {'New σ':>8} {'Δ%':>8} {'N':>6}  Flag")
    print("-" * 60)
    for _, row in df.iterrows():
        new   = f"{row['new_sigma']:.4f}" if row["new_sigma"] is not None else "ERR"
        ref   = f"{row['ref_sigma']:.4f}" if row["ref_sigma"] is not None else "N/A"
        delta = f"{row['delta_pct']:+.1f}%" if row["delta_pct"] is not None else "N/A"
        print(f"  {row['underlying']:<10} {ref:>8} {new:>8} {delta:>8} {row['n_obs']:>6}  {row['flag']}")
    flagged = df[df["flag"].str.startswith(("LARGE", "MODERATE", "ERROR"), na=False)]
    if not flagged.empty:
        print(f"\n  {len(flagged)} items need attention (see flag column).")
    print()


def write_sigma_override(
    df: pd.DataFrame,
    output_path: Path = _SIGMA_OVERRIDE_FILE,
    only_flagged: bool = False,
) -> None:
    """
    Write sigma_override.yaml from recalibration results.

    If only_flagged=True, only includes underlyings where delta_pct > 10%.
    The live config loads this file on startup via _build_live_ps_filters().

    The file format is:
        sigmas:
          BTC: 0.0421
          ETH: 0.0381
          ...
    """
    rows = df[df["new_sigma"].notna()]
    if only_flagged:
        rows = rows[rows["flag"].str.startswith(("LARGE", "MODERATE"), na=False)]

    sigmas = {row["underlying"]: round(row["new_sigma"], 6)
              for _, row in rows.iterrows()}
    if not sigmas:
        print("No sigmas to write.")
        return

    payload = {
        "sigmas": sigmas,
        "_meta": {
            "generated": str(date.today()),
            "n_underlyings": len(sigmas),
        },
    }
    output_path.write_text(yaml.dump(payload, default_flow_style=False))
    print(f"  Wrote {len(sigmas)} sigma overrides → {output_path}")


def recalibrate_sigmas(
    lookback_years: float = 3.0,
    k: float = 1.25,
    confirm: bool = False,
    state_store=None,
) -> pd.DataFrame:
    """
    Full recalibration flow:
      1. Compute new sigmas for all SIGMA underlyings.
      2. Print report.
      3. If confirm=True, write sigma_override.yaml.
      4. If state_store provided, record each sigma to sigma_history table.

    Returns the recalibration DataFrame.
    """
    print(f"\nRecalibrating sigmas ({lookback_years}y lookback, k={k}) ...")
    df = recalibrate_all(lookback_years=lookback_years)
    print_calibration_report(df)

    print("=== PS filter threshold preview (new_sigma × k) ===")
    for _, row in df.sort_values("underlying").iterrows():
        if row["new_sigma"] is not None:
            thr = row["new_sigma"] * k
            ref_thr = (row["ref_sigma"] * k) if row["ref_sigma"] else float("nan")
            print(f"  {row['underlying']:<12} threshold: {thr:.4f}  (was {ref_thr:.4f})")
    print()

    if confirm:
        write_sigma_override(df)
    else:
        print("  Pass confirm=True to apply changes (write sigma_override.yaml).")

    if state_store is not None:
        from datetime import timezone
        UTC = timezone.utc
        from datetime import datetime
        for _, row in df.iterrows():
            if row["new_sigma"] is None:
                continue
            state_store.save_sigma_calibration(
                underlying=row["underlying"],
                sigma=row["new_sigma"],
                lookback_years=lookback_years,
                n_obs=int(row["n_obs"]),
                source="yfinance",
            )

    return df
