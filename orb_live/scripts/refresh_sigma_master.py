"""
scripts/refresh_sigma_master.py — Regenerate sigma_master.csv from FULL,
adjusted historical daily data. Fully self-contained in the live repo: needs
only yfinance (already a dependency) and this script — no dependency on the
backtest repo or its cache.

Sigma matches the curated methodology exactly:
    sigma = close.pct_change().dropna().abs().std()
computed over the WHOLE adjusted (split/dividend-adjusted) daily series. Using
ADJUSTED prices is deliberate — raw/unadjusted closes carry reverse-split-day
jumps that inflate sigma 2–3× for names like XOP/GDXJ/MSTR. This reproduces the
patched sigma_master.csv (e.g. XOP≈0.0180, GDXJ≈0.0196, QQQ≈0.0126).

Because sigma is measured over many years, adding a few weeks of data barely
moves it — so run this monthly/quarterly to let values drift slowly without
window noise. It previews by default; pass --confirm to overwrite. Recomputed
rows are MERGED into the existing CSV so untouched underlyings are preserved.

Usage:
    python -m orb_live.scripts.refresh_sigma_master                 # preview universe
    python -m orb_live.scripts.refresh_sigma_master --all           # preview every CSV row
    python -m orb_live.scripts.refresh_sigma_master --confirm       # write (universe)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

import pandas as pd

import orb_live  # noqa: F401 — path setup
from orb_live.config.live_config import _load_v1_universe, _SIGMA_CSV


# yfinance symbols for tickers whose live/underlying name differs from the feed.
_YF_ALIASES = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
    "ADA": "ADA-USD", "DOGE": "DOGE-USD",
    "VIX": "^VIX", "NYFANG": "^NYFANG",
}

_YF_RETRIES = 3
_YF_BACKOFF_S = 2.0


def _fetch_adjusted_closes(ul: str) -> pd.Series | None:
    """Full-history ADJUSTED daily closes from yfinance, or None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance is required: pip install yfinance")

    ticker = _YF_ALIASES.get(ul, ul)
    for attempt in range(_YF_RETRIES):
        try:
            df = yf.download(ticker, period="max", progress=False,
                             auto_adjust=True, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and not df.empty and "Close" in df.columns:
                return df["Close"].dropna()
        except Exception:
            pass
        time.sleep(_YF_BACKOFF_S * (attempt + 1))
    return None


def _sigma(closes: pd.Series) -> float:
    return float(closes.pct_change().dropna().abs().std())


def refresh(
    targets: list[str] | None = None,
    min_obs: int = 250,
    confirm: bool = False,
    out_path: Path = _SIGMA_CSV,
) -> pd.DataFrame:
    """Recompute sigma for `targets` (default: the live universe underlyings)
    and merge into the existing CSV. Writes only when confirm=True."""
    if targets is None:
        instruments, _seed, universe = _load_v1_universe()
        targets = sorted({
            instruments[s].underlying for s in universe if s in instruments
        })

    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(
        columns=["underlying", "sigma", "n_obs", "source", "computed_at"]
    )
    old_map = dict(zip(existing["underlying"].astype(str),
                       existing["sigma"].astype(float)))

    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated: dict[str, dict] = {}
    skipped: list[str] = []
    for ul in targets:
        closes = _fetch_adjusted_closes(ul)
        n = int(closes.shape[0]) if closes is not None else 0
        if closes is None or n < min_obs:
            skipped.append(f"{ul}(n={n})")
            continue
        updated[ul] = {
            "underlying": ul, "sigma": _sigma(closes),
            "n_obs": n, "source": "yfinance", "computed_at": now,
        }

    # Merge: keep every existing row, overwrite recomputed ones.
    merged = {r["underlying"]: dict(r) for _, r in existing.iterrows()}
    merged.update(updated)
    df = (pd.DataFrame(merged.values())
          .reindex(columns=["underlying", "sigma", "n_obs", "source", "computed_at"])
          .sort_values("underlying").reset_index(drop=True))

    print(f"recomputed {len(updated)} underlyings from yfinance adjusted full history")
    if skipped:
        print(f"skipped {len(skipped)} (no data / < {min_obs} obs): {', '.join(skipped)}")

    diverged = [
        f"{ul}({updated[ul]['sigma']/old_map[ul]:.2f}x)"
        for ul in updated
        if ul in old_map and old_map[ul] and
        abs(updated[ul]['sigma'] / old_map[ul] - 1.0) > 0.20
    ]
    if diverged:
        print(f"\n!! {len(diverged)} underlyings moved >20% vs the current CSV:")
        print("   " + ", ".join(diverged))
        print("   (a >20% move changes PS-filter thresholds — re-check the "
              "backtest if this is unexpected)\n")

    if confirm:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(df)} rows)")
    else:
        print("PREVIEW ONLY — pass --confirm to write. Recomputed rows:")
        print(pd.DataFrame(updated.values())
              .reindex(columns=["underlying", "sigma", "n_obs"])
              .to_string(index=False))
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Regenerate sigma_master.csv from yfinance adjusted full history")
    p.add_argument("--all", action="store_true",
                   help="Recompute every underlying already in the CSV (default: live universe only)")
    p.add_argument("--min-obs", type=int, default=250,
                   help="Skip underlyings with fewer than this many daily closes")
    p.add_argument("--confirm", action="store_true",
                   help="Actually overwrite sigma_master.csv (default is preview only)")
    args = p.parse_args()

    targets = None
    if args.all and _SIGMA_CSV.exists():
        targets = sorted(pd.read_csv(_SIGMA_CSV)["underlying"].astype(str).tolist())

    df = refresh(targets=targets, min_obs=args.min_obs, confirm=args.confirm)
    if df.empty:
        print("ERROR: no sigmas produced — check the data source.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
