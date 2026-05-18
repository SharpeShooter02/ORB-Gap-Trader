"""
ops/quarterly_calibration.py — Quarterly health-check calibration jobs.

Run quarterly (Jan/Apr/Jul/Oct 1st) via cron:
    0 7 1 1,4,7,10 * cd /opt/orb-live && .venv/bin/python -m orb_live.ops.quarterly_calibration full

Functions
---------
  recompute_adv_baselines  — fetch 1yr daily bars, flag illiquid symbols
  detect_metric_drift      — compare last-90-day live metrics to historical mean ± 2σ
  audit_universe_changes   — report symbols added/removed since quarter start
  run_full_calibration     — run all three in sequence
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from orb_live.ops.long_term_logging import _load_parquet, _parquet_path, _archive_dir


# ── recompute_adv_baselines ───────────────────────────────────────────────────

def recompute_adv_baselines(
    universe: list[str],
    alpaca,
    archive_base: Path,
    today: Optional[date] = None,
    logger=None,
) -> dict:
    """
    Fetch 1-year daily bars for every symbol in universe, compute:
      - 1yr ADV (average daily dollar volume)
      - Trailing 90-day ADV
    Emit WARN if 90d ADV < 80% of 1yr; CRITICAL if < 50%.

    Persists results to data/archive/{YYYY}/adv_baseline.parquet.
    Returns a dict {symbol: {"adv_1yr": float, "adv_90d": float, "flag": str}}.
    """
    today     = today or date.today()
    results   = {}
    rows      = []

    for sym in universe:
        try:
            df = alpaca.get_daily_bars(sym, lookback_days=365)
            if df is None or df.empty:
                continue
            df = df.copy()
            if "close" not in df.columns or "volume" not in df.columns:
                continue
            df["dv"] = df["close"] * df["volume"]
            adv_1yr  = float(df["dv"].mean())

            cutoff_90 = pd.Timestamp(today) - pd.Timedelta(days=90)
            df_90 = df[df["date"] >= cutoff_90] if "date" in df.columns else df.tail(90)
            adv_90d = float(df_90["dv"].mean()) if not df_90.empty else adv_1yr

            ratio = adv_90d / adv_1yr if adv_1yr > 0 else 1.0
            if ratio < 0.50:
                flag = "CRITICAL"
                if logger:
                    logger.critical(
                        "adv_critical_illiquid", symbol=sym,
                        adv_1yr=adv_1yr, adv_90d=adv_90d, ratio=round(ratio, 3),
                    )
            elif ratio < 0.80:
                flag = "WARN"
                if logger:
                    logger.warning(
                        "adv_warn_declining", symbol=sym,
                        adv_1yr=adv_1yr, adv_90d=adv_90d, ratio=round(ratio, 3),
                    )
            else:
                flag = "OK"

            results[sym] = {"adv_1yr": adv_1yr, "adv_90d": adv_90d, "flag": flag}
            rows.append({"symbol": sym, "date": today, "adv_1yr": adv_1yr,
                         "adv_90d": adv_90d, "flag": flag})
        except Exception as exc:
            if logger:
                logger.error("adv_fetch_failed", symbol=sym, exc=str(exc))

    if rows:
        path = _parquet_path(archive_base, today.year, "adv_baseline")
        from orb_live.ops.long_term_logging import _upsert_parquet
        _upsert_parquet(path, pd.DataFrame(rows), dedup_cols=["symbol", "date"])

    if logger:
        logger.info("adv_baselines_complete", n=len(results))
    return results


# ── detect_metric_drift ───────────────────────────────────────────────────────

def detect_metric_drift(
    archive_base: Path,
    today: Optional[date] = None,
    logger=None,
) -> dict:
    """
    Pull last 90 days of live daily_summary from archive; compute key metrics.
    Compare current quarter's metrics against the historical mean ± 2σ of the
    same metrics across prior quarters.

    Emits CRITICAL if any metric drifts > 2σ from historical mean.

    Returns a dict {metric_name: {"value": float, "mean": float,
                                   "sigma": float, "z": float, "drifted": bool}}.
    """
    today     = today or date.today()
    archive_base = Path(archive_base)

    # Load all daily_summary data
    all_dfs: list[pd.DataFrame] = []
    for year_dir in sorted(archive_base.iterdir()):
        if not year_dir.is_dir():
            continue
        path = year_dir / "daily_summary.parquet"
        df   = _load_parquet(path)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        if logger:
            logger.warning("detect_metric_drift_no_data")
        return {}

    full = pd.concat(all_dfs, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])

    cutoff_90 = pd.Timestamp(today) - pd.Timedelta(days=90)
    current   = full[full["date"] >= cutoff_90].copy()
    historical = full[full["date"] < cutoff_90].copy()

    if current.empty or len(historical) < 20:
        if logger:
            logger.info("detect_metric_drift_insufficient_history",
                        current_n=len(current), hist_n=len(historical))
        return {}

    def _quarter_metrics(df: pd.DataFrame) -> dict:
        n_trades = df["n_entered"].sum()
        pnl_vals = df["realized_pnl"].values
        returns  = pd.Series(pnl_vals)
        sharpe   = (float(returns.mean() / returns.std() * (252 ** 0.5))
                    if returns.std() > 0 else 0.0)
        win_rate = float((df["n_entered"] > 0).mean())  # approx
        mean_pnl = float(returns.mean())
        max_dd   = float(_compute_max_dd(df))
        return {
            "sharpe":    sharpe,
            "mean_pnl":  mean_pnl,
            "max_dd":    max_dd,
        }

    # Compute per-quarter historical metrics for drift baseline
    historical = historical.copy()
    historical["quarter"] = historical["date"].dt.to_period("Q")
    quarter_groups = historical.groupby("quarter")
    hist_by_quarter: dict = {
        "sharpe": [], "mean_pnl": [], "max_dd": [],
    }
    for _, qdf in quarter_groups:
        qm = _quarter_metrics(qdf)
        for k in hist_by_quarter:
            hist_by_quarter[k].append(qm[k])

    current_metrics = _quarter_metrics(current)
    result: dict = {}

    for metric, vals in hist_by_quarter.items():
        if len(vals) < 4:
            continue
        mean_h = statistics.mean(vals)
        std_h  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cur_v  = current_metrics[metric]
        z      = ((cur_v - mean_h) / std_h) if std_h > 0 else 0.0
        drifted = abs(z) > 2.0

        result[metric] = {
            "value":   cur_v,
            "mean":    mean_h,
            "sigma":   std_h,
            "z":       z,
            "drifted": drifted,
        }

        if drifted and logger:
            logger.critical(
                "metric_drift_detected",
                metric=metric,
                value=round(cur_v, 4),
                historical_mean=round(mean_h, 4),
                z_score=round(z, 2),
            )

    if logger:
        logger.info("detect_metric_drift_complete",
                    drifted=[k for k, v in result.items() if v["drifted"]])
    return result


def _compute_max_dd(df: pd.DataFrame) -> float:
    if "ending_equity" not in df.columns or df.empty:
        return 0.0
    equity = df.sort_values("date")["ending_equity"].dropna()
    if equity.empty:
        return 0.0
    peak  = equity.cummax()
    dd    = (equity - peak) / peak
    return float(dd.min())


# ── audit_universe_changes ────────────────────────────────────────────────────

def audit_universe_changes(
    current_universe: list[str],
    archive_base: Path,
    today: Optional[date] = None,
    logger=None,
) -> dict:
    """
    Compare current universe to the universe at the start of the quarter.
    Returns {added: [sym], removed: [sym]}.
    """
    today = today or date.today()
    archive_base = Path(archive_base)

    # Determine quarter start
    qm  = ((today.month - 1) // 3) * 3 + 1
    q_start = date(today.year, qm, 1)

    # Load candidates from archive to infer prior universe
    prior_syms: set = set()
    path = _parquet_path(archive_base, q_start.year, "candidates")
    df   = _load_parquet(path)
    if not df.empty:
        df["date"] = pd.to_datetime(df.get("session_date", df.get("_session_date", None)))
        mask = (df["date"] >= pd.Timestamp(q_start)) & \
               (df["date"] < pd.Timestamp(today))
        if mask.any():
            prior_syms = set(df.loc[mask, "symbol"].unique())

    current_set = set(current_universe)
    added   = sorted(current_set - prior_syms)
    removed = sorted(prior_syms - current_set)

    result = {"added": added, "removed": removed, "quarter_start": str(q_start)}

    if logger:
        logger.info("universe_audit_complete",
                    added=len(added), removed=len(removed), q_start=str(q_start))
        if added:
            logger.info("universe_symbols_added", symbols=added)
        if removed:
            logger.info("universe_symbols_removed", symbols=removed)

    # Write report to archive
    report_lines = [
        f"# Universe Audit — Q starting {q_start}",
        f"_Run date: {today}_",
        "",
        f"## Added ({len(added)})",
        *([f"- {s}" for s in added] or ["_None_"]),
        "",
        f"## Removed ({len(removed)})",
        *([f"- {s}" for s in removed] or ["_None_"]),
    ]
    report_path = (_archive_dir(archive_base, today.year)
                   / f"Q{(today.month-1)//3+1}_universe_audit.md")
    report_path.write_text("\n".join(report_lines))

    return result


# ── Full calibration run ──────────────────────────────────────────────────────

def run_full_calibration(
    universe: list[str],
    alpaca,
    archive_base: Path,
    today: Optional[date] = None,
    logger=None,
) -> None:
    today = today or date.today()
    if logger:
        logger.info("quarterly_calibration_start", date=str(today))

    recompute_adv_baselines(universe, alpaca, archive_base, today, logger)
    detect_metric_drift(archive_base, today, logger)
    audit_universe_changes(universe, archive_base, today, logger)

    if logger:
        logger.info("quarterly_calibration_complete", date=str(today))


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger("quarterly_calibration")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"

    from orb_live.config.live_config import load_live_config
    from orb_live.data.alpaca_client import build_client_from_env

    cfg     = load_live_config()
    alpaca  = build_client_from_env(paper=True)
    archive = cfg.data_dir.parent / "archive"

    if cmd == "full":
        run_full_calibration(cfg.symbols, alpaca, archive, logger=_log)
    elif cmd == "adv":
        recompute_adv_baselines(cfg.symbols, alpaca, archive, logger=_log)
    elif cmd == "drift":
        detect_metric_drift(archive, logger=_log)
    elif cmd == "universe":
        audit_universe_changes(cfg.symbols, archive, logger=_log)
    else:
        print(f"Unknown command: {cmd}.  Use: full | adv | drift | universe")
        sys.exit(1)
