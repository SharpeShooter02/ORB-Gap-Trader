"""
ops/long_term_logging.py — Daily and monthly archival jobs.

daily_rollup   — append today's session data to year-partitioned parquet archives
monthly_review — generate a Markdown report for the prior month
cleanup_old_logs — compress/delete aged log files

Archive layout
--------------
  data/archive/{YYYY}/trades.parquet
  data/archive/{YYYY}/fills.parquet
  data/archive/{YYYY}/events.parquet
  data/archive/{YYYY}/candidates.parquet
  data/archive/{YYYY}/liquidity_metrics.parquet
  data/archive/{YYYY}/daily_summary.parquet

All parquet appends are idempotent: a (session_date, primary_key) dedup pass
removes duplicate rows before writing.
"""

from __future__ import annotations

import gzip
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Archive helpers ────────────────────────────────────────────────────────────

def _archive_dir(base: Path, year: int) -> Path:
    p = base / str(year)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parquet_path(base: Path, year: int, table: str) -> Path:
    return _archive_dir(base, year) / f"{table}.parquet"


def _load_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def _upsert_parquet(
    path: Path,
    new_df: pd.DataFrame,
    dedup_cols: list[str],
) -> None:
    """Append new rows to an existing parquet file, deduplicating on dedup_cols."""
    existing = _load_parquet(path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    _save_parquet(combined, path)


# ── Core archival tables ───────────────────────────────────────────────────────

def _archive_table(
    store,
    session_date: date,
    archive_base: Path,
    table_name: str,
    query_fn,
    dedup_cols: list[str],
) -> None:
    rows = query_fn(store, session_date)
    if not rows:
        return
    df      = pd.DataFrame(rows)
    df["_session_date"] = session_date
    path    = _parquet_path(archive_base, session_date.year, table_name)
    _upsert_parquet(path, df, dedup_cols)


def _get_closed_trades(store, session_date: date) -> list:
    from orb_live.core.state_store import closed_trades as ct
    with store.conn() as c:
        rows = c.execute(
            ct.select().where(ct.c.trade_date == session_date)
        ).mappings().all()
    return [dict(r) for r in rows]


def _get_fills(store, session_date: date) -> list:
    from orb_live.core.state_store import fills as ft
    with store.conn() as c:
        rows = c.execute(
            ft.select().where(ft.c.session_date == session_date)
        ).mappings().all()
    return [dict(r) for r in rows]


def _get_events(store, session_date: date) -> list:
    from orb_live.core.state_store import system_events as se
    # Approximate by date of created_at
    day_start = datetime.combine(session_date, datetime.min.time())
    day_end   = day_start + timedelta(days=1)
    with store.conn() as c:
        rows = c.execute(
            se.select()
            .where(se.c.created_at >= day_start)
            .where(se.c.created_at <  day_end)
        ).mappings().all()
    return [dict(r) for r in rows]


def _get_candidates(store, session_date: date) -> list:
    return store.get_candidates(session_date)


def _get_liquidity(store, session_date: date) -> list:
    from orb_live.core.state_store import liquidity_metrics as lm
    with store.conn() as c:
        rows = c.execute(
            lm.select().where(lm.c.check_date == session_date)
        ).mappings().all()
    return [dict(r) for r in rows]


# ── daily_rollup ──────────────────────────────────────────────────────────────

def daily_rollup(
    session_date: date,
    store,
    archive_base: Path,
    is_half_day: bool = False,
    logger=None,
) -> None:
    """
    Append today's rows to year-partitioned parquet archives.

    Idempotent: running twice for the same session_date produces the same result.
    Truncates state_store rows older than 90 days after archiving (fast SQLite
    query; archive/ is the source of truth for older data).
    """
    archive_base = Path(archive_base)
    tables = [
        ("trades",            _get_closed_trades, ["trade_date", "id"]),
        ("fills",             _get_fills,         ["session_date", "id"]),
        ("events",            _get_events,        ["id"]),
        ("candidates",        _get_candidates,    ["session_date", "symbol", "phase"]),
        ("liquidity_metrics", _get_liquidity,     ["check_date", "symbol"]),
    ]

    for name, query_fn, dedup_cols in tables:
        try:
            _archive_table(store, session_date, archive_base, name,
                           query_fn, dedup_cols)
        except Exception as exc:
            if logger:
                logger.error("rollup_table_failed", table=name,
                             date=str(session_date), exc=str(exc))

    # Compute and append daily summary row
    _append_daily_summary(store, session_date, archive_base, is_half_day, logger)

    # Truncate state_store tables older than 90 days
    _truncate_old_rows(store, session_date, logger)

    if logger:
        logger.info("daily_rollup_complete", date=str(session_date))


def _append_daily_summary(
    store,
    session_date: date,
    archive_base: Path,
    is_half_day: bool,
    logger,
) -> None:
    from orb_live.core.state_store import (
        closed_trades as ct, fills as ft,
        equity_curve as ec, system_events as se,
    )
    try:
        with store.conn() as c:
            ct_rows = c.execute(
                ct.select().where(ct.c.trade_date == session_date)
            ).mappings().all()
            eq_row  = c.execute(
                ec.select().where(ec.c.trade_date == session_date)
            ).mappings().first()
            fill_rows = c.execute(
                ft.select().where(ft.c.session_date == session_date)
            ).mappings().all()

        trades_df = pd.DataFrame([dict(r) for r in ct_rows])
        n_entered = len(trades_df)
        n_wins    = int((trades_df["dollar_pnl"] > 0).sum()) if n_entered else 0
        exit_counts = trades_df["exit_reason"].value_counts().to_dict() if n_entered else {}
        pnl       = float(trades_df["dollar_pnl"].sum()) if n_entered else 0.0
        equity    = float(eq_row["end_equity"]) if eq_row else 0.0

        n_partial = sum(1 for r in fill_rows if r["reason"] == "partial_unfilled")
        n_repegs  = sum(1 for r in fill_rows if r.get("attempts", 1) > 1)

        # Count reconnects from system_events
        day_start = datetime.combine(session_date, datetime.min.time())
        day_end   = day_start + timedelta(days=1)
        with store.conn() as c:
            ev_rows = c.execute(
                se.select()
                .where(se.c.event_type.like("%reconnect%"))
                .where(se.c.created_at >= day_start)
                .where(se.c.created_at <  day_end)
            ).mappings().all()
        n_reconnects = len(ev_rows)

        candidates_today = store.get_candidates(session_date)
        n_candidates = len({r["symbol"] for r in candidates_today})

        row = {
            "date":              session_date,
            "n_candidates":      n_candidates,
            "n_passed_preflight": sum(1 for r in candidates_today
                                     if r.get("preflight_passed")),
            "n_entered":         n_entered,
            "n_exited_tp1":      exit_counts.get("tp1", 0),
            "n_exited_tp2":      exit_counts.get("tp2", 0),
            "n_exited_tp3":      exit_counts.get("tp3", 0),
            "n_stopped":         exit_counts.get("stop", 0),
            "n_eod":             exit_counts.get("eod_sweep", 0),
            "realized_pnl":      pnl,
            "ending_equity":     equity,
            "max_open_positions": n_entered,  # conservative upper bound
            "n_partial_fills":   n_partial,
            "n_repegs":          n_repegs,
            "n_reconnects":      n_reconnects,
            "max_reconnect_duration": 0.0,    # TODO: track per-reconnect duration
            "was_half_day":      is_half_day,
        }
        path = _parquet_path(archive_base, session_date.year, "daily_summary")
        _upsert_parquet(path, pd.DataFrame([row]), dedup_cols=["date"])

    except Exception as exc:
        if logger:
            logger.error("daily_summary_failed", date=str(session_date), exc=str(exc))


def _truncate_old_rows(store, session_date: date, logger) -> None:
    cutoff = session_date - timedelta(days=90)
    try:
        from orb_live.core.state_store import (
            closed_trades as ct, fills as ft,
            system_events as se, candidates as cd,
        )
        with store.conn() as c:
            c.execute(ct.delete().where(ct.c.trade_date < cutoff))
            c.execute(ft.delete().where(ft.c.session_date < cutoff))
            c.execute(cd.delete().where(cd.c.session_date < cutoff))
            c.commit()
    except Exception as exc:
        if logger:
            logger.warning("truncate_old_rows_failed", exc=str(exc))


# ── monthly_review ────────────────────────────────────────────────────────────

def monthly_review(
    year: int,
    month: int,
    archive_base: Path,
    logger=None,
) -> Path:
    """
    Generate a Markdown review report for the given year/month.

    Reads data from data/archive/{year}/daily_summary.parquet.
    Writes to data/archive/{year}/{month:02d}_review.md.
    Returns the path to the written file.
    """
    archive_base = Path(archive_base)
    summary_path = _parquet_path(archive_base, year, "daily_summary")
    summary_df   = _load_parquet(summary_path)

    # Filter to the target month
    if not summary_df.empty:
        summary_df["date"] = pd.to_datetime(summary_df["date"])
        mask = (summary_df["date"].dt.year == year) & \
               (summary_df["date"].dt.month == month)
        df = summary_df.loc[mask].copy()
    else:
        df = pd.DataFrame()

    lines: list[str] = [
        f"# Monthly Review — {year}-{month:02d}",
        "",
        f"_Generated: {datetime.now().date()}_",
        "",
    ]

    if df.empty:
        lines.append("_No session data found for this month._")
    else:
        n_sessions   = len(df)
        total_pnl    = float(df["realized_pnl"].sum())
        n_trades     = int(df["n_entered"].sum())
        n_tp1        = int(df.get("n_exited_tp1", pd.Series([0])).sum())
        n_tp2        = int(df.get("n_exited_tp2", pd.Series([0])).sum())
        n_tp3        = int(df.get("n_exited_tp3", pd.Series([0])).sum())
        n_stopped    = int(df.get("n_stopped",    pd.Series([0])).sum())
        n_eod        = int(df.get("n_eod",        pd.Series([0])).sum())
        n_reconnects = int(df.get("n_reconnects", pd.Series([0])).sum())
        n_partial    = int(df.get("n_partial_fills", pd.Series([0])).sum())
        n_half_days  = int(df.get("was_half_day", pd.Series([False])).sum())
        avg_pnl      = total_pnl / n_sessions if n_sessions else 0.0

        equity_series = df["ending_equity"].dropna()
        ending_equity = float(equity_series.iloc[-1]) if not equity_series.empty else 0.0

        # Simple Sharpe estimate (daily returns)
        if n_sessions > 1 and ending_equity > 0:
            returns = df["realized_pnl"] / ending_equity
            sharpe  = (returns.mean() / returns.std() * (252 ** 0.5)
                       if returns.std() > 0 else 0.0)
        else:
            sharpe = 0.0

        lines += [
            "## Live Metrics",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Sessions | {n_sessions} |",
            f"| Total P&L | ${total_pnl:,.2f} |",
            f"| Avg daily P&L | ${avg_pnl:,.2f} |",
            f"| Ending equity | ${ending_equity:,.2f} |",
            f"| Est. monthly Sharpe | {sharpe:.3f} |",
            f"| Trades entered | {n_trades} |",
            f"| Exits: TP1/TP2/TP3/Stop/EOD | {n_tp1}/{n_tp2}/{n_tp3}/{n_stopped}/{n_eod} |",
            "",
            "## Operational Stats",
            f"| Stat | Count |",
            f"|------|-------|",
            f"| WS reconnects | {n_reconnects} |",
            f"| Partial fills | {n_partial} |",
            f"| Half-days | {n_half_days} |",
            "",
            "## Daily Breakdown",
            "",
        ]

        for _, row in df.iterrows():
            d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            lines.append(
                f"- `{d}` — entered={int(row.get('n_entered', 0))} "
                f"pnl=${float(row.get('realized_pnl', 0)):,.2f} "
                + ("⚡half-day" if row.get("was_half_day") else "")
            )

    report_path = _archive_dir(archive_base, year) / f"{month:02d}_review.md"
    report_path.write_text("\n".join(lines))
    if logger:
        logger.info("monthly_review_written", path=str(report_path))
    return report_path


# ── cleanup_old_logs ──────────────────────────────────────────────────────────

def cleanup_old_logs(
    log_dir: Path,
    retain_days: int = 90,
    compress_after_days: int = 30,
) -> None:
    """
    Compress log files older than compress_after_days, delete those older
    than retain_days.  Never touches data/archive/.
    """
    log_dir   = Path(log_dir)
    cutoff_del = datetime.now() - timedelta(days=retain_days)
    cutoff_gz  = datetime.now() - timedelta(days=compress_after_days)

    for fpath in log_dir.glob("*.log"):
        try:
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            if mtime < cutoff_del:
                fpath.unlink()
            elif mtime < cutoff_gz:
                gz_path = fpath.with_suffix(".log.gz")
                if not gz_path.exists():
                    with fpath.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    fpath.unlink()
        except Exception:
            pass

    # Also clean up pre-existing .log.gz files past retain_days
    for fpath in log_dir.glob("*.log.gz"):
        try:
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            if mtime < cutoff_del:
                fpath.unlink()
        except Exception:
            pass
