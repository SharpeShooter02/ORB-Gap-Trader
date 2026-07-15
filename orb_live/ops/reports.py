"""
ops/reports.py — End-of-session and diagnostic report generation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from orb_live.core.state_store import StateStore


def daily_summary(store: StateStore, trade_date: date) -> dict:
    """
    Pull all closed trades for trade_date and compute session summary stats.

    Returns dict with keys: date, n_trades, n_wins, win_rate, total_pnl,
    avg_pnl, best_trade, worst_trade.
    """
    with store.conn() as c:
        from orb_live.core.state_store import closed_trades
        rows = c.execute(
            closed_trades.select().where(
                closed_trades.c.trade_date == trade_date
            )
        ).mappings().all()

    if not rows:
        return {"date": trade_date, "n_trades": 0, "message": "no trades"}

    df = pd.DataFrame([dict(r) for r in rows])
    wins = df[df["dollar_pnl"] > 0]
    pnl = df["dollar_pnl"].dropna()
    if pnl.empty:
        best_trade  = {"symbol": "N/A", "pnl": 0.0}
        worst_trade = {"symbol": "N/A", "pnl": 0.0}
    else:
        best_trade  = {"symbol": df.loc[pnl.idxmax(), "symbol"], "pnl": float(pnl.max())}
        worst_trade = {"symbol": df.loc[pnl.idxmin(), "symbol"], "pnl": float(pnl.min())}
    return {
        "date":        trade_date,
        "n_trades":    len(df),
        "n_wins":      len(wins),
        "win_rate":    len(wins) / len(df),
        "total_pnl":   float(df["dollar_pnl"].sum()),
        "avg_pnl":     float(df["dollar_pnl"].mean()),
        "best_trade":  best_trade,
        "worst_trade": worst_trade,
    }


def print_daily_summary(store: StateStore, trade_date: date) -> None:
    s = daily_summary(store, trade_date)
    print(f"\n{'='*50}")
    print(f"  Session Summary — {s['date']}")
    print(f"{'='*50}")
    if s.get("message") == "no trades":
        print("  No trades executed today.")
        return
    print(f"  Trades:    {s['n_trades']}  ({s['n_wins']} wins,  WR {s['win_rate']:.1%})")
    print(f"  Total P&L: ${s['total_pnl']:,.2f}")
    print(f"  Avg/trade: ${s['avg_pnl']:,.2f}")
    print(f"  Best:      {s['best_trade']['symbol']} ${s['best_trade']['pnl']:,.2f}")
    print(f"  Worst:     {s['worst_trade']['symbol']} ${s['worst_trade']['pnl']:,.2f}")
    print()


def equity_report(store: StateStore, last_n: int = 30) -> pd.DataFrame:
    """Return equity curve DataFrame for the last N trading days."""
    with store.conn() as c:
        from orb_live.core.state_store import equity_curve
        rows = c.execute(
            equity_curve.select()
            .order_by(equity_curve.c.trade_date.desc())
            .limit(last_n)
        ).mappings().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    return df.sort_values("trade_date").reset_index(drop=True)


def export_trades_csv(store: StateStore, output_path: Path) -> None:
    """Export all closed_trades to a CSV file."""
    with store.conn() as c:
        from orb_live.core.state_store import closed_trades
        rows = c.execute(closed_trades.select().order_by(
            closed_trades.c.trade_date, closed_trades.c.closed_at
        )).mappings().all()
    df = pd.DataFrame([dict(r) for r in rows])
    df.to_csv(output_path, index=False)
    print(f"  Exported {len(df)} trades → {output_path}")


def generate_daily_report(
    store: StateStore,
    trade_date: date,
    output_dir: Optional[Path] = None,
    logger=None,
) -> dict:
    """
    Generate end-of-session report for trade_date.

    Computes session summary, writes a JSON report file to output_dir
    (default: store's parent directory / reports /), and prints the human-
    readable summary to stdout.

    Returns the report dict for programmatic use.
    """
    summary = daily_summary(store, trade_date)

    report = {
        "trade_date":  str(trade_date),
        "generated_at": date.today().isoformat(),
        **summary,
    }

    # Determine output path
    if output_dir is None:
        # Resolve relative to the DB path stored in the store, or fall back
        # to the current working directory.
        db_path = getattr(store, "_db_path", None) or getattr(store, "db_path", None)
        if db_path is not None:
            output_dir = Path(db_path).parent / "reports"
        else:
            output_dir = Path("reports")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"session_{trade_date}.json"
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print_daily_summary(store, trade_date)

    if logger:
        logger.info(
            "daily_report_written",
            date=str(trade_date),
            path=str(report_path),
            n_trades=summary.get("n_trades", 0),
            total_pnl=summary.get("total_pnl", 0.0),
        )

    return report
