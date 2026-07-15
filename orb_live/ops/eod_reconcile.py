"""
ops/eod_reconcile.py — EOD fill reconciliation.

Compare state_store closed_trades against IB executions for the session date.
Logs a WARNING for every IB fill that has no matching closed_trade record.
"""
from __future__ import annotations
from datetime import date


def reconcile_fills(trade_date: date, store, broker, log=None) -> list[str]:
    """Compare closed_trades vs broker executions; return list of divergence messages.

    Each returned string describes one IB execution that has no matching
    closed_trade row in state_store (the common gap when an OCA bracket fills
    while the runner is restarting or between bar polls).
    """
    divergences: list[str] = []

    if not hasattr(broker, "get_executions"):
        return divergences

    ib_fills = broker.get_executions(trade_date=trade_date)
    db_trades = store.get_closed_trades(trade_date)

    db_by_symbol: dict[str, list[dict]] = {}
    for t in db_trades:
        db_by_symbol.setdefault(t["symbol"], []).append(t)

    for fill in ib_fills:
        sym    = fill.get("symbol", "?")
        qty    = fill.get("qty", 0)
        price  = fill.get("price", 0)
        if qty <= 0:
            continue

        trades_for_sym = db_by_symbol.get(sym, [])
        matched = any(
            abs((t.get("realized_exit_price") or 0) - price) < 0.02
            and abs((t.get("qty") or 0) - qty) < 1
            for t in trades_for_sym
        )
        if not matched:
            msg = (
                f"IB fill not in closed_trades: {sym} qty={qty} price={price:.4f} "
                f"(trade_date={trade_date})"
            )
            divergences.append(msg)
            if log:
                log.warning("eod_reconcile_missing_trade", symbol=sym,
                            qty=qty, fill_price=price, trade_date=str(trade_date))

    return divergences
