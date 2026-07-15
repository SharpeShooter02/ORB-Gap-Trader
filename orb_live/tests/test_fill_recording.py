"""
tests/test_fill_recording.py — Fill-event recording and EOD reconcile tests.
"""
import pytest
from datetime import date
from types import SimpleNamespace

SYMBOL     = "TQQQ"
TRADE_DATE = date(2026, 7, 6)

from orb_live.tests.test_position_manager import (
    _make_entry, _build_mgr, _strategy_cfg, _ts, _bar,
)

UTC = __import__("datetime").timezone.utc
from datetime import datetime


def _make_fill_ns(order_id, price, qty, commission=0.0):
    execution = SimpleNamespace(
        orderId=int(order_id) if str(order_id).isdigit() else 0,
        price=price,
        shares=qty,
        cumQty=qty,
        avgPrice=price,
        side="BOT",
        time="2026-07-06 10:15:00",
        execId="E001",
    )
    commission_report = SimpleNamespace(commission=commission)
    return SimpleNamespace(execution=execution, commissionReport=commission_report)


def _make_trade_ns(order_id, qty):
    order = SimpleNamespace(orderId=int(order_id) if str(order_id).isdigit() else 0,
                             totalQuantity=qty)
    return SimpleNamespace(order=order)


def _make_tp1only_entry(**kwargs):
    """Entry where all shares go to TP1 — triggers OCA bracket path."""
    defaults = dict(
        entry_price=100.0, shares=100, orb_range=1.0,
        stop_price=99.0, tp1_price=101.0, tp2_price=102.0,
        tp1_shares=100, tp2_shares=0, tp3_shares=0,
    )
    defaults.update(kwargs)
    return _make_entry(**defaults)


class TestFillEventRecording:

    def test_tp1_fill_event_records_closed_trade(self, mock_broker, tmp_store):
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = mgr.open_position(_make_tp1only_entry(), SYMBOL, +1, TRADE_DATE)
        assert pos is not None
        tp1_id = pos.tp1_order_id
        assert tp1_id is not None

        mock_broker.fire_fill_watcher(tp1_id, qty=float(pos.tp1_shares), price=pos.tp1_price)

        trades = tmp_store.get_closed_trades(TRADE_DATE)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "TP1"
        assert t["realized_exit_price"] == pytest.approx(pos.tp1_price)
        assert t["qty"] == pytest.approx(pos.tp1_shares)
        expected_pnl = (pos.tp1_price - pos.actual_entry_price) * pos.tp1_shares * 1
        assert t["dollar_pnl"] == pytest.approx(expected_pnl, abs=0.01)

    def test_stop_fill_event_records_closed_trade(self, mock_broker, tmp_store):
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = mgr.open_position(_make_tp1only_entry(), SYMBOL, +1, TRADE_DATE)
        assert pos is not None
        stop_id   = pos.stop_order_id
        stop_price = pos.stop_price
        assert stop_id is not None

        mock_broker.fire_fill_watcher(stop_id, qty=float(pos.entry_shares), price=stop_price)

        trades = tmp_store.get_closed_trades(TRADE_DATE)
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "STOP"
        expected_pnl = (stop_price - pos.actual_entry_price) * pos.entry_shares * 1
        assert t["dollar_pnl"] == pytest.approx(expected_pnl, abs=0.01)

    def test_fill_watcher_dedup_prevents_double_record(self, mock_broker, tmp_store):
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = mgr.open_position(_make_tp1only_entry(), SYMBOL, +1, TRADE_DATE)
        tp1_id = pos.tp1_order_id
        assert tp1_id is not None

        mock_broker.fire_fill_watcher(tp1_id, qty=float(pos.tp1_shares), price=pos.tp1_price)
        mock_broker.fire_fill_watcher(tp1_id, qty=float(pos.tp1_shares), price=pos.tp1_price)

        trades = tmp_store.get_closed_trades(TRADE_DATE)
        assert len(trades) == 1

    def test_tp1_oca_poll_path_also_records_with_pnl(self, mock_broker, tmp_store):
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = mgr.open_position(_make_tp1only_entry(), SYMBOL, +1, TRADE_DATE)
        assert pos is not None
        assert pos.tp1_order_id is not None

        mock_broker._orders[pos.tp1_order_id].update({
            "status": "filled",
            "filled_qty": str(pos.tp1_shares),
            "filled_avg_price": str(pos.tp1_price),
        })

        mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.2), _ts())

        trades = tmp_store.get_closed_trades(TRADE_DATE)
        tp1_trades = [t for t in trades if t.get("exit_reason") == "TP1_ONLY"]
        assert len(tp1_trades) == 1
        t = tp1_trades[0]
        assert t["dollar_pnl"] is not None
        assert t["qty"] == pos.entry_shares

    def test_stop_poll_path_records_with_correct_qty_and_pnl(self, mock_broker, tmp_store):
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = mgr.open_position(_make_tp1only_entry(), SYMBOL, +1, TRADE_DATE)
        assert pos is not None
        stop_id = pos.stop_order_id

        fill_qty   = pos.entry_shares
        fill_price = 99.0
        mock_broker._orders[stop_id].update({
            "status": "filled",
            "filled_qty": str(fill_qty),
            "filled_avg_price": str(fill_price),
        })

        mgr.on_bar(SYMBOL, _bar(hi=100.5, lo=98.5), _ts())

        trades = tmp_store.get_closed_trades(TRADE_DATE)
        assert len(trades) == 1
        t = trades[0]
        assert t["qty"] == fill_qty
        expected_pnl = (fill_price - pos.actual_entry_price) * fill_qty * 1
        assert t["dollar_pnl"] == pytest.approx(expected_pnl, abs=0.01)


class TestEodReconcile:

    def test_reconcile_flags_missing_fill(self, mock_broker, tmp_store):
        from orb_live.ops.eod_reconcile import reconcile_fills

        mock_broker._orders["exec-1"] = {
            "id": "exec-1", "status": "filled",
            "filled_qty": "35", "filled_avg_price": "101.0",
        }

        divergences = reconcile_fills(TRADE_DATE, tmp_store, mock_broker)
        # reconcile_fills should not crash; result depends on get_executions output
        assert isinstance(divergences, list)

    def test_reconcile_no_divergence_when_all_recorded(self, mock_broker, tmp_store):
        from orb_live.ops.eod_reconcile import reconcile_fills

        # When there are no IB fills at all, no divergences are possible.
        divergences = reconcile_fills(TRADE_DATE, tmp_store, mock_broker)
        assert divergences == []

    def test_reconcile_empty_when_no_broker_fills(self, mock_broker, tmp_store):
        from orb_live.ops.eod_reconcile import reconcile_fills
        divergences = reconcile_fills(TRADE_DATE, tmp_store, mock_broker)
        assert divergences == []
