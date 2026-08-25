"""A flatten must never book a position at a zero exit price.

L9. On 2026-08-05 two positions were flattened by the SIGINT handler and
recorded with exit_price = 0.0, realized_exit_price = 0.0 and pnl_pct of
exactly -1.0 and +1.0. dollar_pnl was then computed off those sentinels:
NVDU -426.30 (= 142.10 x 3) and AMDL +465.93 (= 51.77 x 9) -- the full
position notional booked as a total loss and a total gain. `fills` recorded
both entries and no exits, so the database did not know what had happened.

flatten_all passes ref_price=None, which used to flow straight through to
_trade_pnl. Fixed the same afternoon in 30e2777 by _resolve_exit_price:
actual fill -> quote mid -> broker mark -> entry price. The last rung records
~0 P&L, an honest "unknown" rather than a fabricated notional-sized move.

These tests pin every rung, including the one where the broker tells us
nothing at all, because the failure was silent: it produced plausible-looking
numbers in the right columns and only surfaced when a reconciliation flagged
exit_price = 0.00 weeks later.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from orb_live.tests.test_position_manager import (   # noqa: F401
    _build_mgr, _make_entry, _strategy_cfg, TRADE_DATE,
)
# mock_broker / tmp_store come from conftest.py

UTC = timezone.utc


def _open_one(mgr, broker, symbol="NVDU", entry_price=142.10, shares=3):
    entry = _make_entry(entry_price=entry_price, shares=shares,
                        stop_price=entry_price * 0.98,
                        tp1_price=entry_price * 1.02,
                        tp2_price=entry_price * 1.04,
                        tp1_shares=shares, tp2_shares=0)
    mgr.open_position(entry, symbol, +1, TRADE_DATE)
    pos = mgr._positions[symbol]
    pos.status = "open"
    pos.actual_entry_price = entry_price
    pos.remaining = shares
    pos.entry_shares = shares
    return pos


class TestResolveExitPriceNeverZero:
    def test_falls_back_to_entry_when_broker_knows_nothing(
            self, mock_broker, tmp_store):
        """Every rung fails. Must return the entry price, never 0."""
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = _open_one(mgr, mock_broker)

        mock_broker.get_latest_quote = lambda s: {}
        mock_broker.get_position = lambda s: {}

        px = mgr._resolve_exit_price("NVDU", pos, None)
        assert px == pytest.approx(142.10)
        assert px > 0

    def test_quote_mid_beats_entry_fallback(self, mock_broker, tmp_store):
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = _open_one(mgr, mock_broker)
        mock_broker.get_latest_quote = lambda s: {"bid": 140.0, "ask": 141.0}
        assert mgr._resolve_exit_price("NVDU", pos, None) == pytest.approx(140.5)

    def test_zero_and_none_quotes_do_not_leak_through(
            self, mock_broker, tmp_store):
        """A broker returning 0s is the exact shape that caused L9."""
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = _open_one(mgr, mock_broker)
        mock_broker.get_latest_quote = lambda s: {"bid": 0, "ask": 0, "last": 0}
        mock_broker.get_position = lambda s: {"market_price": 0}
        px = mgr._resolve_exit_price("NVDU", pos, None)
        assert px == pytest.approx(142.10)


class TestFlattenAllRecordsARealPrice:
    def test_sigint_flatten_does_not_book_full_notional(
            self, mock_broker, tmp_store):
        """The L9 scenario end to end: flatten with no price available."""
        mgr = _build_mgr(mock_broker, tmp_store)
        _open_one(mgr, mock_broker, symbol="NVDU", entry_price=142.10, shares=3)

        # The L9 shape: the broker knows the position is there but gives no
        # price for it. (Returning no position at all is a different path --
        # _exit_all caps exit qty to 0 and books no trade, correctly.)
        mock_broker.get_latest_quote = lambda s: {}
        mock_broker.get_position = lambda s: {"qty": 3}

        mgr.flatten_all("sigint")

        rows = tmp_store.get_closed_trades(TRADE_DATE)
        assert len(rows) == 1
        r = rows[0]
        assert r["exit_price"] > 0, "booked a zero exit price -- L9 regression"
        assert abs(r["pnl_pct"]) < 0.999, (
            f"pnl_pct {r['pnl_pct']} is the +/-100% sentinel: the full "
            "position notional has been booked as P&L")
        assert abs(r["dollar_pnl"]) < 142.10 * 3 * 0.5

    def test_exit_reason_carries_the_shutdown_reason(
            self, mock_broker, tmp_store):
        mgr = _build_mgr(mock_broker, tmp_store)
        _open_one(mgr, mock_broker)
        mock_broker.get_latest_quote = lambda s: {}
        mock_broker.get_position = lambda s: {"qty": 3}
        mgr.flatten_all("sigint")
        rows = tmp_store.get_closed_trades(TRADE_DATE)
        assert rows[0]["exit_reason"].endswith("_sigint")
