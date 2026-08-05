"""
test_pre_placed_entries.py — Spec tests for pre-placed ORB stop-limit entry approach.

In the pre-placed model:
  1. on_orb_complete → place_entry_order() submits a resting stop-limit at
     orb["high"] (long) or orb["low"] (short) for each candidate.
  2. IB fill event (execDetailsEvent) → _on_resting_entry_fill() callback:
       a. Check risk gate (at fill time, not placement time).
       b. Gate OK  → submit OCA bracket (TP1 + protective stop), track position.
       c. Gate FAIL → submit_market_order to immediately reverse the fill.
  3. EOD (16:00:30) → flatten_all():
       Step 0: cancel all resting (unfilled) entry orders.
       Step 1: market-exit all open positions (unchanged).
       Safety net: broker position scan catches cancel/fill races.

All tests here FAIL until the pre-placed entry feature is implemented.
They define the required behaviour and serve as the acceptance criteria.

Interface targeted on LivePositionManager:
    place_entry_order(entry: dict, symbol: str, direction: int,
                      session_date: date) -> str | None
        Submits resting stop-limit. Returns the IB order_id or None on failure.
        Registers fill watcher; does NOT yet check the risk gate.
        Adds symbol to _resting_entries.

    _resting_entries: dict[str, dict]
        {symbol: {"order_id": str, "entry": dict, "direction": int,
                  "session_date": date}}

    flatten_all(reason) extended:
        Before Step 1, iterate _resting_entries: cancel each order,
        remove from dict.  No closed_trade record on a cancelled-unfilled entry.
"""

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch, call

import pytest

UTC        = timezone.utc
TRADE_DATE = date(2026, 1, 6)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _exe_cfg(**overrides):
    ns = SimpleNamespace(
        session_kill_loss_pct=0.03,
        max_concurrent_positions=0,
        max_gross_exposure_pct=2.0,
        max_position_pct=0.50,
        entry_slippage_bps=10,
        stop_order_type="market",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _strategy_cfg(**overrides):
    ns = SimpleNamespace(
        tp3_mode="ema_crossback",
        eod_exit_hour=16,
        eod_exit_minute=0,
        exit_ratio_tp1=1.0,
        exit_ratio_tp2=0.0,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _build_mgr(mock_broker, tmp_store, exe_cfg=None, strategy_cfg=None, universe=None):
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager

    exc  = exe_cfg      or _exe_cfg()
    scfg = strategy_cfg or _strategy_cfg()

    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=lambda _: None)
    gate   = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TRADE_DATE)

    mgr = LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, indicators_store={}, config=scfg,
        universe=universe or ["TQQQ"],
    )
    return mgr


def _make_entry(entry_price=100.0, shares=10, stop_price=99.0,
                tp1_price=102.0, orb_range=1.0):
    """All-to-TP1 entry (v1 style)."""
    return {
        "entry_price":   entry_price,
        "shares":        shares,
        "orb_range":     orb_range,
        "stop_price":    stop_price,
        "tp1_price":     tp1_price,
        "tp2_price":     tp1_price,
        "tp1_shares":    shares,
        "tp2_shares":    0,
        "tp3_shares":    0,
        "exit_override": {},
    }


# ── pp-a: EOD cancels all unfilled resting orders ────────────────────────────

def test_pp_a_eod_cancels_all_unfilled_resting_orders(mock_broker, tmp_store):
    """All 3 pre-placed entries sit unfilled through the session.
    flatten_all must:
      - cancel each resting order exactly once
      - NOT submit any market orders (nothing to exit)
      - NOT create any closed_trade DB rows
    """
    syms = ["SOXL", "SOXS", "TQQQ"]
    mgr  = _build_mgr(mock_broker, tmp_store, universe=syms)

    for sym in syms:
        mgr.place_entry_order(_make_entry(), sym, +1, TRADE_DATE)

    assert len(mgr._resting_entries) == 3

    cancelled_ids = []
    original_cancel = mock_broker.cancel_order
    def track_cancel(oid):
        cancelled_ids.append(oid)
        return original_cancel(oid)

    with patch.object(mock_broker, "cancel_order", side_effect=track_cancel):
        with patch.object(mock_broker, "submit_market_order") as mock_mkt:
            mgr.flatten_all("eod_sweep")
            mock_mkt.assert_not_called()

    assert len(cancelled_ids) == 3
    assert len(mgr._resting_entries) == 0
    assert tmp_store.get_closed_trades(TRADE_DATE) == []


# ── pp-b: late fill → position opened → EOD flattens it ─────────────────────

def test_pp_b_late_fill_position_opened_and_flattened_at_eod(mock_broker, tmp_store):
    """Entry fires 60 s before EOD (fill watcher arrives just before flatten_all).
    After the fill:
      - Position must be tracked as 'open' with bracket orders.
      - flatten_all must cancel the bracket and submit a market exit.
    """
    mgr      = _build_mgr(mock_broker, tmp_store, universe=["TQQQ"])
    order_id = mgr.place_entry_order(_make_entry(shares=10), "TQQQ", +1, TRADE_DATE)
    assert "TQQQ" in mgr._resting_entries

    # IB triggers the stop-limit entry.
    mock_broker.fire_fill_watcher(order_id, qty=10, price=100.05)

    pos = mgr._positions.get("TQQQ")
    assert pos is not None, "position must be tracked after entry fill"
    assert pos.status == "open"
    assert "TQQQ" not in mgr._resting_entries

    # EOD: flatten_all must exit the position.
    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "eod-flat"}) as mock_mkt:
        mgr.flatten_all("eod_sweep")

    mock_mkt.assert_called_once()
    sym, side, qty = mock_mkt.call_args.args[:3]
    assert sym  == "TQQQ"
    assert side == "sell"
    assert qty  == 10


# ── pp-c: risk gate caps concurrent positions at fill time ───────────────────

def test_pp_c_risk_gate_caps_concurrent_fills(mock_broker, tmp_store):
    """10 orders pre-placed; max_concurrent_positions=3.
    All 10 fill simultaneously.  Only 3 must result in open positions.
    The 7 excess fills must be immediately reversed with a sell market order.
    """
    syms = [f"SYM{i}" for i in range(10)]
    mgr  = _build_mgr(
        mock_broker, tmp_store,
        exe_cfg=_exe_cfg(max_concurrent_positions=3),
        universe=syms,
    )

    order_ids = {}
    for sym in syms:
        oid = mgr.place_entry_order(_make_entry(shares=5), sym, +1, TRADE_DATE)
        order_ids[sym] = oid

    reversals: list[tuple] = []
    def track_mkt(sym, side, qty, **kw):
        reversals.append((sym, side, qty))
        return {"id": f"rev-{sym}"}

    with patch.object(mock_broker, "submit_market_order", side_effect=track_mkt):
        for sym in syms:
            mock_broker.fire_fill_watcher(order_ids[sym], qty=5, price=100.0)

    open_syms = [s for s, p in mgr._positions.items() if p.status == "open"]
    assert len(open_syms) == 3, f"expected 3 open positions, got {len(open_syms)}"
    assert len(reversals) == 7, f"expected 7 reversals, got {len(reversals)}"
    for _, side, _ in reversals:
        assert side == "sell", "excess fill reversal must sell"


# ── pp-d: cancel/fill race → EOD safety net catches orphan position ───────────

def test_pp_d_cancel_fill_race_safety_net_flattens_orphan(mock_broker, tmp_store):
    """Race: flatten_all sends cancel for TQQQ, but IB had already filled it.
    The cancel 'wins' locally (_resting_entries is cleared), but the broker
    holds 10 shares with no bracket.  EOD safety net must detect and flatten it.
    """
    mgr      = _build_mgr(mock_broker, tmp_store, universe=["TQQQ"])
    order_id = mgr.place_entry_order(_make_entry(shares=10), "TQQQ", +1, TRADE_DATE)

    # Simulate: our cancel call succeeds, but IB had already filled the order.
    mock_broker.cancel_order(order_id)          # cancel recorded in mock
    mock_broker.set_broker_position("TQQQ", 10) # IB holds shares anyway

    # position_manager has no knowledge of the position (fill watcher never called)
    assert "TQQQ" not in mgr._positions

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "race-flat"}) as mock_mkt:
        mgr.flatten_all("eod_sweep")

    # Safety net (broker position scan in flatten_all) must catch it.
    mock_mkt.assert_called_once()
    sym, side, qty = mock_mkt.call_args.args[:3]
    assert sym  == "TQQQ"
    assert side == "sell"
    assert qty  == 10


# ── pp-e: gross exposure cap across simultaneous fills ───────────────────────

def test_pp_e_gross_exposure_capped_on_simultaneous_fills(mock_broker, tmp_store):
    """20 orders pre-placed at $5,000 notional each = $100k potential.
    max_gross_exposure_pct=0.30 on $100k equity caps at $30k.
    Total open notional after all fills must not exceed the cap.
    """
    syms = [f"SYM{i}" for i in range(20)]
    mock_broker.set_equity(100_000.0)
    mgr = _build_mgr(
        mock_broker, tmp_store,
        exe_cfg=_exe_cfg(max_gross_exposure_pct=0.30, max_concurrent_positions=0),
        universe=syms,
    )

    order_ids = {}
    for sym in syms:
        # 50 shares @ $100 = $5,000 notional
        oid = mgr.place_entry_order(
            _make_entry(entry_price=100.0, shares=50), sym, +1, TRADE_DATE
        )
        order_ids[sym] = oid

    with patch.object(mock_broker, "submit_market_order", return_value={"id": "rev"}):
        for sym in syms:
            mock_broker.fire_fill_watcher(order_ids[sym], qty=50, price=100.0)

    open_positions = {s: p for s, p in mgr._positions.items() if p.status == "open"}
    total_notional = sum(
        p.actual_entry_price * p.entry_shares for p in open_positions.values()
    )
    cap = 100_000.0 * 0.30

    assert total_notional <= cap + 1e-6, (
        f"notional {total_notional:.0f} exceeds cap {cap:.0f}"
    )


# ── pp-f: unfilled order leaves no closed_trade record ───────────────────────

def test_pp_f_unfilled_order_leaves_no_closed_trade_record(mock_broker, tmp_store):
    """A pre-placed order that is cancelled (never filled) at EOD must not
    produce any row in closed_trades.  Prevents $0 phantom trades in the
    session summary.
    """
    mgr = _build_mgr(mock_broker, tmp_store, universe=["TQQQ"])
    mgr.place_entry_order(_make_entry(), "TQQQ", +1, TRADE_DATE)

    mgr.flatten_all("eod_sweep")

    rows = tmp_store.get_closed_trades(TRADE_DATE)
    assert rows == [], f"expected 0 closed_trade rows, got {len(rows)}"


# ── pp-g: partial entry fill → position not opened until complete ─────────────

def test_pp_g_partial_fill_deferred_until_complete(mock_broker, tmp_store):
    """IB partially fills the entry stop-limit (40 of 100 shares).
    Position must NOT be opened on the partial event (bracket would be for
    wrong qty).  When the remaining 60 fill, position opens for 100 shares.
    """
    mgr      = _build_mgr(mock_broker, tmp_store, universe=["TQQQ"])
    order_id = mgr.place_entry_order(_make_entry(shares=100), "TQQQ", +1, TRADE_DATE)

    # Partial fill: cumQty=40, totalQuantity=100.
    mock_broker.fire_fill_watcher(order_id, qty=40, price=100.0, total_qty=100)

    # Position must not be open yet — bracket placement would be for wrong qty.
    pos = mgr._positions.get("TQQQ")
    is_pending = pos is None or pos.status in ("new", "entering", "pending")
    assert is_pending, f"position must not be 'open' on partial fill; status={getattr(pos, 'status', None)}"

    # Full fill: cumQty=100, totalQuantity=100.
    mock_broker.fire_fill_watcher(order_id, qty=100, price=100.0, total_qty=100)

    pos = mgr._positions.get("TQQQ")
    assert pos is not None, "position must be tracked after full fill"
    assert pos.status == "open"
    assert pos.entry_shares == 100


# ── pp-h: cancel failure on individual order does not block other cancels ─────

def test_pp_h_individual_cancel_failure_does_not_block_eod(mock_broker, tmp_store):
    """If cancel_order raises for one resting entry (IB pacing error / timeout),
    flatten_all must continue cancelling the remaining orders and still exit
    any open positions.  One bad cancel must not abort the EOD sweep.
    """
    syms = ["SOXL", "SOXS", "TQQQ"]
    mgr  = _build_mgr(mock_broker, tmp_store, universe=syms)

    order_ids = {}
    for sym in syms:
        order_ids[sym] = mgr.place_entry_order(_make_entry(), sym, +1, TRADE_DATE)

    # SOXL's cancel raises (simulates IB pacing or timeout).
    soxl_oid = order_ids["SOXL"]
    original_cancel = mock_broker.cancel_order
    def flaky_cancel(oid):
        if oid == soxl_oid:
            raise RuntimeError("IB pacing limit hit")
        return original_cancel(oid)

    succeeded = []
    def counting_cancel(oid):
        try:
            result = flaky_cancel(oid)
            succeeded.append(oid)
            return result
        except RuntimeError:
            raise

    with patch.object(mock_broker, "cancel_order", side_effect=counting_cancel):
        # Must NOT raise; must complete the sweep.
        mgr.flatten_all("eod_sweep")

    # The other two orders must have been cancelled despite the failure on SOXL.
    assert len(succeeded) == 2
    # _resting_entries must be cleared even for the failed cancel.
    assert len(mgr._resting_entries) == 0


# ── pp-i: session kill blocks new fills from becoming positions ───────────────

def test_pp_i_session_kill_blocks_resting_entry_fills(mock_broker, tmp_store):
    """If the session kill is triggered before a resting entry fills, the fill
    must be reversed immediately (same as gate rejection) rather than opening
    a position into a killing session.
    """
    mgr      = _build_mgr(
        mock_broker, tmp_store,
        exe_cfg=_exe_cfg(session_kill_loss_pct=0.01),  # very tight kill threshold
        universe=["TQQQ"],
    )
    order_id = mgr.place_entry_order(_make_entry(), "TQQQ", +1, TRADE_DATE)

    # Trigger the session kill before the entry fills.
    mgr._gate.record_realized_pnl(-2_000.0)  # exceeds 1% of $100k
    assert mgr._gate.is_session_killed()

    reversals = []
    def track_mkt(sym, side, qty, **kw):
        reversals.append((sym, side, qty))
        return {"id": "kill-rev"}

    with patch.object(mock_broker, "submit_market_order", side_effect=track_mkt):
        mock_broker.fire_fill_watcher(order_id, qty=10, price=100.0)

    assert "TQQQ" not in mgr._positions, "kill session must not open new positions"
    assert len(reversals) == 1
    assert reversals[0][1] == "sell"


# ── pp-j: EOD flatten closes orphaned broker position from missed fill event ──

def test_pp_j_flatten_closes_orphaned_broker_position(mock_broker, tmp_store):
    """flatten_all must close positions that exist at the broker but were never
    registered in _positions (e.g. fill event lost during a connectivity blip).

    Regression: prior implementation iterated self._universe (empty when universe
    kwarg not passed to LivePositionManager), so get_positions() was never called
    and the orphaned position was left open overnight.
    """
    # No universe kwarg — reproduces production wiring in main.py
    mgr = _build_mgr(mock_broker, tmp_store)

    # Simulate IB holding 100 shares of XRPT that our code never heard about
    mock_broker.set_broker_position("XRPT", 100)

    market_orders: list[tuple] = []
    original = mock_broker.submit_market_order
    def track(sym, side, qty, **kw):
        market_orders.append((sym, side, qty))
        return original(sym, side, qty, **kw)

    mock_broker.submit_market_order = track
    mgr.flatten_all("eod_sweep")

    assert any(sym == "XRPT" and side == "sell" and qty == 100
               for sym, side, qty in market_orders), (
        "flatten_all must close orphaned XRPT position via get_positions()"
    )


# ── pp-k: resting entries are placed for ALL candidates (no placement cap) ────

def test_pp_k_no_placement_buying_power_cap(mock_broker, tmp_store):
    """Resting stop-limits consume no margin until triggered, so place_entry_order
    must place them for every candidate even when combined notional far exceeds
    buying power. Allocation happens at fill time, not placement."""
    mock_broker.set_equity(1_000.0)  # buying_power = 4_000
    mgr = _build_mgr(mock_broker, tmp_store, universe=["A", "B", "C"])

    for sym in ["A", "B", "C"]:
        # $10k notional each — well beyond the $4k buying power
        oid = mgr.place_entry_order(
            _make_entry(entry_price=100.0, shares=100), sym, +1, TRADE_DATE)
        assert oid is not None, f"{sym} resting order must be placed"

    assert len(mgr._resting_entries) == 3


# ── pp-l: fill-time budget gate reverses an overflow fill ─────────────────────

def test_pp_l_fill_time_budget_reverses_overflow(mock_broker, tmp_store):
    """When simultaneous breaks would over-commit, the fill that overflows the
    margin budget is reversed (flattened). Proactive cancel is disabled here to
    isolate the reversal path (i.e. two orders already in flight at IB)."""
    mgr = _build_mgr(mock_broker, tmp_store, universe=["A", "B"])
    mgr._margin_budget = 600.0
    mgr._margin_rate["A"] = 1.0
    mgr._margin_rate["B"] = 1.0

    oid_a = mgr.place_entry_order(_make_entry(entry_price=100.0, shares=6), "A", +1, TRADE_DATE)
    oid_b = mgr.place_entry_order(_make_entry(entry_price=100.0, shares=6), "B", +1, TRADE_DATE)

    # Simulate both already filled at IB before either proactive-cancel ran.
    mgr._cancel_unaffordable_resting = lambda: None

    reversals: list[tuple] = []
    original = mock_broker.submit_market_order
    def track(sym, side, qty, **kw):
        reversals.append((sym, side, qty))
        return original(sym, side, qty, **kw)
    mock_broker.submit_market_order = track

    mock_broker.fire_fill_watcher(oid_a, qty=6, price=100.0)  # 600 ≤ 600 → accepted
    mock_broker.fire_fill_watcher(oid_b, qty=6, price=100.0)  # 1200 > 600 → reversed

    assert mgr._positions["A"].status == "open"
    assert "B" not in mgr._positions
    assert any(sym == "B" and side == "sell" for sym, side, qty in reversals), (
        "overflow fill B must be reversed with a market sell"
    )


# ── pp-m: proactive cancel removes resting orders that no longer fit ──────────

def test_pp_m_proactive_cancel_of_unaffordable_resting(mock_broker, tmp_store):
    """After an accepted fill consumes the budget, resting orders that no longer
    fit are cancelled so they never trigger (avoids fill-then-reverse churn)."""
    mgr = _build_mgr(mock_broker, tmp_store, universe=["A", "B"])
    mgr._margin_budget = 600.0
    mgr._margin_rate["A"] = 1.0
    mgr._margin_rate["B"] = 1.0

    oid_a = mgr.place_entry_order(_make_entry(entry_price=100.0, shares=6), "A", +1, TRADE_DATE)
    oid_b = mgr.place_entry_order(_make_entry(entry_price=100.0, shares=6), "B", +1, TRADE_DATE)

    cancels: list[str] = []
    original = mock_broker.cancel_order
    def track(oid, **kw):
        cancels.append(oid)
        return original(oid, **kw)
    mock_broker.cancel_order = track

    mock_broker.fire_fill_watcher(oid_a, qty=6, price=100.0)  # consumes full budget

    assert mgr._positions["A"].status == "open"
    assert "B" not in mgr._resting_entries, "B must be proactively cancelled"
    assert oid_b in cancels
