"""
tests/test_margin_budget.py — Margin-budget entry gating + order-reject rollback.

Covers the flood-day buying-power management added on top of open_position:
  1. prewarm_margin caches IB's true initial-margin rate and snapshots budget
  2. the margin-budget gate blocks a concurrent entry that would over-commit
  3. an IB order reject (201) rolls back an 'entering' entry immediately
"""

from datetime import date, datetime, timezone
from types import SimpleNamespace

UTC        = timezone.utc
TRADE_DATE = date(2026, 1, 5)


def _exe_cfg(**overrides):
    ns = SimpleNamespace(
        session_kill_loss_pct=0.03,
        max_concurrent_positions=0,
        max_gross_exposure_pct=10.0,   # loosen so the margin gate is what bites
        max_position_pct=5.0,
        entry_slippage_bps=10,
        stop_order_type="market",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _strategy_cfg(**overrides):
    ns = SimpleNamespace(
        eod_exit_hour=16,
        eod_exit_minute=0,
        exit_ratio_tp1=1.0,
        exit_ratio_tp2=0.0,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _build_mgr(mock_broker, tmp_store, exe_cfg=None, strategy_cfg=None):
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager

    exc  = exe_cfg      or _exe_cfg()
    scfg = strategy_cfg or _strategy_cfg()

    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=lambda _: None)
    gate   = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TRADE_DATE)

    return LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, config=scfg,
    )


def _make_entry(entry_price=100.0, shares=6):
    return {
        "entry_price":   entry_price,
        "shares":        shares,
        "orb_range":     1.0,
        "stop_price":    entry_price - 1.0,
        "tp1_price":     entry_price + 2.0,
        "tp2_price":     entry_price + 2.0,
        "tp1_shares":    shares,
        "tp2_shares":    0,
        "tp3_shares":    0,
        "exit_override": {},
    }


# ── 1. prewarm caches rate + budget ──────────────────────────────────────────

def test_prewarm_sets_rate_and_budget(mock_broker, tmp_store):
    mock_broker.set_equity(50_000.0)
    mock_broker._margin_rate = 0.75          # simulate a 3x-ETF requirement
    mgr = _build_mgr(mock_broker, tmp_store)

    mgr.prewarm_margin("FNGU", "buy", 10, 100.0)

    assert abs(mgr._margin_rate["FNGU"] - 0.75) < 1e-9
    assert mgr._margin_budget == 50_000.0     # available_funds snapshot


# ── 2. budget gate blocks a concurrent over-commit ───────────────────────────

def test_margin_budget_blocks_second_entry(mock_broker, tmp_store):
    # Budget = 1000; each entry needs 600 of margin (rate 1.0 × 6 × $100).
    # First entry opens (600 ≤ 1000); second would push committed to 1200.
    mock_broker.set_equity(1_000.0)
    mock_broker._margin_rate = 1.0
    mgr = _build_mgr(mock_broker, tmp_store)

    mgr.prewarm_margin("AAA", "buy", 6, 100.0)
    mgr.prewarm_margin("BBB", "buy", 6, 100.0)

    pos_a = mgr.open_position(_make_entry(shares=6), "AAA", +1, TRADE_DATE)
    pos_b = mgr.open_position(_make_entry(shares=6), "BBB", +1, TRADE_DATE)

    assert pos_a is not None and mgr._positions["AAA"].status == "open"
    assert pos_b is None                       # rejected by margin budget
    assert "BBB" not in mgr._positions


def test_lower_margin_rate_allows_both_entries(mock_broker, tmp_store):
    # Same budget, but 2x-ETF margin (rate 0.4): 2 × 240 = 480 ≤ 1000 → both fit.
    mock_broker.set_equity(1_000.0)
    mock_broker._margin_rate = 0.4
    mgr = _build_mgr(mock_broker, tmp_store)

    mgr.prewarm_margin("AAA", "buy", 6, 100.0)
    mgr.prewarm_margin("BBB", "buy", 6, 100.0)

    pos_a = mgr.open_position(_make_entry(shares=6), "AAA", +1, TRADE_DATE)
    pos_b = mgr.open_position(_make_entry(shares=6), "BBB", +1, TRADE_DATE)

    assert pos_a is not None and pos_b is not None


# ── 3. 201 reject rolls back an 'entering' entry ─────────────────────────────

def test_order_reject_rolls_back_entering_position(mock_broker, tmp_store):
    mgr = _build_mgr(mock_broker, tmp_store)

    # Force the entry to stay pending so status == 'entering'.
    mock_broker.set_pending_fill(5)
    pos = mgr.open_position(_make_entry(shares=6), "AAA", +1, TRADE_DATE)
    assert pos is None                         # pending → not yet open
    assert mgr._positions["AAA"].status == "entering"
    assert "AAA" in mgr._pending_entries

    # IB rejects the entry for insufficient margin (async, after submit).
    mock_broker.fire_order_error(order_id="x", code=201,
                                 message="insufficient margin", symbol="AAA")

    assert "AAA" not in mgr._positions
    assert "AAA" not in mgr._pending_entries
    assert tmp_store.get_open_position("AAA") is None


def test_order_reject_ignores_unknown_symbol(mock_broker, tmp_store):
    mgr = _build_mgr(mock_broker, tmp_store)
    mock_broker.set_pending_fill(5)
    mgr.open_position(_make_entry(shares=6), "AAA", +1, TRADE_DATE)

    # Reject for a symbol we aren't tracking must not disturb AAA.
    mock_broker.fire_order_error(order_id="y", code=201, message="x", symbol="ZZZ")

    assert mgr._positions["AAA"].status == "entering"
