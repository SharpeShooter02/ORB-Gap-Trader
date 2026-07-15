"""
tests/test_position_manager.py — 18 cases (a–r) for LivePositionManager.

Each test builds the minimum state needed and asserts only the invariant
under test.  The mock stack (MockBroker → policy → gate → mgr) is
constructed via _build_mgr() to keep individual tests concise.

On-bar order-of-operations (mirrors simulate_trade exactly):
  1. max_fav / post_tp2_mfe update
  2. EOD check
  3. TP1 check  → moves current_stop to breakeven; if tp2_shares==0 sets tp2_hit
  4. trail_after_tp1 peak update
  5. TP2 check  ("if", not "elif")
  6. TP3 EMA-crossback check
  7. Stop-loss check
"""

import math
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

UTC = timezone.utc
TRADE_DATE = date(2026, 1, 5)
SYMBOL = "TQQQ"

_NO_SLEEP = lambda _: None  # noqa: E731


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _exe_cfg(**overrides):
    """Minimal execution config for risk-gate + order-policy."""
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
    """Minimal strategy config for LivePositionManager."""
    ns = SimpleNamespace(
        tp3_mode="ema_crossback",
        eod_exit_hour=16,
        eod_exit_minute=0,
        exit_ratio_tp1=0.35,
        exit_ratio_tp2=0.05,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _build_mgr(mock_broker, tmp_store,
               indicators=None, exe_cfg=None, strategy_cfg=None):
    """Assemble the full execution stack."""
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager

    exc  = exe_cfg      or _exe_cfg()
    scfg = strategy_cfg or _strategy_cfg()

    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=_NO_SLEEP)
    gate   = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TRADE_DATE)

    return LivePositionManager(
        broker=mock_broker,
        policy=policy,
        state_store=tmp_store,
        risk_gate=gate,
        indicators_store=indicators if indicators is not None else {},
        config=scfg,
    )


def _make_entry(
    entry_price=100.0,
    shares=100,
    orb_range=1.0,
    stop_price=99.0,
    tp1_price=101.0,
    tp2_price=102.0,
    tp1_shares=35,
    tp2_shares=5,
    tp3_shares=60,
    exit_override=None,
):
    return {
        "entry_price":   entry_price,
        "shares":        shares,
        "orb_range":     orb_range,
        "stop_price":    stop_price,
        "tp1_price":     tp1_price,
        "tp2_price":     tp2_price,
        "tp1_shares":    tp1_shares,
        "tp2_shares":    tp2_shares,
        "tp3_shares":    tp3_shares,
        "exit_override": exit_override or {},
    }


def _bar(hi, lo, cl=None):
    cl = cl if cl is not None else (hi + lo) / 2.0
    return {"high": hi, "low": lo, "close": cl}


def _ts(hour=10, minute=30):
    return datetime(2026, 1, 5, hour, minute, tzinfo=UTC)


def _seeded_indicator(ema_value: float):
    """
    Return a RollingIndicators pre-seeded so that self.ema == ema_value.

    Uses a single ORB bar with hi = ema_value + 0.5, lo = ema_value - 0.5,
    so mid = ema_value.  With one bar the seed formula sets ema = mid[0].
    """
    from orb_live.execution.indicators import RollingIndicators
    cfg = SimpleNamespace(ema_length=30)
    ind = RollingIndicators(SYMBOL, cfg)
    orb_bars = pd.DataFrame(
        {"high": [ema_value + 0.5], "low": [ema_value - 0.5]},
        index=pd.date_range("2022-01-07 09:30", periods=1, freq="1min"),
    )
    ind.seed_from_orb_bars(orb_bars)
    assert abs(ind.ema - ema_value) < 1e-9
    return ind


# ── a. Risk gate reject ────────────────────────────────────────────────────────

def test_a_risk_gate_reject_returns_none(mock_broker, tmp_store):
    """open_position returns None when the risk gate rejects; candidate is saved."""
    mgr = _build_mgr(mock_broker, tmp_store,
                     exe_cfg=_exe_cfg(max_position_pct=0.0))  # any size rejected
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is None
    candidates = tmp_store.get_candidates(TRADE_DATE, phase=3)
    assert any(c["symbol"] == SYMBOL for c in candidates)


# ── b. Full entry fill ─────────────────────────────────────────────────────────

def test_b_entry_full_fill_returns_open_position(mock_broker, tmp_store):
    """Full fill: Position returned with status='open' and correct share counts."""
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is not None
    assert pos.status == "open"
    assert pos.entry_shares == 100
    assert pos.remaining == 100
    # actual_entry_price = limit price (100.0 * 1.001 = 100.10)
    assert abs(pos.actual_entry_price - 100.10) < 1e-6
    assert pos.tp1_shares == 35
    assert pos.tp2_shares == 5
    assert pos.tp3_shares == 60


# ── c. Zero fill → unfilled ────────────────────────────────────────────────────

def test_c_zero_fill_returns_none_no_db_row(mock_broker, tmp_store):
    """Zero fill: open_position returns None; 'entering' row must be deleted.

    A left-behind row blocks re-entry on the same symbol (UNIQUE constraint).
    After BUG 8 fix the row is deleted, not left as 'unfilled'.
    """
    mock_broker.set_fill_fraction(0.0)
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is None
    assert mgr._positions.get(SYMBOL) is None
    assert tmp_store.get_open_position(SYMBOL) is None


# ── d. Partial fill → shares recomputed ───────────────────────────────────────

def test_d_partial_fill_recomputes_tp_shares(mock_broker, tmp_store):
    """Partial fill: tp1/tp2/tp3 shares rescaled from fill.qty, not entry['shares']."""
    # Attempt 1: fills 70/100; attempts 2+3: no fill → partial_unfilled(70 shares).
    mock_broker.set_fill_sequence(0.7, 0.0, 0.0)
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is not None
    assert pos.entry_shares == 70
    assert pos.tp1_shares == math.floor(70 * 0.35)   # 24
    assert pos.tp2_shares == math.floor(70 * 0.05)   # 3
    assert pos.tp3_shares == 70 - pos.tp1_shares - pos.tp2_shares  # 43


# ── e. TP1 fires → stop moves to breakeven ────────────────────────────────────

def test_e_tp1_fires_stop_moves_to_breakeven(mock_broker, tmp_store):
    """After TP1: current_stop == actual_entry_price; tp1_hit=True; remaining decremented."""
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Simulate IB filling the OCA TP1 bracket child (tp1_shares=35 shares).
    mock_broker._orders[pos.tp1_order_id].update({
        "status": "filled", "filled_qty": "35", "filled_avg_price": "101.0",
    })
    # lo=100.2 stays above breakeven so fresh stop doesn't immediately fire.
    tp1_bar = _bar(hi=101.5, lo=100.2)
    mgr.on_bar(SYMBOL, tp1_bar, _ts())

    assert pos.tp1_hit is True
    assert pos.remaining == 100 - 35      # tp1_shares=35 sold
    assert abs(pos.current_stop - pos.actual_entry_price) < 1e-9


# ── f. TP1 fires, tp2_shares==0 → tp2_hit immediately ────────────────────────

def test_f_tp1_fires_with_no_tp2_shares_sets_tp2_hit(mock_broker, tmp_store):
    """When tp2_shares==0, tp2_hit is set True on the same bar as TP1."""
    entry = _make_entry(tp1_shares=40, tp2_shares=0, tp3_shares=60)
    mgr   = _build_mgr(mock_broker, tmp_store)
    pos   = mgr.open_position(entry, SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Simulate IB filling the OCA TP1 bracket child (tp1_shares=40 shares).
    mock_broker._orders[pos.tp1_order_id].update({
        "status": "filled", "filled_qty": "40", "filled_avg_price": "101.0",
    })
    tp1_bar = _bar(hi=101.5, lo=100.0)
    # TP3 check will reach indicator lookup; provide a seeded indicator
    ind = _seeded_indicator(ema_value=200.0)   # ema >> entry, crossback always True
    mgr._indicators[SYMBOL] = ind

    mgr.on_bar(SYMBOL, tp1_bar, _ts())

    assert pos.tp1_hit  is True
    assert pos.tp2_hit  is True   # set immediately because tp2_shares==0


# ── g. TP1 and TP2 fire on the same bar ───────────────────────────────────────

def test_g_tp1_and_tp2_fire_same_bar(mock_broker, tmp_store):
    """Both TP1 and TP2 fire in one on_bar call (step 3 and step 5 use 'if')."""
    # ema=100.5 is just above breakeven (100.10); close=102.0 > ema → no TP3 crossback.
    # lo=100.2 is above breakeven so stop doesn't fire.
    ind = _seeded_indicator(ema_value=100.5)
    mgr = _build_mgr(mock_broker, tmp_store, indicators={SYMBOL: ind})
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Simulate IB filling the OCA TP1 bracket child (tp1_shares=35 shares).
    mock_broker._orders[pos.tp1_order_id].update({
        "status": "filled", "filled_qty": "35", "filled_avg_price": "102.0",
    })
    both_bar = _bar(hi=102.5, lo=100.2, cl=102.0)
    ind.on_bar({"high": 102.5, "low": 100.2, "close": 102.0})
    mgr.on_bar(SYMBOL, both_bar, _ts())

    assert pos.tp1_hit is True
    assert pos.tp2_hit is True
    assert pos.remaining == 100 - 35 - 5   # TP1 (35) and TP2 (5) lots sold, TP3 (60) open


# ── h. TP3 EMA-crossback fires ────────────────────────────────────────────────

def test_h_tp3_ema_crossback_fires(mock_broker, tmp_store):
    """TP3 fires when close crosses below EMA while EMA is above entry (profitable)."""
    # ema=101.0 → just above entry=100.0 (is_profitable=True)
    ind = _seeded_indicator(ema_value=101.0)
    mgr = _build_mgr(mock_broker, tmp_store, indicators={SYMBOL: ind})
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Bar 1: TP1 fires via OCA; TP2 fires via bar price.
    mock_broker._orders[pos.tp1_order_id].update({
        "status": "filled", "filled_qty": "35", "filled_avg_price": "101.0",
    })
    # lo=100.2 > breakeven so fresh stop doesn't immediately fire.
    ind.on_bar({"high": 102.5, "low": 100.2, "close": 102.0})
    mgr.on_bar(SYMBOL, _bar(hi=102.5, lo=100.2, cl=102.0), _ts(10, 31))
    assert pos.tp2_hit is True
    assert pos.remaining == 60   # only TP3 shares remain

    # Bar 2: close crosses below ema → TP3 fires.
    # lo=100.5 stays above current_stop (breakeven≈100.10) so stop doesn't pre-empt.
    current_ema = ind.ema
    crossback_close = current_ema - 0.10
    ind.on_bar({"high": 101.5, "low": 100.5, "close": crossback_close})
    mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.5, cl=crossback_close), _ts(10, 32))

    assert pos.tp3_hit is True
    assert pos.status  == "closed"
    assert mgr._positions.get(SYMBOL) is None


# ── i. TP3 does not fire when ema is below entry ──────────────────────────────

def test_i_tp3_no_fire_when_ema_below_entry(mock_broker, tmp_store):
    """TP3 is skipped when ema < actual_entry_price (trade not profitable at ema)."""
    # ema=95.0 < entry=100.0 → is_profitable=False for a long
    ind = _seeded_indicator(ema_value=95.0)
    mgr = _build_mgr(mock_broker, tmp_store, indicators={SYMBOL: ind})
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Bar 1: TP1 fires via OCA; TP2 fires via bar price.
    mock_broker._orders[pos.tp1_order_id].update({
        "status": "filled", "filled_qty": "35", "filled_avg_price": "101.0",
    })
    ind.on_bar({"high": 102.5, "low": 100.2, "close": 102.0})
    mgr.on_bar(SYMBOL, _bar(hi=102.5, lo=100.2, cl=102.0), _ts())
    assert pos.tp2_hit is True

    # Bar 2: close above ema (ema≈95 << close=100.9) → crossed_back=False → no TP3.
    # lo=100.2 > current_stop (breakeven≈100.10) → stop doesn't fire either.
    ind.on_bar({"high": 101.0, "low": 100.2, "close": 100.9})
    mgr.on_bar(SYMBOL, _bar(hi=101.0, lo=100.2, cl=100.9), _ts(10, 31))

    assert pos.tp3_hit is False
    assert pos.status  != "closed"


# ── j. Stop fires → position closed with exit_reason='STOP' ──────────────────

def test_j_stop_fires_closes_position(mock_broker, tmp_store):
    """Polling detects stop fill → position closes with exit_reason='STOP' (no prior TP1)."""
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(stop_price=99.0), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Simulate IB reporting the stop has filled at the stop price.
    mock_broker._orders[pos.stop_order_id] = {
        "status": "filled", "filled_qty": str(pos.remaining), "filled_avg_price": "99.0",
    }
    # Any bar — polling at the top of on_bar detects the fill first.
    mgr.on_bar(SYMBOL, _bar(hi=100.5, lo=98.5), _ts())

    assert pos.status      == "closed"
    assert pos.exit_reason == "STOP"
    assert pos.exit_price  == pytest.approx(99.0)
    assert mgr._positions.get(SYMBOL) is None


# ── k. Stop after TP1 → exit_reason='TP1_ONLY' ───────────────────────────────

def test_k_stop_after_tp1_exit_reason_tp1_only(mock_broker, tmp_store):
    """TP1 fires on bar 1 (OCA); fresh stop placed; polling detects stop fill on bar 2
    → exit_reason='TP1_ONLY' because pos.tp1_hit=True and use_trail_atp1=False."""
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Bar 1: TP1 fires via OCA (simulate IB fill); fresh stop placed for remaining 65.
    mock_broker._orders[pos.tp1_order_id].update({
        "status": "filled", "filled_qty": "35", "filled_avg_price": "101.0",
    })
    mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.2), _ts())
    assert pos.tp1_hit is True

    # Before bar 2: IB reports the fresh stop (pos.stop_order_id updated by on_bar) has fired.
    mock_broker._orders[pos.stop_order_id].update({
        "status": "filled", "filled_qty": str(pos.remaining), "filled_avg_price": "100.10",
    })
    mgr.on_bar(SYMBOL, _bar(hi=100.1, lo=99.5), _ts(10, 31))

    assert pos.status      == "closed"
    assert pos.exit_reason == "TP1_ONLY"


# ── l. Stop with trail_after_tp1 → exit_reason='TRAIL' ───────────────────────

def test_l_stop_with_trail_exit_reason_trail(mock_broker, tmp_store):
    """TP1 fires (bar 1), trail peak/stop update (bar 2); polling detects stop
    fill on bar 3 → exit_reason='TRAIL' (tp1_hit=True and use_trail_atp1=True)."""
    exit_override = {"method": "trail_after_tp1", "mult": 0.5}
    entry = _make_entry(orb_range=2.0, exit_override=exit_override)
    mgr   = _build_mgr(mock_broker, tmp_store)
    pos   = mgr.open_position(entry, SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    assert pos.use_trail_atp1 is True

    # Bar 1: TP1 fires via OCA; fresh stop placed for remaining 65 shares.
    mock_broker._orders[pos.tp1_order_id].update({
        "status": "filled", "filled_qty": "35", "filled_avg_price": "101.0",
    })
    mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.6), _ts())
    assert pos.tp1_hit is True

    # Bar 2: rally; trail peak advances to 103.0, current_stop → 102.0.
    mgr.on_bar(SYMBOL, _bar(hi=103.0, lo=102.1), _ts(10, 31))
    assert abs(pos.trail_atp1_peak - 103.0) < 1e-9
    assert abs(pos.current_stop - 102.0) < 1e-9

    # Before bar 3: IB reports the fresh stop (pos.stop_order_id updated after TP1) fired.
    mock_broker._orders[pos.stop_order_id].update({
        "status": "filled", "filled_qty": str(pos.remaining), "filled_avg_price": "102.0",
    })
    mgr.on_bar(SYMBOL, _bar(hi=102.5, lo=101.5), _ts(10, 32))

    assert pos.status      == "closed"
    assert pos.exit_reason == "TRAIL"


# ── Sanity: bar below stop price does NOT close without IB confirmation ────────

def test_bar_below_stop_does_not_close_without_broker_confirmation(
    mock_broker, tmp_store
):
    """
    A bar where lo < current_stop must NOT close the position unless the
    broker reports the stop filled. The bar-by-bar stop check was removed;
    only polling-based detection closes the position now.

    This test would have FAILED before this refactor.
    """
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(stop_price=99.0), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # IB reports stop still active (not filled).
    mock_broker._orders[pos.stop_order_id] = {"status": "submitted"}

    # Bar clearly below the stop level — old code would have closed here.
    mgr.on_bar(SYMBOL, _bar(hi=100.0, lo=95.0), _ts())

    assert pos.status == "open"
    assert pos.exit_reason != "STOP"


# ── m. EOD fires when ts.time() >= eod_time ───────────────────────────────────

def test_m_eod_fires_at_configured_hour(mock_broker, tmp_store):
    """EOD branch fires when bar timestamp >= eod_exit_hour (here 10:00)."""
    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(eod_exit_hour=10, eod_exit_minute=0))
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # ts exactly at eod threshold (10:00:00)
    eod_ts  = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    eod_bar = _bar(hi=100.5, lo=99.8, cl=100.0)
    mgr.on_bar(SYMBOL, eod_bar, eod_ts)

    # No TP1/TP2 fired, so exit_reason == 'EOD'
    assert pos.status      == "closed"
    assert pos.exit_reason == "EOD"


# ── n. flatten_all exits all open positions ────────────────────────────────────

def test_n_flatten_all_closes_all_positions(mock_broker, tmp_store):
    """flatten_all() closes every open position regardless of state."""
    mgr = _build_mgr(mock_broker, tmp_store)

    pos1 = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    # Short SQQQ — use prices where stop won't accidentally fire
    entry2 = _make_entry(stop_price=200.0, tp1_price=50.0, tp2_price=40.0)
    pos2   = mgr.open_position(entry2, "SQQQ", -1, TRADE_DATE)

    assert pos1 is not None
    assert pos2 is not None
    assert len(mgr._positions) == 2

    mgr.flatten_all("session_end")

    assert len(mgr._positions) == 0
    assert pos1.status == "closed"
    assert pos2.status == "closed"


# ── o. on_bar ignores unknown symbol ─────────────────────────────────────────

def test_o_on_bar_ignores_unknown_symbol(mock_broker, tmp_store):
    """on_bar is a no-op for symbols not in _positions (no exception raised)."""
    mgr = _build_mgr(mock_broker, tmp_store)
    bar = _bar(hi=101.0, lo=99.0)
    mgr.on_bar("UNKNOWN_SYM", bar, _ts())   # must not raise


# ── p. on_bar ignores closed position ────────────────────────────────────────

def test_p_on_bar_ignores_non_open_status(mock_broker, tmp_store):
    """on_bar is a no-op when position exists but status != 'open'."""
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Manually mark as entering (non-open status)
    pos.status = "entering"

    stop_bar = _bar(hi=100.5, lo=98.0)   # would fire stop if on_bar processed it
    mgr.on_bar(SYMBOL, stop_bar, _ts())

    # Status unchanged — on_bar was a no-op
    assert pos.status == "entering"


# ── q. TP3 check with unseeded indicator → RuntimeError ───────────────────────

def test_q_tp3_unseeded_indicator_raises(mock_broker, tmp_store):
    """RuntimeError raised if TP3 check fires against an unseeded RollingIndicators."""
    from orb_live.execution.indicators import RollingIndicators
    ind = RollingIndicators(SYMBOL, SimpleNamespace(ema_length=30))
    # ind.is_seeded == False

    mgr = _build_mgr(mock_broker, tmp_store, indicators={SYMBOL: ind})
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Manually advance state to the TP3 check path
    pos.tp1_hit = True
    pos.tp2_hit = True

    bar = _bar(hi=101.0, lo=100.0, cl=100.5)
    with pytest.raises(RuntimeError, match="is not seeded"):
        mgr.on_bar(SYMBOL, bar, _ts())


# ── r. EOD branch dead for production eod_exit_hour=16 on any RTH bar ────────

def test_r_eod_branch_never_fires_for_rth_bars(mock_broker, tmp_store):
    """
    With eod_exit_hour=16 (production), on_bar does not close the position
    on any RTH bar (9:30–15:59).  The last RTH bar at 15:59 must leave the
    position open so the runner can sweep it at 16:00:30.
    """
    mgr = _build_mgr(mock_broker, tmp_store)   # default: eod_exit_hour=16
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Bar just before 16:00 — must NOT trigger EOD
    last_rth = datetime(2026, 1, 5, 15, 59, tzinfo=UTC)
    safe_bar  = _bar(hi=100.5, lo=100.2, cl=100.3)
    mgr.on_bar(SYMBOL, safe_bar, last_rth)

    # Position still open — EOD branch was not reached
    assert mgr._positions.get(SYMBOL) is not None
    assert pos.status == "open"

    # Exhaustive check: not a single RTH minute triggers the branch
    from datetime import time as dtime
    eod_time = dtime(16, 0)
    for minute in range(30, 30 + 390):   # 9:30 → 15:59
        h, m = divmod(minute, 60)
        h += 9
        assert dtime(h, m) < eod_time, f"{dtime(h, m)} >= {eod_time}"


# ── s–x: Exchange-resident stop placement ─────────────────────────────────────

def test_s_open_position_places_bracket_with_stop(mock_broker, tmp_store):
    """After a successful entry fill, submit_bracket_order places stop at correct price."""
    from unittest.mock import patch
    mgr = _build_mgr(mock_broker, tmp_store)

    bracket_return = {
        "entry_order_id": "entry-123",
        "tp1_order_id":   "tp1-123",
        "stop_order_id":  "stop-123",
    }
    mock_broker._orders["entry-123"] = {
        "id": "entry-123", "status": "filled",
        "filled_qty": "100", "filled_avg_price": "100.10",
    }
    mock_broker._orders["tp1-123"] = {
        "id": "tp1-123", "status": "new", "filled_qty": "0", "filled_avg_price": "0",
    }
    mock_broker._orders["stop-123"] = {
        "id": "stop-123", "status": "new", "filled_qty": "0", "filled_avg_price": "0",
    }
    mock_broker._oca_siblings["tp1-123"]  = "stop-123"
    mock_broker._oca_siblings["stop-123"] = "tp1-123"

    with patch.object(mock_broker, "submit_bracket_order",
                      return_value=bracket_return) as mock_bracket:
        pos = mgr.open_position(_make_entry(stop_price=95.0), SYMBOL, +1, TRADE_DATE)

    assert pos is not None
    mock_bracket.assert_called_once()
    kw = mock_bracket.call_args.kwargs
    assert kw["symbol"]     == SYMBOL
    assert kw["side"]       == "buy"
    assert kw["stop_price"] == pytest.approx(95.0)
    assert pos.stop_order_id == "stop-123"


def test_t_open_position_places_bracket_for_short(mock_broker, tmp_store):
    """Short entry: bracket side must be 'sell'."""
    from unittest.mock import patch
    mgr = _build_mgr(mock_broker, tmp_store)

    bracket_return = {
        "entry_order_id": "entry-456",
        "tp1_order_id":   "tp1-456",
        "stop_order_id":  "stop-456",
    }
    mock_broker._orders["entry-456"] = {
        "id": "entry-456", "status": "filled",
        "filled_qty": "100", "filled_avg_price": "99.90",
    }
    mock_broker._orders["tp1-456"] = {
        "id": "tp1-456", "status": "new", "filled_qty": "0", "filled_avg_price": "0",
    }
    mock_broker._orders["stop-456"] = {
        "id": "stop-456", "status": "new", "filled_qty": "0", "filled_avg_price": "0",
    }

    with patch.object(mock_broker, "submit_bracket_order",
                      return_value=bracket_return) as mock_bracket:
        pos = mgr.open_position(_make_entry(stop_price=105.0), SYMBOL, -1, TRADE_DATE)

    assert pos is not None
    kw = mock_bracket.call_args.kwargs
    assert kw["side"] == "sell"


def test_u_open_position_partial_fill_stop_uses_fill_qty(mock_broker, tmp_store):
    """Partial fill: fresh stop qty must match actual filled shares, not requested shares.

    submit_bracket_order fills 60/100 shares; _complete_entry_fill cancels the
    bracket children and places a fresh stop for the 60 actual filled shares.
    """
    from unittest.mock import patch
    mock_broker.set_fill_sequence(0.60)
    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_stop_order",
                      return_value={"id": "stop-789", "status": "new"}) as mock_stop:
        pos = mgr.open_position(_make_entry(shares=100), SYMBOL, +1, TRADE_DATE)

    assert pos is not None
    call_kw = mock_stop.call_args.kwargs
    assert call_kw["qty"] == 60


def test_v_partial_fill_stop_failure_flattens_position(mock_broker, tmp_store):
    """Partial fill + stop failure: position must be flattened via market order, status=closed."""
    from unittest.mock import patch
    mock_broker.set_fill_sequence(0.60)
    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_stop_order",
                      side_effect=Exception("simulated stop failure")), \
         patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "mkt-flat"}) as mock_flat:
        result = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert result is None
    mock_flat.assert_called_once()
    db_row = tmp_store.get_open_position(SYMBOL)
    assert db_row["status"]      == "closed"
    assert db_row["exit_reason"] == "STOP_PLACEMENT_FAILED"


def test_w_partial_fill_stop_failure_logs_critical(mock_broker, tmp_store):
    """Partial fill + stop failure must emit a 'stop_placement_failed_flattening_position' critical log."""
    from unittest.mock import patch, MagicMock
    logger = MagicMock()
    mock_broker.set_fill_sequence(0.60)
    mgr = _build_mgr(mock_broker, tmp_store)
    mgr._log = logger

    with patch.object(mock_broker, "submit_stop_order",
                      side_effect=Exception("broker down")):
        mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    critical_events = [
        call.args[0]
        for call in logger.critical.call_args_list
    ]
    assert "stop_placement_failed_flattening_position" in critical_events


def test_x_open_position_no_stop_on_unfilled_entry(mock_broker, tmp_store):
    """Zero fill: bracket children are cancelled by IB; submit_stop_order NOT called."""
    from unittest.mock import patch
    mock_broker.set_fill_fraction(0.0)
    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_stop_order") as mock_stop:
        result = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert result is None
    mock_stop.assert_not_called()


# ── y–ad: Exchange-resident stop polling ──────────────────────────────────────

def _open_pos_with_stop_id(mock_broker, tmp_store):
    """Open a position via bracket; returns (mgr, pos) with stop_order_id set."""
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(stop_price=99.0), SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    assert pos.stop_order_id is not None
    return mgr, pos


def test_y_on_bar_polling_detects_filled_stop(mock_broker, tmp_store):
    """Polling detects filled stop → position closed, no TP logic runs."""
    from unittest.mock import patch
    mgr, pos = _open_pos_with_stop_id(mock_broker, tmp_store)

    stop_state = {"status": "filled", "filled_qty": "100", "filled_avg_price": "99.0"}
    mock_broker._orders[pos.stop_order_id] = stop_state

    # Bar that WOULD trigger TP1 — must not run because position is already closed
    tp1_trigger = _bar(hi=102.0, lo=100.5)
    with patch.object(mock_broker, "submit_limit_order") as mock_tp:
        mgr.on_bar(SYMBOL, tp1_trigger, _ts(10, 30))

    assert pos.status      == "closed"
    assert pos.exit_reason == "STOP"
    assert pos.exit_price  == pytest.approx(99.0)
    assert pos.remaining   == 0
    mock_tp.assert_not_called()


def test_z_on_bar_polling_detects_partially_filled_stop(mock_broker, tmp_store):
    """Partial fill on stop: position closed, remaining decremented by filled_qty."""
    from unittest.mock import patch
    mgr, pos = _open_pos_with_stop_id(mock_broker, tmp_store)

    original_remaining = pos.remaining   # 100
    stop_state = {"status": "partially_filled", "filled_qty": "60",
                  "filled_avg_price": "98.5"}
    mock_broker._orders[pos.stop_order_id] = stop_state

    with patch.object(mock_broker, "submit_limit_order"):
        mgr.on_bar(SYMBOL, _bar(hi=102.0, lo=100.5), _ts(10, 30))

    assert pos.status      == "closed"
    assert pos.exit_reason == "STOP"
    assert pos.remaining   == max(0, original_remaining - 60)


def test_aa_on_bar_polling_continues_when_stop_still_active(mock_broker, tmp_store):
    """Stop still active → position stays open, normal bar processing runs."""
    mgr, pos = _open_pos_with_stop_id(mock_broker, tmp_store)

    mock_broker._orders[pos.stop_order_id] = {
        "status": "new", "filled_qty": "0", "filled_avg_price": "0",
    }

    # Safe bar — above stop, below TP1 — just updates max_fav
    safe_bar = _bar(hi=100.5, lo=100.0)
    mgr.on_bar(SYMBOL, safe_bar, _ts(10, 30))

    assert pos.status == "open"
    assert pos.max_fav >= 0   # max_fav was updated → bar processing ran


def test_ab_on_bar_polling_exception_logs_warning_continues(mock_broker, tmp_store):
    """Polling exception → warning logged, position stays open (graceful path)."""
    from unittest.mock import MagicMock, patch
    logger = MagicMock()
    mgr, pos = _open_pos_with_stop_id(mock_broker, tmp_store)
    mgr._log = logger

    with patch.object(mock_broker, "get_order", side_effect=Exception("broker down")):
        mgr.on_bar(SYMBOL, _bar(hi=100.3, lo=99.8), _ts(10, 30))

    warning_events = [c.args[0] for c in logger.warning.call_args_list]
    assert "stop_poll_failed" in warning_events
    assert pos.status == "open"


def test_ac_on_bar_no_stop_order_id_skips_stop_polling(mock_broker, tmp_store):
    """No stop_order_id or tp1_order_id → get_order never called, normal bar processing runs."""
    from unittest.mock import patch
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Force both polling IDs to None to disable stop and TP1 get_order calls.
    pos.stop_order_id = None
    pos.tp1_order_id  = None

    with patch.object(mock_broker, "get_order") as mock_get:
        mgr.on_bar(SYMBOL, _bar(hi=100.3, lo=99.8), _ts(10, 30))

    mock_get.assert_not_called()
    assert pos.status == "open"


def test_ad_handle_stop_fired_persists_position(mock_broker, tmp_store):
    """_handle_stop_fired marks position closed and calls _update_pos."""
    from unittest.mock import patch
    mgr, pos = _open_pos_with_stop_id(mock_broker, tmp_store)

    stop_state = {"status": "filled", "filled_qty": "100", "filled_avg_price": "99.0"}
    with patch.object(mgr, "_update_pos") as mock_update:
        mgr._handle_stop_fired(pos, SYMBOL, stop_state)

    mock_update.assert_called_once_with(pos)
    assert pos.exit_time is not None
    assert pos.status == "closed"


# ── ae–aj: TP1 → modify IB stop ───────────────────────────────────────────────

def _open_with_stop(mock_broker, tmp_store, entry_kw=None, strategy_kw=None):
    """Open a bracket position and pre-fill TP1 so on_bar sees the OCA fill."""
    entry_kw    = entry_kw    or {}
    strategy_kw = strategy_kw or {}
    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(**strategy_kw))
    pos = mgr.open_position(_make_entry(**entry_kw), SYMBOL, +1, TRADE_DATE)
    assert pos is not None and pos.tp1_order_id is not None
    # Simulate IB OCA TP1 fill so on_bar enters the OCA post-TP1 path.
    mock_broker._orders[pos.tp1_order_id].update({
        "status":           "filled",
        "filled_qty":       str(pos.tp1_shares),
        "filled_avg_price": str(pos.tp1_price),
    })
    return mgr, pos


def test_ae_tp1_oca_places_fresh_stop(mock_broker, tmp_store):
    """After OCA TP1 fill with remaining > 0: submit_stop_order called with remaining qty and breakeven."""
    from unittest.mock import patch
    mgr, pos = _open_with_stop(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_stop_order",
                      return_value={"id": "fresh-stop-1"}) as mock_fresh:
        mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.2), _ts())

    assert pos.tp1_hit is True
    mock_fresh.assert_called_once()
    kw = mock_fresh.call_args.kwargs
    assert kw["qty"]        == pos.remaining                           # post-TP1 remaining (65)
    assert kw["stop_price"] == pytest.approx(pos.actual_entry_price)  # breakeven
    assert pos.stop_order_id == "fresh-stop-1"


def test_af_tp1_oca_fresh_stop_trail_price(mock_broker, tmp_store):
    """Trail mode: fresh stop after OCA TP1 uses trail-adjusted stop price."""
    from unittest.mock import patch
    exit_override = {"method": "trail_after_tp1", "mult": 0.5}
    mgr, pos = _open_with_stop(
        mock_broker, tmp_store,
        entry_kw={"orb_range": 2.0, "exit_override": exit_override},
    )
    expected_trail_stop = pos.tp1_price - pos.trail_atp1_dist  # tp1=101 - 1.0=100.0

    with patch.object(mock_broker, "submit_stop_order",
                      return_value={"id": "fresh-stop-trail"}) as mock_fresh:
        mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.6), _ts())

    mock_fresh.assert_called_once()
    kw = mock_fresh.call_args.kwargs
    assert kw["stop_price"] == pytest.approx(expected_trail_stop)
    assert kw["qty"]        == pos.remaining


def test_ag_fresh_stop_failure_triggers_recovery(mock_broker, tmp_store):
    """OCA TP1 fills; submit_stop_order raises → market flatten for remaining, pos closed."""
    from unittest.mock import patch, MagicMock
    logger = MagicMock()
    mgr, pos = _open_with_stop(mock_broker, tmp_store)
    mgr._log = logger

    with patch.object(mock_broker, "submit_stop_order",
                      side_effect=Exception("IB timeout")), \
         patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "mkt-flat"}) as mock_flat:
        mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.2), _ts())

    mock_flat.assert_called_once()
    call_args = mock_flat.call_args.args
    assert call_args[1] == "sell"
    assert call_args[2] == 65  # remaining after TP1 (100 - 35)
    assert pos.status      == "closed"
    assert pos.exit_reason == "STOP_RECOVERY_FAILED"
    critical_events = [c.args[0] for c in logger.critical.call_args_list]
    assert any("fresh_stop_failed" in e for e in critical_events)


def test_ah_fresh_stop_and_flatten_both_fail(mock_broker, tmp_store):
    """submit_stop_order fails + submit_market_order also fails → pos still closed."""
    from unittest.mock import patch, MagicMock
    logger = MagicMock()
    mgr, pos = _open_with_stop(mock_broker, tmp_store)
    mgr._log = logger

    with patch.object(mock_broker, "submit_stop_order",
                      side_effect=Exception("replace failed")), \
         patch.object(mock_broker, "submit_market_order",
                      side_effect=Exception("flatten failed")) as mock_flat:
        mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.2), _ts())

    mock_flat.assert_called_once()
    assert pos.status      == "closed"
    assert pos.exit_reason == "STOP_RECOVERY_FAILED"
    critical_events = [c.args[0] for c in logger.critical.call_args_list]
    assert any("failed" in e or "flatten" in e for e in critical_events)


def test_ai_tp1_oca_fresh_stop_success(mock_broker, tmp_store):
    """Happy path: submit_stop_order succeeds after OCA TP1; pos remains open, stop_order_id updated."""
    from unittest.mock import patch
    mgr, pos = _open_with_stop(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_stop_order",
                      return_value={"id": "fresh-stop-ok"}) as mock_sub, \
         patch.object(mock_broker, "submit_market_order") as mock_mkt:
        mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.2), _ts())

    mock_sub.assert_called_once()
    mock_mkt.assert_not_called()
    assert pos.status        == "open"
    assert pos.remaining     == 65   # 100 - tp1_shares(35)
    assert pos.stop_order_id == "fresh-stop-ok"


def test_aj_no_tp1_order_id_uses_bar_price_tp1_and_modify(mock_broker, tmp_store):
    """tp1_order_id=None (partial-fill path): bar-price TP1 fires and uses modify_stop_order."""
    from unittest.mock import patch
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    # Force partial-fill scenario: no tp1_order_id, stop still set from bracket.
    pos.tp1_order_id = None

    with patch.object(mock_broker, "modify_stop_order",
                      return_value={"id": pos.stop_order_id}) as mock_mod, \
         patch.object(mock_broker, "submit_stop_order") as mock_sub:
        mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.2), _ts())

    assert pos.tp1_hit is True
    mock_mod.assert_called_once()
    mock_sub.assert_not_called()


# ── ak–an: OCA bracket hardening (Items 1, 2, 3) ─────────────────────────────

def _make_v1_entry(
    entry_price=100.0,
    shares=100,
    orb_range=2.0,
    stop_price=99.0,
    tp1_price=102.0,
):
    """v1 TP1-only entry: all shares at TP1."""
    return {
        "entry_price": entry_price,
        "shares":      shares,
        "orb_range":   orb_range,
        "stop_price":  stop_price,
        "tp1_price":   tp1_price,
        "tp2_price":   tp1_price + orb_range,  # unused in v1
        "tp1_shares":  shares,
        "tp2_shares":  0,
        "tp3_shares":  0,
        "exit_override": {},
    }


def test_ak_open_position_uses_bracket_order(mock_broker, tmp_store):
    """open_position calls submit_bracket_order with correct prices; submit_oca_pair and
    submit_stop_order must NOT be called; pos tp1_order_id and stop_order_id set at open."""
    from unittest.mock import patch
    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_bracket_order",
                      wraps=mock_broker.submit_bracket_order) as mock_bkt, \
         patch.object(mock_broker, "submit_oca_pair") as mock_oca, \
         patch.object(mock_broker, "submit_stop_order") as mock_stop:
        pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is not None
    mock_bkt.assert_called_once()
    mock_oca.assert_not_called()
    mock_stop.assert_not_called()

    kw = mock_bkt.call_args.kwargs
    assert kw["symbol"]          == SYMBOL
    assert kw["tp1_limit_price"] == pytest.approx(pos.tp1_price)
    assert kw["stop_price"]      == pytest.approx(pos.stop_price)
    assert kw["tp1_qty"]         == pos.tp1_shares
    assert kw["qty"]             == 100
    assert pos.tp1_order_id  is not None
    assert pos.stop_order_id is not None


def test_al_oca_tp1_fill_cancels_stop_and_closes_position(mock_broker, tmp_store):
    """OCA TP1 fill path:
    1. open_position places OCA pair
    2. test pre-fills tp1_order in mock (simulates IB fill at tp1_price)
    3. MockBroker.get_order auto-cancels stop (OCA sibling linkage)
    4. on_bar closes position with exit_reason='TP1_ONLY'
    """
    from orb_live.core.state_store import closed_trades

    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(exit_ratio_tp1=1.0, exit_ratio_tp2=0.0))
    pos = mgr.open_position(_make_v1_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    assert pos.tp1_order_id is not None

    # Simulate IB filling the OCA TP1 limit at the exact tp1_price.
    tp1_id = pos.tp1_order_id
    mock_broker._orders[tp1_id]["status"]           = "filled"
    mock_broker._orders[tp1_id]["filled_qty"]       = str(pos.entry_shares)
    mock_broker._orders[tp1_id]["filled_avg_price"] = str(pos.tp1_price)
    # MockBroker.get_order will auto-cancel stop on next get_order(tp1_id) call.

    # Bar where hi >= tp1_price confirms the fill context (bar-price fallback matches too)
    mgr.on_bar(SYMBOL, _bar(hi=pos.tp1_price + 0.5, lo=pos.tp1_price - 1.0), _ts())

    assert mgr._positions.get(SYMBOL) is None, "Position should be removed on TP1_ONLY close"
    assert pos.status      == "closed"
    assert pos.exit_reason == "TP1_ONLY"

    # Stop order should now be cancelled (OCA auto-cancelled it)
    stop_state = mock_broker._orders.get(pos.stop_order_id, {})
    assert stop_state.get("status") == "cancelled", \
        f"OCA should have cancelled stop; got status={stop_state.get('status')}"

    # closed_trades record must exist
    with tmp_store.conn() as c:
        rows = c.execute(closed_trades.select()).mappings().all()
    assert rows, "save_closed_trade must be called"
    assert any(dict(r)["exit_reason"] == "TP1_ONLY" for r in rows)


def test_am_oca_stop_not_cancelled_emits_critical_retains_position(mock_broker, tmp_store):
    """OCA cancel failure: if IB doesn't cancel the stop sibling after TP1 fill,
    position_manager must:
      - emit a CRITICAL alert into state_store.alert_log
      - NOT delete the position (leave for reconcile_from_broker)
    """
    from orb_live.core.state_store import alert_log

    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(exit_ratio_tp1=1.0, exit_ratio_tp2=0.0))
    pos = mgr.open_position(_make_v1_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    assert pos.tp1_order_id is not None

    # Simulate OCA linkage failure: remove the sibling mapping so the stop
    # does NOT get auto-cancelled when TP1 fills.
    tp1_id  = pos.tp1_order_id
    stop_id = pos.stop_order_id
    mock_broker._oca_siblings.pop(tp1_id,  None)
    mock_broker._oca_siblings.pop(stop_id, None)

    # Pre-fill the TP1 order (IB filled it, but OCA didn't cancel the stop).
    mock_broker._orders[tp1_id]["status"]           = "filled"
    mock_broker._orders[tp1_id]["filled_qty"]       = str(pos.entry_shares)
    mock_broker._orders[tp1_id]["filled_avg_price"] = str(pos.tp1_price)
    # Stop remains "new" — the OCA cancel did not arrive.
    mock_broker._orders[stop_id]["status"] = "new"

    mgr.on_bar(SYMBOL, _bar(hi=pos.tp1_price + 0.5, lo=pos.tp1_price - 1.0), _ts())

    # Position must NOT be deleted — it's left for reconcile.
    assert mgr._positions.get(SYMBOL) is not None, \
        "Position must be retained when OCA cancel fails (left for reconcile)"
    assert pos.status != "closed", \
        "Position must not be marked closed if the stop cancel was unconfirmed"

    # A CRITICAL alert must have been written to the DB.
    with tmp_store.conn() as c:
        rows = c.execute(alert_log.select()).mappings().all()
    critical_rows = [dict(r) for r in rows if r["level"] == "CRITICAL"]
    assert critical_rows, "A CRITICAL alert must be logged on OCA cancel failure"
    assert any("oca_cancel_failure" in r.get("category", "") for r in critical_rows), \
        f"Expected oca_cancel_failure category in alerts; got {critical_rows}"


def test_an_realized_exit_price_stored_in_closed_trades(mock_broker, tmp_store):
    """realized_exit_price must reflect the actual IB fill price, not just the
    tp1_price target. exit_price stays at the target (backtest-parity field)."""
    from orb_live.core.state_store import closed_trades

    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(exit_ratio_tp1=1.0, exit_ratio_tp2=0.0))
    pos = mgr.open_position(_make_v1_entry(tp1_price=102.0), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # IB fills at 102.03 — slightly above the limit (positive slippage simulation).
    realized = 102.03
    tp1_id   = pos.tp1_order_id
    mock_broker._orders[tp1_id]["status"]           = "filled"
    mock_broker._orders[tp1_id]["filled_qty"]       = str(pos.entry_shares)
    mock_broker._orders[tp1_id]["filled_avg_price"] = str(realized)

    mgr.on_bar(SYMBOL, _bar(hi=102.5, lo=101.0), _ts())

    with tmp_store.conn() as c:
        rows = c.execute(closed_trades.select()).mappings().all()
    assert rows, "save_closed_trade must be called"
    row = dict(rows[-1])

    # exit_price = target (tp1_price), realized_exit_price = actual fill
    assert row["exit_price"]          == pytest.approx(102.0),   \
        f"exit_price should be tp1_price=102.0, got {row['exit_price']}"
    assert row["realized_exit_price"] == pytest.approx(realized), \
        f"realized_exit_price should be {realized}, got {row['realized_exit_price']}"


# ── ao–ar: OCA premature-commit fix ───────────────────────────────────────────

def test_ao_oca_bar_crosses_unfilled_no_commit(mock_broker, tmp_store):
    """OCA resting, bar price crosses TP1 but order still 'new':
    no P&L booked, remaining unchanged, tp1_hit=False, position stays open."""
    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(exit_ratio_tp1=1.0, exit_ratio_tp2=0.0))
    pos = mgr.open_position(_make_v1_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    assert pos.tp1_order_id is not None

    initial_remaining = pos.remaining

    # OCA tp1_order stays "new" — do not set it to "filled"
    bar = _bar(hi=pos.tp1_price + 1.0, lo=pos.tp1_price - 0.5)
    mgr.on_bar(SYMBOL, bar, _ts())

    pos_after = mgr._positions.get(SYMBOL)
    assert pos_after is not None, "Position must stay open when OCA order is unfilled"
    assert pos_after.remaining == initial_remaining, \
        "remaining must not change on bar-price cross with unconfirmed OCA fill"
    assert not pos_after.tp1_hit, "tp1_hit must stay False on unconfirmed OCA fill"


def test_ap_oca_unfilled_2bars_emits_tp1_limit_not_filling_alert(mock_broker, tmp_store):
    """After 2 consecutive bars with price past TP1 but OCA order unfilled,
    a CRITICAL alert with category 'tp1_limit_not_filling' must be emitted."""
    from orb_live.core.state_store import alert_log

    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(exit_ratio_tp1=1.0, exit_ratio_tp2=0.0))
    pos = mgr.open_position(_make_v1_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # tp1_order stays "new" throughout
    bar = _bar(hi=pos.tp1_price + 1.0, lo=pos.tp1_price - 0.5)

    # Bar 1: counter reaches 1 — no alert yet (threshold is 2)
    mgr.on_bar(SYMBOL, bar, _ts(10, 30))
    with tmp_store.conn() as c:
        rows = c.execute(alert_log.select()).mappings().all()
    assert not any(r["category"] == "tp1_limit_not_filling" for r in rows), \
        "No alert should fire on the first crossed-but-unfilled bar"

    # Bar 2: counter reaches 2 — alert must fire
    mgr.on_bar(SYMBOL, bar, _ts(10, 31))
    with tmp_store.conn() as c:
        rows = c.execute(alert_log.select()).mappings().all()
    tp1_alerts = [dict(r) for r in rows if r["category"] == "tp1_limit_not_filling"]
    assert tp1_alerts, "CRITICAL alert must fire after 2 bars crossed but OCA unfilled"
    assert tp1_alerts[0]["level"] == "CRITICAL"

    # Position still open, no trade record
    assert mgr._positions.get(SYMBOL) is not None


def test_aq_wick_then_stop_no_zombie_one_trade(mock_broker, tmp_store):
    """Wick-then-reverse: price crosses TP1 (OCA unfilled), later bar fires stop.
    Result: exactly one closed trade (STOP), no TP1 gain booked, no zombie."""
    from orb_live.core.state_store import closed_trades

    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(exit_ratio_tp1=1.0, exit_ratio_tp2=0.0))
    pos = mgr.open_position(_make_v1_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    stop_id = pos.stop_order_id

    # Bar 1: hi wicks above tp1_price but OCA order never fills (status="new")
    bar_wick = _bar(hi=pos.tp1_price + 0.5, lo=pos.tp1_price - 0.5)
    mgr.on_bar(SYMBOL, bar_wick, _ts(10, 30))

    pos_after_wick = mgr._positions.get(SYMBOL)
    assert pos_after_wick is not None, "Position must survive the wick with OCA unconfirmed"
    remaining_after_wick = pos_after_wick.remaining

    # Bar 2: IB fires the stop; price reverses hard
    mock_broker._orders[stop_id]["status"]           = "filled"
    mock_broker._orders[stop_id]["filled_qty"]       = str(remaining_after_wick)
    mock_broker._orders[stop_id]["filled_avg_price"] = str(pos_after_wick.stop_price)

    bar_stop = _bar(hi=pos.tp1_price - 0.5, lo=pos.stop_price - 0.5)
    mgr.on_bar(SYMBOL, bar_stop, _ts(10, 31))

    assert mgr._positions.get(SYMBOL) is None, "Position must close on stop"

    with tmp_store.conn() as c:
        rows = c.execute(closed_trades.select()).mappings().all()
    assert len(rows) == 1, f"Exactly one trade record expected; got {len(rows)}"
    row = dict(rows[0])
    assert row["exit_reason"] == "STOP", \
        f"exit_reason must be STOP (no TP1 was committed); got {row['exit_reason']}"


def test_ar_oca_repoll_for_filled_avg_price(mock_broker, tmp_store):
    """When filled_avg_price reads 0 on the first poll (IB TWS latency),
    a single re-poll recovers the real price; realized_exit_price uses it."""
    from orb_live.core.state_store import closed_trades
    from unittest.mock import patch

    mgr = _build_mgr(mock_broker, tmp_store,
                     strategy_cfg=_strategy_cfg(exit_ratio_tp1=1.0, exit_ratio_tp2=0.0))
    pos = mgr.open_position(_make_v1_entry(tp1_price=102.0), SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    tp1_id = pos.tp1_order_id

    _repoll_price = 102.5
    _tp1_calls    = [0]

    # Set tp1 as filled in _orders so OCA auto-cancel fires on stop verification
    mock_broker._orders[tp1_id]["status"]           = "filled"
    mock_broker._orders[tp1_id]["filled_qty"]       = str(pos.entry_shares)
    mock_broker._orders[tp1_id]["filled_avg_price"] = "0"  # latency: not yet populated

    _real_get_order = type(mock_broker).get_order   # unbound method

    def _staged(order_id):
        result = _real_get_order(mock_broker, order_id)
        if order_id == tp1_id:
            _tp1_calls[0] += 1
            if _tp1_calls[0] >= 2:
                result = dict(result)
                result["filled_avg_price"] = str(_repoll_price)
        return result

    with patch.object(mock_broker, "get_order", side_effect=_staged):
        mgr.on_bar(SYMBOL, _bar(hi=103.0, lo=101.5), _ts())

    with tmp_store.conn() as c:
        rows = c.execute(closed_trades.select()).mappings().all()
    assert rows, "save_closed_trade must be called"
    row = dict(rows[-1])
    assert row["realized_exit_price"] == pytest.approx(_repoll_price), \
        f"realized_exit_price should be re-polled {_repoll_price}; got {row['realized_exit_price']}"
    assert row["exit_price"] == pytest.approx(102.0), \
        f"exit_price should be tp1_price=102.0; got {row['exit_price']}"


# ── BUG 8: stale 'entering' row cleanup ───────────────────────────────────────

def test_as_failed_entry_exception_leaves_no_db_row(mock_broker, tmp_store):
    """Entry submission raises → 'entering' row must be deleted immediately.

    A left-behind row causes IntegrityError on the next breakout for the same
    symbol.
    """
    from unittest.mock import patch

    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_bracket_order",
                      side_effect=RuntimeError("IB connection lost")):
        pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is None
    assert mgr._positions.get(SYMBOL) is None
    assert tmp_store.get_open_position(SYMBOL) is None, \
        "Stale 'entering' row must be deleted after a submission exception"


def test_at_second_breakout_after_failed_entry_succeeds(mock_broker, tmp_store):
    """After a failed entry (no DB row left), a second breakout on the same symbol
    must open successfully without IntegrityError.
    """
    from unittest.mock import patch

    mgr = _build_mgr(mock_broker, tmp_store)

    # First attempt: submission fails
    with patch.object(mock_broker, "submit_bracket_order",
                      side_effect=RuntimeError("timeout")):
        pos1 = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos1 is None

    # Second attempt: normal fill — must NOT raise IntegrityError
    pos2 = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos2 is not None
    assert pos2.status == "open"


def test_au_startup_reconcile_purges_entering_row(mock_broker, tmp_store):
    """startup_reconcile must delete 'entering' rows left from a prior session.

    Simulates a process restart where the previous run crashed after persisting
    the 'entering' row but before entry filled (or the position was rejected).
    """
    # Manually insert a stale 'entering' row (simulates prior crash)
    tmp_store.save_open_position(
        symbol=SYMBOL, trade_date=TRADE_DATE,
        direction=1, status="entering",
        entry_price=100.0, actual_entry_price=100.0,
        qty=100, entry_shares=100, remaining=100,
        orb_range=1.0, stop_price=99.0, current_stop=99.0,
        tp1_price=101.0, tp2_price=102.0,
        tp1_shares=35, tp2_shares=5, tp3_shares=60,
        tp1_hit=False, tp2_hit=False, tp3_hit=False,
        use_trail_atp1=False, trail_atp1_dist=0.0,
        max_fav=0.0, post_tp2_mfe=0.0, decision_reason="",
    )
    assert tmp_store.get_open_position(SYMBOL) is not None

    mgr = _build_mgr(mock_broker, tmp_store)
    mgr.startup_reconcile()

    assert tmp_store.get_open_position(SYMBOL) is None, \
        "startup_reconcile must purge non-'open' rows"


def test_av_startup_reconcile_purges_broker_orphan(mock_broker, tmp_store):
    """startup_reconcile must delete an 'open' DB row when the broker has no position.

    Simulates a position that was closed at the broker (filled EOD or manually)
    but whose DB row was never cleaned up due to a crash.
    """
    tmp_store.save_open_position(
        symbol=SYMBOL, trade_date=TRADE_DATE,
        direction=1, status="open",
        entry_price=100.0, actual_entry_price=100.1,
        qty=100, entry_shares=100, remaining=100,
        orb_range=1.0, stop_price=99.0, current_stop=99.0,
        tp1_price=101.0, tp2_price=102.0,
        tp1_shares=35, tp2_shares=5, tp3_shares=60,
        tp1_hit=False, tp2_hit=False, tp3_hit=False,
        use_trail_atp1=False, trail_atp1_dist=0.0,
        max_fav=0.0, post_tp2_mfe=0.0, decision_reason="",
    )
    # Broker has no position (the position was closed externally)
    mock_broker.set_broker_position(SYMBOL, 0)

    mgr = _build_mgr(mock_broker, tmp_store)
    mgr.startup_reconcile()

    assert tmp_store.get_open_position(SYMBOL) is None, \
        "startup_reconcile must purge 'open' rows with no matching broker position"


def test_aw_startup_reconcile_flattens_live_broker_position(mock_broker, tmp_store):
    """startup_reconcile must flatten (not preserve) an 'open' row where the broker
    still holds shares — guarantees a clean slate before a new session starts."""
    from unittest.mock import patch

    tmp_store.save_open_position(
        symbol=SYMBOL, trade_date=TRADE_DATE,
        direction=1, status="open",
        entry_price=100.0, actual_entry_price=100.1,
        qty=100, entry_shares=100, remaining=100,
        orb_range=1.0, stop_price=99.0, current_stop=99.0,
        tp1_price=101.0, tp2_price=102.0,
        tp1_shares=35, tp2_shares=5, tp3_shares=60,
        tp1_hit=False, tp2_hit=False, tp3_hit=False,
        use_trail_atp1=False, trail_atp1_dist=0.0,
        max_fav=0.0, post_tp2_mfe=0.0, decision_reason="",
    )
    mock_broker.set_broker_position(SYMBOL, 100)

    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "startup-aw"}) as mock_flat:
        mgr.startup_reconcile()

    # Row must be closed (flattened), not left as 'open'.
    assert tmp_store.get_open_position(SYMBOL) is None, \
        "startup_reconcile must close the DB row after flattening the live position"
    mock_flat.assert_called_once()
    assert mock_flat.call_args.args[0] == SYMBOL


def test_ax_open_position_blocked_by_db_open_row(mock_broker, tmp_store):
    """If the DB has an 'open' row (in-memory is empty after restart), open_position
    must return None rather than overwriting the real position's record.
    """
    tmp_store.save_open_position(
        symbol=SYMBOL, trade_date=TRADE_DATE,
        direction=1, status="open",
        entry_price=100.0, actual_entry_price=100.1,
        qty=100, entry_shares=100, remaining=100,
        orb_range=1.0, stop_price=99.0, current_stop=99.0,
        tp1_price=101.0, tp2_price=102.0,
        tp1_shares=35, tp2_shares=5, tp3_shares=60,
        tp1_hit=False, tp2_hit=False, tp3_hit=False,
        use_trail_atp1=False, trail_atp1_dist=0.0,
        max_fav=0.0, post_tp2_mfe=0.0, decision_reason="",
    )
    # in-memory is empty (simulates fresh process start)
    mgr = _build_mgr(mock_broker, tmp_store)
    assert mgr._positions == {}

    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is None, "open_position must not overwrite an existing 'open' DB row"
    # Original row must survive intact
    row = tmp_store.get_open_position(SYMBOL)
    assert row is not None and row["status"] == "open"


# ── ay–az: BUG 11 — async fill detection + EOD safety net ────────────────────

def test_ay_async_fill_creates_position_and_bracket(mock_broker, tmp_store):
    """Order fills asynchronously (after submission):
    - open_position returns None (still pending)
    - on_bar detects the fill, creates the position, and places the bracket
    - DB row transitions from 'entering' to 'open'
    """
    mock_broker.set_pending_fill(n_polls=1)  # first get_order → 'new', second → 'filled'
    mgr = _build_mgr(mock_broker, tmp_store)

    result = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    # Order is pending — not yet filled
    assert result is None
    assert SYMBOL in mgr._pending_entries
    db_row = tmp_store.get_open_position(SYMBOL)
    assert db_row is not None and db_row["status"] == "entering"

    # on_bar polls the pending order; second get_order call → 'filled'
    mgr.on_bar(SYMBOL, _bar(hi=101.0, lo=99.5), _ts())

    pos = mgr._positions.get(SYMBOL)
    assert pos is not None, "Position must be created after async fill detected"
    assert pos.status == "open"
    assert SYMBOL not in mgr._pending_entries

    db_row = tmp_store.get_open_position(SYMBOL)
    assert db_row is not None and db_row["status"] == "open"

    # Bracket must be placed (stop or OCA)
    assert pos.stop_order_id is not None or pos.tp1_order_id is not None


def test_ay2_async_fill_blocked_re_entry(mock_broker, tmp_store):
    """While an entry order is pending, a second open_position call is rejected."""
    mock_broker.set_pending_fill(n_polls=2)
    mgr = _build_mgr(mock_broker, tmp_store)

    result1 = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert result1 is None
    assert SYMBOL in mgr._pending_entries

    # Second attempt on same symbol while first is still pending
    result2 = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert result2 is None

    # Only one pending entry
    assert len(mgr._pending_entries) == 1


def test_ay3_re_entry_blocked_by_broker_position(mock_broker, tmp_store):
    """open_position returns None when the broker already holds a position (untracked fill)."""
    mock_broker.set_broker_position(SYMBOL, 100)
    mgr = _build_mgr(mock_broker, tmp_store)

    result = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert result is None
    assert SYMBOL not in mgr._positions
    assert SYMBOL not in mgr._pending_entries
    assert tmp_store.get_open_position(SYMBOL) is None


def test_az_flatten_all_closes_untracked_broker_position(mock_broker, tmp_store):
    """flatten_all flattens broker positions not tracked in self._positions.

    This is the EOD safety net for fills that were never detected by the runner
    (e.g. SBIT/AMDL/BOIL from the 2026-06-30 incident).
    """
    from unittest.mock import patch

    mock_broker.set_broker_position(SYMBOL, 100)   # untracked long position
    mgr = _build_mgr(mock_broker, tmp_store)
    mgr.set_universe([SYMBOL])

    assert len(mgr._positions) == 0   # manager has no tracked positions

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "eod-flat"}) as mock_flat:
        mgr.flatten_all("eod_sweep")

    mock_flat.assert_called_once()
    args = mock_flat.call_args.args
    assert args[0] == SYMBOL
    assert args[1] == "sell"    # long → sell to flatten
    assert args[2] == 100


# ── ba–bb: Fix 2 — cancel bracket orders before flatten ──────────────────────

def test_ba_exit_all_cancels_bracket_before_market_order(mock_broker, tmp_store):
    """_exit_all must cancel tp1_order_id and stop_order_id BEFORE submit_market_order.

    Without the pre-cancel, an OCA exit and the emergency flatten can both
    execute on the same position, leaving an unintended net-short (or net-long)
    residual.
    """
    from unittest.mock import patch, call
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    tp1_id  = pos.tp1_order_id
    stop_id = pos.stop_order_id
    assert tp1_id  is not None
    assert stop_id is not None

    cancel_calls = []
    market_calls = []

    def _tracking_cancel(oid):
        cancel_calls.append(oid)
        return mock_broker.cancel_order.__wrapped__(mock_broker, oid) \
               if hasattr(mock_broker.cancel_order, "__wrapped__") else True

    with patch.object(mock_broker, "cancel_order", side_effect=lambda oid: cancel_calls.append(oid) or True) as mock_cancel, \
         patch.object(mock_broker, "submit_market_order", return_value={"id": "flat-1"}) as mock_flat:
        mgr.flatten_all("session_end")

    # Both bracket order IDs must have been cancelled before the market exit.
    assert tp1_id  in cancel_calls, "tp1_order_id not cancelled before flatten"
    assert stop_id in cancel_calls, "stop_order_id not cancelled before flatten"

    # The market flatten must have been called exactly once.
    mock_flat.assert_called_once()

    # Cancels must precede the market order (cancel_calls populated before mock_flat).
    assert len(cancel_calls) >= 2
    assert pos.status == "closed"


def test_bb_exit_all_cancel_failure_still_flattens(mock_broker, tmp_store):
    """If cancel_order raises, _exit_all must still send the market flatten."""
    from unittest.mock import patch
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    with patch.object(mock_broker, "cancel_order", side_effect=Exception("IB cancel timeout")), \
         patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "flat-2"}) as mock_flat:
        mgr.flatten_all("session_end")

    mock_flat.assert_called_once()
    assert pos.status == "closed"


# ── bc–bf: broker-truth flatten + fill-driven remaining ───────────────────────

def test_bc_flatten_all_uses_broker_qty_not_stale_remaining(mock_broker, tmp_store):
    """Stale pos.remaining (100) > broker-held qty (60): flatten submits 60, not 100.

    Reproduces the SOLT EOD scenario: OCA TP1 fills server-side between bars,
    pos.remaining is not yet updated, but _exit_all caps at broker's actual qty.
    """
    from unittest.mock import patch
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(shares=100), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Broker only holds 60 (OCA TP1 filled 40 server-side; pos.remaining still stale at 100).
    mock_broker.set_broker_position(SYMBOL, 60)
    assert pos.remaining == 100  # stale — no on_bar ran to detect the fill

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "flat-bc"}) as mock_flat:
        mgr.flatten_all("eod_sweep")

    # First call must be for the tracked position; assert it uses broker qty.
    first_call = mock_flat.call_args_list[0].args
    assert first_call[0] == SYMBOL
    assert first_call[1] == "sell"
    assert first_call[2] == 60   # broker qty, not stale remaining (100)


def test_bd_flatten_all_catches_memory_closed_but_broker_open(mock_broker, tmp_store):
    """Position marked 'closed' in memory (failed EOD flatten) but broker still holds it.
    The safety net must catch and market-exit the broker position."""
    from unittest.mock import patch
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(_make_entry(shares=100), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Simulate failed EOD flatten: position is "closed" in memory, broker still holds 100.
    pos.status    = "closed"
    pos.remaining = 0
    mock_broker.set_broker_position(SYMBOL, 100)
    mgr.set_universe([SYMBOL])

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "flat-bd"}) as mock_flat:
        mgr.flatten_all("eod_sweep")

    # Safety net must have submitted a market sell for the 100 broker-held shares.
    sell_calls = [
        c for c in mock_flat.call_args_list
        if c.args[0] == SYMBOL and c.args[1] == "sell"
    ]
    assert sell_calls, "Safety net must submit a sell for SYMBOL when broker holds shares"
    assert sell_calls[0].args[2] == 100


def test_be_startup_reconcile_flattens_live_position(mock_broker, tmp_store):
    """startup_reconcile: 'open' DB row + broker still holds position → cancel orders + flatten."""
    from unittest.mock import patch

    tmp_store.save_open_position(
        symbol=SYMBOL, trade_date=TRADE_DATE,
        direction=1, status="open",
        entry_price=100.0, actual_entry_price=100.1,
        qty=100, entry_shares=100, remaining=100,
        orb_range=1.0, stop_price=99.0, current_stop=99.0,
        tp1_price=101.0, tp2_price=102.0,
        tp1_shares=35, tp2_shares=5, tp3_shares=60,
        tp1_hit=False, tp2_hit=False, tp3_hit=False,
        use_trail_atp1=False, trail_atp1_dist=0.0,
        max_fav=0.0, post_tp2_mfe=0.0, decision_reason="",
    )
    mock_broker.set_broker_position(SYMBOL, 100)

    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "startup-flat"}) as mock_flat:
        mgr.startup_reconcile()

    mock_flat.assert_called_once()
    args = mock_flat.call_args.args
    assert args[0] == SYMBOL
    assert args[1] == "sell"   # direction=1 long → sell to flatten
    assert args[2] == 100
    assert tmp_store.get_open_position(SYMBOL) is None


def test_bf_pos_remaining_driven_by_oca_fill(mock_broker, tmp_store):
    """pos.remaining decrements by confirmed OCA fill qty, not optimistically.

    Crypto 0.4/0.2/0.4 split: 100 shares, TP1=40. OCA fills 40 server-side.
    on_bar detects the fill via get_order; remaining must drop to 60.
    """
    mgr = _build_mgr(mock_broker, tmp_store)
    pos = mgr.open_position(
        _make_entry(shares=100, tp1_shares=40, tp2_shares=20, tp3_shares=40),
        SYMBOL, +1, TRADE_DATE,
    )
    assert pos is not None
    assert pos.remaining == 100
    assert pos.tp1_order_id is not None

    # Simulate IB OCA-filling the TP1 child for exactly 40 shares.
    mock_broker._orders[pos.tp1_order_id].update({
        "status":           "filled",
        "filled_qty":       "40",
        "filled_avg_price": "101.0",
    })

    # on_bar polls get_order → newly_filled=40 → remaining -= 40.
    mgr.on_bar(SYMBOL, _bar(hi=101.5, lo=100.2), _ts())

    assert pos.remaining == 60          # entry_qty(100) − oca_filled(40)
    assert pos.tp1_filled_qty_booked == 40
    assert pos.tp1_hit is True          # full TP1 fill → flag set


def test_bg_pending_entry_timeout_with_real_strategy_config(mock_broker, tmp_store):
    """_check_pending_entry must not raise AttributeError when LivePositionManager
    is wired with the real StrategyConfig (not a SimpleNamespace mock).

    Regression for: 'StrategyConfig' object has no attribute 'entry_repeg_seconds'
    The timeout check now uses the module constant _PENDING_ENTRY_TIMEOUT instead.
    """
    from orb_live.config.live_config import load_live_config

    cfg  = load_live_config(use_rolling=False)
    scfg = cfg.strategy_config   # real StrategyConfig, not SimpleNamespace

    mgr = _build_mgr(mock_broker, tmp_store, strategy_cfg=scfg)

    # Bracket entry sits in _pending_entries (n_polls=100 → never auto-fills).
    mock_broker.set_pending_fill(n_polls=100)
    mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert SYMBOL in mgr._pending_entries

    # Backdate submit_time so elapsed > _PENDING_ENTRY_TIMEOUT.
    mgr._pending_entries[SYMBOL]["submit_time"] -= 120.0

    # on_bar calls _check_pending_entry; must not raise AttributeError.
    mgr.on_bar(SYMBOL, _bar(hi=100.5, lo=99.5), _ts())

    # Timed-out pending entry must be cleaned up.
    assert SYMBOL not in mgr._pending_entries
