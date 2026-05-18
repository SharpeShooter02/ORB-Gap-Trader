"""
tests/test_position_manager.py — 18 cases (a–r) for LivePositionManager.

Each test builds the minimum state needed and asserts only the invariant
under test.  The mock stack (MockAlpaca → policy → gate → mgr) is
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
        entry_repeg_seconds=60.0,
        entry_repeg_max_attempts=3,
        entry_slippage_max_bps=30,
        exit_slippage_bps=5,
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


def _build_mgr(mock_alpaca, tmp_store,
               indicators=None, exe_cfg=None, strategy_cfg=None):
    """Assemble the full execution stack."""
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager

    exc  = exe_cfg      or _exe_cfg()
    scfg = strategy_cfg or _strategy_cfg()

    policy = MarketableLimitPolicy(mock_alpaca, exc, tmp_store, _sleep=_NO_SLEEP)
    gate   = RiskGate(exc, tmp_store, mock_alpaca)
    gate.session_start(100_000.0, TRADE_DATE)

    return LivePositionManager(
        alpaca=mock_alpaca,
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

def test_a_risk_gate_reject_returns_none(mock_alpaca, tmp_store):
    """open_position returns None when the risk gate rejects; candidate is saved."""
    mgr = _build_mgr(mock_alpaca, tmp_store,
                     exe_cfg=_exe_cfg(max_position_pct=0.0))  # any size rejected
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is None
    candidates = tmp_store.get_candidates(TRADE_DATE, phase=3)
    assert any(c["symbol"] == SYMBOL for c in candidates)


# ── b. Full entry fill ─────────────────────────────────────────────────────────

def test_b_entry_full_fill_returns_open_position(mock_alpaca, tmp_store):
    """Full fill: Position returned with status='open' and correct share counts."""
    mgr = _build_mgr(mock_alpaca, tmp_store)
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

def test_c_zero_fill_returns_none_status_unfilled(mock_alpaca, tmp_store):
    """Zero fill: open_position returns None; DB row has status='unfilled'."""
    mock_alpaca.set_fill_fraction(0.0)
    mgr = _build_mgr(mock_alpaca, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is None
    assert mgr._positions.get(SYMBOL) is None
    db_row = tmp_store.get_open_position(SYMBOL)
    assert db_row is not None
    assert db_row["status"] == "unfilled"


# ── d. Partial fill → shares recomputed ───────────────────────────────────────

def test_d_partial_fill_recomputes_tp_shares(mock_alpaca, tmp_store):
    """Partial fill: tp1/tp2/tp3 shares rescaled from fill.qty, not entry['shares']."""
    # Attempt 1: fills 70/100; attempts 2+3: no fill → partial_unfilled(70 shares).
    mock_alpaca.set_fill_sequence(0.7, 0.0, 0.0)
    mgr = _build_mgr(mock_alpaca, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)

    assert pos is not None
    assert pos.entry_shares == 70
    assert pos.tp1_shares == math.floor(70 * 0.35)   # 24
    assert pos.tp2_shares == math.floor(70 * 0.05)   # 3
    assert pos.tp3_shares == 70 - pos.tp1_shares - pos.tp2_shares  # 43


# ── e. TP1 fires → stop moves to breakeven ────────────────────────────────────

def test_e_tp1_fires_stop_moves_to_breakeven(mock_alpaca, tmp_store):
    """After TP1: current_stop == actual_entry_price; tp1_hit=True; remaining decremented."""
    mgr = _build_mgr(mock_alpaca, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # lo=100.2 stays above breakeven (actual_entry_price≈100.10) so stop doesn't fire.
    # hi clears tp1_price=101.0; hi < tp2_price=102.0 so only TP1 fires.
    tp1_bar = _bar(hi=101.5, lo=100.2)
    mgr.on_bar(SYMBOL, tp1_bar, _ts())

    assert pos.tp1_hit is True
    assert pos.remaining == 100 - 35      # tp1_shares=35 sold
    assert abs(pos.current_stop - pos.actual_entry_price) < 1e-9


# ── f. TP1 fires, tp2_shares==0 → tp2_hit immediately ────────────────────────

def test_f_tp1_fires_with_no_tp2_shares_sets_tp2_hit(mock_alpaca, tmp_store):
    """When tp2_shares==0, tp2_hit is set True on the same bar as TP1."""
    entry = _make_entry(tp1_shares=40, tp2_shares=0, tp3_shares=60)
    mgr   = _build_mgr(mock_alpaca, tmp_store)
    pos   = mgr.open_position(entry, SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    tp1_bar = _bar(hi=101.5, lo=100.0)
    # TP3 check will reach indicator lookup; provide a seeded indicator
    ind = _seeded_indicator(ema_value=200.0)   # ema >> entry, crossback always True
    mgr._indicators[SYMBOL] = ind

    mgr.on_bar(SYMBOL, tp1_bar, _ts())

    assert pos.tp1_hit  is True
    assert pos.tp2_hit  is True   # set immediately because tp2_shares==0


# ── g. TP1 and TP2 fire on the same bar ───────────────────────────────────────

def test_g_tp1_and_tp2_fire_same_bar(mock_alpaca, tmp_store):
    """Both TP1 and TP2 fire in one on_bar call (step 3 and step 5 use 'if')."""
    # ema=100.5 is just above breakeven (100.10); close=102.0 > ema → no TP3 crossback.
    # lo=100.2 is above breakeven so stop doesn't fire.
    ind = _seeded_indicator(ema_value=100.5)
    mgr = _build_mgr(mock_alpaca, tmp_store, indicators={SYMBOL: ind})
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    both_bar = _bar(hi=102.5, lo=100.2, cl=102.0)
    ind.on_bar({"high": 102.5, "low": 100.2, "close": 102.0})
    mgr.on_bar(SYMBOL, both_bar, _ts())

    assert pos.tp1_hit is True
    assert pos.tp2_hit is True
    assert pos.remaining == 100 - 35 - 5   # TP1 (35) and TP2 (5) lots sold, TP3 (60) open


# ── h. TP3 EMA-crossback fires ────────────────────────────────────────────────

def test_h_tp3_ema_crossback_fires(mock_alpaca, tmp_store):
    """TP3 fires when close crosses below EMA while EMA is above entry (profitable)."""
    # ema=101.0 → just above entry=100.0 (is_profitable=True)
    ind = _seeded_indicator(ema_value=101.0)
    mgr = _build_mgr(mock_alpaca, tmp_store, indicators={SYMBOL: ind})
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Bar 1: hits TP1 and TP2. lo=100.2 > breakeven (100.10) so stop doesn't fire.
    # close=102.0 > ema≈101.0 so no TP3 crossback yet.
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

def test_i_tp3_no_fire_when_ema_below_entry(mock_alpaca, tmp_store):
    """TP3 is skipped when ema < actual_entry_price (trade not profitable at ema)."""
    # ema=95.0 < entry=100.0 → is_profitable=False for a long
    ind = _seeded_indicator(ema_value=95.0)
    mgr = _build_mgr(mock_alpaca, tmp_store, indicators={SYMBOL: ind})
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Bar 1: hit TP1 + TP2. lo=100.2 > breakeven (100.10) so stop doesn't fire.
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

def test_j_stop_fires_closes_position(mock_alpaca, tmp_store):
    """Stop hit (lo <= current_stop) closes the position with exit_reason='STOP'."""
    mgr = _build_mgr(mock_alpaca, tmp_store)
    pos = mgr.open_position(_make_entry(stop_price=99.0), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    stop_bar = _bar(hi=100.5, lo=98.5)   # lo=98.5 <= stop=99.0
    mgr.on_bar(SYMBOL, stop_bar, _ts())

    assert pos.status      == "closed"
    assert pos.exit_reason == "STOP"
    assert mgr._positions.get(SYMBOL) is None


# ── k. Stop after TP1 → exit_reason='TP1_ONLY' ───────────────────────────────

def test_k_stop_after_tp1_exit_reason_tp1_only(mock_alpaca, tmp_store):
    """Stop hit after TP1 (not trail mode) produces exit_reason='TP1_ONLY'."""
    mgr = _build_mgr(mock_alpaca, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Bar 1: TP1 fires (stop moves to breakeven=100.10)
    tp1_bar = _bar(hi=101.5, lo=100.2)
    mgr.on_bar(SYMBOL, tp1_bar, _ts())
    assert pos.tp1_hit is True
    # current_stop moved to actual_entry_price ≈ 100.10

    # Bar 2: close drops below breakeven stop
    stop_bar = _bar(hi=100.1, lo=99.5)   # lo=99.5 <= current_stop≈100.10
    mgr.on_bar(SYMBOL, stop_bar, _ts(10, 31))

    assert pos.status      == "closed"
    assert pos.exit_reason == "TP1_ONLY"


# ── l. Stop with trail_after_tp1 → exit_reason='TRAIL' ───────────────────────

def test_l_stop_with_trail_exit_reason_trail(mock_alpaca, tmp_store):
    """Trail stop after TP1 produces exit_reason='TRAIL'."""
    exit_override = {"method": "trail_after_tp1", "mult": 0.5}
    entry = _make_entry(orb_range=2.0, exit_override=exit_override)
    mgr   = _build_mgr(mock_alpaca, tmp_store)
    pos   = mgr.open_position(entry, SYMBOL, +1, TRADE_DATE)
    assert pos is not None
    assert pos.use_trail_atp1 is True
    # trail_dist = 0.5 * 2.0 = 1.0

    # Bar 1: TP1 fires. After TP1: trail_peak=101.0, then step-4 updates peak to hi=101.5,
    # so current_stop = 101.5 - 1.0 = 100.5.  lo=100.6 > 100.5 → stop doesn't fire.
    tp1_bar = _bar(hi=101.5, lo=100.6)
    mgr.on_bar(SYMBOL, tp1_bar, _ts())
    assert pos.tp1_hit is True

    # Bar 2: rally. hi=103.0 → trail peak=103.0, stop=102.0.  lo=102.1 > 102.0 → no stop.
    rally_bar = _bar(hi=103.0, lo=102.1)
    mgr.on_bar(SYMBOL, rally_bar, _ts(10, 31))
    assert abs(pos.trail_atp1_peak - 103.0) < 1e-9
    assert abs(pos.current_stop - 102.0) < 1e-9

    # Bar 3: price drops through trail stop. lo=101.5 <= current_stop=102.0 → TRAIL fires.
    stop_bar = _bar(hi=102.5, lo=101.5)
    mgr.on_bar(SYMBOL, stop_bar, _ts(10, 32))

    assert pos.status      == "closed"
    assert pos.exit_reason == "TRAIL"


# ── m. EOD fires when ts.time() >= eod_time ───────────────────────────────────

def test_m_eod_fires_at_configured_hour(mock_alpaca, tmp_store):
    """EOD branch fires when bar timestamp >= eod_exit_hour (here 10:00)."""
    mgr = _build_mgr(mock_alpaca, tmp_store,
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

def test_n_flatten_all_closes_all_positions(mock_alpaca, tmp_store):
    """flatten_all() closes every open position regardless of state."""
    mgr = _build_mgr(mock_alpaca, tmp_store)

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

def test_o_on_bar_ignores_unknown_symbol(mock_alpaca, tmp_store):
    """on_bar is a no-op for symbols not in _positions (no exception raised)."""
    mgr = _build_mgr(mock_alpaca, tmp_store)
    bar = _bar(hi=101.0, lo=99.0)
    mgr.on_bar("UNKNOWN_SYM", bar, _ts())   # must not raise


# ── p. on_bar ignores closed position ────────────────────────────────────────

def test_p_on_bar_ignores_non_open_status(mock_alpaca, tmp_store):
    """on_bar is a no-op when position exists but status != 'open'."""
    mgr = _build_mgr(mock_alpaca, tmp_store)
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Manually mark as entering (non-open status)
    pos.status = "entering"

    stop_bar = _bar(hi=100.5, lo=98.0)   # would fire stop if on_bar processed it
    mgr.on_bar(SYMBOL, stop_bar, _ts())

    # Status unchanged — on_bar was a no-op
    assert pos.status == "entering"


# ── q. TP3 check with unseeded indicator → RuntimeError ───────────────────────

def test_q_tp3_unseeded_indicator_raises(mock_alpaca, tmp_store):
    """RuntimeError raised if TP3 check fires against an unseeded RollingIndicators."""
    from orb_live.execution.indicators import RollingIndicators
    ind = RollingIndicators(SYMBOL, SimpleNamespace(ema_length=30))
    # ind.is_seeded == False

    mgr = _build_mgr(mock_alpaca, tmp_store, indicators={SYMBOL: ind})
    pos = mgr.open_position(_make_entry(), SYMBOL, +1, TRADE_DATE)
    assert pos is not None

    # Manually advance state to the TP3 check path
    pos.tp1_hit = True
    pos.tp2_hit = True

    bar = _bar(hi=101.0, lo=100.0, cl=100.5)
    with pytest.raises(RuntimeError, match="is not seeded"):
        mgr.on_bar(SYMBOL, bar, _ts())


# ── r. EOD branch dead for production eod_exit_hour=16 on any RTH bar ────────

def test_r_eod_branch_never_fires_for_rth_bars(mock_alpaca, tmp_store):
    """
    With eod_exit_hour=16 (production), on_bar does not close the position
    on any RTH bar (9:30–15:59).  The last RTH bar at 15:59 must leave the
    position open so the runner can sweep it at 16:00:30.
    """
    mgr = _build_mgr(mock_alpaca, tmp_store)   # default: eod_exit_hour=16
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
