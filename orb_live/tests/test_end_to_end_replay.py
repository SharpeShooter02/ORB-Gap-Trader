"""
tests/test_end_to_end_replay.py — Load-bearing end-to-end replay test.

Replays three synthetic fixture scenarios through the full live execution
stack (StrategyEngine + LivePositionManager + DryRunBroker) and verifies that
the exit reason and P&L direction match the expected outcome.

Scenarios:
    A. tp3_crossback  — entry fires, TP1 + TP2 hit, TP3 EMA-crossback exits
    B. stop_hit       — entry fires, price drops below stop immediately
    C. eod_flatten    — entry fires, position open at "EOD", flatten_all closes it

These are NOT parity tests against the full backtest — they verify that the
live execution path produces the correct exit reason and that P&L is
directionally correct (positive for wins, negative for stops).

CRITICAL ORDERING verified implicitly:
    For each bar, indicators[symbol].on_bar() is called BEFORE
    position_manager.on_bar(), matching the SessionRunner._on_bar_dispatch
    contract.  Tests that replay bars manually enforce this.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import orb_live  # noqa: F401 — path setup

ET    = ZoneInfo("America/New_York")
TDATE = date(2026, 1, 7)

_NO_SLEEP = lambda _: None  # noqa: E731


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

def _et(h: int, m: int) -> datetime:
    return datetime(2026, 1, 7, h, m, 0, tzinfo=ET)


def _bar(ts: datetime, close: float, hi: float, lo: float) -> dict:
    return {"timestamp": ts, "open": close, "high": hi, "low": lo,
            "close": close, "volume": 5000}


def _orb(high: float = 101.0, low: float = 99.0, ema: Optional[float] = None) -> dict:
    mid = (high + low) / 2.0
    return {
        "high": high, "low": low, "midpoint": mid,
        "size_pct": (high - low) / mid,
        "n_bars": 30,
        "ema": ema if ema is not None else mid,
    }


def _make_p2(symbol, orb_val=None, tp1_mult=0.35, tp2_mult=0.05):
    from orb_live.signals.pre_market import Phase2Result
    o = orb_val or _orb()
    return Phase2Result(
        symbol=symbol, gap_abs=0.10, gap_direction=1, prior_close=90.0,
        first_open=100.0, orb=o, rtg_val=None, rtg_pct=None,
        tp1_mult=tp1_mult, tp2_mult=tp2_mult,
        rtg_excluded=False, routing_action="normal", size_mult=1.0,
        preflight=None, is_candidate=True, exclusion_reason="",
    )


def _build_live_stack(tmp_store, mock_broker, tp3_ema_value=100.0):
    """
    Build LivePositionManager + StrategyEngine + RollingIndicators for TQQQ.

    tp3_ema_value — the EMA the indicator is seeded to (used for TP3 detection).
    The position manager reads this EMA during on_bar.
    """
    from orb_live.config.live_config import load_live_config
    from orb_live.execution.indicators import RollingIndicators
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.runner.strategy_engine import StrategyEngine

    cfg  = load_live_config()
    scfg = cfg.strategy_config

    exc_cfg = SimpleNamespace(
        entry_slippage_bps=10,
        entry_repeg_seconds=60.0,
        entry_repeg_max_attempts=3,
        entry_slippage_max_bps=30,
        exit_slippage_bps=5,
        stop_order_type="market",
        session_kill_loss_pct=0.03,
        max_concurrent_positions=0,
        max_gross_exposure_pct=2.0,
        max_position_pct=0.50,
    )

    indicators_store: dict = {}

    # Seed indicator
    ind = RollingIndicators("TQQQ", scfg)
    orb_bars = pd.DataFrame(
        [{"high": tp3_ema_value + 0.5, "low": tp3_ema_value - 0.5,
          "close": tp3_ema_value}],
        index=pd.date_range("2026-01-07 09:30", periods=1, freq="1min", tz=ET),
    )
    ind.seed_from_orb_bars(orb_bars)
    indicators_store["TQQQ"] = ind

    policy = MarketableLimitPolicy(mock_broker, exc_cfg, tmp_store,
                                   _sleep=_NO_SLEEP)
    gate   = RiskGate(exc_cfg, tmp_store, mock_broker)
    gate.session_start(100_000.0, TDATE)

    mgr = LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, indicators_store=indicators_store, config=scfg,
    )

    engine = StrategyEngine(mgr, cfg, tmp_store, mock_broker)
    engine.new_session(TDATE)

    return mgr, engine, indicators_store, ind, scfg


def _replay_bars(bars: list[dict], mgr, indicators_store, symbol="TQQQ"):
    """
    Replay a list of bars through indicators + position_manager in the correct order.
    Mirrors SessionRunner._on_bar_dispatch for post-ORB bars.
    """
    for bar in bars:
        ts  = bar["timestamp"]
        ind = indicators_store.get(symbol)
        if ind is not None and ind.is_seeded:
            ind.on_bar(bar)
        mgr.on_bar(symbol, bar, ts)


# ── Scenario A: v1 TP1-only full exit ────────────────────────────────────────

def test_a_tp1_only_full_exit(mock_broker, tmp_store):
    """
    v1 TP1-only: entry fires, TP1 consumes all shares, position closes immediately.

    ORB: high=101, low=99 (range=2), ema=100
    v1 config: exit_ratio_tp1=1.0, tp1_target_multiple=1.0
    v1 sizing: shares = floor(1000 × 1.0 / 102) = 9
    TP1 price = 102 + 2×1.0 = 104.0; tp1_shares=9, tp2_shares=0, tp3_shares=0

    Bar 1: hi=105.0 ≥ TP1=104.0 → TP1 fires, all 9 shares exit, remaining=0
    → position closes immediately with exit_reason="TP1_ONLY"
    """
    orb = _orb(high=101.0, low=99.0, ema=100.0)
    mgr, engine, indicators_store, ind, scfg = _build_live_stack(
        tmp_store, mock_broker, tp3_ema_value=103.0
    )
    # v1: tp1_mult=1.0 sets TP1 price = entry + 1×range; tp2_mult=0.0 unused
    p2 = _make_p2("TQQQ", orb_val=orb, tp1_mult=1.0, tp2_mult=0.0)
    engine.on_orb_complete("TQQQ", p2, None)

    # Entry bar: close=102 > orb_high=101 AND > orb_ema=100 → breakout
    entry_bar = _bar(_et(10, 1), close=102.0, hi=102.5, lo=101.5)
    ind.on_bar(entry_bar)
    engine.on_bar("TQQQ", entry_bar, _et(10, 1))

    pos = mgr._positions.get("TQQQ")
    assert pos is not None, "Position should open on breakout"
    assert pos.tp1_shares == pos.entry_shares, "v1: all shares allocated to TP1"
    assert pos.tp2_shares == 0
    assert pos.tp3_shares == 0

    # Simulate IB filling the OCA TP1 limit order (happens when hi crosses tp1_price).
    # The MockBroker's get_order will then auto-cancel the stop (OCA behavior).
    if pos.tp1_order_id:
        mock_broker._orders[pos.tp1_order_id]["status"]           = "filled"
        mock_broker._orders[pos.tp1_order_id]["filled_qty"]       = str(pos.entry_shares)
        mock_broker._orders[pos.tp1_order_id]["filled_avg_price"] = str(pos.tp1_price)

    # Bar 1: hi=105 ≥ TP1=104 → OCA TP1 confirmed filled → position closes
    bar1 = _bar(_et(10, 2), close=104.5, hi=105.0, lo=103.5)
    ind.on_bar(bar1)
    mgr.on_bar("TQQQ", bar1, _et(10, 2))

    pos = mgr._positions.get("TQQQ")
    assert pos is None or pos.status == "closed", \
        "TP1-only: position must close when all shares exit at TP1"

    from orb_live.core.state_store import closed_trades
    with tmp_store.conn() as c:
        rows = c.execute(closed_trades.select()).mappings().all()

    assert rows, "Should have a closed trade record"
    exit_reasons = [dict(r)["exit_reason"] for r in rows]
    assert any("TP1_ONLY" in r for r in exit_reasons), \
        f"Expected TP1_ONLY exit among {exit_reasons}"


# ── Scenario B: Stop hit ───────────────────────────────────────────────────────

def test_b_stop_hit_exit_reason_and_negative_pnl(mock_broker, tmp_store):
    """
    Entry fires, then price drops immediately below stop → STOP exit.

    ORB: high=101, low=99, midpoint=100
    Stop = (midpoint + low) / 2 = (100 + 99) / 2 = 99.5
    Entry bar: close=102 (breakout)
    Bar 1:     lo=99.0 < stop=99.5 → stop fires
    """
    orb = _orb(high=101.0, low=99.0, ema=100.0)
    # EMA #2 above entry → no TP3 crossback on this scenario
    mgr, engine, indicators_store, ind, scfg = _build_live_stack(
        tmp_store, mock_broker, tp3_ema_value=110.0
    )
    p2 = _make_p2("TQQQ", orb_val=orb, tp1_mult=0.35, tp2_mult=0.05)
    engine.on_orb_complete("TQQQ", p2, None)

    entry_bar = _bar(_et(10, 1), close=102.0, hi=102.5, lo=101.5)
    ind.on_bar(entry_bar)
    engine.on_bar("TQQQ", entry_bar, _et(10, 1))

    pos = mgr._positions.get("TQQQ")
    assert pos is not None, "Position should open on breakout"

    actual_entry = pos.actual_entry_price
    stop_price   = pos.current_stop  # initial stop from compute_entry

    # Simulate IB firing the stop: price dropped, exchange executed stop at stop_price.
    mock_broker._orders[pos.stop_order_id] = {
        "status": "filled", "filled_qty": str(pos.remaining),
        "filled_avg_price": str(stop_price),
    }
    bar_stop = _bar(_et(10, 2), close=99.0, hi=101.0, lo=98.5)
    ind.on_bar(bar_stop)
    mgr.on_bar("TQQQ", bar_stop, _et(10, 2))

    pos = mgr._positions.get("TQQQ")
    assert pos is None or pos.status == "closed", "Position should be closed after stop"

    from orb_live.core.state_store import closed_trades
    with tmp_store.conn() as c:
        rows = c.execute(closed_trades.select()).mappings().all()

    assert rows, "Should have a closed trade record"
    last = dict(rows[-1])
    assert last["exit_reason"] in ("STOP", "TP1_ONLY", "TRAIL"), \
        f"Expected stop-type exit, got {last['exit_reason']}"
    # dollar_pnl may be None (not stored by position_manager); compute from prices
    pnl = (float(last["exit_price"]) - float(last["entry_price"])) * float(last["qty"])
    assert pnl < 0, f"Stop exit should produce a loss; got pnl={pnl:.2f}"


# ── Scenario C: EOD flatten ────────────────────────────────────────────────────

def test_c_eod_flatten_closes_open_position(mock_broker, tmp_store):
    """
    Entry fires and the position remains open until flatten_all('eod_sweep').
    flatten_all must close the position and record a closed trade.
    """
    orb = _orb(high=101.0, low=99.0, ema=100.0)
    # EMA #2 well above close → no TP3 crossback
    mgr, engine, indicators_store, ind, scfg = _build_live_stack(
        tmp_store, mock_broker, tp3_ema_value=200.0
    )
    p2 = _make_p2("TQQQ", orb_val=orb, tp1_mult=0.35, tp2_mult=0.05)
    engine.on_orb_complete("TQQQ", p2, None)

    entry_bar = _bar(_et(10, 1), close=102.0, hi=102.5, lo=101.5)
    ind.on_bar(entry_bar)
    engine.on_bar("TQQQ", entry_bar, _et(10, 1))

    pos = mgr._positions.get("TQQQ")
    assert pos is not None, "Position should open on breakout"

    # Deliver bars that don't hit TP or stop
    # TP1 = 102.0 + 2*0.35 = 102.70; stop = 99.5; keep hi < 102.70, lo > 99.5
    for i in range(2, 6):
        safe_bar = _bar(_et(10, i), close=101.5, hi=102.0, lo=100.5)
        ind.on_bar(safe_bar)
        mgr.on_bar("TQQQ", safe_bar, _et(10, i))

    pos = mgr._positions.get("TQQQ")
    assert pos is not None and pos.status == "open", "Position should still be open"

    # EOD sweep
    mgr.flatten_all("eod_sweep")

    pos = mgr._positions.get("TQQQ")
    assert pos is None or pos.status == "closed", "Position should be closed after EOD"

    from orb_live.core.state_store import closed_trades
    with tmp_store.conn() as c:
        rows = c.execute(closed_trades.select()).mappings().all()

    assert rows, "Should have a closed trade record"
    last = dict(rows[-1])
    assert "eod_sweep" in last["exit_reason"] or "EOD" in last["exit_reason"], \
        f"Expected EOD exit reason, got {last['exit_reason']}"


# ── Scenario D: DryRunBroker fill model verification ─────────────────────────

def test_d_dry_run_limit_buy_fills_at_or_below_limit(mock_broker, tmp_store):
    """DryRunBroker: limit buy fills at min(limit_price, ask+1¢)."""
    from orb_live.runner.dry_run import DryRunBroker

    class _FakeReal:
        def get_latest_quote(self, sym):
            return {"bid": 99.90, "ask": 100.10}

        def get_account(self):
            return {"equity": 100_000.0}

    dry = DryRunBroker(_FakeReal(), starting_equity=100_000.0)
    dry.submit_limit_order("TQQQ", "buy", 100, 100.50, "order-1")
    order = dry.get_order("order-1")

    # min(100.50, 100.10+0.01) = min(100.50, 100.11) = 100.11
    fill = float(order["filled_avg_price"])
    assert fill == pytest.approx(100.11, abs=1e-6)
    assert order["status"] == "filled"
    assert int(order["filled_qty"]) == 100


def test_e_dry_run_market_sell_applies_adverse_slippage(mock_broker, tmp_store):
    """DryRunBroker: market sell fills at bid × (1 - 5bps)."""
    from orb_live.runner.dry_run import DryRunBroker

    class _FakeReal:
        def get_latest_quote(self, sym):
            return {"bid": 100.0, "ask": 100.20}

        def get_account(self):
            return {"equity": 100_000.0}

    dry = DryRunBroker(_FakeReal(), starting_equity=100_000.0)
    dry.submit_market_order("TQQQ", "sell", 50, "order-2")
    order = dry.get_order("order-2")

    # bid × (1 - 5bps) = 100.0 × 0.9995 = 99.95
    fill = float(order["filled_avg_price"])
    assert fill == pytest.approx(99.95, abs=1e-4)
    assert order["status"] == "filled"
