"""
tests/test_session_runner.py — SessionRunner and StrategyEngine unit tests.

All tests are synchronous — no real market data, no broker credentials.
Components are assembled from the existing conftest.py fixtures
(MockBroker, tmp_store) with stub pre-market / clock / bar_cache / bar_router.
"""

from datetime import date, datetime, time as dtime, timedelta
from types import SimpleNamespace
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import orb_live  # noqa: F401 — path setup

ET    = ZoneInfo("America/New_York")
TDATE = date(2026, 1, 7)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _et(h, m) -> datetime:
    return datetime(2026, 1, 7, h, m, 0, tzinfo=ET)


def _bar(ts: datetime, close: float, hi: Optional[float] = None,
         lo: Optional[float] = None) -> dict:
    hi = hi or close + 0.5
    lo = lo or close - 0.5
    return {"timestamp": ts, "open": close, "high": hi, "low": lo,
            "close": close, "volume": 1000}


def _minimal_orb(high: float = 101.0, low: float = 99.0) -> dict:
    mid = (high + low) / 2.0
    return {
        "high": high, "low": low, "midpoint": mid,
        "size_pct": (high - low) / mid,
        "n_bars": 30, "ema": mid,
    }


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _StubClock:
    def now_et(self): return _et(10, 5)
    def orb_end_et(self, orb_minutes=30): return _et(10, 0)
    def eod_exit_et(self, h=16, m=0): return _et(16, 0)
    def is_half_day(self): return False
    def effective_close(self): return dtime(16, 0)
    def next_open_eval_start(self): return _et(9, 31)  # in the past → no sleep


class _StubBarCache:
    def __init__(self, bars: dict = None):
        self._bars = bars or {}
        self.added = []

    def add_bar(self, symbol, bar):
        self.added.append((symbol, bar))

    def get_bars(self, symbol):
        rows = self._bars.get(symbol, [])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)


class _StubBarRouter:
    def __init__(self):
        self.subscribed = []
        self.unsubscribed = False
        self._listeners = []

    def register_listener(self, fn):
        self._listeners.append(fn)

    def subscribe(self, symbols):
        self.subscribed.extend(symbols)

    def unsubscribe(self):
        self.unsubscribed = True

    def push_bar(self, symbol, bar):
        for fn in self._listeners:
            fn(symbol, bar)


class _StubPreMarket:
    def __init__(self, p1_results=None, p2_results=None):
        self._p1 = p1_results or []
        self._p2 = p2_results or []

    def run_phase1(self, trade_date, *a, **kw):
        return self._p1

    def run_phase2(self, trade_date, phase1, *a, **kw):
        return self._p2


def _make_p1(symbol):
    from orb_live.signals.pre_market import Phase1Result
    return Phase1Result(symbol=symbol, gap_abs=0.10, gap_direction=1,
                        prior_close=90.0, ps_filter_passed=True)


def _make_p2(symbol, is_candidate=True, rtg_excluded=False, size_mult=1.0,
              orb=None, tp1_mult=0.35, tp2_mult=0.05):
    from orb_live.signals.pre_market import Phase2Result
    return Phase2Result(
        symbol=symbol, gap_abs=0.10, gap_direction=1, prior_close=90.0,
        first_open=100.0,
        orb=orb if orb is not None else _minimal_orb(),
        rtg_val=None, rtg_pct=None,
        tp1_mult=tp1_mult, tp2_mult=tp2_mult,
        rtg_excluded=rtg_excluded,
        routing_action="normal",
        size_mult=size_mult,
        preflight=None,
        is_candidate=is_candidate,
        exclusion_reason="" if is_candidate else "no_orb",
    )


def _build_engine(mock_broker, tmp_store):
    from orb_live.config.live_config import load_live_config
    from orb_live.execution.indicators import RollingIndicators
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.runner.strategy_engine import StrategyEngine

    cfg  = load_live_config()
    scfg = cfg.strategy_config

    exc = SimpleNamespace(
        entry_slippage_bps=10, entry_repeg_seconds=60.0,
        entry_repeg_max_attempts=3, entry_slippage_max_bps=30,
        exit_slippage_bps=5, stop_order_type="market",
        session_kill_loss_pct=0.03,
        max_concurrent_positions=0,
        max_gross_exposure_pct=2.0,
        max_position_pct=0.50,
    )
    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=lambda _: None)
    gate   = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TDATE)
    indicators_store = {}

    mgr = LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, indicators_store=indicators_store, config=scfg,
    )
    engine = StrategyEngine(mgr, cfg, tmp_store, mock_broker)
    return engine, mgr, indicators_store, cfg


# ── Tests: StrategyEngine state machine ───────────────────────────────────────

def test_a_waiting_for_orb_is_default_state(mock_broker, tmp_store):
    """A freshly constructed engine has WAITING_FOR_ORB for any symbol."""
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState
    engine, _, _, cfg = _build_engine(mock_broker, tmp_store)
    engine.new_session(TDATE)
    assert engine.get_state("TQQQ") == SymbolState.WAITING_FOR_ORB


def test_b_not_candidate_skips_immediately(mock_broker, tmp_store):
    """on_orb_complete with is_candidate=False → EXITED_OR_SKIPPED."""
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState
    engine, _, _, _ = _build_engine(mock_broker, tmp_store)
    engine.new_session(TDATE)
    p2 = _make_p2("TQQQ", is_candidate=False)
    engine.on_orb_complete("TQQQ", p2, None)
    assert engine.get_state("TQQQ") == SymbolState.EXITED_OR_SKIPPED



def test_d_routing_skip_skips(mock_broker, tmp_store):
    """on_orb_complete with size_mult=0.0 → EXITED_OR_SKIPPED."""
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState
    engine, _, _, _ = _build_engine(mock_broker, tmp_store)
    engine.new_session(TDATE)
    p2 = _make_p2("TQQQ", size_mult=0.0)
    engine.on_orb_complete("TQQQ", p2, None)
    assert engine.get_state("TQQQ") == SymbolState.EXITED_OR_SKIPPED


def test_e_valid_p2_transitions_to_orb_complete(mock_broker, tmp_store):
    """A valid Phase2Result with is_candidate=True → ORB_COMPLETE."""
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState
    engine, _, _, _ = _build_engine(mock_broker, tmp_store)
    engine.new_session(TDATE)
    p2 = _make_p2("TQQQ")
    engine.on_orb_complete("TQQQ", p2, None)
    assert engine.get_state("TQQQ") == SymbolState.ORB_COMPLETE


def test_f_breakout_bar_transitions_to_in_position(mock_broker, tmp_store):
    """
    A bar that passes check_breakout on an ORB_COMPLETE symbol → IN_POSITION.
    ORB: high=101, low=99.  Breakout long: close > 101 AND close > ema(100).
    """
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState
    engine, _, indicators_store, cfg = _build_engine(mock_broker, tmp_store)
    engine.new_session(TDATE)

    orb = _minimal_orb(high=101.0, low=99.0)  # ema=100
    p2  = _make_p2("TQQQ", orb=orb)
    engine.on_orb_complete("TQQQ", p2, None)

    # Seed indicator before calling on_bar
    from orb_live.execution.indicators import RollingIndicators
    ind = RollingIndicators("TQQQ", cfg.strategy_config)
    orb_df = pd.DataFrame(
        [{"high": 101.0, "low": 99.0, "close": 100.5}],
        index=pd.date_range("2026-01-07 09:30", periods=1, freq="1min", tz=ET),
    )
    ind.seed_from_orb_bars(orb_df)
    indicators_store["TQQQ"] = ind

    # Breakout bar: close=102.0 > orb_high=101 AND > ema≈100
    bar = _bar(_et(10, 1), close=102.0, hi=102.5, lo=101.5)
    engine.on_bar("TQQQ", bar, _et(10, 1))

    assert engine.get_state("TQQQ") == SymbolState.IN_POSITION


def test_g_latest_entry_minute_cutoff(mock_broker, tmp_store):
    """Bars arriving after latest_entry_minute transition to EXITED_OR_SKIPPED."""
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState
    engine, _, _, cfg = _build_engine(mock_broker, tmp_store)
    engine.new_session(TDATE)

    # Temporarily patch latest_entry_minute to 30 (only bars up to 10:00 ET)
    scfg = cfg.strategy_config
    original = scfg.latest_entry_minute
    scfg.latest_entry_minute = 30  # 30 min after open = 10:00 ET

    try:
        p2 = _make_p2("TQQQ")
        engine.on_orb_complete("TQQQ", p2, None)

        # Bar at 10:31 (61 min after 9:30) → past cutoff
        bar = _bar(_et(10, 31), close=102.0)
        engine.on_bar("TQQQ", bar, _et(10, 31))
        assert engine.get_state("TQQQ") == SymbolState.EXITED_OR_SKIPPED
    finally:
        scfg.latest_entry_minute = original


def test_h_in_position_state_does_not_try_second_entry(mock_broker, tmp_store):
    """Once IN_POSITION, on_bar is a no-op for entry detection."""
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState
    engine, mgr, indicators_store, cfg = _build_engine(mock_broker, tmp_store)
    engine.new_session(TDATE)

    orb = _minimal_orb(high=101.0, low=99.0)
    p2  = _make_p2("TQQQ", orb=orb)
    engine.on_orb_complete("TQQQ", p2, None)

    from orb_live.execution.indicators import RollingIndicators
    ind = RollingIndicators("TQQQ", cfg.strategy_config)
    orb_df = pd.DataFrame(
        [{"high": 101.0, "low": 99.0, "close": 100.5}],
        index=pd.date_range("2026-01-07 09:30", periods=1, freq="1min", tz=ET),
    )
    ind.seed_from_orb_bars(orb_df)
    indicators_store["TQQQ"] = ind

    # First breakout → IN_POSITION
    bar1 = _bar(_et(10, 1), close=102.0, hi=102.5, lo=101.5)
    engine.on_bar("TQQQ", bar1, _et(10, 1))
    assert engine.get_state("TQQQ") == SymbolState.IN_POSITION

    # Track how many open_position calls are made
    open_calls_before = sum(1 for s, p in mgr._positions.items() if p is not None)

    # Second bar (also above breakout) — should not try to open again
    bar2 = _bar(_et(10, 2), close=103.0, hi=103.5, lo=102.5)
    engine.on_bar("TQQQ", bar2, _et(10, 2))

    # Still IN_POSITION, not EXITED_OR_SKIPPED
    assert engine.get_state("TQQQ") == SymbolState.IN_POSITION


# ── Tests: SessionRunner bar dispatch order ───────────────────────────────────

def test_i_indicator_updated_before_position_manager(mock_broker, tmp_store):
    """
    LOAD-BEARING ORDER TEST: indicators[symbol].on_bar must be called before
    position_manager.on_bar so that position_manager reads the current EMA.

    Verify by recording call timestamps via side-effect ordering.
    """
    from orb_live.config.live_config import load_live_config
    from orb_live.execution.indicators import RollingIndicators
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.runner.bar_router import BarRouter
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.runner.strategy_engine import StrategyEngine

    cfg  = load_live_config()
    scfg = cfg.strategy_config

    exc = SimpleNamespace(
        entry_slippage_bps=10, entry_repeg_seconds=60.0,
        entry_repeg_max_attempts=3, entry_slippage_max_bps=30,
        exit_slippage_bps=5, stop_order_type="market",
        session_kill_loss_pct=0.03, max_concurrent_positions=0,
        max_gross_exposure_pct=2.0, max_position_pct=0.50,
    )
    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=lambda _: None)
    gate   = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TDATE)
    indicators_store = {}

    mgr = LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, indicators_store=indicators_store, config=scfg,
    )

    router = _StubBarRouter()
    pre    = _StubPreMarket()
    engine = StrategyEngine(mgr, cfg, tmp_store, mock_broker)

    # Seed indicator for TQQQ
    ind = RollingIndicators("TQQQ", scfg)
    orb_df = pd.DataFrame(
        [{"high": 101.0, "low": 99.0, "close": 100.0}],
        index=pd.date_range("2026-01-07 09:30", periods=1, freq="1min", tz=ET),
    )
    ind.seed_from_orb_bars(orb_df)
    indicators_store["TQQQ"] = ind

    call_order = []
    original_ind_on_bar = ind.on_bar
    original_mgr_on_bar = mgr.on_bar

    def _track_ind(bar):
        call_order.append("indicator")
        return original_ind_on_bar(bar)

    def _track_mgr(symbol, bar, ts):
        call_order.append("position_manager")
        return original_mgr_on_bar(symbol, bar, ts)

    ind.on_bar = _track_ind
    mgr.on_bar = _track_mgr

    runner = SessionRunner(
        config=cfg, broker=mock_broker, state_store=tmp_store,
        bar_cache=_StubBarCache(), bar_router=router,
        pre_market_job=pre, strategy_engine=engine,
        position_manager=mgr, risk_gate=gate,
        indicators_store=indicators_store,
        underlying_store=SimpleNamespace(),
        clock=_StubClock(), _sleep=lambda _: None,
    )

    # Dispatch a post-ORB bar (10:05 > orb_end=10:00)
    post_orb_bar = _bar(_et(10, 5), close=100.5)
    runner._on_bar_dispatch("TQQQ", post_orb_bar)

    # indicator must appear before position_manager
    assert "indicator" in call_order
    assert "position_manager" in call_order
    assert call_order.index("indicator") < call_order.index("position_manager")


def test_j_orb_window_bars_not_dispatched_to_engine(mock_broker, tmp_store):
    """Bars with timestamp < 10:00 ET are cached but NOT sent to the engine."""
    from orb_live.config.live_config import load_live_config
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState

    cfg  = load_live_config()
    exc = SimpleNamespace(
        entry_slippage_bps=10, entry_repeg_seconds=60.0,
        entry_repeg_max_attempts=3, entry_slippage_max_bps=30,
        exit_slippage_bps=5, stop_order_type="market",
        session_kill_loss_pct=0.03, max_concurrent_positions=0,
        max_gross_exposure_pct=2.0, max_position_pct=0.50,
    )
    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=lambda _: None)
    gate   = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TDATE)

    mgr = LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, indicators_store={}, config=cfg.strategy_config,
    )
    engine = StrategyEngine(mgr, cfg, tmp_store, mock_broker)
    engine.new_session(TDATE)

    p2 = _make_p2("TQQQ")
    engine.on_orb_complete("TQQQ", p2, None)
    assert engine.get_state("TQQQ") == SymbolState.ORB_COMPLETE

    cache = _StubBarCache()
    runner = SessionRunner(
        config=cfg, broker=mock_broker, state_store=tmp_store,
        bar_cache=cache, bar_router=_StubBarRouter(),
        pre_market_job=_StubPreMarket(), strategy_engine=engine,
        position_manager=mgr, risk_gate=gate,
        indicators_store={}, underlying_store=SimpleNamespace(),
        clock=_StubClock(), _sleep=lambda _: None,
    )

    # ORB window bar: 9:45 < 10:00
    orb_bar = _bar(_et(9, 45), close=100.0)
    runner._on_bar_dispatch("TQQQ", orb_bar)

    # Cached but engine state unchanged (still ORB_COMPLETE, not triggered)
    assert len(cache.added) == 1
    assert engine.get_state("TQQQ") == SymbolState.ORB_COMPLETE
