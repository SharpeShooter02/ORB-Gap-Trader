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
        self._entry_listeners = []

    def register_listener(self, fn):
        self._listeners.append(fn)

    def register_entry_listener(self, fn):
        self._entry_listeners.append(fn)

    def subscribe(self, symbols):
        self.subscribed.extend(symbols)

    def unsubscribe(self):
        self.unsubscribed = True

    def bars_received(self, symbol: str) -> int:
        return 0

    def push_bar(self, symbol, bar):
        for fn in self._listeners:
            fn(symbol, bar)

    def push_entry_bar(self, symbol, bar):
        for fn in self._entry_listeners:
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
        entry_slippage_bps=10, stop_order_type="market",
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
    ORB: high=101, low=99.  Breakout long: close > 101.
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
        entry_slippage_bps=10, stop_order_type="market",
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
        entry_slippage_bps=10, stop_order_type="market",
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


def test_k_dispatch_path_places_order_on_breakout(mock_broker, tmp_store):
    """
    End-to-end dispatch path: runner._on_bar_dispatch → engine.on_bar → order placed.

    Existing tests a-h call engine.on_bar() directly.  This test goes through
    the full live bar-dispatch chain (the path that was silently dead in production
    because subscribing all 59 symbols exceeded IB's market-data line cap).
    """
    from orb_live.config.live_config import load_live_config
    from orb_live.execution.indicators import RollingIndicators
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState

    cfg  = load_live_config()
    scfg = cfg.strategy_config

    exc = SimpleNamespace(
        entry_slippage_bps=10, stop_order_type="market",
        session_kill_loss_pct=0.03, max_concurrent_positions=0,
        max_gross_exposure_pct=2.0, max_position_pct=0.50,
    )
    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=lambda _: None)
    gate   = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TDATE)
    indicators_store: dict = {}

    mgr = LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, indicators_store=indicators_store, config=scfg,
    )
    engine = StrategyEngine(mgr, cfg, tmp_store, mock_broker)
    engine.new_session(TDATE)

    # Seed TQQQ as ORB_COMPLETE candidate (high=101, low=99, ema=100)
    orb = _minimal_orb(high=101.0, low=99.0)
    p2  = _make_p2("TQQQ", orb=orb)
    engine.on_orb_complete("TQQQ", p2, None)

    # Seed rolling indicator so check_breakout has a live EMA
    ind = RollingIndicators("TQQQ", scfg)
    orb_df = pd.DataFrame(
        [{"high": 101.0, "low": 99.0, "close": 100.0}],
        index=pd.date_range("2026-01-07 09:30", periods=1, freq="1min", tz=ET),
    )
    ind.seed_from_orb_bars(orb_df)
    indicators_store["TQQQ"] = ind

    from orb_live.core.logger import get_logger
    runner = SessionRunner(
        config=cfg, broker=mock_broker, state_store=tmp_store,
        bar_cache=_StubBarCache(), bar_router=_StubBarRouter(),
        pre_market_job=_StubPreMarket(), strategy_engine=engine,
        position_manager=mgr, risk_gate=gate,
        indicators_store=indicators_store,
        underlying_store=SimpleNamespace(),
        clock=_StubClock(), _sleep=lambda _: None,
        logger=get_logger(__name__),  # exercise the debug-log branch (bar_time)
    )

    # Post-ORB breakout bar (10:05 > orb_end=10:00): close=102 > orb_high=101 > ema≈100
    breakout_bar = _bar(_et(10, 5), close=102.0, hi=103.0, lo=101.5)
    runner._on_bar_dispatch("TQQQ", breakout_bar)

    assert engine.get_state("TQQQ") == SymbolState.IN_POSITION, (
        "_on_bar_dispatch must trigger breakout detection → IN_POSITION"
    )
    assert mock_broker._orders, (
        "_on_bar_dispatch breakout must place an order in the broker"
    )


def test_k2_entry_path_places_order_on_5sec_breakout(mock_broker, tmp_store):
    """The 5-sec entry path (_on_entry_bar_dispatch) must trigger a breakout and
    place an order WITHOUT any 1-min bar being dispatched — proving entry no
    longer waits for minute close. A pre-ORB 5-sec bar must be ignored."""
    from orb_live.config.live_config import load_live_config
    from orb_live.execution.indicators import RollingIndicators
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState

    cfg  = load_live_config()
    scfg = cfg.strategy_config

    exc = SimpleNamespace(
        entry_slippage_bps=10, stop_order_type="market",
        session_kill_loss_pct=0.03, max_concurrent_positions=0,
        max_gross_exposure_pct=2.0, max_position_pct=0.50,
    )
    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=lambda _: None)
    gate   = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TDATE)
    indicators_store: dict = {}

    mgr = LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, indicators_store=indicators_store, config=scfg,
    )
    engine = StrategyEngine(mgr, cfg, tmp_store, mock_broker)
    engine.new_session(TDATE)

    orb = _minimal_orb(high=101.0, low=99.0)
    p2  = _make_p2("TQQQ", orb=orb)
    engine.on_orb_complete("TQQQ", p2, None)

    ind = RollingIndicators("TQQQ", scfg)
    orb_df = pd.DataFrame(
        [{"high": 101.0, "low": 99.0, "close": 100.0}],
        index=pd.date_range("2026-01-07 09:30", periods=1, freq="1min", tz=ET),
    )
    ind.seed_from_orb_bars(orb_df)
    indicators_store["TQQQ"] = ind

    runner = SessionRunner(
        config=cfg, broker=mock_broker, state_store=tmp_store,
        bar_cache=_StubBarCache(), bar_router=_StubBarRouter(),
        pre_market_job=_StubPreMarket(), strategy_engine=engine,
        position_manager=mgr, risk_gate=gate,
        indicators_store=indicators_store,
        underlying_store=SimpleNamespace(),
        clock=_StubClock(), _sleep=lambda _: None,
    )

    # Pre-ORB 5-sec bar (09:59:55 < 10:00) that touches the boundary → ignored
    early = _bar(_et(9, 59).replace(second=55), close=102.0, hi=103.0, lo=101.5)
    runner._on_entry_bar_dispatch("TQQQ", early)
    assert engine.get_state("TQQQ") == SymbolState.ORB_COMPLETE
    assert not mock_broker._orders

    # Post-ORB 5-sec bar (10:00:05) touching the boundary → entry fires now
    tick = _bar(_et(10, 0).replace(second=5), close=102.0, hi=103.0, lo=101.5)
    runner._on_entry_bar_dispatch("TQQQ", tick)

    assert engine.get_state("TQQQ") == SymbolState.IN_POSITION, (
        "5-sec entry path must trigger breakout → IN_POSITION without a 1-min bar"
    )
    assert mock_broker._orders

    # breakout_signal row must be written
    from orb_live.core.state_store import breakout_signal as bs_table
    with tmp_store.conn() as c:
        rows = c.execute(bs_table.select()).mappings().all()
    assert len(rows) == 1, "breakout detection must write a breakout_signal row"
    assert rows[0]["symbol"] == "TQQQ"
    assert rows[0]["direction"] == 1


def test_l_broker_sleep_pumps_loop_for_bar_delivery(tmp_store):
    """
    Regression for BUG 0b: runner._sleep must pump the IB event loop.

    ib_insync's reqRealTimeBars updateEvent callbacks only fire while the event
    loop is being pumped.  time.sleep() starves the loop; ib.sleep() (forwarded
    via broker.sleep) keeps it alive so bars arrive during wait phases.

    This test uses a real asyncio loop to verify the pump behaviour — a pure Mock
    cannot catch this because it doesn't exercise blocking-vs-pumping.
    """
    import asyncio
    import threading
    from orb_live.config.live_config import load_live_config
    from orb_live.runner.session_runner import SessionRunner
    from orb_live.runner.strategy_engine import StrategyEngine
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager
    from orb_live.execution.risk_gate import RiskGate

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    class LoopAwareBroker:
        _quote = {"bid": 99.90, "ask": 100.10}
        _orders: dict = {}

        def get_account(self):         return {"equity": 100_000.0}
        def get_latest_quote(self, s): return dict(self._quote)
        def get_positions(self):       return {}
        def get_open_orders(self):     return []

        def sleep(self, seconds: float) -> None:
            fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(seconds), loop)
            fut.result(timeout=seconds + 2)

    broker = LoopAwareBroker()
    cfg    = load_live_config()
    scfg   = cfg.strategy_config
    exc    = SimpleNamespace(
        entry_slippage_bps=10, stop_order_type="market",
        session_kill_loss_pct=0.03, max_concurrent_positions=0,
        max_gross_exposure_pct=2.0, max_position_pct=0.50,
    )
    policy = MarketableLimitPolicy(broker, exc, tmp_store, _sleep=lambda _: None)
    gate   = RiskGate(exc, tmp_store, broker)
    gate.session_start(100_000.0, TDATE)
    ind_store: dict = {}
    mgr = LivePositionManager(
        broker=broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, indicators_store=ind_store, config=scfg,
    )
    engine = StrategyEngine(mgr, cfg, tmp_store, broker)
    engine.new_session(TDATE)

    # _sleep=None → SessionRunner picks up broker.sleep via getattr fallback
    runner = SessionRunner(
        config=cfg, broker=broker, state_store=tmp_store,
        bar_cache=_StubBarCache(), bar_router=_StubBarRouter(),
        pre_market_job=_StubPreMarket(), strategy_engine=engine,
        position_manager=mgr, risk_gate=gate,
        indicators_store=ind_store, underlying_store=SimpleNamespace(),
        clock=_StubClock(), _sleep=None,
    )

    dispatched: list = []

    async def _deliver_after_delay():
        await asyncio.sleep(0.05)  # fire 50ms into the 200ms sleep
        runner._on_bar_dispatch("TQQQ", _bar(_et(10, 5), close=100.5))
        dispatched.append(True)

    asyncio.run_coroutine_threadsafe(_deliver_after_delay(), loop)

    # broker.sleep pumps the real loop — the 50ms callback must fire before this returns
    runner._sleep(0.2)

    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=5)
    loop.close()

    assert dispatched, (
        "broker.sleep must pump the asyncio loop so bars scheduled inside it reach "
        "_on_bar_dispatch; failure means the runner is using time.sleep() which "
        "starves the ib_insync event loop (BUG 0b)"
    )


# ── m. EOD flatten runs BEFORE the close (RTH liquidity) ──────────────────────

class _ClockAt:
    def __init__(self, now):
        self._now = now
    def now_et(self): return self._now
    def orb_end_et(self, orb_minutes=30): return _et(10, 0)
    def is_half_day(self): return False
    def effective_close(self): return dtime(16, 0)


def _build_runner_for_eod(mock_broker, tmp_store, now, lead):
    from orb_live.runner.session_runner import SessionRunner

    engine, mgr, ind_store, cfg = _build_engine(mock_broker, tmp_store)
    cfg.eod_flatten_lead_secs = lead

    sleeps: list[float] = []
    runner = SessionRunner(
        config=cfg, broker=mock_broker, state_store=tmp_store,
        bar_cache=_StubBarCache(), bar_router=_StubBarRouter(),
        pre_market_job=_StubPreMarket(), strategy_engine=engine,
        position_manager=mgr, risk_gate=SimpleNamespace(),
        indicators_store=ind_store, underlying_store=SimpleNamespace(),
        clock=_ClockAt(now), _sleep=lambda s: sleeps.append(s),
    )
    return runner, mgr, sleeps


def test_m_wait_until_eod_wakes_before_close(mock_broker, tmp_store):
    """_wait_until_eod must wake `eod_flatten_lead_secs` BEFORE the close so the
    flatten runs during liquid RTH (market exits after 16:00 don't fill)."""
    runner, _mgr, sleeps = _build_runner_for_eod(
        mock_broker, tmp_store, now=_et(15, 50), lead=120)

    runner._wait_until_eod(TDATE)

    # 15:50 → 16:00 is 600s; wake 120s early → sleep ~480s.
    assert len(sleeps) == 1
    assert abs(sleeps[0] - 480.0) < 1.0


def test_m_run_eod_flattens_without_post_close_sleep(mock_broker, tmp_store):
    """_run_eod must flatten immediately (no sleep past the boundary)."""
    runner, mgr, sleeps = _build_runner_for_eod(
        mock_broker, tmp_store, now=_et(15, 58), lead=120)

    flattened: list[str] = []
    mgr.flatten_all = lambda reason: flattened.append(reason)

    runner._run_eod(TDATE)

    assert flattened == ["eod_sweep"]
    assert sleeps == []          # no 30s post-close sleep anymore


def test_base_notional_scales_with_equity(mock_broker, tmp_store):
    """base_notional_pct sizes the per-unit notional as a fraction of live equity
    (compounds with the account); falls back to the fixed value when unusable."""
    from types import SimpleNamespace
    from orb_live.runner.strategy_engine import StrategyEngine

    cfg = SimpleNamespace(base_notional_pct=0.10, v1_base_notional=1000.0)
    eng = StrategyEngine(None, cfg, tmp_store, mock_broker)

    assert eng._base_notional(50_000.0) == pytest.approx(5_000.0)   # 10% of equity
    assert eng._base_notional(0.0) == 1000.0                        # equity invalid → fixed


def test_base_notional_fixed_when_pct_zero(mock_broker, tmp_store):
    """base_notional_pct=0 keeps the fixed-dollar base regardless of equity."""
    from types import SimpleNamespace
    from orb_live.runner.strategy_engine import StrategyEngine

    cfg = SimpleNamespace(base_notional_pct=0.0, v1_base_notional=1000.0)
    eng = StrategyEngine(None, cfg, tmp_store, mock_broker)

    assert eng._base_notional(50_000.0) == 1000.0


def test_fetch_start_equity_retries_past_zero(mock_broker, tmp_store):
    """A 0.0 equity read (account not ready) must be retried, not trusted."""
    runner, _mgr, _sleeps = _build_runner_for_eod(
        mock_broker, tmp_store, now=_et(8, 30), lead=60)
    calls = {"n": 0}
    def acct():
        calls["n"] += 1
        return {"equity": 0.0 if calls["n"] < 3 else 10_000.0}
    mock_broker.get_account = acct

    assert runner._fetch_start_equity() == pytest.approx(10_000.0)
    assert calls["n"] == 3    # two zero reads, then a valid one


def test_fetch_start_equity_none_when_always_zero(mock_broker, tmp_store):
    """If equity never becomes valid, return None so the baseline stays unset
    (EOD then records a flat session, not a fake +$equity spike)."""
    runner, _mgr, _sleeps = _build_runner_for_eod(
        mock_broker, tmp_store, now=_et(8, 30), lead=60)
    mock_broker.get_account = lambda: {"equity": 0.0}

    assert runner._fetch_start_equity(retries=3) is None


def test_m_run_eod_records_real_session_pnl(mock_broker, tmp_store):
    """_run_eod must record session_pnl = end_equity − start_equity (captured at
    pre-market), not the old hardcoded 0 with start==end."""
    from orb_live.core.state_store import equity_curve

    runner, mgr, _sleeps = _build_runner_for_eod(
        mock_broker, tmp_store, now=_et(15, 58), lead=120)
    mgr.flatten_all = lambda reason: None

    runner._session_start_equity = 10_000.0     # baseline captured at pre-market
    mock_broker.set_equity(9_982.13)            # end equity after the day

    runner._run_eod(TDATE)

    with tmp_store.conn() as c:
        row = dict(c.execute(equity_curve.select()).mappings().all()[-1])
    assert row["start_equity"] == pytest.approx(10_000.0)
    assert row["end_equity"]   == pytest.approx(9_982.13)
    assert row["session_pnl"]  == pytest.approx(-17.87)


# ── n. use_resting_entries flag: place at ORB close, no reactive bar-watch ────

def test_n_resting_entries_flag_places_order_and_skips_bar_watch(mock_broker, tmp_store):
    """With use_resting_entries=True, on_orb_complete places a resting stop-limit
    and moves the symbol to RESTING — the reactive on_bar entry path must not
    fire (no double entry)."""
    from orb_live.runner.strategy_engine import SymbolState

    engine, mgr, _ind_store, cfg = _build_engine(mock_broker, tmp_store)
    cfg.use_resting_entries = True
    engine.new_session(TDATE)

    p2 = _make_p2("TQQQ")   # is_candidate, size_mult=1.0, gap_direction=1, orb set
    orb_df = pd.DataFrame(
        [{"high": 101.0, "low": 99.0, "close": 100.5}],
        index=pd.date_range("2026-01-07 09:30", periods=1, freq="1min", tz=ET),
    )
    engine.on_orb_complete("TQQQ", p2, orb_df)

    assert engine.get_state("TQQQ") == SymbolState.RESTING
    assert "TQQQ" in mgr._resting_entries          # resting order placed at 10:00

    # A breakout bar must NOT open a position via the reactive path.
    engine.on_bar("TQQQ", _bar(_et(10, 5), close=105.0), _et(10, 5))
    assert "TQQQ" not in mgr._positions


# ── connection resilience ─────────────────────────────────────────────────────

def _build_runner_with_sleep(mock_broker, tmp_store, sleep_fn):
    from orb_live.runner.session_runner import SessionRunner
    engine, mgr, ind_store, cfg = _build_engine(mock_broker, tmp_store)
    runner = SessionRunner(
        config=cfg, broker=mock_broker, state_store=tmp_store,
        bar_cache=_StubBarCache(), bar_router=_StubBarRouter(),
        pre_market_job=_StubPreMarket(), strategy_engine=engine,
        position_manager=mgr, risk_gate=SimpleNamespace(),
        indicators_store=ind_store, underlying_store=SimpleNamespace(),
        clock=_ClockAt(_et(10, 5)), _sleep=sleep_fn,
    )
    return runner


def test_resilient_sleep_reconnects_and_resumes(mock_broker, tmp_store):
    """A socket drop mid-sleep must reconnect + re-subscribe and finish the
    wait in place — not crash the session."""
    calls = {"n": 0}
    def flaky_sleep(_secs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("Socket disconnect")   # drop on first wait

    runner = _build_runner_with_sleep(mock_broker, tmp_store, flaky_sleep)
    runner._watched = ["TQQQ", "SOXL"]

    runner._sleep(1.0)   # must NOT raise

    assert mock_broker.reconnect_calls == 1          # reconnected once
    assert calls["n"] == 2                           # resumed the wait after reconnect
    assert runner._router.subscribed[-2:] == ["TQQQ", "SOXL"]  # bars re-subscribed


def test_resilient_sleep_reraises_when_reconnect_fails(mock_broker, tmp_store):
    """If the broker can't reconnect, the disconnect is surfaced (to the daemon
    backstop) rather than silently swallowed."""
    mock_broker._reconnect_ok = False
    def always_drops(_secs):
        raise ConnectionError("Socket disconnect")

    runner = _build_runner_with_sleep(mock_broker, tmp_store, always_drops)
    with pytest.raises(ConnectionError):
        runner._sleep(1.0)
