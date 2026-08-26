"""Margin-aware selection, rate seeding, and partial fills.

Three changes, all driven by measured IB rates rather than the FINRA minima the
model used to assume (see orb_live/scripts/measure_universe_margin.py):

  1. Rates are seeded from master_universe.csv at session start instead of
     depending on prewarm_margin. That path never ran in production — the
     post-ORB loop reads a bar cache the subscription only fills afterwards —
     so _margin_budget stayed None and the whole margin gate was dead code.

  2. Entries that do not fit the remaining budget are sized down rather than
     dropped. Leaving budget idle is strictly worse than a smaller position.

  3. Siblings whose margin rate makes them unfillable are removed before
     skip-cheap picks the most expensive. Shorting ETHU costs 4.09x notional:
     at weight 3.0 on a $32k account that needs ~$39k against ~$32k available,
     so it can never fill, and the backtest was booking 96 such trades.

Rates are keyed by SIDE because the two differ and not by a constant: SQQQ is
0.79 long and 0.95 short, TSLL is 1.00 long and 0.72 short.
"""

from datetime import date, timezone
from types import SimpleNamespace

UTC = timezone.utc
TRADE_DATE = date(2026, 1, 5)


def _exe_cfg(**overrides):
    ns = SimpleNamespace(
        session_kill_loss_pct=0.03,
        max_concurrent_positions=0,
        max_gross_exposure_pct=10.0,
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


def _build_mgr(mock_broker, tmp_store, strategy_cfg=None, exe_cfg=None):
    from orb_live.execution.risk_gate import RiskGate
    from orb_live.execution.order_policy import MarketableLimitPolicy
    from orb_live.execution.position_manager import LivePositionManager

    exc = exe_cfg or _exe_cfg()
    scfg = strategy_cfg or _strategy_cfg()
    policy = MarketableLimitPolicy(mock_broker, exc, tmp_store, _sleep=lambda _: None)
    gate = RiskGate(exc, tmp_store, mock_broker)
    gate.session_start(100_000.0, TRADE_DATE)
    return LivePositionManager(
        broker=mock_broker, policy=policy, state_store=tmp_store,
        risk_gate=gate, config=scfg,
    )


def _make_entry(entry_price=100.0, shares=6):
    return {
        "entry_price": entry_price,
        "shares": shares,
        "orb_range": 1.0,
        "stop_price": entry_price - 1.0,
        "tp1_price": entry_price + 2.0,
        "tp2_price": entry_price + 2.0,
        "tp1_shares": shares,
        "tp2_shares": 0,
        "tp3_shares": 0,
        "exit_override": {},
    }


# ── 1. Rates seeded from the profile, no broker round-trip ───────────────────

def test_seed_margin_rates_sets_budget_without_prewarm(mock_broker, tmp_store):
    """The gate must arm from the profile alone.

    prewarm_margin is unreachable in production, so a manager that only arms
    via prewarm has its margin gate permanently disabled.
    """
    mock_broker.set_equity(50_000.0)
    mgr = _build_mgr(mock_broker, tmp_store)

    assert mgr._margin_budget is None          # not armed yet

    mgr.seed_margin_rates({("FNGU", 1): 0.79, ("FNGU", -1): 0.95})

    assert mgr._margin_budget == 50_000.0
    assert abs(mgr.margin_rate_for("FNGU", 1) - 0.79) < 1e-9
    assert abs(mgr.margin_rate_for("FNGU", -1) - 0.95) < 1e-9


def test_seeded_rate_is_side_specific(mock_broker, tmp_store):
    """TSLL is 1.00 long and 0.72 short — one number per symbol is wrong."""
    mock_broker.set_equity(50_000.0)
    mgr = _build_mgr(mock_broker, tmp_store)
    mgr.seed_margin_rates({("TSLL", 1): 1.00, ("TSLL", -1): 0.72})

    assert mgr.margin_rate_for("TSLL", 1) > mgr.margin_rate_for("TSLL", -1)


def test_prewarm_still_overrides_a_seeded_rate(mock_broker, tmp_store):
    """A live measurement beats the stored table when we can get one."""
    mock_broker.set_equity(50_000.0)
    mock_broker._margin_rate = 0.60
    mgr = _build_mgr(mock_broker, tmp_store)
    mgr.seed_margin_rates({("FNGU", 1): 0.79})

    mgr.prewarm_margin("FNGU", "buy", 10, 100.0)

    assert abs(mgr.margin_rate_for("FNGU", 1) - 0.60) < 1e-9


def test_unknown_symbol_falls_back_to_conservative_rate(mock_broker, tmp_store):
    mock_broker.set_equity(50_000.0)
    mgr = _build_mgr(mock_broker, tmp_store)
    mgr.seed_margin_rates({("FNGU", 1): 0.79})

    assert mgr.margin_rate_for("NOPE", 1) == 1.0


# ── 2. Partial fills ─────────────────────────────────────────────────────────

def test_partial_fill_uses_the_remaining_budget(mock_broker, tmp_store):
    """Budget 1000, rate 1.0. First entry takes 600, leaving 400.

    The second entry wants 600 and would currently be dropped entirely. It
    should instead be cut to the 4 shares the remaining 400 supports.

    Slippage is disabled so the arithmetic is exact — at the default 10bps the
    fill is 100.1 and only 3 shares fit, which is correct but obscures what is
    being pinned here.
    """
    mock_broker.set_equity(1_000.0)
    mgr = _build_mgr(
        mock_broker, tmp_store,
        _strategy_cfg(allow_partial_entries=True),
        _exe_cfg(entry_slippage_bps=0),
    )
    mgr.seed_margin_rates({("AAA", 1): 1.0, ("BBB", 1): 1.0})

    pos_a = mgr.open_position(_make_entry(shares=6), "AAA", +1, TRADE_DATE)
    pos_b = mgr.open_position(_make_entry(shares=6), "BBB", +1, TRADE_DATE)

    assert pos_a is not None
    assert pos_b is not None, "second entry should be partially filled, not dropped"
    assert mgr._positions["BBB"].remaining == 4


def test_partial_fill_never_exceeds_the_budget(mock_broker, tmp_store):
    """The invariant that matters, independent of fill price.

    Whatever size the partial lands on, committed margin must stay inside the
    budget — a partial that overshoots is worse than the rejection it replaced.
    """
    mock_broker.set_equity(1_000.0)
    mgr = _build_mgr(mock_broker, tmp_store, _strategy_cfg(allow_partial_entries=True))
    mgr.seed_margin_rates({("AAA", 1): 1.0, ("BBB", 1): 1.0})

    mgr.open_position(_make_entry(shares=6), "AAA", +1, TRADE_DATE)
    pos_b = mgr.open_position(_make_entry(shares=6), "BBB", +1, TRADE_DATE)

    assert pos_b is not None
    assert 1 <= mgr._positions["BBB"].remaining < 6
    assert mgr._committed_margin() <= mgr._margin_budget + 1e-6


def test_partial_fill_below_floor_is_rejected(mock_broker, tmp_store):
    """A sliver is a rounding artifact, not a trade — live buys whole shares."""
    mock_broker.set_equity(1_000.0)
    mgr = _build_mgr(
        mock_broker, tmp_store,
        _strategy_cfg(allow_partial_entries=True, partial_entry_min_frac=0.5),
    )
    mgr.seed_margin_rates({("AAA", 1): 1.0, ("BBB", 1): 1.0})

    mgr.open_position(_make_entry(shares=9), "AAA", +1, TRADE_DATE)   # uses 900
    pos_b = mgr.open_position(_make_entry(shares=6), "BBB", +1, TRADE_DATE)

    # Only 100 left => 1 of 6 shares => 0.17 < 0.5 floor.
    assert pos_b is None
    assert "BBB" not in mgr._positions


def test_partial_fills_off_by_default_preserves_all_or_nothing(mock_broker, tmp_store):
    mock_broker.set_equity(1_000.0)
    mgr = _build_mgr(mock_broker, tmp_store)
    mgr.seed_margin_rates({("AAA", 1): 1.0, ("BBB", 1): 1.0})

    mgr.open_position(_make_entry(shares=6), "AAA", +1, TRADE_DATE)
    pos_b = mgr.open_position(_make_entry(shares=6), "BBB", +1, TRADE_DATE)

    assert pos_b is None


# ── 3. Unfillable siblings are dropped before skip-cheap ─────────────────────

def _instruments():
    from orb_live.strategy.v1_strategy import Instrument
    return {
        # Bull and bear on one underlying. The bull is pricier, so skip-cheap
        # picks it — but on a gap DOWN it would be shorted, and its short rate
        # makes it unfillable.
        "BULL": Instrument(symbol="BULL", underlying="UL", leverage=2,
                           inverse=False, margin_long=1.00, margin_short=4.09),
        "BEAR": Instrument(symbol="BEAR", underlying="UL", leverage=2,
                           inverse=True, margin_long=1.01, margin_short=1.12),
    }


def _plan(max_rate):
    from orb_live.strategy import v1_strategy as v1
    return v1.compute_candidates(
        universe=["BULL", "BEAR"],
        instruments=_instruments(),
        sigmas={},
        overnight_gaps={"UL": -0.05},        # gap DOWN
        prior_two_closes={},
        prior_etf_close={"BULL": 100.0, "BEAR": 20.0},
        max_margin_rate=max_rate,
    )


def test_without_a_cap_skip_cheap_picks_the_unfillable_short():
    """Current behaviour, pinned so the change is visible."""
    cands = _plan(max_rate=None)
    assert [c.symbol for c in cands] == ["BULL"]
    assert cands[0].etf_dir == -1             # shorting the bull fund


def test_cap_drops_the_unfillable_sibling_and_takes_the_bear_long():
    cands = _plan(max_rate=2.0)
    assert [c.symbol for c in cands] == ["BEAR"]
    assert cands[0].etf_dir == +1             # long the inverse fund instead


def test_cap_keeps_the_price_pick_when_every_sibling_is_capped():
    """Two impossible trades is not a reason to invent a preference."""
    cands = _plan(max_rate=0.5)
    assert [c.symbol for c in cands] == ["BULL"]


# ── 4. Live rates are re-pulled from IB each session ─────────────────────────

def test_margin_is_prewarmed_from_ib_even_with_an_empty_bar_cache(
    mock_broker, tmp_store, monkeypatch,
):
    """House requirements move, so the stored table is a fallback, not a source.

    prewarm_margin has to run every session. It previously sat after a
    bar-cache read in _run_post_orb, and the market-data subscription that
    fills that cache is deliberately deferred until after the loop (IB's line
    cap). So every symbol hit the `raw.empty` branch and continued first, and
    no session ever measured a rate.
    """
    from types import SimpleNamespace
    from datetime import datetime, timezone as _tz

    from orb_live.runner.session_runner import SessionRunner
    from orb_live.tests.test_session_runner import (
        _StubBarCache, _StubBarRouter, _StubClock, _StubPreMarket, _make_p1, _make_p2,
    )

    prewarmed = []

    class _Mgr:
        def __init__(self):
            self._margin_budget = None

        def prewarm_margin(self, symbol, side, qty, price):
            prewarmed.append(symbol)

        def seed_margin_rates(self, rates, budget=None):
            pass

        def flatten_all(self, *a, **kw):
            return []

    cfg = SimpleNamespace(
        orb_minutes=30, eod_exit_hour=16, eod_exit_minute=0,
        eod_flatten_lead_secs=120, v1_base_notional=1000.0,
        base_notional_pct=0.10, instruments={},
        strategy_config=SimpleNamespace(),
    )

    runner = SessionRunner(
        config=cfg, broker=mock_broker, state_store=tmp_store,
        bar_cache=_StubBarCache(),            # EMPTY — the production condition
        bar_router=_StubBarRouter(),
        pre_market_job=_StubPreMarket([_make_p1("TQQQ")], [_make_p2("TQQQ")]),
        strategy_engine=SimpleNamespace(on_orb_complete=lambda *a, **kw: None),
        position_manager=_Mgr(), risk_gate=SimpleNamespace(),
        underlying_store=SimpleNamespace(),
        clock=_StubClock(), _sleep=lambda _: None,
    )
    runner._phase1_results = [_make_p1("TQQQ")]
    monkeypatch.setattr(runner, "_wait_until_eod", lambda *_a, **_kw: None)

    runner._run_post_orb(datetime.now(_tz.utc).date())

    assert prewarmed == ["TQQQ"], (
        "prewarm must run for every candidate, not only those with cached bars"
    )
