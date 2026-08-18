"""
tests/test_pre_market.py — Unit tests for Phase 1 gap reference price and
open-eval clock trigger.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")
TDATE = date(2026, 1, 7)


def test_phase2_tp1_mult_is_class_based():
    """Phase-2 sets tp1_mult per the class map: C1 (crypto) → 2× ORB, C2/C3 → 1×.
    Regression guard: tp1_mult must NOT be hardcoded — it is passed to
    compute_entry as an override and so silently beats config if wrong (was once
    pinned to 1.0, cutting every winner to 1× ORB). Uses the REAL config."""
    from orb_live.config.live_config import load_live_config
    from orb_live.signals.pre_market import PreMarketJob, Phase1Result

    cfg = load_live_config()
    job = PreMarketJob(cfg, MagicMock(), MagicMock(), MagicMock())
    orb = {"high": 156.0, "low": 153.0, "midpoint": 154.5}

    def _tp1(symbol):
        p1 = Phase1Result(symbol=symbol, gap_abs=0.05, gap_direction=-1,
                          prior_close=150.0, ps_filter_passed=True)
        return job._make_p2(p1, orb=orb, first_open=153.0, size_mult=1.0,
                            preflight=None, is_candidate=True).tp1_mult

    assert _tp1("BITX") == 2.0, "C1 (crypto) TP1 must be 2× ORB range"
    assert _tp1("NUGT") == 1.0, "C2 (gold) TP1 must be 1× ORB range"
    assert _tp1("TQQQ") == 1.0, "C3 (broad leveraged) TP1 must be 1× ORB range"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bar_df(close: float, ts_hour: int = 9, ts_minute: int = 30) -> pd.DataFrame:
    ts = datetime(TDATE.year, TDATE.month, TDATE.day, ts_hour, ts_minute, tzinfo=ET)
    return pd.DataFrame(
        {"open": [close - 0.5], "high": [close + 0.5],
         "low": [close - 0.5], "close": [close], "volume": [10_000]},
        index=[ts],
    )


def _make_job(client) -> object:
    """Construct a PreMarketJob with only _client populated (no other deps needed)."""
    from orb_live.signals.pre_market import PreMarketJob
    job = PreMarketJob.__new__(PreMarketJob)
    job._client = client
    return job


# ── _get_ref_price ─────────────────────────────────────────────────────────────

class TestGetRefPrice:
    def test_uses_930_bar_close(self):
        """_get_ref_price must return the close of the first RTH bar (09:30)."""
        client = MagicMock()
        client.get_intraday_bars.return_value = _make_bar_df(185.50)

        job = _make_job(client)
        ref = job._get_ref_price("TQQQ", None, TDATE)

        assert ref == pytest.approx(185.50)

    def test_fetch_starts_at_0930(self):
        """The fetch must start at 09:30 ET to guarantee the first bar is the RTH open bar."""
        client = MagicMock()
        client.get_intraday_bars.return_value = _make_bar_df(200.0)

        job = _make_job(client)
        job._get_ref_price("TQQQ", None, TDATE)

        args, kwargs = client.get_intraday_bars.call_args
        # Second positional arg is `start`
        start = args[1]
        assert start.hour == 9 and start.minute == 30, (
            f"Expected start=09:30 (RTH filter), got {start.hour}:{start.minute:02d}"
        )

    def test_none_when_bar_unavailable(self):
        """Empty bars → None so the symbol is recorded as no_ref_price."""
        client = MagicMock()
        client.get_intraday_bars.return_value = pd.DataFrame()

        job = _make_job(client)
        assert job._get_ref_price("TQQQ", None, TDATE) is None

    def test_none_on_broker_exception(self):
        """Broker exception → None (fail gracefully, not with an unhandled error)."""
        client = MagicMock()
        client.get_intraday_bars.side_effect = RuntimeError("timeout")

        job = _make_job(client)
        assert job._get_ref_price("TQQQ", None, TDATE) is None

    def test_injected_ref_price_bypasses_broker(self):
        """Pre-injected ref_prices dict takes priority over the broker fetch."""
        client = MagicMock()

        job = _make_job(client)
        ref = job._get_ref_price("TQQQ", {"TQQQ": 99.99}, TDATE)

        assert ref == pytest.approx(99.99)
        client.get_intraday_bars.assert_not_called()


# ── Gap formula parity with backtest ─────────────────────────────────────────

class TestGapFormulaBacktestParity:
    def test_phase1_gap_matches_backtest_formula(self):
        """
        The live gap calculation must produce the same result as the backtest:
            gap = (bar_close - prior_close) / prior_close
        where bar_close is the close of the 9:30 bar.
        """
        from orb_live.signals.strategy_signals import compute_gap

        prior_close = 100.0
        bar_close   = 107.0

        daily_df = pd.DataFrame({
            "date":  [pd.Timestamp("2026-01-06")],
            "close": [prior_close],
        })

        gap_result = compute_gap(TDATE, daily_df, bar_close)
        assert gap_result is not None

        gap_abs, direction, pc = gap_result
        expected_gap = (bar_close - prior_close) / prior_close

        assert gap_abs == pytest.approx(abs(expected_gap))
        assert direction == 1    # gap up
        assert pc == pytest.approx(prior_close)

    def test_gap_down_direction(self):
        prior_close = 100.0
        bar_close   = 93.0

        from orb_live.signals.strategy_signals import compute_gap
        daily_df = pd.DataFrame({
            "date":  [pd.Timestamp("2026-01-06")],
            "close": [prior_close],
        })

        gap_result = compute_gap(TDATE, daily_df, bar_close)
        assert gap_result is not None
        gap_abs, direction, _ = gap_result
        assert direction == -1
        assert gap_abs == pytest.approx((prior_close - bar_close) / prior_close)


# ── Open-eval clock trigger ───────────────────────────────────────────────────

class TestOpenEvalTrigger:
    def test_open_eval_trigger_is_0931(self):
        """next_open_eval_start() must return 09:31:00 ET on the current trading day
        when called before 09:31."""
        from orb_live.core.clock import MarketClock, OPEN_EVAL_START

        fixed = datetime(2026, 5, 19, 8, 45, 0, tzinfo=ET)  # Monday 08:45
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: fixed  # type: ignore[method-assign]

        trigger = mc.next_open_eval_start()

        assert trigger.date() == date(2026, 5, 19)
        assert trigger.hour   == OPEN_EVAL_START.hour   == 9
        assert trigger.minute == OPEN_EVAL_START.minute == 31
        assert trigger.second == 0
        assert trigger.tzinfo is not None

    def test_open_eval_skips_holidays(self):
        """next_open_eval_start() must skip Memorial Day (2026-05-25) to 2026-05-26."""
        from orb_live.core.clock import MarketClock

        # Friday 2026-05-22 evening — session closed; next trading day skips holiday
        fixed = datetime(2026, 5, 22, 18, 0, 0, tzinfo=ET)
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: fixed  # type: ignore[method-assign]

        trigger = mc.next_open_eval_start()

        # May 25 = Memorial Day; expect May 26 (Tuesday)
        assert trigger.date() == date(2026, 5, 26)
        assert trigger.hour   == 9
        assert trigger.minute == 31

    def test_open_eval_after_trigger_time_returns_past(self):
        """After 09:31 on a trading day next_open_eval_start returns today at 09:31
        (in the past), so the runner proceeds without sleeping."""
        from orb_live.core.clock import MarketClock

        fixed = datetime(2026, 5, 19, 10, 5, 0, tzinfo=ET)  # 10:05 — already past
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: fixed  # type: ignore[method-assign]

        trigger = mc.next_open_eval_start()

        assert trigger.date() == date(2026, 5, 19)
        assert trigger < fixed  # confirms it's in the past → no sleep


# ── P0: ul_gap sign and regime parity ────────────────────────────────────────

def _make_phase1_job(symbols, instruments, sigmas, underlying_dfs, ref_prices, daily_bars_map):
    """Build a PreMarketJob with mocked dependencies for run_phase1() unit tests."""
    from orb_live.signals.pre_market import PreMarketJob
    from orb_live.config.live_config import LiveConfig
    from orb_live.strategy.v1_strategy import build_universe, _make_ps_filters_from_instruments

    cfg = MagicMock(spec=LiveConfig)
    cfg.symbols = symbols
    cfg.instruments = instruments
    cfg.sigmas = sigmas
    cfg.direction_filters = {}
    # Build prior_session_filters from instruments (no thresholds needed for these tests)
    cfg.prior_session_filters = {s: (instruments[s].underlying, 0.0)
                                  for s in symbols if s in instruments}
    cfg.strategy_config = MagicMock()

    store = MagicMock()
    store.save_gap_scan = MagicMock()
    store.save_ps_filter = MagicMock()
    store.save_candidate = MagicMock()

    client = MagicMock()
    client.get_daily_bars.side_effect = lambda sym, **kw: daily_bars_map.get(sym, pd.DataFrame())

    ul_store = MagicMock()
    ul_store.get.side_effect = lambda ul: underlying_dfs.get(ul, pd.DataFrame())
    ul_store.warn_if_stale.return_value = None

    job = PreMarketJob(cfg, store, client, ul_store)
    return job


def _instrument(sym, ul, leverage, inverse=False):
    from orb_live.strategy.v1_strategy import Instrument
    return Instrument(symbol=sym, underlying=ul, leverage=leverage, inverse=inverse)


def _daily(dates_closes: list[tuple]) -> pd.DataFrame:
    dates  = [pd.Timestamp(d) for d, _ in dates_closes]
    closes = [c for _, c in dates_closes]
    return pd.DataFrame({"date": dates, "close": closes, "volume": [100_000] * len(dates)})


class TestUlGapSignFix:
    """P0: inverse ETF ul_gap must have the SAME sign as the underlying's move."""

    def test_inverse_etf_ul_gap_sign(self):
        """When a bear ETF gaps DOWN (UL gapped UP), ul_gap fed to overnight_gaps
        must be positive (matching the UL direction), not negative."""
        from orb_live.signals.pre_market import PreMarketJob
        from orb_live.strategy.v1_strategy import Instrument
        from orb_live.config.live_config import LiveConfig

        session_date = date(2026, 6, 22)

        # Bull IBB ETF (non-inverse, 3x) — gapped UP 6.6%
        # Bear IBB ETF (inverse, 3x)    — gapped DOWN 7.1% (UL went up)
        instrs = {
            "LABU": Instrument("LABU", "IBB", 3, False),
            "LABD": Instrument("LABD", "IBB", 3, True),
        }
        sigmas = {"IBB": 0.01}  # tight threshold — PS should pass when prior move is ~0

        # IBB prior two closes: flat (prior session ~0% move)
        ibb_df = _daily([
            ("2026-06-17", 173.44),
            ("2026-06-18", 173.64),
        ])
        underlying_dfs = {"IBB": ibb_df}

        # ETF daily bars: prior close used by compute_gap
        labu_bars = _daily([("2026-06-18", 160.0)])
        labd_bars = _daily([("2026-06-18",  50.0)])
        daily_bars_map = {"LABU": labu_bars, "LABD": labd_bars}

        cfg = MagicMock(spec=LiveConfig)
        cfg.symbols = ["LABU", "LABD"]
        cfg.instruments = instrs
        cfg.sigmas = sigmas
        cfg.direction_filters = {"LABU": +1, "LABD": -1}
        cfg.prior_session_filters = {
            "LABU": ("IBB", 0.01),
            "LABD": ("IBB", 0.01, True),
        }
        cfg.strategy_config = MagicMock()

        store = MagicMock()
        store.save_gap_scan = MagicMock()
        store.save_ps_filter = MagicMock()

        client = MagicMock()
        client.get_daily_bars.side_effect = lambda sym, **kw: daily_bars_map.get(sym, pd.DataFrame())

        ul_store = MagicMock()
        ul_store.get.side_effect = lambda ul: underlying_dfs.get(ul, pd.DataFrame())
        ul_store.warn_if_stale.return_value = None

        job = PreMarketJob(cfg, store, client, ul_store)

        # Inject ref_prices: LABU at 171.2 (gap-up 7% on 160 close, well above 3×2%=6%)
        #                    LABD at 46.5  (gap-down 7% on 50 close, inverse)
        ref_prices = {"LABU": 171.2, "LABD": 46.5}

        results = job.run_phase1(session_date, ref_prices=ref_prices, daily_bars=daily_bars_map)

        # With the ul_gap sign fix, IBB contributes to overnight_gaps with the correct
        # positive sign, so plan_session() sees IBB as a qualifying UL.
        # With the skip-cheap N==2 fix, only the pricier ETF is kept: LABU (160) > LABD (50).
        symbols_in_plan = {r.symbol for r in results}
        assert "LABU" in symbols_in_plan, (
            "LABU must be a Phase1 candidate — ul_gap sign bug would cause direction "
            "filter to reject it in plan_session()"
        )
        assert "LABD" not in symbols_in_plan, (
            "LABD must be dropped by skip-cheap (N==2: keep pricier only; LABU 160 > LABD 50)"
        )

    def test_n_uls_regime_quiet_to_active(self):
        """plan_session: n_uls < 5 → quiet; n_uls >= 5 → active."""
        from orb_live.strategy.v1_strategy import plan_session, Instrument, ACTIVE_MIN

        instrs = {}
        uls = ["SPY", "QQQ", "IWM", "IBB", "SOXX", "GDX", "AMD"]
        for ul in uls:
            bull = f"{ul}_B"
            instrs[bull] = Instrument(bull, ul, 3, False)

        universe = list(instrs.keys())
        sigmas   = {ul: 0.015 for ul in uls}

        # Gap each UL at exactly 2.5% UP; flat prior session → PS passes
        overnight_gaps    = {ul: 0.025 for ul in uls}
        prior_two_closes  = {ul: (100.0, 100.0) for ul in uls}  # 0% prior session
        prior_etf_close   = {f"{ul}_B": 50.0 for ul in uls}

        for n in range(1, len(uls) + 1):
            sub_uls  = uls[:n]
            sub_gaps = {ul: overnight_gaps[ul] for ul in sub_uls}
            sub_ptc  = {ul: prior_two_closes[ul] for ul in sub_uls}
            sub_etc  = {f"{ul}_B": 50.0 for ul in sub_uls}
            sub_univ = [f"{ul}_B" for ul in sub_uls]

            plan = plan_session(sub_univ, instrs, sigmas, sub_gaps, sub_ptc, sub_etc)
            expected_regime = "active" if n >= ACTIVE_MIN else "quiet"
            assert plan.regime == expected_regime, (
                f"n_uls={n}: expected {expected_regime}, got {plan.regime} "
                f"(n_uls in plan={plan.n_uls})"
            )
            assert plan.n_uls == n, f"Expected n_uls={n}, got {plan.n_uls}"

    def test_gap_threshold_parity_phase1_vs_plan_session(self):
        """Phase-1 ETF gap threshold (leverage × 2%) is equivalent to |ul_gap| ≥ 2%
        used by plan_session. Verify with a non-inverse 3x ETF at exactly the boundary."""
        from orb_live.strategy.v1_strategy import GAP_THRESHOLD, plan_session, Instrument

        leverage = 3
        ul       = "SOXX"
        sym      = "SOXL"
        inst     = Instrument(sym, ul, leverage, False)

        # ETF gap = 1% above the boundary (leverage × GAP_THRESHOLD + 1%), should pass
        above_boundary_etf_gap = leverage * GAP_THRESHOLD + 0.01   # 0.07

        # ul_gap > GAP_THRESHOLD: plan_session should qualify
        ul_gap = above_boundary_etf_gap / leverage   # 0.0233... > 0.02

        plan = plan_session(
            [sym], {sym: inst}, {"SOXX": 0.015},
            {ul: ul_gap}, {ul: (100.0, 100.0)}, {sym: 50.0},
        )
        assert sym in plan.candidates, (
            f"ETF gap above boundary ({above_boundary_etf_gap*100:.1f}%) "
            f"should produce |ul_gap| > GAP_THRESHOLD and qualify"
        )
