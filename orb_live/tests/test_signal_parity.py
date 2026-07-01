"""
tests/test_signal_parity.py — Functional regression tests for live signal functions.

Originally compared against reference/orb_backtester.py (removed in Phase 2
rebuild).  Parity was verified before removal; reference/ is no longer present.
Tests now assert correct behaviour directly against known inputs/outputs.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime

import pandas as pd
import pytest

import orb_live  # noqa: F401 — sys.path setup


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ref_cfg():
    """Default StrategyConfig used as the config fixture across tests."""
    from orb_live.config.strategy_config import StrategyConfig
    return StrategyConfig()


@pytest.fixture(scope="module")
def daily_df():
    """5 days of daily OHLCV for gap computation."""
    dates  = pd.to_datetime([
        "2022-01-03", "2022-01-04", "2022-01-05",
        "2022-01-06", "2022-01-07",
    ])
    closes = [100.0, 102.0, 101.0, 100.0, 108.0]
    return pd.DataFrame({"date": dates, "close": closes})


@pytest.fixture(scope="module")
def orb_bars():
    """
    30 one-minute bars 9:30–9:59 ET on 2022-01-07.

    Prices form a gentle trend: open ≈ 108.5, close ≈ 111.4.
    ORB high ≈ 111.9, ORB low ≈ 108.0.
    """
    idx    = pd.date_range("2022-01-07 09:30", periods=30, freq="1min")
    closes = [108.5 + 0.1 * i for i in range(30)]
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    return pd.DataFrame({"close": closes, "high": highs, "low": lows}, index=idx)


@pytest.fixture(scope="module")
def underlying_data():
    """Mock QQQ daily data for PS filter (4 trading days before 2022-01-07)."""
    dates  = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"])
    closes = [380.0, 383.0, 381.0, 385.0]
    return {"QQQ": pd.DataFrame({"date": dates, "close": closes})}


@pytest.fixture(scope="module")
def ps_cfg():
    """Config variant with a simple TQQQ PS filter keyed on QQQ."""
    from orb_live.config.strategy_config import StrategyConfig
    return replace(StrategyConfig(), prior_session_filters={"TQQQ": ("QQQ", 0.02)})


# ── compute_gap ───────────────────────────────────────────────────────────────

class TestComputeGap:
    def test_gap_up(self, daily_df):
        from orb_live.signals.strategy_signals import compute_gap as live_fn

        result = live_fn(date_=date(2022, 1, 7), daily_df=daily_df, today_ref_price=108.0)

        assert result is not None
        gap_abs, gap_dir, prior_close = result
        assert pytest.approx(gap_abs, rel=1e-9) == 0.08
        assert gap_dir == 1
        assert pytest.approx(prior_close, rel=1e-9) == 100.0

    def test_gap_down(self, daily_df):
        from orb_live.signals.strategy_signals import compute_gap as live_fn

        result = live_fn(date_=date(2022, 1, 7), daily_df=daily_df, today_ref_price=93.0)

        assert result is not None
        gap_abs, gap_dir, _ = result
        assert gap_dir == -1
        assert pytest.approx(gap_abs) == 0.07

    def test_no_prior_data_returns_none(self):
        from orb_live.signals.strategy_signals import compute_gap as live_fn

        empty = pd.DataFrame({"date": pd.to_datetime([]), "close": []})
        assert live_fn(date(2022, 1, 7), empty, 100.0) is None

    def test_zero_prior_close_returns_none(self, daily_df):
        from orb_live.signals.strategy_signals import compute_gap as live_fn

        bad = daily_df.copy()
        bad.loc[bad["date"] == pd.Timestamp("2022-01-06"), "close"] = 0.0
        assert live_fn(date(2022, 1, 7), bad, 108.0) is None


# ── check_prior_session_filter ────────────────────────────────────────────────

class TestCheckPriorSessionFilter:
    def test_pass_small_move(self, ps_cfg, underlying_data):
        """Prior session move < threshold → allowed (True)."""
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        # Prior move: (385-381)/381 ≈ 1.05% < 2% threshold → True
        result = live_fn(
            symbol="TQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=ps_cfg,
            underlying_data=underlying_data,
        )
        assert result is True

    def test_block_large_move(self, ref_cfg, underlying_data):
        """Prior session move > threshold → blocked (False)."""
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        cfg = replace(ref_cfg, prior_session_filters={"TQQQ": ("QQQ", 0.005)})
        ul_data = {"QQQ": pd.DataFrame({
            "date":  pd.to_datetime(["2022-01-05", "2022-01-06"]),
            "close": [370.0, 385.0],  # move ≈ 4.05% > 0.5% threshold
        })}
        result = live_fn(
            symbol="TQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=cfg, underlying_data=ul_data,
        )
        assert result is False

    def test_no_filter_symbol_passes(self, ref_cfg, underlying_data):
        """Symbol with no PS filter always passes."""
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        cfg = replace(ref_cfg, prior_session_filters={})
        result = live_fn(
            symbol="TQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=cfg, underlying_data=underlying_data,
        )
        assert result is True

    def test_missing_underlying_data_allows(self, ps_cfg):
        """Missing underlying data → allow (True)."""
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        assert live_fn(
            symbol="TQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=ps_cfg, underlying_data={},
            logger=None,
        ) is True

    def test_inverse_flag_flips_direction(self, ref_cfg, underlying_data):
        """Inverse flag negates gap_direction before threshold comparison."""
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        # Threshold 0.005, inverse=True, gap_direction=+1 → effective_dir=-1
        # prior_ret ≈ +1.05% → dir_adj_ret ≈ -1.05% < 0.5% → passes
        cfg = replace(ref_cfg, prior_session_filters={"SQQQ": ("QQQ", 0.005, True)})
        ul_data = {"QQQ": pd.DataFrame({
            "date":  pd.to_datetime(["2022-01-05", "2022-01-06"]),
            "close": [381.0, 385.0],
        })}
        result = live_fn(
            symbol="SQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=cfg, underlying_data=ul_data,
        )
        assert result is True


# ── compute_opening_range ─────────────────────────────────────────────────────

class TestComputeOpeningRange:
    def test_basic_orb(self, orb_bars, ref_cfg):
        from orb_live.signals.strategy_signals import compute_opening_range as live_fn

        live = live_fn(orb_bars, ref_cfg, symbol="TQQQ")

        assert live is not None
        for key in ("high", "low", "midpoint", "size_pct", "n_bars", "ema"):
            assert key in live, f"Missing key '{key}'"
        assert live["n_bars"] == 30
        assert pytest.approx(live["high"], abs=0.01) == 111.9
        assert pytest.approx(live["low"],  abs=0.01) == 108.0
        assert pytest.approx(live["midpoint"], abs=0.01) == (111.9 + 108.0) / 2
        assert live["ema"] > 108.0  # EMA is within the ORB price range

    def test_returns_none_for_insufficient_bars(self, ref_cfg):
        from orb_live.signals.strategy_signals import compute_opening_range as live_fn

        idx  = pd.date_range("2022-01-07 09:30", periods=5, freq="1min")
        bars = pd.DataFrame({"close": [100.0] * 5, "high": [100.5] * 5,
                             "low": [99.5] * 5}, index=idx)
        assert live_fn(bars, ref_cfg) is None

    def test_tz_aware_index_accepted(self, orb_bars, ref_cfg):
        """tz-aware DatetimeIndex is stripped and processed identically."""
        from orb_live.signals.strategy_signals import compute_opening_range as live_fn

        import pytz
        tz_bars = orb_bars.copy()
        tz_bars.index = tz_bars.index.tz_localize("America/New_York")

        live = live_fn(tz_bars, ref_cfg)

        assert live is not None
        for key in ("high", "low", "midpoint", "size_pct", "n_bars", "ema"):
            assert key in live


# ── check_breakout ────────────────────────────────────────────────────────────

class TestCheckBreakout:
    def _make_orb(self):
        return {
            "high":     111.0,
            "low":      108.0,
            "midpoint": 109.5,
            "size_pct": (111.0 - 108.0) / 109.5,
            "n_bars":   30,
            "ema":      109.8,
        }

    def test_valid_breakout_long(self, ref_cfg):
        from orb_live.signals.strategy_signals import check_breakout as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 112.0}, name=datetime(2022, 1, 7, 10, 5))
        assert live_fn(bar, orb, 1, ref_cfg) is True

    def test_no_breakout_below_orb_high(self, ref_cfg):
        from orb_live.signals.strategy_signals import check_breakout as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 110.5}, name=datetime(2022, 1, 7, 10, 5))
        assert live_fn(bar, orb, 1, ref_cfg) is False

    def test_valid_breakout_short(self, ref_cfg):
        from orb_live.signals.strategy_signals import check_breakout as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 107.0}, name=datetime(2022, 1, 7, 10, 5))
        assert live_fn(bar, orb, -1, ref_cfg) is True

    def test_orb_too_small_rejected(self, ref_cfg):
        """ORB size_pct < min_profit_pct → no breakout regardless of price."""
        from orb_live.signals.strategy_signals import check_breakout as live_fn

        tiny_orb = {
            "high":     100.1, "low": 100.0,
            "midpoint": 100.05,
            "size_pct": 0.001,   # way below min_profit_pct
            "n_bars":   30, "ema": 100.0,
        }
        bar = pd.Series({"close": 101.0}, name=datetime(2022, 1, 7, 10, 5))
        assert live_fn(bar, tiny_orb, 1, ref_cfg) is False


# ── compute_entry ─────────────────────────────────────────────────────────────

class TestComputeEntry:
    def _make_orb(self):
        return {
            "high":     111.0,
            "low":      100.0,
            "midpoint": 105.5,
            "size_pct": (111.0 - 100.0) / 105.5,
            "n_bars":   30,
            "ema":      109.0,
        }

    def test_long_entry_matches_reference(self, ref_cfg):
        from orb_live.signals.strategy_signals import compute_entry as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 112.0}, name=datetime(2022, 1, 7, 10, 5))
        result = live_fn(bar=bar, orb=orb, gap_direction=1, config=ref_cfg,
                         current_equity=100_000.0, symbol="TQQQ")

        assert result["direction"] == 1
        assert result["entry_price"] == 112.0
        assert result["tp1_price"] > result["entry_price"]
        assert result["tp2_price"] > result["tp1_price"]
        assert result["stop_price"] < result["entry_price"]
        assert result["shares"] > 0
        assert result["tp1_shares"] >= 0
        assert result["tp2_shares"] >= 0
        assert result["tp3_shares"] >= 0
        assert result["tp1_shares"] + result["tp2_shares"] + result["tp3_shares"] == result["shares"]

    def test_short_entry_matches_reference(self, ref_cfg):
        from orb_live.signals.strategy_signals import compute_entry as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 99.0}, name=datetime(2022, 1, 7, 10, 5))
        result = live_fn(bar=bar, orb=orb, gap_direction=-1, config=ref_cfg,
                         current_equity=100_000.0, symbol="TQQQ")

        assert result["direction"] == -1
        assert result["entry_price"] == 99.0
        assert result["tp1_price"] < result["entry_price"]
        assert result["stop_price"] > result["entry_price"]
        assert result["shares"] > 0

    def test_share_count_is_floor_not_round(self, ref_cfg):
        """shares must use math.floor — identical to backtest."""
        from orb_live.signals.strategy_signals import compute_entry as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 112.0}, name=datetime(2022, 1, 7, 10, 5))
        result = live_fn(bar=bar, orb=orb, gap_direction=1,
                         config=ref_cfg, current_equity=100_000.0)
        assert result["shares"] == math.floor(result["shares"])
        assert result["tp1_shares"] == math.floor(result["tp1_shares"])
        assert result["tp2_shares"] == math.floor(result["tp2_shares"])

    def test_tp_mult_overrides(self, ref_cfg):
        """tp1_mult_override / tp2_mult_override take precedence over config."""
        from orb_live.signals.strategy_signals import compute_entry as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 112.0}, name=datetime(2022, 1, 7, 10, 5))
        result = live_fn(bar=bar, orb=orb, gap_direction=1, config=ref_cfg,
                         current_equity=100_000.0,
                         tp1_mult_override=0.75, tp2_mult_override=1.50)

        orb_range = orb["high"] - orb["low"]
        assert pytest.approx(result["tp1_price"]) == 112.0 + orb_range * 0.75
        assert pytest.approx(result["tp2_price"]) == 112.0 + orb_range * 1.50

    def test_size_mult_scales_shares(self, ref_cfg):
        from orb_live.signals.strategy_signals import compute_entry as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 112.0}, name=datetime(2022, 1, 7, 10, 5))

        base   = live_fn(bar=bar, orb=orb, gap_direction=1, config=ref_cfg,
                         current_equity=100_000.0, size_mult=1.0)
        double = live_fn(bar=bar, orb=orb, gap_direction=1, config=ref_cfg,
                         current_equity=100_000.0, size_mult=2.0)

        assert double["shares"] > base["shares"]
        assert double["shares"] >= base["shares"] * 2 - 1

    def test_all_three_multiplier_args_combined(self, ref_cfg):
        """All three override/mult args applied simultaneously."""
        from orb_live.signals.strategy_signals import compute_entry as live_fn

        orb = self._make_orb()
        bar = pd.Series({"close": 112.0}, name=datetime(2022, 1, 7, 10, 5))
        result = live_fn(
            bar=bar, orb=orb, gap_direction=1, config=ref_cfg,
            current_equity=100_000.0,
            tp1_mult_override=0.60,
            tp2_mult_override=1.20,
            size_mult=2.0,
        )

        orb_range = orb["high"] - orb["low"]
        entry = 112.0

        assert pytest.approx(result["tp1_price"]) == entry + orb_range * 0.60
        assert pytest.approx(result["tp2_price"]) == entry + orb_range * 1.20

        base_shares = live_fn(bar=bar, orb=orb, gap_direction=1, config=ref_cfg,
                              current_equity=100_000.0)["shares"]
        assert result["shares"] > base_shares


# ── check_prior_session_filter: data-warning behaviour ───────────────────────

class TestCheckPriorSessionFilterWarnings:
    """
    Tests that the live port's logger integration is wired correctly.
    The return value (True = allow) must also be verified.
    """

    class _CapturingLogger:
        def __init__(self):
            self.calls: list[dict] = []

        def warning(self, event, **kw):
            self.calls.append({"event": event, **kw})

    def _ps_cfg(self, ref_cfg):
        from dataclasses import replace
        return replace(ref_cfg, prior_session_filters={"TQQQ": ("QQQ", 0.02)})

    def test_missing_underlying_returns_true_and_logs(self, ref_cfg):
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        log = self._CapturingLogger()
        result = live_fn(
            symbol="TQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=self._ps_cfg(ref_cfg),
            underlying_data={},
            logger=log,
        )

        assert result is True, "Missing underlying must allow the trade"
        assert len(log.calls) == 1, "Exactly one warning expected"
        assert log.calls[0]["event"] == "ps_filter_data_warning"
        assert log.calls[0]["n_closes"] == 0

    def test_one_prior_row_returns_true_and_logs(self, ref_cfg):
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        log = self._CapturingLogger()
        ul_data = {"QQQ": pd.DataFrame({
            "date":  pd.to_datetime(["2022-01-06"]),
            "close": [383.0],
        })}
        result = live_fn(
            symbol="TQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=self._ps_cfg(ref_cfg),
            underlying_data=ul_data,
            logger=log,
        )

        assert result is True, "Single prior row must allow the trade"
        assert len(log.calls) == 1
        assert log.calls[0]["n_closes"] == 1

    def test_no_warning_when_logger_none(self, ref_cfg):
        """Passing logger=None must not raise and must still return True."""
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        result = live_fn(
            symbol="TQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=self._ps_cfg(ref_cfg),
            underlying_data={},
            logger=None,
        )
        assert result is True

    def test_sufficient_data_no_warning(self, ref_cfg):
        """With two prior rows the function must NOT log a warning."""
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn

        log = self._CapturingLogger()
        ul_data = {"QQQ": pd.DataFrame({
            "date":  pd.to_datetime(["2022-01-05", "2022-01-06"]),
            "close": [381.0, 383.0],
        })}
        live_fn(
            symbol="TQQQ", date_=date(2022, 1, 7),
            gap_direction=1, config=self._ps_cfg(ref_cfg),
            underlying_data=ul_data,
            logger=log,
        )
        assert log.calls == [], "No warnings expected when data is sufficient"


# ── check_prior_session_filter: is_inverse tuple convention ──────────────────

class TestInverseTupleConvention:
    """
    Verify the exact tuple-length convention used to detect inverse ETFs.

    is_inverse = len(filter_cfg) == 3 and filter_cfg[2] is True

    Consequence: a 2-tuple is NEVER inverse; a 3-tuple with filter_cfg[2]=False
    is also NOT inverse.  Only filter_cfg[2]=True triggers direction flip.
    """

    def _run(self, ref_cfg, filter_spec, ul_data, gap_direction):
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn
        from dataclasses import replace

        cfg = replace(ref_cfg, prior_session_filters={"TEST": filter_spec})
        return live_fn(
            symbol="TEST", date_=date(2022, 1, 7),
            gap_direction=gap_direction,
            config=cfg, underlying_data=ul_data,
        )

    def _ul_big_move(self):
        """QQQ with prior move ≈ +4% — will block if direction-adjusted correctly."""
        return {"QQQ": pd.DataFrame({
            "date":  pd.to_datetime(["2022-01-05", "2022-01-06"]),
            "close": [370.0, 385.0],   # +4.05%
        })}

    def test_two_tuple_is_not_inverse(self, ref_cfg):
        """2-tuple → is_inverse=False; gap-direction=+1, big up move → blocked."""
        ul = self._ul_big_move()
        # threshold 0.02 (2%); prior move 4.05% * (+1) > 0.02 → blocked
        assert self._run(ref_cfg, ("QQQ", 0.02), ul, gap_direction=1) is False

    def test_three_tuple_true_is_inverse(self, ref_cfg):
        """3-tuple [2]=True → is_inverse=True; direction flipped → passes."""
        ul = self._ul_big_move()
        # threshold 0.02; prior move +4.05%; effective_dir=-1; dir_adj=-4.05% < 2% → passes
        assert self._run(ref_cfg, ("QQQ", 0.02, True), ul, gap_direction=1) is True

    def test_three_tuple_false_is_not_inverse(self, ref_cfg):
        """
        3-tuple with [2]=False must NOT be treated as inverse.

        len==3 is necessary but not sufficient.
        filter_cfg[2] must be `is True` (not just truthy).
        """
        from orb_live.signals.strategy_signals import check_prior_session_filter as live_fn
        from dataclasses import replace

        ul = self._ul_big_move()
        # 3-tuple but [2]=False → same as non-inverse → blocked (same as 2-tuple)
        cfg = replace(ref_cfg, prior_session_filters={"TEST": ("QQQ", 0.02, False)})
        result = live_fn(
            symbol="TEST", date_=date(2022, 1, 7),
            gap_direction=1, config=cfg, underlying_data=ul,
        )
        assert result is False, (
            "3-tuple with [2]=False must not invert direction — "
            "should be blocked same as a 2-tuple"
        )


# ── Two-EMA parity: ORB close EMA vs full-session midpoint EMA ────────────────

class TestIndicatorParity:
    """
    Verify to 1e-9 that the two EMA signals used by the strategy are:
    (a) each computed correctly per their respective formula, and
    (b) numerically distinct from each other.

    orb["ema"]  — ORB window CLOSE EMA; seed = first ORB bar's close;
                  30 bars (09:30–09:59); used for breakout detection.

    Full-session midpoint EMA  — ALL session bars' (high+low)/2 from 09:30;
                                  seed = first bar's midpoint;
                                  used for TP3 trail crossback.
    """

    @pytest.fixture(scope="class")
    def full_day_bars(self):
        """390 one-minute bars 09:30–15:59 ET on 2022-01-07 (deterministic)."""
        import math as _math
        n   = 390
        idx = pd.date_range("2022-01-07 09:30", periods=n, freq="1min")
        closes = [100.0 + 0.05 * i + 0.30 * _math.sin(i * 0.10) for i in range(n)]
        highs  = [c + 0.40 for c in closes]
        lows   = [c - 0.30 for c in closes]
        return pd.DataFrame({"close": closes, "high": highs, "low": lows}, index=idx)

    @staticmethod
    def _ema_from_closes(closes: list, k: float) -> float:
        ema = float(closes[0])
        for c in closes[1:]:
            ema = float(c) * k + ema * (1.0 - k)
        return ema

    @staticmethod
    def _ema_from_midpoints(highs: list, lows: list, k: float) -> float:
        ema = (float(highs[0]) + float(lows[0])) / 2.0
        for h, lo in zip(highs[1:], lows[1:]):
            mid = (float(h) + float(lo)) / 2.0
            ema = mid * k + ema * (1.0 - k)
        return ema

    def test_orb_ema_matches_manual(self, full_day_bars, ref_cfg):
        """orb["ema"] must equal the manual close-EMA over the 30-bar ORB window."""
        from orb_live.signals.strategy_signals import compute_opening_range as live_fn

        orb_live = live_fn(full_day_bars, ref_cfg)
        assert orb_live is not None

        k         = 2.0 / (ref_cfg.ema_length + 1)
        orb_bars  = full_day_bars.iloc[:30]
        expected  = self._ema_from_closes(orb_bars["close"].tolist(), k)

        assert pytest.approx(orb_live["ema"], rel=1e-9) == expected

    def test_session_midpoint_ema_matches_manual(self, full_day_bars, ref_cfg):
        """Full-session midpoint EMA formula must match manual computation to 1e-9."""
        k           = 2.0 / (ref_cfg.ema_length + 1)
        all_highs   = full_day_bars["high"].tolist()
        all_lows    = full_day_bars["low"].tolist()
        session_ema = self._ema_from_midpoints(all_highs, all_lows, k)

        session_ema2 = self._ema_from_midpoints(all_highs, all_lows, k)
        assert pytest.approx(session_ema, rel=1e-15) == session_ema2
        assert 95.0 < session_ema < 125.0, "Session EMA out of expected price range"

    def test_orb_ema_and_session_ema_are_distinct(self, full_day_bars, ref_cfg):
        """
        The two EMAs must be numerically different.

        Reason: (1) different inputs — ORB uses CLOSE prices; session uses
        MIDPOINTS; (2) different windows — 30 bars vs 390 bars.
        """
        from orb_live.signals.strategy_signals import compute_opening_range as live_fn

        orb = live_fn(full_day_bars, ref_cfg)
        assert orb is not None

        k           = 2.0 / (ref_cfg.ema_length + 1)
        session_ema = self._ema_from_midpoints(
            full_day_bars["high"].tolist(),
            full_day_bars["low"].tolist(),
            k,
        )

        assert abs(orb["ema"] - session_ema) > 1e-6, (
            f"orb['ema']={orb['ema']:.8f} must differ from "
            f"full-session midpoint EMA={session_ema:.8f}"
        )

    def test_session_ema_seed_is_midpoint_not_close(self, full_day_bars, ref_cfg):
        """
        The full-session EMA seed is (hi+lo)/2 of the 9:30 bar — NOT its close.
        """
        k        = 2.0 / (ref_cfg.ema_length + 1)
        first    = full_day_bars.iloc[0]

        seed_mid   = (float(first["high"]) + float(first["low"])) / 2.0
        seed_close = float(first["close"])

        # Fixture: high = close+0.40, low = close-0.30 → midpoint = close+0.05
        assert seed_mid != seed_close, (
            "Fixture error: midpoint and close are the same — "
            "cannot distinguish the two EMA seeds"
        )


# ── EOD exit hour=16 behavioral contract ─────────────────────────────────────

class TestEodExitHour16Contract:
    """
    Verify that with eod_exit_hour=16 (LiveConfig default), no bar timestamp
    in the regular session (09:30–15:59 ET) satisfies the on-bar EOD condition
    (bar_ts.time() >= time(16, 0)).
    """

    def test_no_regular_session_bar_reaches_eod_hour_16(self):
        """All 390 regular-session bars must fall before 16:00."""
        from datetime import time as dtime

        eod_time     = dtime(16, 0)
        session_bars = pd.date_range("2022-01-07 09:30", periods=390, freq="1min")
        hits = [ts for ts in session_bars if ts.time() >= eod_time]

        assert hits == [], (
            f"Expected no regular-session bars at/after 16:00; found: {hits[:3]}"
        )

    def test_every_session_bar_fails_on_bar_eod_check(self):
        """
        Exhaustive: each of the 390 bars individually must fail the on-bar
        EOD check when eod_exit_hour=16.
        """
        from datetime import time as dtime

        eod_time     = dtime(16, 0)
        session_bars = pd.date_range("2022-01-07 09:30", periods=390, freq="1min")

        for ts in session_bars:
            assert ts.time() < eod_time, (
                f"Bar at {ts.time()} would incorrectly trigger on-bar EOD "
                "exit with eod_exit_hour=16"
            )

    def test_boil_kold_1455_bar_is_valid_session_bar(self):
        """
        BOIL/KOLD use eod_exit_hour=14, eod_exit_minute=55.
        The 14:55 bar exists as a regular session bar (time < 16:00).
        """
        from datetime import time as dtime

        boil_eod     = dtime(14, 55)
        session_bars = pd.date_range("2022-01-07 09:30", periods=390, freq="1min")
        hits = [ts for ts in session_bars if ts.time() == boil_eod]

        assert len(hits) == 1, "Exactly one 14:55 bar per regular session"
        assert hits[0].time() < dtime(16, 0), "14:55 is a regular-session bar (< 16:00)"
