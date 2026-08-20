"""The gap scan, and the coverage the golden-session gate used to be missing.

The point of extracting scan_gaps() from PreMarketJob was that the layer where
live and the backtest can actually disagree — the ETF gap, the
leverage × GAP_THRESHOLD test, the ETF→UL reconstruction, most-extreme-wins,
the PS filter — was unreachable without a database and a broker, and was
recorded in fixtures only as its own output.

test_leverage_change_is_caught_by_the_replay is the regression that matters:
the same edit that made GDXU a 3x fund left all 52 backtest tests green under
the schema-1 fixture.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from orb_live.signals.gap_scan import scan_gaps
from orb_live.strategy import v1_strategy as v1
from orb_live.strategy.session_fixture import (
    build_fixture,
    diff_scan,
    has_raw_inputs,
    replay_scan,
    serialise_raw_inputs,
)

TRADE_DATE = date(2026, 8, 18)


def _daily(closes: list[float], end: date = TRADE_DATE) -> pd.DataFrame:
    """Daily bars ending the session before `end`."""
    idx = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=len(closes))
    return pd.DataFrame({"date": idx, "close": closes})


class _Cfg:
    """Stand-in for LiveConfig — scan_gaps only reads these two."""

    def __init__(self, direction_filters=None, prior_session_filters=None):
        self.direction_filters = direction_filters or {}
        self.prior_session_filters = prior_session_filters or {}


@pytest.fixture
def scene():
    """Two ETFs on QQQ (one inverse) and one on GDX.

    QQQ closed at 400; TQQQ's 9:30 print is 6% up on its prior close, which is
    a 2% underlying gap at 3x — exactly on the threshold's right side.
    """
    instruments = {
        "TQQQ": v1.Instrument(symbol="TQQQ", underlying="QQQ", leverage=3, inverse=False),
        "SQQQ": v1.Instrument(symbol="SQQQ", underlying="QQQ", leverage=3, inverse=True),
        "GDXU": v1.Instrument(symbol="GDXU", underlying="GDX", leverage=3, inverse=False),
    }
    return dict(
        symbols=["TQQQ", "SQQQ", "GDXU"],
        instruments=instruments,
        ref_prices={"TQQQ": 74.20, "SQQQ": 11.28, "GDXU": 55.00},
        etf_daily={
            "TQQQ": _daily([69.0, 70.0]),
            "SQQQ": _daily([12.1, 12.0]),
            "GDXU": _daily([49.5, 50.0]),
        },
        ul_daily={
            "QQQ": _daily([398.0, 400.0]),
            "GDX": _daily([30.6, 30.0]),
        },
        config=_Cfg(),
    )


def test_gap_is_measured_on_the_etf_and_converted_by_leverage(scene):
    res = scan_gaps(TRADE_DATE, **scene)
    tqqq = next(s for s in res.scans if s.symbol == "TQQQ")

    assert tqqq.etf_gap_abs == pytest.approx(0.06, abs=1e-9)
    assert tqqq.ul_gap == pytest.approx(0.02, abs=1e-9)
    assert tqqq.qualifies


def test_inverse_etf_reconstruction_flips_the_sign(scene):
    """SQQQ gapping down means QQQ gapped up. Both ETFs must agree on QQQ."""
    res = scan_gaps(TRADE_DATE, **scene)
    sqqq = next(s for s in res.scans if s.symbol == "SQQQ")

    assert sqqq.gap_direction == -1
    assert sqqq.ul_gap > 0
    assert res.overnight_gaps["QQQ"] > 0


def test_most_extreme_gap_wins_when_etfs_share_an_underlying(scene):
    """Two ETFs, one underlying — the larger |reconstruction| is the one kept."""
    res = scan_gaps(TRADE_DATE, **scene)
    per_etf = {s.symbol: s.ul_gap for s in res.scans if s.ul_gap is not None}

    assert res.overnight_gaps["QQQ"] == pytest.approx(
        max((per_etf["TQQQ"], per_etf["SQQQ"]), key=abs)
    )


def test_gap_below_leverage_threshold_is_rejected_with_a_reason(scene):
    """A 3x fund needs 6%, not 2%. Rejections carry their reason, not silence."""
    scene["ref_prices"]["GDXU"] = 51.0  # +2% on the ETF => 0.67% on GDX
    res = scan_gaps(TRADE_DATE, **scene)
    gdxu = next(s for s in res.scans if s.symbol == "GDXU")

    assert not gdxu.qualifies
    assert gdxu.filter_reason == "gap_too_small"
    assert "GDX" not in res.overnight_gaps


def test_missing_inputs_are_reported_not_dropped(scene):
    """Every symbol comes back, so the caller can persist why it failed."""
    scene["ref_prices"]["GDXU"] = None
    scene["etf_daily"]["SQQQ"] = pd.DataFrame(columns=["date", "close"])
    res = scan_gaps(TRADE_DATE, **scene)
    reasons = {s.symbol: s.filter_reason for s in res.scans}

    assert len(res.scans) == 3
    assert reasons["GDXU"] == "no_ref_price"
    assert reasons["SQQQ"] == "no_daily_bars"


def test_direction_filter_rejects_the_wrong_side(scene):
    scene["config"] = _Cfg(direction_filters={"TQQQ": -1})
    res = scan_gaps(TRADE_DATE, **scene)
    tqqq = next(s for s in res.scans if s.symbol == "TQQQ")

    assert not tqqq.qualifies
    assert tqqq.filter_reason == "direction_filtered"


def test_ps_filter_blocks_and_records_the_move(scene):
    """PS filter reads real UL closes: QQQ rose 0.5% into a gap up.

    Both QQQ ETFs are filtered, because blocking only one would leave the
    underlying in overnight_gaps via its sibling.
    """
    scene["config"] = _Cfg(prior_session_filters={
        "TQQQ": ("QQQ", 0.001),
        "SQQQ": ("QQQ", 0.001, True),
    })
    res = scan_gaps(TRADE_DATE, **scene)
    tqqq = next(s for s in res.scans if s.symbol == "TQQQ")

    assert tqqq.ps_checked
    assert not tqqq.ps_passed
    assert tqqq.filter_reason == "ps_filter"
    assert tqqq.ul_move_pct == pytest.approx(0.005025, abs=1e-6)
    assert "QQQ" not in res.overnight_gaps


# ── The fixture contract ──────────────────────────────────────────────────────


def _fixture(scene):
    res = scan_gaps(TRADE_DATE, **scene)
    plan = v1.plan_session(
        universe=res.qualified_symbols,
        instruments=scene["instruments"],
        sigmas={"QQQ": 0.012, "GDX": 0.020},
        overnight_gaps=res.overnight_gaps,
        prior_two_closes=res.prior_two_closes,
        prior_etf_close=res.prior_etf_close,
    )
    return build_fixture(
        TRADE_DATE,
        universe=res.qualified_symbols,
        instruments=scene["instruments"],
        sigmas={"QQQ": 0.012, "GDX": 0.020},
        overnight_gaps=res.overnight_gaps,
        prior_two_closes=res.prior_two_closes,
        prior_etf_close=res.prior_etf_close,
        plan=plan,
        raw=serialise_raw_inputs(
            symbols=scene["symbols"],
            ref_prices=scene["ref_prices"],
            etf_daily=scene["etf_daily"],
            ul_daily=scene["ul_daily"],
            prior_session_filters=scene["config"].prior_session_filters,
        ),
    )


def test_scan_replays_from_raw_inputs(scene):
    payload = _fixture(scene)
    assert has_raw_inputs(payload)
    assert diff_scan(payload, replay_scan(payload)) == []


def test_leverage_change_is_caught_by_the_replay(scene):
    """The GDXU regression.

    Under the schema-1 fixture this change was invisible: leverage is consumed
    before plan_session ever runs, so no recorded input moved and every replay
    stayed green. Starting the replay above the conversion makes it visible.
    """
    payload = _fixture(scene)
    payload["inputs"]["instruments"]["GDXU"]["leverage"] = 2

    diffs = diff_scan(payload, replay_scan(payload))
    assert diffs, "a leverage change must not replay clean"
    assert any("overnight_gaps[GDX]" in d for d in diffs)


def test_schema_1_fixture_has_nothing_to_replay(scene):
    """Old fixtures still load; they just cannot be scan-replayed."""
    payload = _fixture(scene)
    payload["inputs"]["raw"] = None

    assert not has_raw_inputs(payload)
    with pytest.raises(ValueError):
        replay_scan(payload)
