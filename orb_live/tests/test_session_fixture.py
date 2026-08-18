"""Golden-session fixture round-trip.

The fixture is the synchronisation contract between this repo and the
backtest repo, so its serialisation has to be exact. These tests pin that:
a plan written out and replayed must come back identical.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from orb_live.strategy import v1_strategy as v1
from orb_live.strategy.session_fixture import (
    SCHEMA_VERSION,
    build_fixture,
    diff_plan,
    load_fixture,
    replay,
    write_fixture,
)


@pytest.fixture
def session_inputs():
    """A small session: two underlyings, one of them with two ETFs."""
    instruments = {
        "TQQQ": v1.Instrument(symbol="TQQQ", underlying="QQQ", leverage=3, inverse=False),
        "SQQQ": v1.Instrument(symbol="SQQQ", underlying="QQQ", leverage=3, inverse=True),
        "NUGT": v1.Instrument(symbol="NUGT", underlying="GDX", leverage=2, inverse=False),
    }
    return dict(
        universe=["TQQQ", "SQQQ", "NUGT"],
        instruments=instruments,
        sigmas={"QQQ": 0.012, "GDX": 0.020},
        overnight_gaps={"QQQ": 0.031, "GDX": -0.045},
        prior_two_closes={"QQQ": (400.0, 399.0), "GDX": (30.0, 30.6)},
        prior_etf_close={"TQQQ": 70.0, "SQQQ": 12.0, "NUGT": 45.0},
    )


def _plan(inputs):
    return v1.plan_session(**inputs)


def test_replay_reproduces_the_plan(session_inputs):
    plan = _plan(session_inputs)
    payload = build_fixture(date(2026, 8, 18), plan=plan, **session_inputs)
    assert diff_plan(payload["plan"], replay(payload)) == []


def test_fixture_survives_json_round_trip(session_inputs):
    """Serialisation must not lose precision or reorder anything material."""
    plan = _plan(session_inputs)
    payload = build_fixture(date(2026, 8, 18), plan=plan, **session_inputs)
    reloaded = json.loads(json.dumps(payload))
    assert diff_plan(reloaded["plan"], replay(reloaded)) == []


def test_write_and_load(tmp_path, session_inputs):
    plan = _plan(session_inputs)
    ok = write_fixture(date(2026, 8, 18), plan=plan, out_dir=tmp_path, **session_inputs)
    assert ok
    written = tmp_path / "2026-08-18.json"
    assert written.exists()
    payload = load_fixture(written)
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["trade_date"] == "2026-08-18"
    assert diff_plan(payload["plan"], replay(payload)) == []


def test_profile_snapshot_is_recorded(session_inputs):
    """A retune must be distinguishable from an input change on replay."""
    plan = _plan(session_inputs)
    prof = build_fixture(date(2026, 8, 18), plan=plan, **session_inputs)["profile"]
    assert prof["k_sigma"] == v1.K_SIGMA
    assert prof["cap_units"] == v1.CAP_UNITS
    assert prof["weights"]["C2|flood"] == v1.WEIGHTS[("C2", "flood")]


def test_diff_plan_detects_disagreement(session_inputs):
    plan = _plan(session_inputs)
    payload = build_fixture(date(2026, 8, 18), plan=plan, **session_inputs)
    payload["plan"]["regime"] = "flood"
    payload["plan"]["cap_factor"] = 0.5
    diffs = diff_plan(payload["plan"], replay(payload))
    assert any("regime" in d for d in diffs)
    assert any("cap_factor" in d for d in diffs)


def test_write_never_raises_on_bad_target(session_inputs):
    """A fixture failure must not be able to abort a trading session."""
    plan = _plan(session_inputs)
    # A path that cannot be created as a directory (NUL is reserved on Windows,
    # and an empty-name path fails on POSIX) — write_fixture must absorb it.
    assert write_fixture(
        date(2026, 8, 18), plan=plan, out_dir="\0invalid", **session_inputs
    ) is False
