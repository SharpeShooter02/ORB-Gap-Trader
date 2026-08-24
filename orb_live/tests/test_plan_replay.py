"""Replay the plan for every RECORDED session, not just synthetic ones.

`diff_plan` already compares the sizing layer -- candidates, regime, n_uls,
cap_factor and per-symbol multipliers -- and `replay()` already re-runs
plan_session from a fixture's recorded inputs. Both were only ever exercised
against a three-symbol fixture built inside test_session_fixture.py. Nothing
replayed the sessions live actually recorded.

That is the gap that mattered when gold moved to C3 weights and GDX/GDXJ began
sharing an allotment: both changes live in the sizing path, and the only thing
confirming live and backtest agreed was that the same rule was written twice and
the aggregates matched. A recorded session replayed against current code is the
check that actually closes it.

The recorded fixtures are thin today -- 2026-08-21 is the only schema-2 one and
it is an empty session -- so a passing run here proves little until a day with
candidates is recorded. The drift cases below are what give it teeth now: each
mutates one piece of the sizing path and asserts the replay notices.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from orb_live.strategy import v1_strategy as v1
from orb_live.strategy.session_fixture import (build_fixture, diff_plan,
                                               diff_profile, replay)

SESSIONS = Path(__file__).resolve().parents[2] / "data" / "sessions"


def _recorded() -> list[tuple[str, dict]]:
    if not SESSIONS.is_dir():
        return []
    out = []
    for f in sorted(SESSIONS.glob("*.json")):
        try:
            p = json.loads(f.read_text())
        except Exception:
            continue
        if "plan" in p and "inputs" in p:
            out.append((f.name, p))
    return out


class TestRecordedSessionsReplay:
    def test_every_recorded_session_reproduces_or_says_why(self):
        """A session may stop reproducing for two reasons: the code drifted,
        or the strategy was deliberately retuned. Only the first is a bug, and
        the recorded profile snapshot is what separates them."""
        rec = _recorded()
        if not rec:
            pytest.skip("no recorded sessions on disk")
        for name, payload in rec:
            d = diff_plan(payload["plan"], replay(payload))
            if not d:
                continue
            changed = diff_profile(payload.get("profile", {}))
            assert changed, (
                f"{name} no longer reproduces and the profile is unchanged — "
                f"this is drift, not a retune: {d}")

    def test_unexplained_drift_would_fail(self):
        """Guard the guard: an identical profile with a different plan must
        be reported, or the check above is vacuous."""
        p = _gold_payload()
        p["plan"]["multipliers"]["SOXL"] = 99.0
        assert diff_plan(p["plan"], replay(p))
        assert diff_profile(p["profile"]) == []

    def test_at_least_one_session_is_recorded(self):
        """Guards against this suite passing because the directory is empty."""
        assert _recorded(), "no session fixtures to replay"

    def test_reports_which_sessions_had_candidates(self):
        """Not an assertion — visibility. A gate that only ever replays empty
        sessions is green for the wrong reason."""
        rec = _recorded()
        if not rec:
            pytest.skip("no recorded sessions")
        withc = [n for n, p in rec if p["plan"].get("multipliers")]
        print(f"\n  {len(rec)} recorded, {len(withc)} with candidates: {withc}")


def _gold_payload():
    """A session where both gold names qualify, so the shared allotment bites."""
    inst = {
        "NUGT": v1.Instrument(symbol="NUGT", underlying="GDX",
                              leverage=2, inverse=False),
        "JNUG": v1.Instrument(symbol="JNUG", underlying="GDXJ",
                              leverage=2, inverse=False),
        "SOXL": v1.Instrument(symbol="SOXL", underlying="SOXX",
                              leverage=3, inverse=False),
    }
    inputs = dict(
        universe=list(inst), instruments=inst,
        sigmas={"GDX": 0.02, "GDXJ": 0.02, "SOXX": 0.02},
        overnight_gaps={"GDX": 0.05, "GDXJ": 0.05, "SOXX": 0.05},
        prior_two_closes={u: (100.0, 100.0) for u in ("GDX", "GDXJ", "SOXX")},
        prior_etf_close={s: 50.0 for s in inst},
    )
    plan = v1.plan_session(**inputs)
    return build_fixture(date(2026, 8, 21), plan=plan, **inputs)


class TestSizingDriftIsCaught:
    def test_the_fixture_actually_sizes_something(self):
        assert _gold_payload()["plan"]["multipliers"]

    def test_clean_replay(self):
        assert diff_plan(_gold_payload()["plan"], replay(_gold_payload())) == []

    def test_catches_a_changed_weight(self, monkeypatch):
        p = _gold_payload()
        w = dict(v1.WEIGHTS); w[("C3", "quiet")] = 1.0
        monkeypatch.setattr(v1, "WEIGHTS", w)
        assert diff_plan(p["plan"], replay(p))

    def test_catches_a_reclassified_symbol(self, monkeypatch):
        p = _gold_payload()
        monkeypatch.setattr(v1, "CLASS_2_SYMS", v1.CLASS_2_SYMS | {"NUGT"})
        assert diff_plan(p["plan"], replay(p))

    def test_catches_a_dropped_shared_allotment(self, monkeypatch):
        """The exact change made 2026-08-23 — if it were reverted on one side
        only, this is what would notice."""
        p = _gold_payload()
        monkeypatch.setattr(v1, "SHARED_ALLOTMENT", {})
        d = diff_plan(p["plan"], replay(p))
        assert d, "removing the shared allotment slipped through"
        assert any("NUGT" in x or "JNUG" in x for x in d), d

    def test_profile_records_the_sizing_inputs(self):
        """SHARED_ALLOTMENT and class membership move multipliers, so a
        fixture that omits them cannot explain its own divergence."""
        prof = _gold_payload()["profile"]
        for key in ("shared_allotment", "class_1", "class_2", "weights"):
            assert key in prof, f"profile does not record {key}"

    def test_profile_diff_names_a_reclassification(self, monkeypatch):
        prof = _gold_payload()["profile"]
        monkeypatch.setattr(v1, "CLASS_2_SYMS", v1.CLASS_2_SYMS | {"NUGT"})
        assert any("class_2" in x for x in diff_profile(prof))

    def test_profile_diff_names_a_dropped_allotment(self, monkeypatch):
        prof = _gold_payload()["profile"]
        monkeypatch.setattr(v1, "SHARED_ALLOTMENT", {})
        assert any("shared_allotment" in x for x in diff_profile(prof))

    def test_names_the_symbol_that_moved(self):
        p = _gold_payload()
        p["plan"]["multipliers"]["SOXL"] = 99.0
        assert any("SOXL" in x for x in diff_plan(p["plan"], replay(p)))
