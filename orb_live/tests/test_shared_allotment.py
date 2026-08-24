"""GDX and GDXJ share one sizing allotment (OPEN_ISSUES R4).

They are one exposure wearing five tickers. Their ETFs correlate 0.88-0.96 in
daily strategy P&L -- including bull against bear, because the strategy trades
the gap DIRECTION, so NUGT and DUST follow through or fail together on the same
gold move. At underlying level GDX/GDXJ is 0.935 over 115 shared days with 83%
overlap, a clear outlier (next pair: AMD/SOXX at 0.812 on 23 days).

Treating them as two underlyings let one gold move open two full-size
positions. Measured under the real allocation policy, sharing one allotment and
weighting gold as C3 was the best of four arrangements: same P&L, Sharpe
2.267 -> 2.296, MaxDD -8.91% -> -7.99%, Calmar +12%, drawdown equal-or-better
in 5 of 7 years.

Two things had to change together -- splitting at C2 weights was WORSE on P&L
(29,970 vs 30,657), because gold's quiet/flood weight of 0.5 is what the notch
imposes and gold is the sub-group that contradicts it.

Deliberately NOT generalised to a correlation threshold: correlation chains, and
at 0.60 single-linkage {AMD, SOXX, XLK, QQQ, XLC, FXI, EEM} collapses into one
seven-underlying cluster. GDX/GDXJ is a hard-coded exception on its merits.
"""
from __future__ import annotations

import pytest

from orb_live.strategy.v1_strategy import (
    CLASS_2_SYMS, SHARED_ALLOTMENT, WEIGHTS, allotment_group, classify,
    compute_cap_factor, position_multiplier,
)

GOLD = ["NUGT", "JNUG", "JDST", "DUST", "GDXU"]


class _C:
    """Minimal stand-in for Candidate — only .symbol and .underlying are read."""
    def __init__(self, symbol, underlying):
        self.symbol = symbol
        self.underlying = underlying


class TestGoldIsC3:
    @pytest.mark.parametrize("sym", GOLD)
    def test_gold_classifies_c3(self, sym):
        assert classify(sym) == "C3"

    @pytest.mark.parametrize("sym", GOLD)
    def test_gold_left_class_two(self, sym):
        assert sym not in CLASS_2_SYMS

    @pytest.mark.parametrize("regime", ["quiet", "active", "flood"])
    def test_gold_gets_full_weight_every_regime(self, regime):
        assert WEIGHTS[("C3", regime)] == 3.0

    def test_c2_notch_survives_for_everyone_else(self):
        """The notch is right about international and single-stock."""
        assert WEIGHTS[("C2", "quiet")] == 0.5
        assert WEIGHTS[("C2", "flood")] == 0.5
        assert WEIGHTS[("C2", "active")] == 3.0
        assert classify("KORU") == "C2" and classify("TSLL") == "C2"


class TestAllotmentGroup:
    def test_gdxj_maps_to_gdx(self):
        assert allotment_group("GDXJ") == "GDX"

    def test_gdx_maps_to_itself(self):
        assert allotment_group("GDX") == "GDX"

    def test_unrelated_underlyings_are_their_own_group(self):
        for ul in ("QQQ", "SPY", "BTC", "UNG", "SOXX"):
            assert allotment_group(ul) == ul

    def test_mapping_is_the_only_exception(self):
        assert SHARED_ALLOTMENT == {"GDXJ": "GDX"}


class TestSharedSizing:
    def test_one_gold_candidate_gets_full_weight(self):
        c = [_C("NUGT", "GDX")]
        assert position_multiplier("NUGT", "quiet", 1.0, c) == pytest.approx(3.0)

    def test_two_gold_candidates_each_get_half(self):
        c = [_C("NUGT", "GDX"), _C("JNUG", "GDXJ")]
        for s in ("NUGT", "JNUG"):
            assert position_multiplier(s, "quiet", 1.0, c) == pytest.approx(1.5)

    def test_combined_gold_equals_one_position(self):
        c = [_C("NUGT", "GDX"), _C("JNUG", "GDXJ")]
        total = sum(position_multiplier(x.symbol, "flood", 1.0, c) for x in c)
        assert total == pytest.approx(3.0)

    def test_bull_and_bear_still_share(self):
        """DUST and JNUG are opposite directions and still one bet."""
        c = [_C("DUST", "GDX"), _C("JNUG", "GDXJ")]
        for s in ("DUST", "JNUG"):
            assert position_multiplier(s, "active", 1.0, c) == pytest.approx(1.5)

    def test_non_gold_unaffected_by_a_gold_pair(self):
        c = [_C("NUGT", "GDX"), _C("JNUG", "GDXJ"), _C("SOXL", "SOXX")]
        assert position_multiplier("SOXL", "quiet", 1.0, c) == pytest.approx(3.0)

    def test_two_siblings_on_ONE_underlying_do_not_split(self):
        """top_n_per_ul=1 means this cannot happen, but if it did the split is
        per allotment group, not per candidate."""
        c = [_C("NUGT", "GDX"), _C("GDXU", "GDX")]
        assert position_multiplier("NUGT", "quiet", 1.0, c) == pytest.approx(1.5)

    def test_omitting_candidates_keeps_old_behaviour(self):
        assert position_multiplier("NUGT", "quiet", 1.0) == pytest.approx(3.0)

    def test_cap_factor_applies_on_top(self):
        c = [_C("NUGT", "GDX"), _C("JNUG", "GDXJ")]
        assert position_multiplier("NUGT", "quiet", 0.5, c) == pytest.approx(0.75)


class TestCapFactorAccounting:
    def test_cap_factor_counts_gold_once(self):
        """Exposure must not be overstated by counting both halves whole."""
        c = [_C("NUGT", "GDX"), _C("JNUG", "GDXJ")]
        cf = compute_cap_factor(c, "quiet", cap_units=3.0)
        assert cf == pytest.approx(1.0)

    def test_cap_factor_still_binds_when_it_should(self):
        c = [_C("NUGT", "GDX"), _C("JNUG", "GDXJ"), _C("SOXL", "SOXX")]
        cf = compute_cap_factor(c, "quiet", cap_units=3.0)
        assert cf == pytest.approx(0.5)

    def test_sized_exposure_never_exceeds_cap(self):
        c = [_C("NUGT", "GDX"), _C("JNUG", "GDXJ"), _C("SOXL", "SOXX"),
             _C("TQQQ", "QQQ")]
        cap = 4.0
        cf = compute_cap_factor(c, "flood", cap_units=cap)
        total = sum(position_multiplier(x.symbol, "flood", cf, c) for x in c)
        assert total <= cap + 1e-9


class TestPlanSessionAppliesTheSplit:
    """The rule must actually reach production, not just pass unit tests.

    position_multiplier takes `candidates` optionally, so plan_session could
    call it without them and the split would silently vanish while every unit
    test above still passed. That is precisely how prewarm_margin sat dead for
    months. These tests exercise the real entrypoint.
    """

    def _plan(self):
        from orb_live.strategy.v1_strategy import Instrument, plan_session
        inst = {
            "NUGT": Instrument("NUGT", "GDX", 2, False),
            "JNUG": Instrument("JNUG", "GDXJ", 2, False),
            "SOXL": Instrument("SOXL", "SOXX", 3, False),
        }
        sig = {"GDX": 0.02, "GDXJ": 0.02, "SOXX": 0.02}
        # Gaps well clear of leverage x 2%; prior sessions flat so the PS
        # filter cannot reject them.
        gaps = {"GDX": 0.05, "GDXJ": 0.05, "SOXX": 0.05}
        closes = {u: (100.0, 100.0) for u in sig}
        etf_close = {s: 50.0 for s in inst}
        return plan_session(list(inst), inst, sig, gaps, closes, etf_close)

    def test_gold_pair_splits_in_a_real_plan(self):
        plan = self._plan()
        got = {s: m for s, m in plan.multipliers.items() if s in ("NUGT", "JNUG")}
        if len(got) < 2:
            pytest.skip(f"both gold names not selected: {plan.candidates}")
        assert got["NUGT"] == pytest.approx(got["JNUG"])
        assert got["NUGT"] < plan.multipliers.get("SOXL", float("inf"))

    def test_gold_pair_totals_one_unshared_position(self):
        plan = self._plan()
        if not {"NUGT", "JNUG"} <= set(plan.multipliers):
            pytest.skip("both gold names not selected")
        gold = plan.multipliers["NUGT"] + plan.multipliers["JNUG"]
        assert gold == pytest.approx(plan.multipliers["SOXL"])
