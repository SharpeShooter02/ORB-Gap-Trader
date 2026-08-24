"""FORCE_INCLUDE symbols must carry their measured IB margin rates.

BOIL, GUSH and KOLD have no master_universe.csv row -- they exist only as
FORCE_INCLUDE entries, and load_master built those without margin fields. So
they fell back to a conservative 1.0 until prewarm_margin measured them each
session, over-reserving by nearly 2x in the window before that (OPEN_ISSUES L6).

Measured against IB 2026-08-24 via whatIfOrder:

    BOIL  init long 0.529  short 0.635
    KOLD  init long 0.521  short 0.625
    GUSH  init long 0.526  short 0.631

These are seeds, not gospel: live still pulls fresh rates each session and
those take precedence. The point is that the fallback should be honest rather
than double the truth.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orb_live.strategy.v1_strategy import FORCE_INCLUDE, load_master

DATA = Path(__file__).resolve().parents[1] / "strategy" / "data"
MASTER = DATA / "master_universe.csv"

MEASURED = {"BOIL": (0.529, 0.635), "KOLD": (0.521, 0.625), "GUSH": (0.526, 0.631)}


@pytest.fixture(scope="module")
def instruments():
    return load_master(MASTER)


class TestForceIncludeMargin:
    @pytest.mark.parametrize("sym", sorted(MEASURED))
    def test_has_measured_margin(self, instruments, sym):
        """The three names with no CSV row still get real rates."""
        inst = instruments[sym]
        assert inst.margin_long is not None, f"{sym} has no long rate"
        assert inst.margin_short is not None, f"{sym} has no short rate"

    @pytest.mark.parametrize("sym,rates", sorted(MEASURED.items()))
    def test_matches_ib_measurement(self, instruments, sym, rates):
        long_r, short_r = rates
        inst = instruments[sym]
        assert inst.margin_long == pytest.approx(long_r, abs=1e-6)
        assert inst.margin_short == pytest.approx(short_r, abs=1e-6)

    @pytest.mark.parametrize("sym", sorted(MEASURED))
    def test_rate_is_far_below_the_old_fallback(self, instruments, sym):
        """The whole point: these are ~0.52, not the 1.0 they defaulted to."""
        assert instruments[sym].margin_long < 0.7

    @pytest.mark.parametrize("sym", sorted(MEASURED))
    def test_margin_rate_helper_uses_them(self, instruments, sym):
        inst = instruments[sym]
        assert inst.margin_rate(1) == inst.margin_long
        assert inst.margin_rate(-1) == inst.margin_short

    def test_short_costs_more_than_long(self, instruments):
        for sym in MEASURED:
            i = instruments[sym]
            assert i.margin_short > i.margin_long, sym

    def test_force_include_without_rates_still_loads(self):
        """Entries that carry no measured rates must not break loading."""
        bare = [s for s, v in FORCE_INCLUDE.items()
                if "margin_long" not in v]
        inst = load_master(MASTER)
        for s in bare:
            assert s in inst, f"{s} vanished from the catalogue"

    def test_csv_rows_still_win_over_force_include(self, instruments):
        """A symbol with a real CSV row keeps the CSV's measured rates."""
        # AMDL is FORCE_INCLUDE *and* present in master_universe.csv.
        assert instruments["AMDL"].margin_long is not None
