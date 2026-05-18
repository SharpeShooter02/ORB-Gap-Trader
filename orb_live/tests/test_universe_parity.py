"""
tests/test_universe_parity.py — Verify live universe matches reference ACTIVE.

These are canary tests: if someone edits _production_run.ACTIVE without
updating the live config or vice versa, this suite catches the drift.
"""

import pytest


def test_live_symbols_match_active(live_cfg):
    from reference._production_run import ACTIVE
    assert set(live_cfg.symbols) == set(ACTIVE), (
        f"Symbol drift detected.\n"
        f"  In ACTIVE but not live: {set(ACTIVE) - set(live_cfg.symbols)}\n"
        f"  In live but not ACTIVE: {set(live_cfg.symbols) - set(ACTIVE)}"
    )


def test_symbol_count(live_cfg):
    from reference._production_run import ACTIVE
    assert len(live_cfg.symbols) == len(ACTIVE), (
        f"Count mismatch: live={len(live_cfg.symbols)}, ACTIVE={len(ACTIVE)}"
    )


def test_all_live_symbols_in_instruments(live_cfg):
    from reference.config import INSTRUMENTS
    missing = [s for s in live_cfg.symbols if s not in INSTRUMENTS]
    assert not missing, f"Live symbols missing from INSTRUMENTS: {missing}"


def test_gap_filters_present_for_all_symbols(live_cfg):
    missing = [s for s in live_cfg.symbols if s not in live_cfg.instrument_gap_filters]
    assert not missing, f"Symbols with no gap filter: {missing}"


def test_all_gap_filters_positive(live_cfg):
    bad = {s: v for s, v in live_cfg.instrument_gap_filters.items() if v <= 0}
    assert not bad, f"Non-positive gap filters: {bad}"


def test_direction_filters_subset_of_universe(live_cfg):
    unknown = [s for s in live_cfg.direction_filters if s not in live_cfg.symbols]
    assert not unknown, f"direction_filters reference symbols not in live universe: {unknown}"


def test_dow_exclusions_valid(live_cfg):
    for sym, days in live_cfg.day_of_week_exclusions.items():
        assert sym in live_cfg.symbols, f"DOW exclusion for unknown symbol: {sym}"
        for d in days:
            assert 0 <= d <= 6, f"Invalid weekday {d} for {sym}"


def test_ethu_ethd_monday_excluded(live_cfg):
    for sym in ("ETHU", "ETHD"):
        if sym in live_cfg.symbols:
            assert 0 in live_cfg.day_of_week_exclusions.get(sym, set()), (
                f"{sym} must have Monday (0) excluded (65h weekend gap)"
            )
