"""
tests/test_universe_parity.py — Verify live universe matches v1 build_universe().

These are canary tests: if someone edits the v1 strategy CSVs or live_config
without keeping them in sync, this suite catches the drift.
"""

import pytest


def test_live_symbols_match_v1_universe(live_cfg):
    from orb_live.strategy.v1_strategy import load_master, load_sigmas, build_universe
    from pathlib import Path

    data_dir  = Path(__file__).parent.parent / "strategy" / "data"
    instr     = load_master(data_dir / "master_universe.csv")
    sigmas    = load_sigmas(data_dir / "sigma_master.csv")
    expected  = set(build_universe(instr, sigmas, data_dir / "master_universe.csv"))
    actual    = set(live_cfg.symbols)

    assert actual == expected, (
        f"Symbol drift.\n"
        f"  In v1 but not live: {expected - actual}\n"
        f"  In live but not v1: {actual - expected}"
    )


def test_symbol_count(live_cfg):
    n = len(live_cfg.symbols)
    assert 50 <= n <= 75, f"Expected ~60 symbols, got {n}"


def test_all_live_symbols_in_instruments(live_cfg):
    missing = [s for s in live_cfg.symbols if s not in live_cfg.instruments]
    assert not missing, f"Live symbols missing from instruments: {missing}"


def test_direction_filters_subset_of_universe(live_cfg):
    unknown = [s for s in live_cfg.direction_filters if s not in live_cfg.symbols]
    assert not unknown, f"direction_filters reference symbols not in live universe: {unknown}"


def test_no_day_of_week_exclusions(live_cfg):
    """v1 removes DOW exclusions; strategy_config.day_of_week_exclusions must be empty."""
    assert live_cfg.strategy_config.day_of_week_exclusions == {}, (
        "v1 config must have no DOW exclusions"
    )


def test_no_instrument_gap_filters(live_cfg):
    """v1 uses per-ETF leverage × GAP_THRESHOLD; instrument_gap_filters must be empty."""
    assert live_cfg.strategy_config.instrument_gap_filters == {}, (
        "v1 config must have no instrument_gap_filters"
    )


class TestSkipCheapParity:
    """Parity test for the skip-cheap-top-2 pruning rule.

    Backtest rule (skip_cheap_by_class.py :: skip_cheap_then_top2_when_3plus):
      N==1 → keep 1; N==2 → drop cheaper, keep 1; N>=3 → keep top-2 by prior close
    """

    def _run(self, syms_with_prices: list[tuple[str, float]]) -> list[str]:
        from orb_live.strategy.v1_strategy import (
            compute_candidates, Instrument,
        )
        ul = "TESTUL"
        instruments = {
            sym: Instrument(sym, ul, 2, False)
            for sym, _ in syms_with_prices
        }
        universe = [s for s, _ in syms_with_prices]
        prior_etf_close = {s: p for s, p in syms_with_prices}
        overnight_gaps = {ul: 0.05}  # 5% gap, above 2% threshold
        prior_two_closes = {ul: (100.0, 100.0)}  # flat — PS filter always passes

        cands = compute_candidates(
            universe, instruments, {"TESTUL": 0.02},
            overnight_gaps, prior_two_closes, prior_etf_close,
        )
        return [c.symbol for c in cands]

    def test_n1_keeps_1(self):
        kept = self._run([("A", 100.0)])
        assert kept == ["A"]

    def test_n2_keeps_1_most_expensive(self):
        # N==2: classic skip-cheap — keep only the pricier one
        kept = self._run([("CHEAP", 50.0), ("PRICEY", 150.0)])
        assert len(kept) == 1
        assert kept[0] == "PRICEY"

    def test_n3_keeps_2_most_expensive(self):
        kept = self._run([("A", 30.0), ("B", 100.0), ("C", 200.0)])
        assert len(kept) == 2
        assert set(kept) == {"B", "C"}

    def test_n4_keeps_2_most_expensive(self):
        kept = self._run([("A", 10.0), ("B", 50.0), ("C", 100.0), ("D", 200.0)])
        assert len(kept) == 2
        assert set(kept) == {"C", "D"}
