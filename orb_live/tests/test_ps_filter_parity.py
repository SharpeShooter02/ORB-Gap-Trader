"""
tests/test_ps_filter_parity.py — Verify live PS filter config matches v1 sigmas.

All tests here explicitly use use_rolling=False (frozen sigma_master.csv seeds)
so that parity is guaranteed regardless of whether parquet files exist.
"""

import pytest


@pytest.fixture
def live_cfg():
    """Frozen-seed config — override the session-scoped conftest fixture."""
    from orb_live.config.live_config import load_live_config
    return load_live_config(use_rolling=False)


def test_ps_filter_keys_are_live_symbols(live_cfg):
    unknown = [s for s in live_cfg.prior_session_filters if s not in live_cfg.symbols]
    assert not unknown, f"PS filters reference symbols not in live universe: {unknown}"


def test_ps_filter_thresholds_positive(live_cfg):
    for sym, spec in live_cfg.prior_session_filters.items():
        threshold = spec[1]
        assert threshold > 0, f"Non-positive threshold for {sym}: {threshold}"


def test_ps_filter_k_applied(live_cfg):
    """Thresholds equal sigma_master sigma × ps_filter_k (1.0)."""
    from orb_live.strategy.v1_strategy import load_sigmas
    from pathlib import Path

    sigma_csv = Path(__file__).parent.parent / "strategy" / "data" / "sigma_master.csv"
    sigmas    = load_sigmas(sigma_csv)
    k         = live_cfg.ps_filter_k

    for sym, spec in live_cfg.prior_session_filters.items():
        ul, threshold = spec[0], spec[1]
        base_sigma = sigmas.get(ul)
        if base_sigma is None:
            continue
        expected = base_sigma * k
        assert abs(threshold - expected) < 1e-9, (
            f"{sym}: threshold={threshold:.6f} != sigma*k={expected:.6f} "
            f"(sigma={base_sigma}, k={k})"
        )


def test_inverse_etfs_have_inverse_flag(live_cfg):
    """Inverse ETFs get a 3-tuple with True; non-inverse get a 2-tuple."""
    for sym, spec in live_cfg.prior_session_filters.items():
        inst = live_cfg.instruments.get(sym)
        if inst is None:
            continue
        is_inv   = getattr(inst, "inverse", False)
        has_flag = len(spec) >= 3 and spec[2] is True
        assert is_inv == has_flag, (
            f"{sym}: inverse={is_inv} but ps_filter inverse_flag={has_flag}"
        )


def test_ps_filters_non_empty(live_cfg):
    """At least one PS filter must exist (sanity: CSVs loaded correctly)."""
    assert live_cfg.prior_session_filters, "prior_session_filters is empty"
