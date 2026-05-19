"""
tests/test_ps_filter_parity.py — Verify live PS filter config matches reference.

All tests here explicitly use use_rolling=False (frozen SIGMA seeds) so that
parity against the reference PS_FILTERS is guaranteed regardless of whether
parquet files exist on the test machine.
"""

import pytest


@pytest.fixture
def live_cfg():
    """Frozen-seed config — override the session-scoped conftest fixture."""
    from orb_live.config.live_config import load_live_config
    return load_live_config(use_rolling=False)


def test_ps_filters_match_reference(live_cfg):
    from reference._production_run import PS_FILTERS
    assert live_cfg.prior_session_filters == PS_FILTERS, (
        "PS filter mismatch between live config and reference:\n"
        + "\n".join(
            f"  {s}: live={live_cfg.prior_session_filters.get(s)!r}  "
            f"ref={PS_FILTERS.get(s)!r}"
            for s in set(live_cfg.prior_session_filters) | set(PS_FILTERS)
            if live_cfg.prior_session_filters.get(s) != PS_FILTERS.get(s)
        )
    )


def test_ps_filter_keys_are_live_symbols(live_cfg):
    unknown = [s for s in live_cfg.prior_session_filters if s not in live_cfg.symbols]
    assert not unknown, f"PS filters reference symbols not in live universe: {unknown}"


def test_ps_filter_thresholds_positive(live_cfg):
    for sym, spec in live_cfg.prior_session_filters.items():
        threshold = spec[1]
        assert threshold > 0, f"Non-positive threshold for {sym}: {threshold}"


def test_ps_filter_k_applied(live_cfg):
    from reference._production_run import SIGMA
    k = live_cfg.ps_filter_k
    for sym, spec in live_cfg.prior_session_filters.items():
        ul, threshold = spec[0], spec[1]
        base_sigma = SIGMA.get(ul)
        if base_sigma is not None:
            expected = base_sigma * k
            assert abs(threshold - expected) < 1e-9, (
                f"{sym}: threshold={threshold:.6f} != sigma*k={expected:.6f} "
                f"(sigma={base_sigma}, k={k})"
            )


def test_inverse_etfs_have_inverse_flag(live_cfg):
    from reference.config import INSTRUMENTS
    for sym, spec in live_cfg.prior_session_filters.items():
        if sym not in INSTRUMENTS:
            continue
        is_inv = INSTRUMENTS[sym]["inverse"]
        has_flag = len(spec) >= 3 and spec[2] is True
        assert is_inv == has_flag, (
            f"{sym}: inverse={is_inv} but ps_filter flag={has_flag}"
        )
