"""
tests/test_sigma_source_parity.py — Lock live's sigma source to the vendored
long-history sigma_master.csv (the backtest's methodology), NOT a rolling
30-day sigma computed at startup.

Guards the revert of the rolling-sigma path:
  - sigma_master.csv exists and loads
  - live loads those exact values (bit-for-bit) — no rolling override
  - no code path computes rolling sigma (_build_rolling_sigma_map is gone)
  - nothing is written to sigma_runtime.yaml at runtime
"""

from pathlib import Path

import pandas as pd

from orb_live.config import live_config as m
from orb_live.config.live_config import load_live_config, _SIGMA_CSV
from orb_live.strategy.v1_strategy import load_sigmas


_SIGMA_RUNTIME = Path(m.__file__).parent.parent / "data" / "sigma_runtime.yaml"


def test_sigma_master_csv_exists_and_loads():
    assert _SIGMA_CSV.exists(), f"missing authoritative sigma file: {_SIGMA_CSV}"
    sigmas = load_sigmas(_SIGMA_CSV)
    assert sigmas, "sigma_master.csv loaded no rows"
    # Long-history file, not a 30-day window: many underlyings present.
    assert len(sigmas) >= 50


def test_rolling_sigma_path_removed():
    """No code path may compute rolling sigma or write sigma_runtime.yaml."""
    assert not hasattr(m, "_build_rolling_sigma_map"), \
        "rolling sigma computation must be removed"
    assert not hasattr(m, "_write_sigma_runtime"), \
        "sigma_runtime writer must be removed"


def test_loaded_sigmas_match_csv_bit_for_bit():
    """Every underlying sigma the live config exposes equals the CSV value
    exactly — proving no rolling recomputation altered them."""
    csv = pd.read_csv(_SIGMA_CSV)
    csv_map = dict(zip(csv["underlying"].astype(str), csv["sigma"].astype(float)))

    cfg = load_live_config()
    checked = 0
    for ul, sig in cfg.sigmas.items():
        if ul in csv_map:
            assert sig == csv_map[ul], f"{ul}: live sigma {sig} != CSV {csv_map[ul]}"
            checked += 1
    assert checked >= 50, f"only {checked} sigmas checked against CSV"

    # use_rolling is a no-op: both call styles yield identical values.
    assert load_live_config(use_rolling=False).sigmas == cfg.sigmas


def test_ps_thresholds_are_seed_sigma_times_k():
    """PS-filter thresholds must be CSV sigma × k — not a rolling-derived value."""
    csv = pd.read_csv(_SIGMA_CSV)
    csv_map = dict(zip(csv["underlying"].astype(str), csv["sigma"].astype(float)))

    cfg = load_live_config()
    for sym, spec in cfg.prior_session_filters.items():
        inst = cfg.instruments.get(sym)
        if inst is None or inst.underlying not in csv_map:
            continue
        expected = csv_map[inst.underlying] * cfg.ps_filter_k
        assert abs(spec[1] - expected) < 1e-12, (
            f"{sym}: threshold {spec[1]} != seed*k {expected}"
        )


def test_no_sigma_runtime_written_at_startup():
    """Building the config must not create sigma_runtime.yaml."""
    if _SIGMA_RUNTIME.exists():
        _SIGMA_RUNTIME.unlink()
    load_live_config()
    assert not _SIGMA_RUNTIME.exists(), \
        "sigma_runtime.yaml must not be written at runtime"
