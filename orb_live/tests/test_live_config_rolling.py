"""
tests/test_live_config_rolling.py — Tests for rolling sigma computation.

All tests use tmp_path fixture parquets so they run without real market data
on any machine (CI, dev, VPS).
"""

import logging

import numpy as np
import pandas as pd
import pytest
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_parquet(data_dir, ul: str, n: int = 252, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.02, n))
    df = pd.DataFrame({
        "date":   pd.date_range("2020-01-02", periods=n, freq="B"),
        "open":   closes * 0.99,
        "high":   closes * 1.01,
        "low":    closes * 0.98,
        "close":  closes,
        "volume": 1_000_000,
    })
    df.to_parquet(data_dir / f"{ul}.parquet", index=False)
    return df


def _expected_sigma(df: pd.DataFrame) -> float:
    return float(df["close"].pct_change().dropna().abs().std())


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_rolling_sigmas_computed_from_parquet(tmp_path):
    from orb_live.config.live_config import (
        _load_v1_universe, _build_rolling_sigma_map, _build_ps_filters,
    )

    instruments, sigmas_seed, universe = _load_v1_universe()
    df_qqq = _make_parquet(tmp_path, "QQQ", seed=1)
    df_spy = _make_parquet(tmp_path, "SPY", seed=2)

    sigma_map = _build_rolling_sigma_map(instruments, universe, sigmas_seed, tmp_path)
    filters   = _build_ps_filters(instruments, sigma_map, universe, k=1.25,
                                  sigma_override_path=tmp_path / "nonexistent.yaml")

    assert "TQQQ" in filters
    assert abs(filters["TQQQ"][1] - _expected_sigma(df_qqq) * 1.25) < 1e-9

    assert "UPRO" in filters
    assert abs(filters["UPRO"][1] - _expected_sigma(df_spy) * 1.25) < 1e-9


def test_rolling_falls_back_to_seed_on_missing_parquet(tmp_path, caplog):
    from orb_live.config.live_config import (
        _load_v1_universe, _build_rolling_sigma_map, _build_ps_filters,
    )

    instruments, sigmas_seed, universe = _load_v1_universe()

    with caplog.at_level(logging.WARNING, logger="orb_live.config.live_config"):
        sigma_map = _build_rolling_sigma_map(instruments, universe, sigmas_seed, tmp_path)
    filters = _build_ps_filters(instruments, sigma_map, universe, k=1.25,
                                sigma_override_path=tmp_path / "nonexistent.yaml")

    assert "TQQQ" in filters
    assert abs(filters["TQQQ"][1] - sigmas_seed["QQQ"] * 1.25) < 1e-9
    assert any("rolling_sigma_fallback" in r.message for r in caplog.records)


def test_rolling_falls_back_to_seed_on_insufficient_data(tmp_path, caplog):
    from orb_live.config.live_config import (
        _load_v1_universe, _build_rolling_sigma_map, _build_ps_filters,
    )

    instruments, sigmas_seed, universe = _load_v1_universe()

    # Only 20 rows — below the 30-row minimum
    df = pd.DataFrame({
        "date":  pd.date_range("2024-01-02", periods=20, freq="B"),
        "close": np.linspace(100, 110, 20),
    })
    df.to_parquet(tmp_path / "QQQ.parquet", index=False)

    with caplog.at_level(logging.WARNING, logger="orb_live.config.live_config"):
        sigma_map = _build_rolling_sigma_map(instruments, universe, sigmas_seed, tmp_path)
    filters = _build_ps_filters(instruments, sigma_map, universe, k=1.25,
                                sigma_override_path=tmp_path / "nonexistent.yaml")

    assert "TQQQ" in filters
    assert abs(filters["TQQQ"][1] - sigmas_seed["QQQ"] * 1.25) < 1e-9
    assert any("rolling_sigma_fallback" in r.message for r in caplog.records)


def test_use_rolling_false_uses_seed_values():
    from orb_live.config.live_config import _load_v1_universe, load_live_config

    instruments, sigmas_seed, universe = _load_v1_universe()
    cfg = load_live_config(use_rolling=False)

    for sym, spec in cfg.prior_session_filters.items():
        inst = instruments.get(sym)
        if inst is None:
            continue
        seed = sigmas_seed.get(inst.underlying)
        if seed is None:
            continue
        expected = seed * cfg.ps_filter_k
        actual   = spec[1]
        assert abs(actual - expected) < 1e-9, (
            f"{sym}: threshold={actual:.6f} != seed*k={expected:.6f}"
        )


def test_sigma_runtime_yaml_written(tmp_path):
    from orb_live.config.live_config import (
        _load_v1_universe, _build_rolling_sigma_map, _write_sigma_runtime,
    )

    instruments, sigmas_seed, universe = _load_v1_universe()
    _make_parquet(tmp_path, "QQQ", seed=1)
    sigma_map = _build_rolling_sigma_map(instruments, universe, sigmas_seed, tmp_path)

    runtime_path = tmp_path / "sigma_runtime.yaml"
    _write_sigma_runtime(sigma_map, runtime_path)

    assert runtime_path.exists(), "sigma_runtime.yaml was not created"
    content = yaml.safe_load(runtime_path.read_text())

    assert "last_computed" in content
    assert "source" in content
    assert content["source"] == "rolling"
    assert "sigmas" in content
    assert isinstance(content["sigmas"], dict)
    assert "QQQ" in content["sigmas"]
    assert isinstance(content["sigmas"]["QQQ"], float)


def test_parallel_rolling_parity(tmp_path):
    """Rolling mode produces thresholds equal to std(abs_returns)*k for parqueted underlyings."""
    from orb_live.config.live_config import (
        _load_v1_universe, _build_rolling_sigma_map, _build_ps_filters,
    )

    instruments, sigmas_seed, universe = _load_v1_universe()
    test_uls = ["QQQ", "SPY", "GDX"]
    dfs = {ul: _make_parquet(tmp_path, ul, seed=i) for i, ul in enumerate(test_uls)}

    sigma_map = _build_rolling_sigma_map(instruments, universe, sigmas_seed, tmp_path)
    filters   = _build_ps_filters(instruments, sigma_map, universe, k=1.25,
                                  sigma_override_path=tmp_path / "nonexistent.yaml")

    for sym in universe:
        inst = instruments.get(sym)
        if inst is None or sym not in filters:
            continue
        ul = inst.underlying
        if ul not in dfs:
            continue
        expected = _expected_sigma(dfs[ul]) * 1.25
        assert abs(filters[sym][1] - expected) < 1e-9, (
            f"{sym} threshold mismatch: {filters[sym][1]:.8f} != {expected:.8f}"
        )
