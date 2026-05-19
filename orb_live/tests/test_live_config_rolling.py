"""
tests/test_live_config_rolling.py — Tests for rolling-sigma build_live_ps_filters.

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
    from orb_live.config.live_config import build_live_ps_filters

    df_qqq = _make_parquet(tmp_path, "QQQ", seed=1)
    df_spy = _make_parquet(tmp_path, "SPY", seed=2)

    filters = build_live_ps_filters(
        use_rolling=True,
        data_dir=tmp_path,
        sigma_override_path=tmp_path / "nonexistent.yaml",
        sigma_runtime_path=tmp_path / "sigma_runtime.yaml",
        k=1.25,
    )

    # TQQQ tracks QQQ; UPRO tracks SPY
    assert "TQQQ" in filters
    assert abs(filters["TQQQ"][1] - _expected_sigma(df_qqq) * 1.25) < 1e-9

    assert "UPRO" in filters
    assert abs(filters["UPRO"][1] - _expected_sigma(df_spy) * 1.25) < 1e-9


def test_rolling_falls_back_to_seed_on_missing_parquet(tmp_path, caplog):
    from orb_live.config.live_config import build_live_ps_filters
    from reference._production_run import SIGMA

    # No parquets — every underlying must fall back to SIGMA seed
    with caplog.at_level(logging.WARNING, logger="orb_live.config.live_config"):
        filters = build_live_ps_filters(
            use_rolling=True,
            data_dir=tmp_path,
            sigma_override_path=tmp_path / "nonexistent.yaml",
            sigma_runtime_path=tmp_path / "sigma_runtime.yaml",
            k=1.25,
        )

    assert "TQQQ" in filters
    assert abs(filters["TQQQ"][1] - SIGMA["QQQ"] * 1.25) < 1e-9
    assert any("rolling_sigma_fallback" in r.message for r in caplog.records)


def test_rolling_falls_back_to_seed_on_insufficient_data(tmp_path, caplog):
    from orb_live.config.live_config import build_live_ps_filters
    from reference._production_run import SIGMA

    # Create parquet with only 20 rows — below the 30-row minimum
    df = pd.DataFrame({
        "date":  pd.date_range("2024-01-02", periods=20, freq="B"),
        "close": np.linspace(100, 110, 20),
    })
    df.to_parquet(tmp_path / "QQQ.parquet", index=False)

    with caplog.at_level(logging.WARNING, logger="orb_live.config.live_config"):
        filters = build_live_ps_filters(
            use_rolling=True,
            data_dir=tmp_path,
            sigma_override_path=tmp_path / "nonexistent.yaml",
            sigma_runtime_path=tmp_path / "sigma_runtime.yaml",
            k=1.25,
        )

    assert "TQQQ" in filters
    assert abs(filters["TQQQ"][1] - SIGMA["QQQ"] * 1.25) < 1e-9
    assert any("rolling_sigma_fallback" in r.message for r in caplog.records)


def test_use_rolling_false_uses_seed_values(tmp_path):
    from orb_live.config.live_config import build_live_ps_filters
    from reference._production_run import PS_FILTERS

    filters = build_live_ps_filters(
        use_rolling=False,
        k=1.25,
        sigma_override_path=tmp_path / "nonexistent.yaml",
    )
    assert filters == PS_FILTERS


def test_universe_underlyings_discovered(tmp_path):
    from orb_live.config.live_config import build_live_ps_filters
    from reference._production_run import SIGMA

    # ASHR/EWW/ITA/IYR/XLU are in UNIVERSE but absent from SIGMA.
    # When their parquets exist, they must produce ps_filter entries.
    missing_from_sigma = ["ASHR", "EWW", "ITA", "IYR", "XLU"]
    for ul in missing_from_sigma:
        assert ul not in SIGMA, f"test assumption broken: {ul} is now in SIGMA"
        _make_parquet(tmp_path, ul, seed=abs(hash(ul)) % 10_000)

    filters = build_live_ps_filters(
        use_rolling=True,
        data_dir=tmp_path,
        sigma_override_path=tmp_path / "nonexistent.yaml",
        sigma_runtime_path=tmp_path / "sigma_runtime.yaml",
        k=1.25,
    )

    # The ETFs that use these underlyings
    ul_to_sym = {
        "ASHR": "CHAU",
        "EWW":  "MEXX",
        "ITA":  "DFEN",
        "IYR":  "URE",
        "XLU":  "UTSL",
    }
    for ul, sym in ul_to_sym.items():
        assert sym in filters, (
            f"{sym} (underlying={ul}) should have a ps_filter when parquet exists"
        )


def test_sigma_runtime_yaml_written(tmp_path):
    from orb_live.config.live_config import build_live_ps_filters

    runtime_path = tmp_path / "sigma_runtime.yaml"
    _make_parquet(tmp_path, "QQQ", seed=1)

    build_live_ps_filters(
        use_rolling=True,
        data_dir=tmp_path,
        sigma_override_path=tmp_path / "nonexistent.yaml",
        sigma_runtime_path=runtime_path,
        k=1.25,
    )

    assert runtime_path.exists(), "sigma_runtime.yaml was not created"
    content = yaml.safe_load(runtime_path.read_text())

    assert "last_computed" in content
    assert "source" in content
    assert content["source"] == "rolling"
    assert "sigmas" in content
    assert "fallback_underlyings" in content
    assert isinstance(content["sigmas"], dict)
    # QQQ had a parquet so it should appear with a computed float value
    assert "QQQ" in content["sigmas"]
    assert isinstance(content["sigmas"]["QQQ"], float)


def test_parallel_rolling_parity(tmp_path):
    """
    Parallel to test_ps_filter_parity: rolling mode with seeded parquets
    produces thresholds equal to std(abs_returns)*k for each parqueted underlying.
    """
    from orb_live.config.live_config import build_live_ps_filters
    from reference._production_run import UNIVERSE

    # Seed parquets for a representative subset of underlyings
    test_uls = ["QQQ", "SPY", "IWM", "GDX", "BTC"]
    dfs = {ul: _make_parquet(tmp_path, ul, seed=i) for i, ul in enumerate(test_uls)}

    filters = build_live_ps_filters(
        use_rolling=True,
        data_dir=tmp_path,
        sigma_override_path=tmp_path / "nonexistent.yaml",
        sigma_runtime_path=tmp_path / "sigma_runtime.yaml",
        k=1.25,
    )

    # For every ETF whose underlying had a parquet, verify threshold = sigma * k
    for sym, info in UNIVERSE.items():
        ul = info["underlying"]
        if ul not in dfs or sym not in filters:
            continue
        expected = _expected_sigma(dfs[ul]) * 1.25
        assert abs(filters[sym][1] - expected) < 1e-9, (
            f"{sym} threshold mismatch: {filters[sym][1]:.8f} != {expected:.8f}"
        )
