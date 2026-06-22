"""
config/live_config.py — v1 strategy bridge.

Sources all strategy constants from the vendored orb_live/strategy/v1_strategy.py.
No runtime imports from reference/_production_run.

Verification:
    python -c "from orb_live.config.live_config import load_live_config; \
        cfg = load_live_config(use_rolling=False); \
        print(len(cfg.symbols), 'symbols')"
    → ~60 symbols
"""

import datetime
import logging
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import orb_live  # noqa: F401 — triggers orb_live/__init__.py path setup

from reference.orb_backtester import StrategyConfig, DEFAULT_CONFIG
from orb_live.strategy.v1_strategy import (
    load_master,
    load_sigmas,
    build_universe,
    K_SIGMA,
    CAP_UNITS,
    DIRECTION_FILTERS,
    Instrument,
)

_STRATEGY_DATA  = Path(__file__).parent.parent / "strategy" / "data"
_MASTER_CSV     = _STRATEGY_DATA / "master_universe.csv"
_SIGMA_CSV      = _STRATEGY_DATA / "sigma_master.csv"

_OVERRIDES_FILE       = Path(__file__).parent / "overrides.yaml"
_SIGMA_OVERRIDE_FILE  = Path(__file__).parent / "sigma_override.yaml"
_DEFAULT_DATA_DIR     = Path(__file__).parent.parent / "data" / "underlyings"
_DEFAULT_DB_PATH      = Path(__file__).parent.parent / "state" / "live.db"
_SIGMA_RUNTIME_FILE   = Path(__file__).parent.parent / "data" / "sigma_runtime.yaml"

_logger = logging.getLogger(__name__)


# ── Build static v1 objects (at import time) ───────────────────────────────────

def _load_v1_universe() -> tuple[dict, dict, list]:
    """Load instruments, sigmas, and the v1 symbol list from vendored CSVs."""
    instruments = load_master(_MASTER_CSV)
    sigmas      = load_sigmas(_SIGMA_CSV)
    universe    = build_universe(instruments, sigmas, _MASTER_CSV)
    return instruments, sigmas, universe


# ── PS-filter builder (v1 style) ───────────────────────────────────────────────

def _build_ps_filters(
    instruments: dict[str, Instrument],
    sigmas: dict[str, float],
    universe: list[str],
    k: float = K_SIGMA,
    sigma_override_path: Path = _SIGMA_OVERRIDE_FILE,
) -> dict:
    """Build prior_session_filters in the legacy (ul, threshold[, True]) tuple format.

    Every ETF in universe whose UL has a sigma gets an entry.
    Threshold = sigma × k.  Inverse ETFs get a third element True.
    """
    effective_sigma = dict(sigmas)
    if sigma_override_path.exists():
        try:
            with sigma_override_path.open() as f:
                override = yaml.safe_load(f) or {}
            if "sigmas" in override:
                effective_sigma = {**effective_sigma, **override["sigmas"]}
        except Exception:
            pass

    filters: dict = {}
    for sym in universe:
        inst = instruments.get(sym)
        if inst is None:
            continue
        sig = effective_sigma.get(inst.underlying)
        if sig is None:
            continue
        threshold = sig * k
        if inst.inverse:
            filters[sym] = (inst.underlying, threshold, True)
        else:
            filters[sym] = (inst.underlying, threshold)
    return filters


def _build_rolling_sigma_map(
    instruments: dict[str, Instrument],
    universe: list[str],
    sigmas: dict[str, float],
    data_dir: Path,
) -> dict[str, float]:
    """Compute rolling sigma from UL parquets; fall back to CSV seeds."""
    all_uls = sorted({instruments[s].underlying for s in universe if s in instruments})
    sigma_map: dict[str, float] = {}
    for ul in all_uls:
        path = data_dir / f"{ul}.parquet"
        try:
            if not path.exists():
                raise FileNotFoundError
            df     = pd.read_parquet(path)
            closes = df["close"].dropna()
            if len(closes) < 30:
                raise ValueError(f"only {len(closes)} rows")
            returns = closes.pct_change().dropna().abs()
            sigma_map[ul] = float(returns.std())
        except Exception as exc:
            seed = sigmas.get(ul)
            if seed is not None:
                _logger.warning("rolling_sigma_fallback ul=%s reason=%s seed=%.6f", ul, exc, seed)
                sigma_map[ul] = seed
            else:
                _logger.warning("rolling_sigma_no_seed ul=%s — ps_filter disabled for UL", ul, exc)
    return sigma_map


def _write_sigma_runtime(sigma_map: dict, path: Path) -> None:
    payload = {
        "last_computed": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source": "rolling",
        "sigmas": {ul: round(s, 6) for ul, s in sorted(sigma_map.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False))


# ── v1 StrategyConfig ──────────────────────────────────────────────────────────

def _make_v1_strategy_config(
    universe: list[str],
    ps_filters: dict,
) -> StrategyConfig:
    """Return a StrategyConfig tuned for v1 — TP1-only, no quality gates."""
    from dataclasses import replace
    return replace(
        DEFAULT_CONFIG,
        symbols=universe,
        gap_filter_pct=0.0,          # handled per-ETF via leverage × GAP_THRESHOLD
        instrument_gap_filters={},
        prior_session_filters=ps_filters,
        min_profit_pct=0.0,          # v1: all breakouts taken
        min_increment_pct=0.0,
        min_entry_excess=0.0,
        exit_ratio_tp1=1.0,          # v1: TP1-only (full position)
        exit_ratio_tp2=0.0,
        exit_ratio_tp3=0.0,
        tp1_target_multiple=1.0,     # 1× ORB range
        use_rtg_scaling=False,
        rtg_gap_exclusion=False,
        rtg_pair_routing=False,
        day_of_week_exclusions={},   # v1: none
        direction_filters=dict(DIRECTION_FILTERS),
        use_risk_based_sizing=False,
        daily_risk_pct=0.0,          # v1 sizing uses v1_base_notional instead
    )


# ── LiveConfig ─────────────────────────────────────────────────────────────────

@dataclass
class LiveConfig:
    """Production live-trading configuration (v1 brain)."""

    # ── v1 strategy layer ─────────────────────────────────────────────────────
    instruments: dict = field(default_factory=dict)   # dict[str, Instrument]
    sigmas:      dict = field(default_factory=dict)   # dict[str, float]  (UL → σ)

    strategy_config: StrategyConfig = field(
        default_factory=lambda: DEFAULT_CONFIG
    )
    symbols: List[str] = field(default_factory=list)

    prior_session_filters: Dict[str, tuple] = field(default_factory=dict)
    direction_filters:     Dict[str, int]   = field(default_factory=lambda: dict(DIRECTION_FILTERS))

    # v1 sizing: shares = floor(v1_base_notional × multiplier / entry_price)
    v1_base_notional: float = 1_000.0
    cap_units:        float = CAP_UNITS   # 20.0 → 200% cap
    ps_filter_k:      float = K_SIGMA     # 1.00

    # ── Execution parameters ──────────────────────────────────────────────────
    slippage_pct:            float = 0.001
    adv_limit_fraction:      float = 0.005
    stop_order_type:         str   = "market"

    entry_slippage_bps:      int   = 10
    entry_repeg_seconds:     float = 30.0
    entry_repeg_max_attempts: int  = 3
    entry_slippage_max_bps:  int   = 30
    exit_slippage_bps:       int   = 5

    max_gross_exposure_pct:  float = 2.0
    max_position_pct:        float = 0.50

    # ── Pre-flight liquidity check ────────────────────────────────────────────
    # Gate C thresholds are neutralised for v1: the only liquidity filter is the
    # universe-build ADV floor.  Set to non-zero values to re-enable per-trade
    # liquidity gating (reversible via overrides.yaml).
    adv_lookback_days:       int   = 20
    min_dollar_volume_floor: float = 0.0
    max_pct_of_adv:          float = 1.0
    min_yesterday_dv_ratio:  float = 0.0
    allow_htb_shorts:        bool  = False

    # ── Risk management ───────────────────────────────────────────────────────
    session_kill_loss_pct:   float = 0.03
    max_concurrent_positions: int  = 0

    # ── Account ───────────────────────────────────────────────────────────────
    paper_trading:    bool = True
    fractional_shares: bool = True

    # ── ORB timing ────────────────────────────────────────────────────────────
    orb_minutes:     int = 30
    eod_exit_hour:   int = 16
    eod_exit_minute: int = 0

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir:            Path = field(default_factory=lambda: _DEFAULT_DATA_DIR)
    sigma_override_path: Path = field(default_factory=lambda: _SIGMA_OVERRIDE_FILE)
    db_path:             Path = field(default_factory=lambda: _DEFAULT_DB_PATH)

    # ── Operator exclusions (from overrides.yaml) ─────────────────────────────
    excluded_symbols:   List[str]        = field(default_factory=list)
    date_exclusions:    Dict[str, List[str]] = field(default_factory=dict)
    force_long_only:    List[str]        = field(default_factory=list)


# ── Loader ─────────────────────────────────────────────────────────────────────

def _load_overrides() -> dict:
    if not _OVERRIDES_FILE.exists():
        return {}
    with _OVERRIDES_FILE.open() as f:
        return yaml.safe_load(f) or {}


def load_live_config(
    overrides: Optional[dict] = None,
    use_rolling: bool = True,
) -> LiveConfig:
    """Build and return a LiveConfig instance sourced from v1.

    use_rolling=True (default): recompute sigmas from UL parquets.
    use_rolling=False: use sigma_master.csv seeds only (for tests).
    """
    instruments, sigmas_seed, universe = _load_v1_universe()

    # Rolling sigma update
    _env_flag = os.getenv("USE_ROLLING_SIGMAS", "1").strip().lower()
    _rolling  = use_rolling and (_env_flag not in ("0", "false"))

    if _rolling:
        effective_sigma = _build_rolling_sigma_map(
            instruments, universe, sigmas_seed, _DEFAULT_DATA_DIR
        )
        try:
            _write_sigma_runtime(effective_sigma, _SIGMA_RUNTIME_FILE)
        except Exception as exc:
            _logger.warning("sigma_runtime_write_failed reason=%s", exc)
    else:
        effective_sigma = dict(sigmas_seed)

    ps_filters = _build_ps_filters(
        instruments, effective_sigma, universe,
        k=K_SIGMA, sigma_override_path=_SIGMA_OVERRIDE_FILE,
    )
    scfg = _make_v1_strategy_config(universe, ps_filters)

    cfg = LiveConfig(
        instruments=instruments,
        sigmas=effective_sigma,
        strategy_config=scfg,
        symbols=list(universe),
        prior_session_filters=ps_filters,
        direction_filters=dict(DIRECTION_FILTERS),
    )

    # Apply overrides.yaml then caller overrides
    merged = {**_load_overrides(), **(overrides or {})}
    for key in ("excluded_symbols", "date_exclusions", "force_long_only",
                "paper_trading", "v1_base_notional", "cap_units", "ps_filter_k"):
        if key in merged:
            setattr(cfg, key, merged.pop(key))
    for key, val in merged.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    return cfg
