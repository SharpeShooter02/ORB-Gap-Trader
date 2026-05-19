"""
config/live_config.py — THE CRITICAL FILE.

Bridges the backtest reference layer to the live system.  Every live
component imports from here rather than importing reference.* directly,
so that config divergence is caught in one place.

Verification:
    python -c "from orb_live.config.live_config import load_live_config; \
        cfg = load_live_config(); \
        print(len(cfg.symbols), 'symbols'); \
        print(len(cfg.prior_session_filters), 'ps_filters')"
    → 47 symbols
    → <n> ps_filters
"""

import datetime
import logging
import os
import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# Ensure BacktestingGaps root is on sys.path before reference imports.
import orb_live  # noqa: F401 — triggers orb_live/__init__.py path setup

from reference.orb_backtester import StrategyConfig, DEFAULT_CONFIG
from reference.config import INSTRUMENTS, UNDERLYINGS
from reference._production_run import (
    ACTIVE,
    UNIVERSE,
    ETH_SYMS,
    GAP_FILTER,
    DOW_EXCL,
    SIGMA,
    ps_filter,
    PS_FILTERS,
    SYMS,
    CLASS_A_EXCL_SYMS,
    CRYPTO_UL,
    SINGLESTK_UL,
    NATGAS_UL,
    INTERNATIONAL_SYMS,
    BIOTECH_SYMS,
    BANKING_SYMS,
    DRIFT_SYMS,
    CLASS_A_UL,
    CLASS_A_SYMS,
    FLOOD_LOSERS,
    _is_class_a,
    BASELINE_PCT,
    CAP_UNITS,
    apply_sizing,
    cfg as _prod_cfg,
)

_OVERRIDES_FILE      = Path(__file__).parent / "overrides.yaml"
_SIGMA_OVERRIDE_FILE = Path(__file__).parent / "sigma_override.yaml"

# Default paths (relative to package root).
_DEFAULT_DATA_DIR    = Path(__file__).parent.parent / "data" / "underlyings"
_DEFAULT_DB_PATH     = Path(__file__).parent.parent / "state" / "live.db"
_SIGMA_RUNTIME_FILE  = Path(__file__).parent.parent / "data" / "sigma_runtime.yaml"

_logger = logging.getLogger(__name__)


# ── Rolling sigma helpers ─────────────────────────────────────────────────────

def _compute_rolling_sigma_map(
    data_dir: Path,
) -> tuple[dict, dict, list]:
    """
    Compute sigma (std of abs daily returns) for every underlying in UNIVERSE
    from its parquet file in data_dir.

    Returns (sigma_map, n_obs_map, fallback_underlyings).
    Falls back to SIGMA seed values when a parquet is missing or has <30 rows.
    Logs WARN for each fallback.
    """
    all_uls = sorted({info["underlying"] for info in UNIVERSE.values()})
    sigma_map:    dict = {}
    n_obs_map:    dict = {}
    fallback_uls: list = []

    for ul in all_uls:
        path = data_dir / f"{ul}.parquet"
        try:
            if not path.exists():
                raise FileNotFoundError("parquet not found")
            df = pd.read_parquet(path)
            closes = df["close"].dropna()
            if len(closes) < 30:
                raise ValueError(f"only {len(closes)} rows")
            returns = closes.pct_change().dropna().abs()
            sigma_map[ul] = float(returns.std())
            n_obs_map[ul] = len(returns)
        except Exception as exc:
            seed = SIGMA.get(ul)
            if seed is not None:
                _logger.warning(
                    "rolling_sigma_fallback ul=%s reason=%s seed=%.6f", ul, exc, seed
                )
                sigma_map[ul] = seed
            else:
                _logger.warning(
                    "rolling_sigma_no_seed ul=%s reason=%s — ps_filter disabled", ul, exc
                )
            n_obs_map[ul] = 0
            fallback_uls.append(ul)

    return sigma_map, n_obs_map, fallback_uls


def _write_sigma_runtime(
    sigma_map: dict,
    fallback_uls: list,
    path: Path,
) -> None:
    payload = {
        "last_computed": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source": "rolling",
        "sigmas": {ul: round(s, 6) for ul, s in sorted(sigma_map.items())},
        "fallback_underlyings": sorted(fallback_uls),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False))


# ── PS-filter builder ─────────────────────────────────────────────────────────

def build_live_ps_filters(
    sigma_map: Optional[dict] = None,
    k: float = 1.25,
    use_rolling: bool = True,
    data_dir: Optional[Path] = None,
    sigma_override_path: Optional[Path] = None,
    sigma_runtime_path: Optional[Path] = None,
    state_store=None,
) -> dict:
    """
    Build the prior_session_filters dict.

    use_rolling=True (default): sigmas computed from full historical data in
    data_dir/*.parquet. Falls back to SIGMA seeds for underlyings whose parquet
    is missing or has <30 rows; logs WARN. Writes sigma_runtime.yaml and
    optionally audits state_store.sigma_history.

    use_rolling=False: uses sigma_map (or SIGMA seeds). Identical to the
    pre-rolling frozen-seed behaviour. Used by tests and backtest validation.

    sigma_override.yaml is applied on top of either path when it exists.
    """
    _override_path = sigma_override_path if sigma_override_path is not None else _SIGMA_OVERRIDE_FILE

    if use_rolling:
        _data_dir = data_dir or _DEFAULT_DATA_DIR
        computed_map, n_obs_map, fallback_uls = _compute_rolling_sigma_map(_data_dir)

        _runtime_path = sigma_runtime_path or _SIGMA_RUNTIME_FILE
        try:
            _write_sigma_runtime(computed_map, fallback_uls, _runtime_path)
        except Exception as exc:
            _logger.warning("sigma_runtime_write_failed path=%s reason=%s", _runtime_path, exc)

        if state_store is not None:
            for ul, sig in computed_map.items():
                source = "runtime_fallback" if ul in fallback_uls else "runtime_rolling"
                try:
                    state_store.save_sigma_calibration(
                        underlying=ul,
                        sigma=sig,
                        n_obs=n_obs_map.get(ul, 0),
                        source=source,
                    )
                except Exception:
                    pass

        effective_sigma = computed_map
    else:
        effective_sigma = dict(sigma_map) if sigma_map is not None else dict(SIGMA)

    if _override_path.exists():
        with _override_path.open() as f:
            override = yaml.safe_load(f) or {}
        if "sigmas" in override:
            effective_sigma = {**effective_sigma, **override["sigmas"]}

    filters: dict = {}
    for sym, info in UNIVERSE.items():
        ul  = info["underlying"]
        sig = effective_sigma.get(ul)
        if sig is None:
            continue
        threshold = sig * k
        filters[sym] = (ul, threshold, True) if info["inverse"] else (ul, threshold)
    return filters


def _build_live_ps_filters(
    sigma_override_path: Path = _SIGMA_OVERRIDE_FILE,
    k: float = 1.25,
) -> dict:
    """Backward-compat wrapper — frozen seed mode (no parquet access)."""
    return build_live_ps_filters(
        use_rolling=False, k=k, sigma_override_path=sigma_override_path
    )


# ── LiveConfig ────────────────────────────────────────────────────────────────

@dataclass
class LiveConfig:
    """
    Production live-trading configuration.

    Composes the validated StrategyConfig (backtest-parity layer) with
    live-only operational parameters.  All field defaults represent safe,
    conservative values for a first live run.
    """

    # ── Backtest parity ───────────────────────────────────────────────────────
    strategy_config: StrategyConfig = field(default_factory=lambda: _prod_cfg)

    # Convenience shortcuts (mirrors strategy_config fields so callers don't
    # need to reach through strategy_config.xxx for hot-path lookups).
    symbols: List[str] = field(default_factory=lambda: list(SYMS))
    prior_session_filters: Dict[str, tuple] = field(
        default_factory=lambda: _build_live_ps_filters()
    )
    instrument_gap_filters: Dict[str, float] = field(default_factory=lambda: dict(GAP_FILTER))
    day_of_week_exclusions: Dict[str, set]   = field(default_factory=lambda: dict(DOW_EXCL))
    direction_filters: Dict[str, int]        = field(default_factory=dict)

    # ── Execution parameters ──────────────────────────────────────────────────
    slippage_pct: float   = 0.001
    adv_limit_fraction: float = 0.005
    stop_order_type: str  = "market"   # "market" or "stop_limit"

    # Marketable-limit order policy (used by MarketableLimitPolicy)
    entry_slippage_bps: int    = 10     # bps above ask for entry buy limit
    entry_repeg_seconds: float = 30.0   # poll window per repeg attempt (seconds)
    entry_repeg_max_attempts: int = 3   # max repegs before accepting partial
    entry_slippage_max_bps: int  = 30   # final repeg bps cap
    exit_slippage_bps: int   = 5        # bps for exit (TP/EOD) limit orders

    # Gross-exposure and per-position caps for RiskGate
    max_gross_exposure_pct: float = 2.0  # max sum(position_value) / equity
    max_position_pct: float       = 0.50 # max single position_value / equity

    # ── Pre-flight liquidity check ────────────────────────────────────────────
    adv_lookback_days: int          = 20
    min_dollar_volume_floor: float  = 1_000_000.0   # $1M minimum ADV
    max_pct_of_adv: float           = 0.01           # max 1% of ADV per order
    min_yesterday_dv_ratio: float   = 0.30           # yesterday DV >= 30% of ADV
    allow_htb_shorts: bool          = False           # hard-to-borrow shorts

    # ── Risk management ───────────────────────────────────────────────────────
    session_kill_loss_pct: float  = 0.03
    max_concurrent_positions: int = 0   # 0 = unlimited

    # ── Account ───────────────────────────────────────────────────────────────
    paper_trading: bool    = True       # flip False only after full validation
    fractional_shares: bool = True

    # ── ORB timing ────────────────────────────────────────────────────────────
    orb_minutes: int    = 30
    eod_exit_hour: int  = 16
    eod_exit_minute: int = 0

    # ── PS filter sigma multiplier ────────────────────────────────────────────
    ps_filter_k: float = 1.25   # k=1.25 matches production backtest

    # ── Sizing (mirrors _production_run constants) ────────────────────────────
    baseline_pct: float = BASELINE_PCT
    cap_units: float    = CAP_UNITS

    # ── Alpaca credentials (resolved at runtime from env / .env) ─────────────
    alpaca_api_key: str    = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str   = "https://paper-api.alpaca.markets"

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir: Path           = field(default_factory=lambda: _DEFAULT_DATA_DIR)
    sigma_override_path: Path = field(default_factory=lambda: _SIGMA_OVERRIDE_FILE)
    db_path: Path            = field(default_factory=lambda: _DEFAULT_DB_PATH)

    # ── Instrument taxonomy (for sizing logic) ────────────────────────────────
    crypto_ul: frozenset   = field(default_factory=lambda: frozenset(CRYPTO_UL))
    singlestk_ul: frozenset = field(default_factory=lambda: frozenset(SINGLESTK_UL))
    natgas_ul: frozenset   = field(default_factory=lambda: frozenset(NATGAS_UL))
    class_a_ul: frozenset  = field(default_factory=lambda: frozenset(CLASS_A_UL))
    class_a_syms: frozenset = field(default_factory=lambda: frozenset(CLASS_A_SYMS))
    flood_losers: frozenset = field(default_factory=lambda: frozenset(FLOOD_LOSERS))

    # ── Operator exclusions (loaded from overrides.yaml) ─────────────────────
    excluded_symbols: List[str]        = field(default_factory=list)
    date_exclusions: Dict[str, List[str]] = field(default_factory=dict)
    force_long_only: List[str]         = field(default_factory=list)

    def is_class_a(self, sym: str, underlying: str) -> bool:
        return _is_class_a(sym, underlying)


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_overrides() -> dict:
    """Read optional overrides.yaml; return empty dict if file missing."""
    if not _OVERRIDES_FILE.exists():
        return {}
    with _OVERRIDES_FILE.open() as f:
        return yaml.safe_load(f) or {}


def _apply_direction_filters(cfg: LiveConfig) -> None:
    """Populate direction_filters from the production StrategyConfig."""
    cfg.direction_filters = dict(cfg.strategy_config.direction_filters)


def load_live_config(
    overrides: Optional[dict] = None,
    use_rolling: bool = True,
) -> LiveConfig:
    """
    Build and return a LiveConfig instance.

    1. Starts from all-default fields (sourced from reference layer).
    2. Applies overrides.yaml (if present).
    3. Applies any caller-supplied overrides dict.
    4. Recomputes prior_session_filters using rolling parquet sigmas when
       use_rolling=True (default). Set USE_ROLLING_SIGMAS=0 in env to force
       frozen-seed mode without changing calling code.

    Override keys must match LiveConfig field names exactly.
    """
    file_overrides   = _load_overrides()
    caller_overrides = overrides or {}
    merged           = {**file_overrides, **caller_overrides}

    cfg = LiveConfig()
    _apply_direction_filters(cfg)

    # Resolve Alpaca credentials from environment if not in overrides.
    cfg.alpaca_api_key    = merged.pop("alpaca_api_key",    os.getenv("ALPACA_API_KEY",    ""))
    cfg.alpaca_secret_key = merged.pop("alpaca_secret_key", os.getenv("ALPACA_SECRET_KEY", ""))

    if "paper_trading" in merged:
        val = merged.pop("paper_trading")
        cfg.paper_trading = bool(val)
        cfg.alpaca_base_url = (
            "https://paper-api.alpaca.markets" if cfg.paper_trading
            else "https://api.alpaca.markets"
        )

    # Operator exclusion lists from overrides.yaml
    for key in ("excluded_symbols", "date_exclusions", "force_long_only"):
        if key in merged:
            setattr(cfg, key, merged.pop(key))

    for key, val in merged.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    # Recompute prior_session_filters with rolling parquet sigmas unless
    # caller or env explicitly opts out.
    _env_flag = os.getenv("USE_ROLLING_SIGMAS", "1").strip().lower()
    _rolling  = use_rolling and (_env_flag not in ("0", "false"))
    cfg.prior_session_filters = build_live_ps_filters(
        use_rolling=_rolling,
        data_dir=cfg.data_dir,
        k=cfg.ps_filter_k,
        sigma_runtime_path=cfg.data_dir.parent / "sigma_runtime.yaml",
    )

    return cfg
