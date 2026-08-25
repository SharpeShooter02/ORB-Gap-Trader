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

import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import orb_live  # noqa: F401 — triggers orb_live/__init__.py path setup

from orb_live.config.strategy_config import StrategyConfig, DEFAULT_CONFIG
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
        tp1_target_multiple=2.0,     # 2× ORB range, every class
        # Empty = no per-class override; every candidate takes the scalar above.
        #
        # A split {C1: 2.0, C2: 1.0, C3: 1.0} ran 2026-08-18 to 2026-08-25.
        # Measured over the current universe it cost 9.0% of P&L and dropped
        # Calmar 6.67 -> 5.13 for +0.064 Sharpe -- capping C2/C3 winners at 1×
        # lowers daily vol but deepens drawdowns, and it gave up the large
        # momentum days the strategy depends on. It was also unreproducible:
        # orb_backtester has no by-class TP1, so no single backtest run could
        # express it, and for a week every quoted figure used the scalar while
        # live used the split (P5). Reverted; R7 tracks whether a per-class
        # target deserves another look on evidence that is not this sample.
        tp1_target_multiple_by_class={},
        require_ema_confirmation=False,   # v1 boundary-fill variant
        entry_at_boundary=True,           # v1 boundary-fill variant
        instrument_exit_overrides={},
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

    # Sizing base unit. When base_notional_pct > 0 the per-unit notional is a
    # fraction of live equity (10% here) so sizing compounds with the account;
    # v1_base_notional is the fixed-dollar fallback when the pct is 0.
    # Per-position notional = base × size_mult; total gross is capped at
    # max_gross_exposure_pct × equity by the risk gate.
    base_notional_pct: float = 0.10
    v1_base_notional: float = 1_000.0
    cap_units:        float = CAP_UNITS   # 20.0 → 200% cap
    ps_filter_k:      float = K_SIGMA     # 1.00

    # ── Execution parameters ──────────────────────────────────────────────────
    slippage_pct:            float = 0.001
    adv_limit_fraction:      float = 0.005
    stop_order_type:         str   = "market"

    entry_slippage_bps:      int   = 10

    # Entry-limit room as a fraction of the ORB range. Price tends to move fast
    # on a 30-min ORB break, so the marketable-limit entry is placed at
    # boundary ± (entry_buffer_orb_frac × orb_range) rather than a fixed bps of
    # price — this auto-scales with each day's volatility and bounds the R:R
    # cost of slippage. When > 0 it supersedes entry_slippage_bps for entries.
    # 0.35 → fills unless price runs >35% of the ORB range past the boundary
    # (realized R:R ≈ 1.50 vs the 2.67 boundary-fill backtest).
    entry_buffer_orb_frac:   float = 0.35

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

    # SAFETY NET ONLY. The real exit is bar-driven, in position_manager, off
    # StrategyConfig.eod_exit_* (15:58) — NOT the eod_exit_* fields above,
    # which LivePositionManager never sees. This flatten exists to catch
    # positions the bar exit missed (bar never delivered, exit order rejected).
    #
    # It must land AFTER the 15:58 bar is delivered, ~15:59:05. At 120s it
    # landed at 15:58:00 and would have pre-empted the bar exit, quietly
    # becoming the real exit path at a different price. 30s → 15:59:30, which
    # is after delivery and still inside RTH: leveraged ETFs have thin/no
    # after-hours books and IB rejects market orders past 16:00.
    #
    # Exit timing is worth real money and the value is front-loaded. Measured
    # over the full sample: 15:59 +19.172, 15:58 +18.880, 15:57 +18.837,
    # 15:55 +18.115 — the last 5.5% worse and negative in all 7 years. Live ran
    # at 15:55 for four sessions because StrategyConfig defaulted there.
    #
    # test_eod_timing_parity.py asserts the ordering and the backtest match.
    eod_flatten_lead_secs: int = 30

    # Entry mechanism. False (default) = reactive: detect the ORB break on a
    # closed 1-min bar, then place a limit at the boundary. True = pre-placed:
    # rest a stop-limit at each candidate's ORB boundary at 10:00 so it fills
    # the instant price crosses. Resting orders are placed for ALL candidates
    # (no placement cap); buying-power overload is allocated first-come-first-
    # served at fill time (see LivePositionManager._on_resting_entry_fill).
    use_resting_entries: bool = False

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
    use_rolling: bool = True,   # deprecated/ignored — see note below
) -> LiveConfig:
    """Build and return a LiveConfig instance sourced from v1.

    Sigmas are loaded verbatim from the vendored strategy/data/sigma_master.csv
    — the same long-history (multi-year) file the backtest uses, refreshed
    out-of-band by scripts/refresh_sigma_master.py. Live NO LONGER computes a
    rolling 30-day sigma at startup (that diverged 2–3× from the backtest for
    some underlyings and silently changed PS-filter thresholds). Nothing is
    written to sigma_runtime.yaml.

    The `use_rolling` argument is retained only for call-site compatibility and
    has no effect; every path now uses the CSV seeds.
    """
    instruments, sigmas_seed, universe = _load_v1_universe()

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
