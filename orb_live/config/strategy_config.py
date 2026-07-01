"""
config/strategy_config.py — StrategyConfig dataclass (moved from orb_backtester.py).

Defines all per-run strategy parameters.  In the live system, _make_v1_strategy_config
in live_config.py overrides the v1-specific fields; the defaults here only matter for
fields NOT overridden there (ema_length, sl_method, tp2_target_multiple, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    # ── Instrument definition ─────────────────────────────────────────────────
    symbols: list = field(default_factory=lambda: [
        "BOIL", "KOLD", "UVXY", "SOXL", "SOXS", "BITX", "ETHU", "JNUG", "NUGT"
    ])
    mechanism_bin: str = "universal"

    # ── Gap filter ────────────────────────────────────────────────────────────
    gap_filter_pct: float = 0.06
    instrument_gap_filters: dict = field(default_factory=dict)

    # ── Prior session filters ─────────────────────────────────────────────────
    prior_session_filters: dict = field(default_factory=lambda: {
        "JNUG": ("GDXJ", 0.0513),
        "NUGT": ("GDX",  0.0182),
        "KOLD": None,
        "ETHU": ("ETH",  0.0329),
        "BITX": ("BTC",  0.0264),
        "UVXY": None,
        "SOXL": ("SOXX", 0.0136),
        "SOXS": ("SOXX", 0.0136),
        "FNGD": ("QQQ",  0.0102, True),
        "EDZ":  ("VWO",  0.0141, True),
        "LABD": ("IBB",  0.0102, True),
    })

    # ── Opening range ─────────────────────────────────────────────────────────
    orb_minutes: int = 30
    min_orb_bars: int = 25
    min_orb_bars_sparse: int = 10
    sparse_data_symbols: list = field(default_factory=lambda: ["BOIL", "KOLD"])

    # ── EMA filter ────────────────────────────────────────────────────────────
    ema_length: int = 30

    # ── Minimum trade quality filters ─────────────────────────────────────────
    min_profit_pct: float = 0.005
    min_increment_pct: float = 0.012
    min_entry_excess: float = 0.0

    # ── Exit structure ────────────────────────────────────────────────────────
    exit_ratio_tp1: float = 0.35
    exit_ratio_tp2: float = 0.05
    exit_ratio_tp3: float = 0.60

    # ── TP3 exit mode ────────────────────────────────────────────────────────
    tp3_mode: str = "ema_crossback"
    atr_length: int = 14
    atr_mult: float = 2.0

    # ── RTG scaling ───────────────────────────────────────────────────────────
    use_rtg_scaling: bool = False
    rtg_tp1_min: float = 0.50
    rtg_tp1_max: float = 1.00
    rtg_tp2_min: float = 1.00
    rtg_tp2_max: float = 2.00
    rtg_scale_gap_cap: float = 0.12
    rtg_min_history: int = 60
    rtg_scale_threshold: float = 0.60

    # ── RTG gap exclusion ─────────────────────────────────────────────────────
    rtg_gap_exclusion: bool = False
    rtg_gap_exclusion_threshold: float = 0.06
    rtg_gap_exclusion_symbols: tuple = ()

    # ── RTG pair routing ──────────────────────────────────────────────────────
    rtg_pair_routing: bool = False
    rtg_routing_pairs: tuple = ()
    rtg_routing_fixed: dict = field(default_factory=dict)

    tp1_target_multiple: float = 1.0
    tp2_target_multiple: float = 2.0

    # ── Stop loss ─────────────────────────────────────────────────────────────
    sl_method: str = "tight"
    sl_slippage_factor: float = 0.0
    instrument_exit_overrides: dict = field(default_factory=dict)

    # ── Position sizing ───────────────────────────────────────────────────────
    daily_risk_pct: float = 0.20
    use_risk_based_sizing: bool = False
    risk_pct_at_stop: float = 0.02
    max_position_pct: float = 0.40
    initial_equity: float = 100_000.0
    max_entry_price: float = 100_000.0

    # ── Session timing ────────────────────────────────────────────────────────
    market_open_hour:   int = 9
    market_open_minute: int = 30
    eod_exit_hour:      int = 15
    eod_exit_minute:    int = 55
    latest_entry_minute: int | None = None

    # ── Date range ───────────────────────────────────────────────────────────
    start_date: str | None = None
    end_date:   str | None = None

    # ── Day-of-week exclusions ────────────────────────────────────────────────
    day_of_week_exclusions: dict = field(default_factory=dict)

    # ── Direction filters ─────────────────────────────────────────────────────
    direction_filters: dict = field(default_factory=dict)

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "results_backtester"
    risk_free_rate_annual: float = 0.043


DEFAULT_CONFIG = StrategyConfig()
