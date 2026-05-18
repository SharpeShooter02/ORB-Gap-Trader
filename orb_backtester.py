#!/usr/bin/env python3
"""
orb_backtester.py
=================
Canonical backtesting engine for the Opening Range Breakout (ORB) strategy.

Self-contained: imports nothing from any other strategy file in this project.
Serves as the foundation for mechanism-specific parameter sets going forward.

Architecture
------------
  Layer 1 — Configuration  StrategyConfig dataclass, DEFAULT_CONFIG
  Layer 2 — Data Loading   load_intraday, load_daily, _daily_from_intraday
  Layer 3 — Trade Logic    compute_gap, check_prior_session_filter,
                           compute_opening_range, check_breakout,
                           compute_entry, simulate_trade
  Layer 4 — Runner/Report  run_backtest, compute_metrics,
                           print_report, save_results

Reference results (universal configuration, 7-instrument portfolio):
  ~1,382 trades | WR ~63% | Sharpe ~1.49

These reference values came from orb_success_model.py, which operated on the
pre-processed feature_matrix.parquet rather than raw intraday data. This
backtester computes everything from the raw intraday cache and will therefore
produce different numbers for two documented reasons:

  1. UVXY data quality: the intraday cache contains un-adjusted historical
     prices for UVXY from 2011–2013 (before five cumulative reverse splits
     totalling approximately 50,000:1). These appear as prices in the hundreds
     of billions per share and are caught by the max_entry_price gate, reducing
     UVXY's trade count from the reference ~200 to ~82 qualifying sessions.

  2. Day selection: the feature_matrix was built with a curated filtering
     pipeline that excluded lower-quality gap days. This backtester includes
     all gap ≥ threshold days from the raw cache, producing more total trades
     (BOIL/KOLD in particular) but at lower average WR than the pre-filtered set.

The trade logic, exit structure, EMA computation, and position sizing are
consistent with the validated strategy. If you need to exactly replicate the
reference, load the feature_matrix as the source of qualifying days instead
of computing gaps from raw intraday. If you are developing mechanism-specific
parameter sets from scratch, this backtester's raw-data computation is the
appropriate foundation.
"""

import io
import sys
import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Windows console may not support box-drawing characters in cp1252.
# Force UTF-8 so print_report renders correctly.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ── Cache path resolution ─────────────────────────────────────────────────────
# Primary: Alpha Vantage cache written by the orb_event_study pipeline.
# Fallback: legacy local cache that produced the reference results.
_OEB_STUDY        = Path(__file__).parent / "orb_event_study"
_AV_INTRADAY_DIR  = _OEB_STUDY / "cache" / "intraday"
_AV_DAILY_DIR     = _OEB_STUDY / "cache" / "daily"
try:
    sys.path.insert(0, str(_OEB_STUDY))
    import config as _ev_config
    _LEGACY_INTRADAY = Path(_ev_config.LEGACY_CACHE_DIR)
except Exception:
    _LEGACY_INTRADAY = Path(r"C:\Users\buttn\Documents\Projects\ORB Algo\data_cache")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyConfig:
    """
    All parameters that define one variant of the ORB strategy.

    The defaults here represent the validated universal configuration that has
    produced Sharpe ~1.49 over 16 years on the 7-instrument portfolio.
    Mechanism-specific variants (delayed drift, information continuation) are
    created by instantiating this class with different values rather than by
    modifying this default.

    WHY a dataclass: passing a single config object through the call stack is
    cleaner than threading many individual parameters through every function
    signature. It also makes it trivial to log the exact configuration used for
    any backtest run, which is essential for reproducibility.
    """

    # ── Instrument definition ─────────────────────────────────────────────────
    symbols: list = field(default_factory=lambda: [
        "BOIL", "KOLD", "UVXY", "SOXL", "SOXS", "BITX", "ETHU", "JNUG", "NUGT"
    ])
    # The mechanism bin this config represents. Used in output filenames and
    # reports to distinguish runs. Values: "universal", "double_reversal",
    # "delayed_drift", "continuation".
    mechanism_bin: str = "universal"

    # ── Gap filter ────────────────────────────────────────────────────────────
    gap_filter_pct: float = 0.06
    # WHY 6%: empirically identified as the minimum gap size where the
    # fading-compression mechanism produces economically viable TP geometry.
    # Below 6%, the opening range is typically too narrow for the TP targets
    # to exceed friction costs even when the directional signal is correct.
    # This is an economic threshold, not a mechanical one — the mechanism often
    # activates at smaller gaps, but the profit potential doesn't justify the
    # risk until the gap is large enough to widen the range sufficiently.

    instrument_gap_filters: dict = field(default_factory=dict)
    # Per-instrument gap filter overrides. Maps symbol -> minimum
    # gap fraction (e.g. 0.04 for 4%). When a symbol appears here,
    # its value overrides gap_filter_pct for that instrument only.
    # Symbols not in this dict use gap_filter_pct as their threshold.
    #
    # WHY per-instrument filters: the gap bucket analysis showed
    # that the optimal gap threshold differs by mechanism and
    # underlying. DR instruments on equity underlyings (SOXL, SOXS,
    # UVXY, BITX) require 6%+ gaps for positive EV. Gold miners
    # (JNUG, NUGT) and commodity/crypto instruments (BOIL, KOLD,
    # ETHU) show positive EV from 4%+ gaps. A universal 6% filter
    # excludes profitable setups for the latter group while a
    # universal 4% filter includes unprofitable setups for the former.
    # Pre-registered 2026-05-06 before running this backtest.

    # ── Prior session filters ─────────────────────────────────────────────────
    # Maps symbol -> (underlying_symbol, direction_adjusted_threshold) or None.
    # None means no filter applied for that symbol.
    prior_session_filters: dict = field(default_factory=lambda: {
        # ── k=1.0 sigma rule: filter when underlying prior-session abs move >= 1σ.
        #    Applied to all depletion-mechanism instruments (2026-05-12).
        #    σ = std(daily abs returns) over full available history.
        #    Drift/continuation instruments (KOLD) have no depletion model → None.
        "JNUG": ("GDXJ", 0.0513),  # k=1.0; GDXJ σ=5.13%
        "NUGT": ("GDX",  0.0182),  # k=1.0; GDX σ=1.82%
        "KOLD": None,               # drift mechanism — UNG direction is the signal, not depletion
        "ETHU": ("ETH",  0.0329),  # k=1.0; ETH σ=3.29%
        "BITX": ("BTC",  0.0264),  # k=1.0; BTC σ=2.64%
        "UVXY": None,               # not a depletion instrument — VIX direction does not predict UVXY ORB quality
        "SOXL": ("SOXX", 0.0136),  # k=1.0; SOXX σ=1.36%
        "SOXS": ("SOXX", 0.0136),  # paired with SOXL
        "FNGD": ("QQQ",  0.0102, True),  # k=1.0; QQQ σ=1.02%; is_inverse=True (short 3x QQQ)
        "EDZ":  ("VWO",  0.0141, True),  # k=1.0; VWO σ=1.41%; is_inverse=True (short 3x EEM)
        "LABD": ("IBB",  0.0102, True),  # k=1.0; IBB σ=1.02%; is_inverse=True (short 3x IBB)
    })
    # WHY these filters exist: the prior session direction-adjusted return was
    # the strongest predictor found in the event study (p < 0.0000004, 55pp
    # win-rate difference at EOD). When the underlying moved strongly in the
    # same direction as today's gap during the prior session, the pool of
    # participants who want to establish positions in that direction has been
    # partially depleted. Filtering these days out improves per-trade quality
    # at the cost of lower trade frequency.
    #
    # WHY close-to-close for the underlying return: we want the full prior
    # session's directional move, not just the intraday component. An overnight
    # gap in the underlying that continued through the session is captured by
    # close-to-close but missed by open-to-close.
    #
    # WHY JNUG/NUGT/FNGU are None despite Grade B/A Stage 5 signal: Stage 5
    # detects depletion at the instrument level via MFE>1x and breakout-rate
    # metrics. However, the portfolio-level backtest (2026-05-08) showed that
    # the trade-reduction cost of filtering exceeds the per-trade quality gain
    # for these instruments. JNUG and NUGT depletion is primarily in the
    # breakout-rate channel (fewer breakouts on depleted days) rather than the
    # win-rate channel; those breakouts that do occur are still profitable.
    # The Stage 5 finding is valid but not actionable here.

    # ── Opening range ─────────────────────────────────────────────────────────
    orb_minutes: int = 30
    # WHY 30 minutes: the window calibration study tested every window from 5
    # to 60 minutes across all portfolio instruments and confirmed 30 minutes
    # as universally optimal. This is the period during which the fading-and-
    # compression mechanism plays out for double reversal instruments, and the
    # period during which low-energy directionless consolidation occurs for
    # drift instruments. Shorter windows capture too little of the range-
    # formation process; longer windows eat into the breakout profit window.
    min_orb_bars: int = 25
    # WHY 25 not 30: allows for minor data gaps (late prints, exchange pauses)
    # without discarding an otherwise clean day. A range computed from 25 bars
    # is reliable; below this the high/low are too sensitive to outlier bars.
    min_orb_bars_sparse: int = 10
    # WHY 10 for sparse instruments: BOIL and KOLD had 6-22 bars/day in their
    # early years (pre-2021, thin natural gas futures market). A 10-bar range
    # spanning the ORB window still captures genuine price extremes; below 10
    # the range high/low is determined by too few data points to be reliable.
    sparse_data_symbols: list = field(default_factory=lambda: ["BOIL", "KOLD"])
    # Instruments whose pre-2021 intraday data has fewer bars per session than
    # the standard min_orb_bars threshold, triggering use of min_orb_bars_sparse.

    # ── EMA filter ────────────────────────────────────────────────────────────
    ema_length: int = 30
    # WHY EMA rather than SMA: the EMA gives more weight to recent bars, making
    # it more responsive to the directional shift that occurs when the fading
    # period ends and the breakout begins. An SMA would lag the transition and
    # sit on the wrong side of price for longer after the actual directional
    # change.
    #
    # The breakout bar must close on the correct side of the EMA: long
    # breakouts require close > EMA, short breakouts require close < EMA.
    # The EMA is computed across the ORB window and then used as a static
    # threshold — it reflects the prevailing short-term directional bias at
    # the moment the breakout fires.

    # ── Minimum trade quality filters ─────────────────────────────────────────
    min_profit_pct: float = 0.005
    # WHY 0.3%: the TP1 target must be at least 0.3% away from entry. Below
    # this, the target is so close that normal bid-ask spread and slippage
    # consume the profit. This prevents entries on very narrow opening ranges
    # where the risk-reward geometry is mechanically unfavorable regardless of
    # which direction the breakout fires.
    min_increment_pct: float = 0.012
    0
    # Minimum additional EMA move beyond the prior TP level required to fire
    # TP2 or TP3.  Prevents multiple TPs firing in rapid succession on a single
    # short-lived EMA spike.  0.075% matches the Flux original.
    min_entry_excess: float = 0.0
    # Minimum distance the breakout bar must close above the ORB
    # high (for longs) or below the ORB low (for shorts), expressed
    # as a fraction of the ORB range.
    #
    # 0.0 = no filter (current behavior — any close beyond the
    # range boundary qualifies as a breakout).
    #
    # 0.08 = breakout bar must close at least 8% of the ORB range
    # beyond the boundary. Pre-registered hypothesis from the
    # breakout selection study (2026-05-06): winners closed
    # 0.075x range above ORB high vs 0.060x for losers
    # (p=0.018, Cohen's d=0.22). Effect size is modest — validate
    # OOS before raising above 0.0 in DEFAULT_CONFIG.
    #
    # This field is intentionally defaulted to 0.0 so DEFAULT_CONFIG
    # behavior is unchanged. Set to 0.08 or 0.10 in experimental
    # configs only.

    # ── Exit structure ────────────────────────────────────────────────────────
    exit_ratio_tp1: float = 0.35
    exit_ratio_tp2: float = 0.05
    exit_ratio_tp3: float = 0.60
    # WHY 35/05/60: full grid search (31 + 21 extended combos, 2026-05-08/09)
    # found w3 is the dominant lever — surface rises monotonically toward large
    # TP3 weight. TP2 is kept at 5% not for its exit value but as a momentum
    # qualification gate: reaching 2x ORB range filters for trending conditions
    # before arming the TP3 EMA crossback. Removing TP2 entirely (2-tier) drops
    # Sharpe by -0.027 and goes negative in 2022 walk-forward. The 60% TP3
    # stub held to EMA crossback captures the full trending move on qualified
    # days. Validated: Sharpe=1.355 vs prior 50/35/15 Sharpe=1.320 (+0.035);
    # walk-forward 5/8 years improve; 2021 and 2022 degrade slightly.

    # ── TP3 exit mode ────────────────────────────────────────────────────────
    tp3_mode: str = "ema_crossback"
    # "ema_crossback" — original: exit when close crosses back below EMA (longs).
    # "atr_trail"     — trail at peak − atr_mult × ATR(atr_length) after TP2 fires.
    atr_length: int = 14
    atr_mult: float = 2.0

    # ── RTG continuous scaling ────────────────────────────────────────────────
    # RTG continuous scaling pre-registered 2026-05-08.
    # Replaces discrete T1/T2/T3 tertile approach.
    # TP1 scales from 1.0x (low RTG) to 0.5x (high RTG).
    # TP2 scales from 2.0x (low RTG) to 1.0x (high RTG).
    # Calibrated to T3 survivor MFE p50=0.46x (TP1 min)
    # and T3 survivor MFE p75=0.99x (TP2 min).
    # Gap cap 0.12: standard targets for 12%+ gap trades.
    use_rtg_scaling: bool = False
    # If True, TP1 and TP2 distances scale continuously with
    # the trade's RTG percentile rank in the rolling historical
    # distribution. Default False — DEFAULT_CONFIG sets True.
    # Pre-registered 2026-05-08.

    rtg_tp1_min: float = 0.50
    # TP1 multiplier at RTG percentile = 1.0 (highest RTG).
    # Calibrated to T3 survivor MFE p50 = 0.46x range.

    rtg_tp1_max: float = 1.00
    # TP1 multiplier at RTG percentile = 0.0 (lowest RTG).
    # Standard 1x ORB range.

    rtg_tp2_min: float = 1.00
    # TP2 multiplier at RTG percentile = 1.0 (highest RTG).
    # Calibrated to T3 survivor MFE p75 = 0.99x range.

    rtg_tp2_max: float = 2.00
    # TP2 multiplier at RTG percentile = 0.0 (lowest RTG).
    # Standard 2x ORB range.

    rtg_scale_gap_cap: float = 0.12
    # Do not apply RTG scaling for gaps >= this threshold.
    # At 12%+ gaps RTG does not affect WR — use standard targets.

    rtg_min_history: int = 60
    # Minimum historical gap days required before RTG scaling
    # activates. Fewer days available → use standard targets.

    rtg_scale_threshold: float = 0.60
    # RTG percentile rank below which standard targets apply.
    # Above this threshold, TP targets scale continuously from
    # standard (at threshold) to fully compressed (at pct=1.0).
    # Default 0.60 — bottom 60% of RTG trades unaffected.
    #
    # RTG quartile analysis (2026-05-08): compression hurts mid-RTG
    # trades (25th-75th pct, AvgPnL +0.307-0.359% vs +0.592% baseline).
    # Only the top ~25-40% benefit from compressed targets.
    #
    # At threshold=0.60:
    #   RTG pct 0.00-0.60: standard 1x/2x (60% of trades unchanged)
    #   RTG pct 0.60-1.00: linear compression 1x→0.5x / 2x→1.0x
    #   RTG pct 0.80:      tp1=0.75x, tp2=1.50x
    #   RTG pct 1.00:      tp1=0.50x, tp2=1.00x (fully compressed)

    # ── RTG gap exclusion ─────────────────────────────────────────────────────
    # Motivated by path analysis (orb_path_by_setup.py 2026-05-08):
    # High-RTG + low-gap cell: WR=44.7%, AvgPnL=-0.184%, STOP=42.1% —
    # the only negative-EV cell in the dataset. Wide ORBs on small gaps
    # tend to chop rather than trend; the ORB range consumes the gap
    # premium and the breakout has no momentum.
    rtg_gap_exclusion: bool = False
    # If True, skip gap days where RTG pct >= rtg_scale_threshold AND
    # gap_abs < rtg_gap_exclusion_threshold. No position taken on those days.

    rtg_gap_exclusion_threshold: float = 0.06
    # Gap size (as fraction of price) below which a high-RTG day is excluded.
    # Default 0.06: excludes the 2-6% gap range for high-RTG trades.

    rtg_gap_exclusion_symbols: tuple = ()
    # Symbols for which high-RTG low-gap exclusion applies.
    # Empty tuple = apply to all symbols (universal exclusion).
    # Set to DR instruments only to avoid excluding
    # BOIL/KOLD drift trades and ETHU continuation trades
    # where RTG has different implications.

    # ── RTG pair routing ──────────────────────────────────────────────────────
    rtg_pair_routing: bool = False
    # If True, when two instruments in rtg_routing_pairs both qualify on
    # the same gap day (gap passes, prior session passes, ORB valid, RTG
    # exclusion passes), route the full combined allocation to the instrument
    # with the higher RTG percentile rank. The lower-RTG instrument is skipped
    # that day and the winner receives 2× daily_risk_pct.
    # On days where only one instrument qualifies, normal sizing applies.
    # Default False — behavior unchanged. Pre-registered 2026-05-10.

    rtg_routing_pairs: tuple = ()
    # Pairs of instruments to apply RTG routing to.
    # Each element: (symbol_a, symbol_b). Both must be in config.symbols.
    # Example: (("JNUG", "NUGT"),)

    rtg_routing_fixed: dict = field(default_factory=dict)
    # Optional per-pair fixed winner override (bypasses RTG comparison).
    # Key: "SYM_A_SYM_B" or "SYM_B_SYM_A" (both orderings searched).
    # Value: the symbol to always select on shared qualifying days.
    # Use for Configs C/D where the goal is to test a fixed instrument.

    tp1_target_multiple: float = 1.0  # TP1 at 1x ORB range from entry
    tp2_target_multiple: float = 2.0  # TP2 at 2x ORB range from entry

    # ── Stop loss ─────────────────────────────────────────────────────────────
    sl_method: str = "tight"
    # "tight": stop halfway between the far side of the range and the midpoint.
    # Long stop = (orb_mid + orb_low) / 2; short stop = (orb_mid + orb_high) / 2.
    # This places the stop at the inner quarter of the range — closer to the
    # breakdown level than the midpoint, giving the trade more room to breathe
    # while still invalidating the setup if price reclaims the inner quarter.

    sl_slippage_factor: float = 0.0
    # Fraction of the bar's overshoot past the stop that becomes extra loss.
    # fill = stop - factor * (stop - bar_low)   [long]
    # fill = stop + factor * (bar_high - stop)   [short]
    # 0.0 = exact fill at stop (current/optimistic). 0.25 = fill a quarter of
    # the way from stop to bar extreme. 1.0 = worst-case fill at bar extreme.
    # Models stop-market execution on fast 1-min bars where price has already
    # moved past the stop level before the order is processed.

    # Per-instrument exit overrides. Each entry maps a symbol to a dict with:
    #   method: "trail_A"  — trailing stop replaces all TP tiers
    #   mult:   float      — trailing distance as a multiple of the ORB range
    instrument_exit_overrides: dict = field(default_factory=lambda: {})
    # WHY no override: three-config UVXY trailing stop comparison (2026-05-04)
    # showed default tiered exits (Sharpe=0.439) outperform trail_A at 2.0x
    # (Sharpe=0.421) and 1.5x (Sharpe=-0.200). Default exits are retained.

    # ── Position sizing ───────────────────────────────────────────────────────
    daily_risk_pct: float = 0.20
    # Fraction of account equity allocated per trade (capital-allocation model).
    # Used when use_risk_based_sizing=False (default).
    use_risk_based_sizing: bool = False
    # When True, size each trade so that a stop-out costs exactly
    # risk_pct_at_stop × equity, regardless of ORB range or price level.
    # shares = floor((equity × risk_pct_at_stop) / (entry_price - stop_price))
    # Capped at max_position_pct × equity / entry_price to prevent
    # oversizing when the stop is very tight.
    risk_pct_at_stop: float = 0.02
    max_position_pct: float = 0.40
    initial_equity: float = 100_000.0
    max_entry_price: float = 100_000.0
    # Data quality gate: skip any trade where the breakout bar's close exceeds
    # this price. Leveraged ETFs should never trade above $100k/share in any
    # realistic scenario. Prices above this threshold indicate un-split-adjusted
    # historical data corruption — e.g., UVXY's intraday cache shows ~$350B/share
    # for 2011-2012 data due to multiple reverse splits not being applied
    # retroactively. Affected dates are logged but do not halt the backtest.

    # ── Session timing ────────────────────────────────────────────────────────
    market_open_hour:   int = 9
    market_open_minute: int = 30
    eod_exit_hour:      int = 15
    eod_exit_minute:    int = 55
    latest_entry_minute: int | None = None
    # If set, any breakout bar whose timestamp is >= this many minutes after
    # market open is skipped (no trade taken). E.g. 30 = only take trades
    # that fire in the first 30 minutes after the ORB (10:00–10:30).
    # WHY 3:55pm not 4:00pm: avoids the final 5 minutes of extreme illiquidity
    # and wide spreads near the literal close. The 3:55pm exit is the validated
    # TP3 timing used throughout this study.

    # ── Date range ───────────────────────────────────────────────────────────
    start_date: str | None = None
    end_date:   str | None = None
    # ISO format strings ("YYYY-MM-DD") or None for no bound.
    # Applied per-symbol: days outside the range are skipped before any
    # gap/filter/ORB logic runs. Useful for recency analysis or OOS splits.

    # ── Day-of-week exclusions ────────────────────────────────────────────────
    day_of_week_exclusions: dict = field(default_factory=dict)
    # Maps symbol -> list of weekday integers to skip (0=Monday, 6=Sunday).
    # e.g. {'ETHU': [0], 'ETHD': [0]} skips Monday trades for ETHU/ETHD.
    # Applied after gap filter, before prior session filter.
    # WHY needed: crypto 24/7 trading makes Monday gaps structurally larger
    # (65-hour weekend vs 17-hour weekday). ETHU/ETHD at 0% gap filter admit
    # all routine weekend drift; Monday exclusion removes this degraded setup
    # type. Validated 2026-05-13: +0.260 Sharpe vs baseline, WF 3/3 years.

    # ── Direction filters ─────────────────────────────────────────────────────
    direction_filters: dict = field(default_factory=dict)
    # Maps symbol -> allowed gap_direction (+1 or -1).
    # +1 = only trade when ETF gaps UP (long direction).
    # -1 = only trade when ETF gaps DOWN (short direction).
    # Symbols not in this dict trade both directions (default behaviour).
    # Applied after gap filter and day-of-week exclusion, before prior session filter.

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "results_backtester"
    risk_free_rate_annual: float = 0.043
    # Current approximate risk-free rate for Sharpe computation. Should be
    # updated periodically to reflect actual short-term Treasury yields.
    # A strategy earning 4% annually can have negative Sharpe if rf > 4%.


# ── RTG EXIT SCALING & GAP EXCLUSION ─────────────────────────────────────────
# Pre-registered and validated 2026-05-08.
#
# RTG = orb_range / gap_abs — knowable at 10:00am before entry.
# Measures how much of the gap was explored during the ORB.
# High RTG = wide ORB relative to gap = spring loaded fully.
# Low RTG  = narrow ORB relative to gap = fast momentum trade.
#
# SCALING (use_rtg_scaling=True):
#   TP targets scale continuously with RTG percentile rank.
#   Below rtg_scale_threshold (60th pct): standard 1x/2x targets.
#   Above threshold: linear compression toward 0.5x/1.0x.
#   Calibrated to T3 survivor MFE p50=0.46x (TP1) and
#   p75=0.99x (TP2) from orb_t3_survivor_analysis.py.
#   Gap cap 0.12: standard targets for 12%+ gap trades.
#
# EXCLUSION (rtg_gap_exclusion=True):
#   Skip breakout scan when RTG pct >= threshold AND
#   gap_abs < 8%. High-RTG low-gap setups have negative or
#   near-zero EV — the ORB consumed most of the gap energy,
#   leaving insufficient room for post-breakout movement.
#   Finding: 2-4% T3 trades showed WR=44.7%, AvgPnL=-0.184%.
#   Universal exclusion (empty symbols tuple) validated as
#   better than DR-only exclusion.
#
# VALIDATION:
#   Full history: Sharpe 1.194→1.299→1.355, DD -6.2%→-3.5%→-3.4%,
#                 Calmar 1.301→2.057→2.265
#   RTG+gap excl walk-forward: improves all weak years (2022,2023,2026).
#   Exit weight 35/05/60 (2026-05-09): 5/8 WF years improve;
#                 2021 and 2022 degrade slightly vs prior 50/35/15.
DEFAULT_CONFIG = StrategyConfig(
    symbols=[
        # Gold miners — depletion mechanism; GDXJ/GDX pre-market sets direction
        "JNUG", "NUGT",
        # Natural gas — drift/continuation mechanism; no depletion filter
        "KOLD",
        # Crypto — ETHU continuation at 2%, BITX double reversal at 6%
        "ETHU", "BITX",
        # Volatility — UVXY double reversal at 6%
        "UVXY",
        # Semis (3x) — DR mechanism, gap-size sensitive; 6% only
        "SOXL",
        # FANG+ inverse (3x) — DR mechanism, gap-size sensitive; 6% only
        "FNGD",
        # Emerging markets short (3x) — DR; VWO prior session filter
        "EDZ",
        # Biotech short (3x) — DR; IBB binary FDA events drive large gaps
        "LABD",
    ],
    instrument_gap_filters={
        # Gold miners — 2% vs 6% sweep (2026-05-06): both improve at 2%.
        "JNUG": 0.04,  # Sharpe 1.529 at 2% vs 0.975 at 6%
        "NUGT": 0.04,  # Sharpe 1.236 at 2% vs 0.968 at 6%
        # Natural gas — formula: 2% threshold × 2x leverage = 4%
        "KOLD": 0.04,
        # Crypto
        "ETHU": 0.02,  # Sharpe 1.871 at 2% vs 1.460 at 6%
        "BITX": 0.02,  # Sharpe 0.879 at 6% vs 0.535 at 2%
        # Volatility
        "UVXY": 0.06,  # Sharpe 0.664 at 6% vs 0.480 at 2%
        # Semis — formula: 2% threshold × 3x leverage = 6%
        "SOXL": 0.06,
        # FANG+ inverse — Sharpe 0.970 at 6% vs 0.280 at 2% (measured on FNGU; FNGD swap 2026-05-10)
        "FNGD": 0.06,
        # Emerging markets short — formula: 2% threshold × 3x leverage = 6%
        "EDZ":  0.06,
        # Biotech short — formula: 2% threshold × 3x leverage = 6%
        "LABD": 0.06,  # Sharpe 1.410 incremental (+0.055 vs base); WF 6/8 years
    },
    # RTG scaling — pre-registered 2026-05-08; threshold=0.60 validated.
    use_rtg_scaling=True,
    rtg_scale_threshold=0.60,
    rtg_tp1_min=0.50,
    rtg_tp1_max=1.00,
    rtg_tp2_min=1.00,
    rtg_tp2_max=2.00,
    rtg_scale_gap_cap=0.12,
    # RTG gap exclusion — pre-registered 2026-05-08; universal excl<8% validated.
    rtg_gap_exclusion=True,
    rtg_gap_exclusion_threshold=0.08,
    rtg_gap_exclusion_symbols=(),
)
# Production portfolio — 10 instruments with per-instrument gap filters.
# Rebuilt 2026-05-06 after overlap analysis:
#   Removed JDST (98.6% overlap with JNUG), DUST (87.7% overlap),
#   KOLD (96.0% overlap with BOIL), SBIT/SQQQ/QLD/QID (overlap/Sharpe cuts).
#   Added BOIL, SOXL, FNGU, EDZ after evaluator A-grades confirmed.
# 2026-05-08: Added LABD — biotech 3x short; Sharpe 1.355->1.410 (+0.055),
#   WF 6/8 years; MaxJ=0.19 vs FNGU (genuinely independent); no DD increase.
# 2026-05-10: Swapped FNGU → FNGD — direction momentum analysis shows FNGD wins
#   in both gap-up (short, +0.247pp) and gap-down (long, +0.247pp) vs FNGU.
# 2026-05-12: Swapped BOIL → KOLD (KOLD Sharpe 0.543 vs BOIL 0.349; same 4% gap filter).
#   Prior session filters unified to k=1.0 sigma rule across all depletion instruments;
#   KOLD left None (drift mechanism). Sharpe 1.424 → 1.290 (cost of principled consistency
#   over selective empirical filters). Forward defensibility prioritised over backtest fit.


# ── Crypto-only portfolio ─────────────────────────────────────────────────────
# All instruments trade the same 24/7 NAV-convergence gap mechanism:
#   the ETF gap reflects an overnight move already priced in the live underlying.
#   This makes the gap highly informational vs equity gaps which are still being
#   discovered at open.
#
# Gap filter rationale: crypto non-gap days are deeply negative for BTC (-0.8 to
#   -1.1 Sharpe) and near-zero for XRP. ETH is more tolerant (non-gap ~0.79
#   Sharpe) but gap days are still materially better. 2% is the validated floor
#   for ETH/BTC; XRP/SOL use 4% conservatively until swept.
#
# Prior session filters: ETH>2.87% (validated Grade A on ETHU) and BTC>2.5%
#   (validated Grade B on BITX) are applied to all instruments sharing that
#   underlying. BTC filter also applied to SOLT — BTC is the dominant SOL driver
#   (validated 2026-05-11: SOLT Sharpe 0.806→1.495, portfolio +0.043).
#   Tested universal BTC for ETH coins — rejected: ETH-specific depletion is
#   more precise (ETHU -0.134, ETHD -0.349 vs ETH filter). XRP filters unvalidated.
#
# Instruments flagged (*) have <60 gap-day trades — metrics are directionally
#   reliable but parameter sweeps should wait for more history to accumulate.
#
# Added 2026-05-11: full crypto universe scan + gap/non-gap split confirmed
#   crypto instruments have highest gap-day lift of any asset class (ΔShrp
#   1.3-3.3 vs 0.1-0.8 for equity sectors).
CRYPTO_CONFIG = StrategyConfig(
    symbols=[
        # ETH (2x) — ETHU bull / ETHD bear; 2% gap filter matches validated ETHU
        "ETHU", "ETHD",
        # BTC (2x bear) — BTCZ only; BITX and BITU (2x bull) dropped 2026-05-11.
        #   Both sat at Sharpe ~1.1, below portfolio quality floor, and shared 77
        #   days at 0.989 correlation — concentrated BTC bull cluster with no
        #   diversification value. Dropping both: Sharpe 1.801→2.449, MaxDD
        #   -3.74%→-2.60%. BTCZ retained as it fires on independent gap-down days.
        "BTCZ",  # BTCZ*: 50 gap trades
        # SOL (2x) — SOLT; 2024+ history only, 74 gap trades
        "SOLT",
        # XRP (2x) — XRPT (Volatility Shares) and UXRP (ProShares Ultra) are both
        #   2x long XRP. Both kept as independent issuers with different inception dates.
        #   Thin history (43 / 32 gap trades*); treat as exploratory.
        "XRPT", "UXRP",
    ],
    instrument_gap_filters={
        # ETH instruments — 2% gap filter + Monday exclusion (2026-05-13).
        #   Monday gaps are structurally larger for crypto (65h weekend vs 17h
        #   weekday). At 0% filter, routine weekend drift admitted on Mondays
        #   produces degraded setups. Combined fix: 2% floor + skip Mondays.
        #   Validated: Sharpe +0.260, WF 3/3 years (grid winner over 0%+excl,
        #   8%+no-excl, and all other combinations). See day_of_week_exclusions.
        "ETHU": 0.02,
        "ETHD": 0.02,
        # BTC instruments — 2% threshold (validated on BITX; others inherit)
        # BTCZ uses 4%: newer instrument, conservative until gap sweep run
        "BTCZ": 0.04,
        # SOL/XRP — 4% default; not yet swept
        "SOLT": 0.04,
        "XRPT": 0.04,
        "UXRP": 0.04,
    },
    prior_session_filters={
        # ETH prior session — Grade A validated on ETHU (long 2x).
        #   ETHD is inverse (-2x); is_inverse=True applied for mechanical consistency.
        "ETHU": ("ETH", 0.0329),        # k=1.0; ETH σ=3.29%
        "ETHD": ("ETH", 0.0329, True),  # k=1.0; ETH σ=3.29%; is_inverse=True
        # BTC prior session — BTCZ is inverse (-2x); is_inverse=True negates gap_direction
        #   so the filter fires on same-direction continuation (prior BTC down = BTCZ up)
        #   rather than reversal setups. k=1.0 threshold 0.0264 = BTC σ.
        #   Inverted+k=1.0: BTCZ Sharpe 1.130→1.677, port 2.566→2.624, Calmar 6.4→8.1 (2026-05-12).
        "BTCZ": ("BTC", 0.0264, True),
        # SOL — SOL>2.0% adopted 2026-05-11; theoretically correct (actual underlying).
        #   Both available years confirm SOL>2.0% > BTC>2.5% (1.509 vs 1.073 in 2025,
        #   3.777 vs 2.687 in 2026). Only 38 gap trades — re-evaluate when 2027 data
        #   accumulates. BTC>2.5% was the validated stand-in before SOL data existed.
        "SOLT": ("SOL", 0.02),
        # XRP — XRP>5.0% validated 2026-05-12; portfolio 2.449->2.566 (+0.117),
        #   XRPT Sharpe 3.133->4.263. Removes only 3 XRPT / 1 UXRP trades —
        #   those filtered days were large losses. Peak confirmed at 5% (4.5%=2.477,
        #   5.5%=2.557). XRP own-coin filter; BTC-based filters rejected (-0.100).
        "XRPT": ("XRP", 0.05),
        "UXRP": ("XRP", 0.05),
    },
    # RTG scaling — same thresholds as DEFAULT_CONFIG (validated universally)
    use_rtg_scaling=True,
    rtg_scale_threshold=0.60,
    rtg_tp1_min=0.50,
    rtg_tp1_max=1.00,
    rtg_tp2_min=1.00,
    rtg_tp2_max=2.00,
    rtg_scale_gap_cap=0.12,
    # RTG gap exclusion — same as DEFAULT_CONFIG; worth testing crypto-specific
    #   threshold separately since BTC/ETH frequently gap 4-8%
    rtg_gap_exclusion=True,
    rtg_gap_exclusion_threshold=0.08,
    rtg_gap_exclusion_symbols=(),
    # Monday exclusion for ETH instruments — crypto trades 65h over weekend
    # vs 17h on weekdays; routine weekend drift is not a depletion signal.
    # Combined with 2% gap filter above. Validated 2026-05-13: +0.260 Sharpe,
    # WF 3/3 (2024: +2.95, 2025: +0.52, 2026: +1.77).
    day_of_week_exclusions={"ETHU": [0], "ETHD": [0]},
    initial_equity=20_000.0,
    daily_risk_pct=1.0,
)


# ── Mechanism-specific configurations ────────────────────────────────────────
# Each config targets a single mechanism bin, overriding only the parameters
# that the mechanism's theory justifies changing. Every deviation from
# DEFAULT_CONFIG is commented with the mechanical reason it was chosen.
# These are pre-registered hypotheses — parameters are committed before any
# backtest runs, not adjusted after seeing results.

DOUBLE_REVERSAL_CONFIG = StrategyConfig(
    symbols=["SOXL", "SOXS", "UVXY", "BITX", "JNUG"],
    mechanism_bin="double_reversal",

    prior_session_filters={
        "SOXL": ("SOXX", 0.015),
        "SOXS": ("SOXX", 0.015),
        "UVXY": None,
        "BITX": ("BTC", 0.025),  # Grade B: BTC>2.5%, +15pp WR, port Sharpe +0.021 (2026-05-08)
    },

    # All other parameters (gap_filter_pct=0.06, orb_minutes=30,
    # sl_method="midpoint", exit ratios 58/38/4, tp multiples 1x/2x)
    # are inherited from StrategyConfig defaults.
    #
    # WHY inherit defaults: the window calibration study confirmed
    # 30 minutes as optimal across all portfolio instruments. The
    # sizing rules validation study confirmed the tight stop and
    # 58/38/4 exit structure on this exact instrument set. These
    # instruments ARE the double-reversal mechanism the defaults
    # were derived from — there is no validated basis for deviating
    # from them here.
    #
    # PENDING EXPERIMENT (pre-registered, do not implement yet):
    # SOXL trailing stop after TP1 at 2.0x ORB range. Pre-registered
    # during the TP exit study when n was insufficient to validate.
    # Test when n > 200 SOXL trades have accumulated.
)

DELAYED_DRIFT_CONFIG = StrategyConfig(
    symbols=["BOIL", "KOLD"],
    mechanism_bin="delayed_drift",

    prior_session_filters={
        "BOIL": None,
        "KOLD": None,
    },
    # WHY no prior session filter: BOIL and KOLD gap from overnight
    # natural gas futures moves. Unlike equity underlyings where the
    # prior session depletes directional participation, nat gas supply
    # and demand information arrives continuously from global markets
    # and is not directionally exhausted by prior session moves in
    # the same way. No prior session filter has been validated for
    # these instruments.

    # WHY eod exit at 14:55 not 15:55: the BOIL/KOLD path profile
    # showed mean directional return peaking at approximately 295
    # minutes from open (roughly 2:25pm ET). After this peak, the
    # drift fades back toward zero into the close. Exiting at 2:55pm
    # captures the drift peak before the reversal consumes gains.
    # The 15:55 default was calibrated on double-reversal instruments
    # where momentum persists to the close — it is structurally
    # wrong for a drift mechanism that peaks at 295 minutes.
    eod_exit_hour=14,
    eod_exit_minute=55,

    # All other parameters (gap_filter=6%, orb_minutes=30,
    # exit ratios 58/38/4) are inherited from defaults.
    # WHY keep 30-minute ORB: the window calibration study confirmed
    # 30 minutes is optimal even for BOIL/KOLD — the drift mechanism
    # does not change the optimal range-formation window because the
    # ORB is still capturing a genuine opening range, just with
    # different post-breakout dynamics than double reversal.
    # WHY keep 6% gap filter: unlike the continuation instruments,
    # BOIL/KOLD need a large gap to produce economically meaningful
    # drift. Below 6%, the drift signal exists but the expected move
    # is too small to clear friction costs after the ORB mechanics.
)

CONTINUATION_CONFIG = StrategyConfig(
    symbols=["ETHU"],
    mechanism_bin="continuation",

    prior_session_filters={
        "ETHU": ("ETH", 0.0287),
    },

    # WHY 4% gap filter vs default 6%: ETHU's continuation mechanism
    # is driven by persistent overnight ETH price action carrying
    # into the equity session. Unlike the double-reversal mechanism
    # where the 6% threshold ensures adequate TP geometry from the
    # compressed range, continuation moves are not range-geometry
    # dependent — they are trend-following. Even a 4% ETH gap can
    # produce a continuation move that exceeds friction costs if the
    # underlying trend is genuine. This is a testable hypothesis:
    # if sub-6% gaps on ETHU show positive EV, the filter is too
    # conservative. If they show negative EV, restore the 6% default.
    gap_filter_pct=0.04,

    # WHY shift exit ratios toward TP2: continuation moves driven
    # by persistent ETH trend regularly extend past the 1x range
    # target. The 58% allocation to TP1 was calibrated on double-
    # reversal instruments where the spring release exhausts near 1x.
    # On continuation days, price often moves through TP1 without
    # pausing. Shifting 8pp from TP1 to TP2 (50/46/4 vs 58/38/4)
    # captures more of the extended move on strong trend days.
    # This is a hypothesis: if ETHU's winners are routinely passing
    # TP2, this shift adds value. If most trades stall at TP1, the
    # shift reduces value.
    exit_ratio_tp1=0.50,
    exit_ratio_tp2=0.46,
    exit_ratio_tp3=0.04,

    # WHY tp2_target_multiple=3.0 vs default 2.0: ETH trend days
    # can carry the leveraged ETF substantially further than the
    # double-reversal mechanism's natural 2x ceiling. A 3x target
    # captures the tail of strong trend days. This is the most
    # speculative parameter change here — if TP2 is rarely reached
    # at 3x, the 46% allocation to it will just convert to EOD
    # exits at whatever price prevails, which may be above or below
    # the 2x level. Compare TP2 hit rate between the default config
    # and this config on ETHU to evaluate.
    tp2_target_multiple=3.0,

    # All other parameters (orb_minutes=30, sl_method=midpoint,
    # eod_exit=15:55) inherited from defaults.
    # WHY keep 30-minute ORB: ETHU zero-crosses at approximately
    # 40 minutes — the 30-minute window captures the uncertainty
    # period and the breakout fires at exactly the right moment
    # when the continuation is beginning to establish. This was
    # the mechanism-window alignment insight from the resolution
    # speed analysis.
)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_intraday(symbol: str) -> dict[date, pd.DataFrame]:
    """
    Load all available intraday 1-minute bars for a symbol.

    Returns a dict mapping each trading date to a DataFrame of session bars.

    WHY return a dict keyed by date: the backtester processes one day at a time,
    so O(1) lookup by date avoids repeatedly slicing a large DataFrame. Memory
    usage is equivalent either way.

    Each per-day DataFrame has:
      - DatetimeIndex in Eastern Time (timezone-naive — tz stripped for speed)
      - Columns: open, high, low, close (+ volume if present)
      - Rows covering 9:30am through 3:59pm only
        (pre-market and after-hours bars are excluded because the strategy
        only operates during regular session hours)

    Cache lookup order:
      1. Alpha Vantage cache: orb_event_study/cache/intraday/{symbol}/
      2. Legacy cache (fallback): LEGACY_CACHE_DIR/{symbol}/
    """
    sym_dir = _AV_INTRADAY_DIR / symbol
    if not sym_dir.exists() or not any(sym_dir.glob("*.parquet")):
        legacy = _LEGACY_INTRADAY / symbol
        if legacy.exists():
            sym_dir = legacy
        else:
            return {}

    files = sorted(f for f in sym_dir.glob("*.parquet") if f.stat().st_size > 0)
    if not files:
        return {}

    frames: list[pd.DataFrame] = []
    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
            if df.empty:
                continue

            # Normalise timestamp — handle column name variants and index storage.
            if "timestamp" not in df.columns:
                if df.index.name in ("timestamp", "datetime") or \
                   pd.api.types.is_datetime64_any_dtype(df.index):
                    df = df.reset_index()
                    if df.columns[0] != "timestamp":
                        df = df.rename(columns={df.columns[0]: "timestamp"})
                elif "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "timestamp"})
            if "timestamp" not in df.columns:
                continue

            ts = pd.to_datetime(df["timestamp"])
            # Strip timezone — all time comparisons use tz-naive Eastern values.
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
            df["timestamp"] = ts

            needed = ["timestamp"] + [c for c in ["open", "high", "low", "close", "volume"]
                                       if c in df.columns]
            if "close" not in needed:
                continue
            frames.append(df[needed])
        except Exception:
            continue

    if not frames:
        return {}

    all_bars = (pd.concat(frames, ignore_index=True)
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True))

    all_bars = all_bars.set_index("timestamp")

    # Keep only regular session: 9:30am through 3:59pm.
    h = all_bars.index.hour
    m = all_bars.index.minute
    session_mask = (
        ((h == 9) & (m >= 30)) |
        ((h >= 10) & (h <= 14)) |
        ((h == 15) & (m <= 59))
    )
    all_bars = all_bars[session_mask]

    result: dict[date, pd.DataFrame] = {}
    for d, grp in all_bars.groupby(all_bars.index.date):
        result[d] = grp.copy()
    return result


def load_daily(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load official daily OHLCV data for a symbol from the project's daily cache.

    Returns a DataFrame with columns [date, open, high, low, close, volume]
    sorted ascending by date, or None if the file does not exist.

    The daily cache is used for underlying instruments (SOXX, ETH) referenced
    by the prior session filter. Leveraged ETF portfolio instruments have no
    entry in the daily cache — their effective daily close is derived from
    intraday bars via _daily_from_intraday.

    WHY official close for underlyings: the prior session filter evaluates
    the underlying's full-session directional move. Official closing prices
    appear in news, brokerage apps, and market participant mental models —
    they are the reference that determines whether the prior day's trend has
    exhausted the directional participant pool.
    """
    fpath = _AV_DAILY_DIR / f"{symbol}.parquet"
    if not fpath.exists():
        return None
    df = pd.read_parquet(fpath)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _daily_from_intraday(intraday: dict[date, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a daily OHLCV summary from the intraday dict, using the last
    session bar's close as the effective daily close.

    Used for gap computation on leveraged ETFs that have no entry in the
    daily cache. The last traded price before 4pm is a reliable proxy for
    the ETF's settlement price because ETF closing auctions follow the
    primary market close closely, and the last 1-minute bar captures that.
    """
    rows = []
    for d, bars in sorted(intraday.items()):
        if bars.empty:
            continue
        rows.append({
            "date":   pd.Timestamp(d),
            "open":   float(bars["open"].iloc[0]) if "open" in bars.columns
                      else float(bars["close"].iloc[0]),
            "high":   float(bars["high"].max()) if "high" in bars.columns
                      else float(bars["close"].max()),
            "low":    float(bars["low"].min()) if "low" in bars.columns
                      else float(bars["close"].min()),
            "close":  float(bars["close"].iloc[-1]),
            "volume": float(bars["volume"].sum()) if "volume" in bars.columns
                      else np.nan,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — TRADE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def compute_gap(
    date_: date,
    daily_df: pd.DataFrame,
    today_ref_price: float,
) -> Optional[tuple[float, int]]:
    """
    Compute the overnight gap for an instrument on the given date.

    today_ref_price: the close of the 9:30 bar — the price where the first
    minute of two-sided trading settled. This is consistent with the event
    study pipeline and avoids the single odd-lot print that the raw open
    field can represent.

    Returns (gap_abs_frac, gap_direction) where:
      gap_abs_frac  — absolute gap as a fraction (e.g. 0.08 for an 8% gap)
      gap_direction — +1 for gap-up, -1 for gap-down

    Returns None if the prior close is unavailable (first day of data or
    a missing daily record).

    WHY measure gap from the prior session's last close: all market
    participants arrive at 9:30am with the prior day's closing price as their
    reference point. The overnight gap from this reference is what determines
    how far price has moved beyond the consensus anchoring point, driving the
    over-extension and subsequent fading that powers the ORB setup.
    """
    prior_rows = daily_df[daily_df["date"] < pd.Timestamp(date_)]
    if prior_rows.empty:
        return None

    prior_close = float(prior_rows.iloc[-1]["close"])
    if prior_close <= 0:
        return None

    gap_pct = (today_ref_price - prior_close) / prior_close
    return abs(gap_pct), (1 if gap_pct >= 0 else -1), prior_close


def check_prior_session_filter(
    symbol: str,
    date_: date,
    gap_direction: int,
    config: StrategyConfig,
    underlying_data: dict[str, pd.DataFrame],
) -> bool:
    """
    Returns True if the trade is ALLOWED, False if BLOCKED.

    Computes the direction-adjusted prior session return:
      dir_adj_ret = prior_close_to_close_return * effective_direction

    where effective_direction = gap_direction for long ETFs,
    or -gap_direction for inverse ETFs (3rd tuple element = True).

    Blocks the trade when dir_adj_ret exceeds the configured threshold,
    meaning the underlying moved strongly in the same direction as the ETF
    during the prior session. This signals potential participant-pool
    depletion in the gap direction.

    For inverse ETFs the underlying moves OPPOSITE to the ETF, so gap_direction
    is negated before the check — depletion fires when the underlying moved
    against the gap (= ETF moved with the gap, same-direction continuation).

    A negative dir_adj_ret (prior session moved against today's gap from the
    ETF's perspective) is actually favorable — no filtering applied.

    No filter configured (None) or missing data → trade is always allowed.
    """
    filter_cfg = config.prior_session_filters.get(symbol)
    if filter_cfg is None:
        return True

    is_inverse = len(filter_cfg) == 3 and filter_cfg[2] is True
    ul_sym, threshold = filter_cfg[0], filter_cfg[1]
    ul_df = underlying_data.get(ul_sym)
    if ul_df is None:
        return True  # Missing underlying data: allow trade (conservative default)

    # Need two consecutive prior closes to compute a return.
    prior_rows = ul_df[ul_df["date"] < pd.Timestamp(date_)].tail(2)
    if len(prior_rows) < 2:
        return True

    close_t1 = float(prior_rows.iloc[-1]["close"])  # most recent prior close
    close_t2 = float(prior_rows.iloc[-2]["close"])  # one session earlier
    if close_t2 <= 0:
        return True

    prior_session_ret = (close_t1 - close_t2) / close_t2
    effective_direction = -gap_direction if is_inverse else gap_direction
    dir_adj_ret = prior_session_ret * effective_direction
    return dir_adj_ret <= threshold


def compute_opening_range(
    bars: pd.DataFrame,
    config: StrategyConfig,
    symbol: str = "",
) -> Optional[dict]:
    """
    Compute the opening range from the first orb_minutes of the session
    (bars with timestamp hour==9, minute 30–59 for a 30-minute window).

    Returns a dict: {high, low, midpoint, size_pct, n_bars, ema}
    Returns None if fewer than min_orb_bars are available.

    WHY high/low rather than open/close of the ORB window: the range captures
    the full price excursion including intrabar wicks. A breakout above the ORB
    high means price exceeded the highest point reached by any participant during
    the range-formation period — a cleaner signal of genuine directional
    commitment than merely exceeding the last bar's close.

    EMA computation:
      multiplier = 2 / (ema_length + 1)
      ema_t = close_t * mult + ema_{t-1} * (1 - mult)
    Seeded from the close of the first ORB bar and applied across all ORB bars
    in chronological order. The value at the close of the last ORB bar is stored
    as the breakout confirmation threshold.
    """
    from datetime import datetime as _dt, timedelta as _td
    _market_open = time(config.market_open_hour, config.market_open_minute)
    _orb_end = (_dt(2000, 1, 1, _market_open.hour, _market_open.minute)
                + _td(minutes=config.orb_minutes)).time()
    orb_bars = bars[
        (bars.index.time >= _market_open) &
        (bars.index.time < _orb_end)
    ]

    effective_min = (config.min_orb_bars_sparse
                     if symbol in config.sparse_data_symbols
                     else config.min_orb_bars)
    if len(orb_bars) < effective_min:
        return None

    orb_high = float(orb_bars["high"].max()) if "high" in orb_bars.columns \
               else float(orb_bars["close"].max())
    orb_low  = float(orb_bars["low"].min()) if "low" in orb_bars.columns \
               else float(orb_bars["close"].min())

    if orb_low <= 0 or orb_high <= orb_low:
        return None

    midpoint = (orb_high + orb_low) / 2.0
    size_pct = (orb_high - orb_low) / midpoint  # range as fraction of price

    mult   = 2.0 / (config.ema_length + 1)
    closes = orb_bars["close"].values
    ema    = float(closes[0])
    for c in closes[1:]:
        ema = float(c) * mult + ema * (1.0 - mult)

    return {
        "high":     orb_high,
        "low":      orb_low,
        "midpoint": midpoint,
        "size_pct": size_pct,
        "n_bars":   len(orb_bars),
        "ema":      ema,
    }


def check_breakout(
    bar: pd.Series,
    orb: dict,
    gap_direction: int,
    config: StrategyConfig,
) -> bool:
    """
    Returns True if the given bar constitutes a valid ORB breakout in the
    gap direction.

    Long (gap_direction=+1) requires ALL of:
      bar.close > orb.high   — price exceeded the range high
      bar.close > orb.ema    — EMA confirms upward short-term bias
      orb.size_pct >= min_profit_pct    — range is wide enough to be profitable
      orb.size_pct >= min_increment_pct — market has sufficient momentum

    Short (gap_direction=-1): symmetric conditions on the downside.

    WHY EMA confirmation: a close above the ORB high when price is below the
    EMA suggests a false breakout into a downward trend — price is touching the
    range boundary before reversing. The EMA at the close of the last ORB bar
    tracks the prevailing short-term bias; a breakout that aligns with it has
    clean momentum behind it.

    WHY two quality gates: min_profit_pct catches mechanically unfavorable
    ranges where TP1 is too close to cover costs; min_increment_pct catches
    low-volatility environments where post-breakout momentum is insufficient
    to reach TP1 even if the range is otherwise adequate.
    """
    close = float(bar["close"])

    if orb["size_pct"] < config.min_profit_pct:
        return False
    if orb["size_pct"] < config.min_increment_pct:
        return False

    # Entry excess filter: breakout bar must close sufficiently
    # beyond the ORB boundary to confirm minimum momentum.
    orb_range = orb["high"] - orb["low"]
    if orb_range > 0 and config.min_entry_excess > 0:
        if gap_direction == 1:
            excess = (close - orb["high"]) / orb_range
        else:
            excess = (orb["low"] - close) / orb_range
        if excess < config.min_entry_excess:
            return False

    if gap_direction == 1:
        return close > orb["high"] and close > orb["ema"]
    else:
        return close < orb["low"] and close < orb["ema"]


def compute_entry(
    bar: pd.Series,
    orb: dict,
    gap_direction: int,
    config: StrategyConfig,
    current_equity: float,
    symbol: str = "",
    tp1_mult_override: Optional[float] = None,
    tp2_mult_override: Optional[float] = None,
    size_mult: float = 1.0,
) -> dict:
    """
    Compute all entry and exit levels for a trade that fires on this bar.

    Entry price: bar.close — represents a stop-market order placed at the
    range boundary that fills approximately at the close of the breakout bar.

    WHY bar.close as entry (not next-bar open): using next-bar open introduces
    a systematic adverse selection bias — the next bar frequently gaps further
    in the breakout direction, making the model entry better than what is
    actually achievable. Closing-bar entry better reflects live execution.

    Share sizing: floor(equity * daily_risk_pct / entry_price)
    WHY floor: fractional shares are not universally supported. Flooring is
    conservative and consistent across different price levels.

    tp1_mult_override: when set (by RTG-conditional logic in run_backtest),
    replaces config.tp1_target_multiple for this trade only.
    tp2_mult_override: when set, replaces config.tp2_target_multiple for this trade only.
    """
    entry_price = float(bar["close"])
    orb_range   = orb["high"] - orb["low"]
    tp1_mult    = tp1_mult_override if tp1_mult_override is not None \
                  else config.tp1_target_multiple
    tp2_mult    = tp2_mult_override if tp2_mult_override is not None \
                  else config.tp2_target_multiple

    if gap_direction == 1:
        tp1_price  = entry_price + orb_range * tp1_mult
        tp2_price  = entry_price + orb_range * tp2_mult
        stop_price = (orb["midpoint"] + orb["low"]) / 2.0
    else:
        tp1_price  = entry_price - orb_range * tp1_mult
        tp2_price  = entry_price - orb_range * tp2_mult
        stop_price = (orb["midpoint"] + orb["high"]) / 2.0

    if config.use_risk_based_sizing:
        stop_dist = abs(entry_price - stop_price)
        if stop_dist > 0:
            shares = math.floor(current_equity * config.risk_pct_at_stop * size_mult / stop_dist)
        else:
            shares = 0
        cap = math.floor(current_equity * config.max_position_pct * size_mult / entry_price)
        shares = min(shares, cap)
    else:
        shares = math.floor(current_equity * config.daily_risk_pct * size_mult / entry_price)
    override = config.instrument_exit_overrides.get(symbol, {})
    r_tp1 = override.get("exit_ratio_tp1", config.exit_ratio_tp1)
    r_tp2 = override.get("exit_ratio_tp2", config.exit_ratio_tp2)
    tp1_shares = math.floor(shares * r_tp1)
    tp2_shares = math.floor(shares * r_tp2)
    tp3_shares = max(0, shares - tp1_shares - tp2_shares)

    return {
        "entry_price":   entry_price,
        "stop_price":    stop_price,
        "tp1_price":     tp1_price,
        "tp2_price":     tp2_price,
        "orb_range":     orb_range,
        "shares":        shares,
        "tp1_shares":    tp1_shares,
        "tp2_shares":    tp2_shares,
        "tp3_shares":    tp3_shares,
        "direction":     gap_direction,
        "exit_override": override,
        "entry_time":    bar.name,
    }


def simulate_trade(
    bars_after_entry: pd.DataFrame,
    entry: dict,
    config: StrategyConfig,
    ema_map: dict,
    atr_map: dict = None,
) -> dict:
    """
    Flux exit mechanism: three EMA-crossback TP tiers (no fixed price targets).

    For a LONG trade, TP1 fires on a bar when ALL of:
      - EMA > entry_price (position is currently profitable)
      - abs(EMA - entry_price) / entry_price >= min_profit_pct
      - bar.close < EMA  (price crossed back below the rising EMA)
    TP1 exits tp1_shares at the EMA value.

    TP2 fires after TP1, requiring EMA to have moved at least min_increment_pct
    beyond the EMA at TP1, then close < EMA again.  Exits tp2_shares at EMA.

    TP3 fires after TP2, same crossback pattern relative to TP2 EMA level.
    Exits all remaining shares at EMA; position is fully closed.

    SHORT trades: all directional comparisons are reversed.

    Stop loss: original tight-stop level; never moved to breakeven (ADAPTIVE_SL=False).
    Checked AFTER each bar's TP checks.

    EOD exit: remaining shares exit at bar.close if eod_time reached.

    Trailing stop override (trail_A): replaces Flux tiers entirely.
    """
    direction    = entry["direction"]
    entry_price  = entry["entry_price"]
    stop_price   = entry["stop_price"]
    tp1_shares   = entry["tp1_shares"]
    tp2_shares   = entry["tp2_shares"]
    total_shares = entry["shares"]
    override     = entry.get("exit_override", {})
    orb_range    = entry["orb_range"]
    eod_time     = time(config.eod_exit_hour, config.eod_exit_minute)

    total_pnl_dollar = 0.0
    exit_time        = None
    exit_reason      = "EOD"

    # ── Trailing stop override path ───────────────────────────────────────────
    if override.get("method") == "trail_A":
        mult       = float(override.get("mult", 2.0))
        trail_dist = mult * orb_range
        peak       = entry_price
        trail_stop = (entry_price - trail_dist) if direction == 1 \
                     else (entry_price + trail_dist)
        max_fav    = 0.0

        for bar_ts, bar in bars_after_entry.iterrows():
            lo = float(bar["low"])  if "low"  in bar.index else float(bar["close"])
            hi = float(bar["high"]) if "high" in bar.index else float(bar["close"])
            cl = float(bar["close"])

            bar_fav = (hi - entry_price) * direction
            if bar_fav > max_fav:
                max_fav = bar_fav

            new_peak = max(peak, hi) if direction == 1 else min(peak, lo)
            if new_peak != peak:
                peak       = new_peak
                trail_stop = (peak - trail_dist) if direction == 1 \
                             else (peak + trail_dist)

            if bar_ts.time() >= eod_time:
                total_pnl_dollar = (cl - entry_price) * total_shares * direction
                exit_time   = bar_ts
                exit_reason = "EOD"
                break

            stop_hit = (direction == 1 and lo <= trail_stop) or \
                       (direction == -1 and hi >= trail_stop)
            if stop_hit:
                if direction == 1:
                    sl_fill = trail_stop - config.sl_slippage_factor * (trail_stop - lo)
                else:
                    sl_fill = trail_stop + config.sl_slippage_factor * (hi - trail_stop)
                total_pnl_dollar = (sl_fill - entry_price) * total_shares * direction
                exit_time   = bar_ts
                exit_reason = "STOP"
                break

        if exit_time is None and not bars_after_entry.empty:
            last = bars_after_entry.iloc[-1]
            total_pnl_dollar = (float(last["close"]) - entry_price) * total_shares * direction
            exit_time   = bars_after_entry.index[-1]
            exit_reason = "EOD"

        position_value = entry_price * total_shares
        pnl_pct      = (total_pnl_dollar / position_value) if position_value > 0 else 0.0
        hold_minutes = int(
            (exit_time - entry["entry_time"]).total_seconds() / 60
        ) if exit_time and entry["entry_time"] else 0
        return {
            "exit_reason":  exit_reason,
            "pnl_pct":      pnl_pct,
            "pnl_dollar":   total_pnl_dollar,
            "tp1_hit":      False,
            "tp2_hit":      False,
            "tp3_hit":      False,
            "tp1_price":    0.0,
            "tp2_price":    0.0,
            "tp3_price":    0.0,
            "exit_time":    exit_time,
            "hold_minutes": hold_minutes,
            "success":      1 if total_pnl_dollar > 0 else 0,
            "mfe_dollar":        round(max(max_fav, 0.0), 4),
            "mfe_pct":           round(max(max_fav, 0.0) / entry_price * 100, 4) if entry_price > 0 else 0.0,
            "post_tp2_mfe_dollar": 0.0,
            "post_tp2_mfe_pct":    0.0,
        }

    # ── Hybrid tiered exit path ───────────────────────────────────────────────
    # TP1 and TP2: fixed price targets (original static logic, preserves W/L ratio).
    # TP3: EMA crossback on the remaining stub after TP2 fires.
    tp1_price    = entry["tp1_price"]
    tp2_price    = entry["tp2_price"]
    remaining    = total_shares
    tp1_hit      = False
    tp2_hit      = False
    tp3_hit      = False
    current_stop    = stop_price  # moves to breakeven after TP1
    max_fav         = 0.0
    post_tp2_mfe    = 0.0         # max favorable move measured from tp2_price, post-TP2 only
    tp3_peak        = None        # running peak used by atr_trail mode

    # trail_after_tp1: trailing stop activates on remaining shares once TP1 fires
    use_trail_atp1  = (override.get("method") == "trail_after_tp1")
    trail_atp1_mult = float(override.get("mult", 1.0)) if use_trail_atp1 else 0.0
    trail_atp1_dist = trail_atp1_mult * orb_range
    trail_atp1_peak = None

    for bar_ts, bar in bars_after_entry.iterrows():
        lo  = float(bar["low"])  if "low"  in bar.index else float(bar["close"])
        hi  = float(bar["high"]) if "high" in bar.index else float(bar["close"])
        cl  = float(bar["close"])

        bar_fav = (hi - entry_price) * direction
        if bar_fav > max_fav:
            max_fav = bar_fav

        if tp2_hit:
            bar_post = (hi - tp2_price) * direction
            if bar_post > post_tp2_mfe:
                post_tp2_mfe = bar_post

        # EOD exit
        if bar_ts.time() >= eod_time:
            total_pnl_dollar += (cl - entry_price) * remaining * direction
            remaining   = 0
            exit_time   = bar_ts
            exit_reason = "TP2" if tp2_hit else ("TP1_ONLY" if tp1_hit else "EOD")
            break

        # TP1 — fixed price target (1× ORB range)
        if not tp1_hit and tp1_shares > 0:
            if (direction == 1 and hi >= tp1_price) or (direction == -1 and lo <= tp1_price):
                total_pnl_dollar += (tp1_price - entry_price) * tp1_shares * direction
                remaining        -= tp1_shares
                tp1_hit           = True
                if use_trail_atp1:
                    trail_atp1_peak = tp1_price
                    current_stop    = (tp1_price - trail_atp1_dist) if direction == 1 \
                                      else (tp1_price + trail_atp1_dist)
                else:
                    current_stop = entry_price  # move stop to breakeven after TP1
                if tp2_shares == 0:
                    tp2_hit = True  # no TP2 allocation — let TP3 fire directly after TP1

        # trail_after_tp1: update trailing peak and stop on remaining shares
        if tp1_hit and use_trail_atp1 and remaining > 0:
            new_peak = max(trail_atp1_peak, hi) if direction == 1 else min(trail_atp1_peak, lo)
            if new_peak != trail_atp1_peak:
                trail_atp1_peak = new_peak
                current_stop    = (new_peak - trail_atp1_dist) if direction == 1 \
                                  else (new_peak + trail_atp1_dist)

        # TP2 — fixed price target (2× ORB range); "if" not "elif" so same bar possible
        if tp1_hit and not tp2_hit and tp2_shares > 0 and not use_trail_atp1:
            if (direction == 1 and hi >= tp2_price) or (direction == -1 and lo <= tp2_price):
                total_pnl_dollar += (tp2_price - entry_price) * tp2_shares * direction
                remaining        -= tp2_shares
                tp2_hit           = True

        # TP3 — after TP2; two exit modes depending on config.tp3_mode
        if tp2_hit and not tp3_hit and remaining > 0 and not use_trail_atp1:
            if config.tp3_mode == "atr_trail" and atr_map:
                # ATR trailing stop: trail peak − atr_mult × ATR
                if tp3_peak is None:
                    tp3_peak = hi if direction == 1 else lo
                if direction == 1:
                    if hi > tp3_peak:
                        tp3_peak = hi
                    _atr_val = atr_map.get(bar_ts, float("nan"))
                    if _atr_val == _atr_val and _atr_val > 0:  # not NaN
                        _trail = tp3_peak - config.atr_mult * _atr_val
                        if lo <= _trail:
                            _exit_px = max(_trail, current_stop)
                            total_pnl_dollar += (_exit_px - entry_price) * remaining * direction
                            remaining   = 0
                            tp3_hit     = True
                            exit_time   = bar_ts
                            exit_reason = "TP3"
                            break
                else:  # direction == -1
                    if lo < tp3_peak:
                        tp3_peak = lo
                    _atr_val = atr_map.get(bar_ts, float("nan"))
                    if _atr_val == _atr_val and _atr_val > 0:
                        _trail = tp3_peak + config.atr_mult * _atr_val
                        if hi >= _trail:
                            _exit_px = min(_trail, current_stop)
                            total_pnl_dollar += (entry_price - _exit_px) * remaining
                            remaining   = 0
                            tp3_hit     = True
                            exit_time   = bar_ts
                            exit_reason = "TP3"
                            break
            else:
                # EMA crossback (original mode)
                ema = ema_map.get(bar_ts, entry_price)
                is_profitable = (ema > entry_price) if direction == 1 else (ema < entry_price)
                crossed_back  = (cl  < ema)          if direction == 1 else (cl  > ema)
                if is_profitable and crossed_back:
                    total_pnl_dollar += (cl - entry_price) * remaining * direction
                    remaining   = 0
                    tp3_hit     = True
                    exit_time   = bar_ts
                    exit_reason = "TP3"
                    break

        # Stop loss — after all TP checks; current_stop is breakeven once TP1 fires
        if remaining > 0:
            if (direction == 1 and lo <= current_stop) or (direction == -1 and hi >= current_stop):
                if direction == 1:
                    sl_fill = current_stop - config.sl_slippage_factor * (current_stop - lo)
                else:
                    sl_fill = current_stop + config.sl_slippage_factor * (hi - current_stop)
                total_pnl_dollar += (sl_fill - entry_price) * remaining * direction
                remaining   = 0
                exit_time   = bar_ts
                exit_reason = ("TRAIL" if use_trail_atp1 else "TP1_ONLY") if tp1_hit else "STOP"
                break

    # Guard: session data ended without explicit exit
    if remaining > 0 and not bars_after_entry.empty:
        last = bars_after_entry.iloc[-1]
        total_pnl_dollar += (float(last["close"]) - entry_price) * remaining * direction
        remaining   = 0
        exit_time   = bars_after_entry.index[-1]
        exit_reason = "TP2" if tp2_hit else ("TP1_ONLY" if tp1_hit else "EOD")

    position_value = entry_price * total_shares
    pnl_pct      = (total_pnl_dollar / position_value) if position_value > 0 else 0.0
    hold_minutes = int(
        (exit_time - entry["entry_time"]).total_seconds() / 60
    ) if exit_time and entry["entry_time"] else 0

    return {
        "exit_reason":  exit_reason,
        "pnl_pct":      pnl_pct,
        "pnl_dollar":   total_pnl_dollar,
        "tp1_hit":      tp1_hit,
        "tp2_hit":      tp2_hit,
        "tp3_hit":      tp3_hit,
        "tp1_price":    tp1_price if tp1_hit else 0.0,
        "tp2_price":    tp2_price if tp2_hit else 0.0,
        "tp3_price":    0.0,
        "exit_time":    exit_time,
        "hold_minutes": hold_minutes,
        "success":      1 if total_pnl_dollar > 0 else 0,
        "mfe_dollar":          round(max(max_fav, 0.0), 4),
        "mfe_pct":             round(max(max_fav, 0.0) / entry_price * 100, 4) if entry_price > 0 else 0.0,
        "post_tp2_mfe_dollar": round(max(post_tp2_mfe, 0.0), 4),
        "post_tp2_mfe_pct":    round(max(post_tp2_mfe, 0.0) / tp2_price * 100, 4) if tp2_price > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — BACKTEST RUNNER AND REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _prequalify_symbol_for_routing(
    symbol: str,
    config: StrategyConfig,
    intraday: dict,
    daily_df: pd.DataFrame,
    underlying_data: dict,
    start_date,
    end_date,
) -> "dict[date, dict]":
    """
    Dry-run qualification pass for RTG pair routing.

    Runs gap → prior-session → ORB → RTG-exclusion logic for every day in
    intraday without executing any trade simulation.  Returns a dict mapping
    each qualifying date to:
        qualifies   bool   — True after all filters pass (breakout not required)
        rtg_val     float  — raw orb_range / gap_dollar (None if unavailable)
        rtg_pct     float  — percentile rank in rolling prior history (None if
                             insufficient data — uses rtg_min_history threshold)
        gap_abs     float
        gap_direction int
    """
    effective_gap_filter = config.instrument_gap_filters.get(symbol, config.gap_filter_pct)

    # Build rolling RTG history (identical logic to run_backtest pre-scan).
    rtg_series: list = []
    _start = start_date
    _end   = end_date
    for _d in sorted(intraday.keys()):
        if _start and _d < _start: continue
        if _end   and _d > _end:   continue
        _bars = intraday[_d]
        if _bars.empty: continue
        _fo = float(_bars.iloc[0]["close"])
        _gr = compute_gap(_d, daily_df, _fo)
        if _gr is None: continue
        _ga, _, _ = _gr
        if _ga < effective_gap_filter: continue
        _orb = compute_opening_range(_bars, config, symbol=symbol)
        if _orb is None: continue
        _gd = _ga * _fo
        if _gd > 0:
            rtg_series.append((_d, (_orb["high"] - _orb["low"]) / _gd))

    _WIN = 252
    rtg_history_map: dict = {}
    for _i, (_d, _) in enumerate(rtg_series):
        if _i >= config.rtg_min_history:
            rtg_history_map[_d] = [r for _, r in rtg_series[max(0, _i - _WIN):_i]]

    result: dict = {}
    for day_date in sorted(intraday.keys()):
        if _start and day_date < _start: continue
        if _end   and day_date > _end:   continue
        day_bars = intraday[day_date]
        if day_bars.empty: continue

        first_open = float(day_bars.iloc[0]["close"])
        gap_result = compute_gap(day_date, daily_df, first_open)
        if gap_result is None: continue
        gap_abs, gap_direction, _ = gap_result
        if gap_abs < effective_gap_filter: continue

        if not check_prior_session_filter(symbol, day_date, gap_direction,
                                          config, underlying_data):
            continue

        orb = compute_opening_range(day_bars, config, symbol=symbol)
        if orb is None: continue

        rtg_val = None
        rtg_pct = None
        if gap_abs > 0:
            gap_dollar = gap_abs * first_open
            if gap_dollar > 0:
                rtg_val = (orb["high"] - orb["low"]) / gap_dollar
                _hist = rtg_history_map.get(day_date)
                if _hist is not None and gap_abs < config.rtg_scale_gap_cap:
                    try:
                        from scipy import stats as _sc
                        rtg_pct = _sc.percentileofscore(_hist, rtg_val, kind="weak") / 100.0
                    except ImportError:
                        rtg_pct = sum(1 for h in _hist if h <= rtg_val) / len(_hist)

        excluded = (
            config.rtg_gap_exclusion
            and rtg_pct is not None
            and rtg_pct >= config.rtg_scale_threshold
            and gap_abs < config.rtg_gap_exclusion_threshold
            and (len(config.rtg_gap_exclusion_symbols) == 0
                 or symbol in config.rtg_gap_exclusion_symbols)
        )

        result[day_date] = {
            "qualifies":     not excluded,
            "rtg_val":       rtg_val,
            "rtg_pct":       rtg_pct,
            "gap_abs":       gap_abs,
            "gap_direction": gap_direction,
        }

    return result


def run_backtest(
    config: StrategyConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Run the full backtest for all symbols in config.symbols.
    Returns a DataFrame of all trades taken (one row per trade).

    Per-symbol flow:
      1. Load intraday bars.
      2. Build daily summary (for gap computation via prior session close).
      3. For each trading day in chronological order:
         a. Compute the gap. Skip if below config.gap_filter_pct.
         b. Check the prior session filter. Skip if blocked.
         c. Compute the opening range. Skip if invalid.
         d. Scan post-ORB bars (from 10:00am) for a valid breakout.
         e. On first breakout: compute entry, simulate trade, record result.
            Only one trade per symbol per day.
      4. Update equity after each trade (affects position sizing going forward).

    WHY sequential processing: symbols are independent and the dataset fits
    comfortably in memory. Sequential processing is simpler and avoids shared-
    state bugs that parallel execution would introduce.
    """
    # Load underlying daily data for prior session filters.
    underlying_data: dict[str, pd.DataFrame] = {}
    for sym_cfg in config.prior_session_filters.values():
        if sym_cfg is None:
            continue
        ul_sym = sym_cfg[0]
        if ul_sym not in underlying_data:
            ul_df = load_daily(ul_sym)
            if ul_df is not None:
                underlying_data[ul_sym] = ul_df

    # ── RTG pair routing — pre-qualification pass ─────────────────────────────
    # For each routing pair where both symbols are in the portfolio, scan all
    # qualifying days (gap + prior-session + ORB) and record RTG.  Then build
    # routing_decisions[symbol][date] = "skip" | "double" | "normal".
    # "double" means this symbol is the RTG winner — gets 2× daily_risk_pct.
    # "skip"   means the paired instrument won — no trade today.
    routing_decisions: dict[str, dict] = {}

    if config.rtg_pair_routing and config.rtg_routing_pairs:
        _rstart = date.fromisoformat(config.start_date) if config.start_date else None
        _rend   = date.fromisoformat(config.end_date)   if config.end_date   else None

        _pair_qualify: dict[str, dict] = {}
        for _pair in config.rtg_routing_pairs:
            for _sym in _pair[:2]:
                if _sym not in config.symbols or _sym in _pair_qualify:
                    continue
                _intra = load_intraday(_sym)
                if not _intra:
                    continue
                _daily = _daily_from_intraday(_intra)
                _pair_qualify[_sym] = _prequalify_symbol_for_routing(
                    _sym, config, _intra, _daily, underlying_data, _rstart, _rend
                )

        for _pair in config.rtg_routing_pairs:
            _sym_a, _sym_b = _pair[0], _pair[1]
            if _sym_a not in _pair_qualify or _sym_b not in _pair_qualify:
                continue
            for _sym in (_sym_a, _sym_b):
                if _sym not in routing_decisions:
                    routing_decisions[_sym] = {}

            _fixed = (config.rtg_routing_fixed.get(f"{_sym_a}_{_sym_b}")
                      or config.rtg_routing_fixed.get(f"{_sym_b}_{_sym_a}"))

            _qa, _qb = _pair_qualify[_sym_a], _pair_qualify[_sym_b]
            for _d in sorted(set(_qa) & set(_qb)):
                if not _qa[_d]["qualifies"] or not _qb[_d]["qualifies"]:
                    continue
                if _fixed:
                    _winner = _fixed
                else:
                    def _score(q):
                        return (q["rtg_pct"] if q["rtg_pct"] is not None
                                else (q["rtg_val"] if q["rtg_val"] is not None else 0.0))
                    _winner = _sym_a if _score(_qa[_d]) >= _score(_qb[_d]) else _sym_b
                _loser = _sym_b if _winner == _sym_a else _sym_a
                routing_decisions[_winner][_d] = "double"
                routing_decisions[_loser][_d]  = "skip"

    all_trades: list[dict] = []

    for symbol in config.symbols:
        effective_gap_filter = config.instrument_gap_filters.get(
            symbol, config.gap_filter_pct
        )
        print(f"  {symbol:<6}: loading... (gap>={effective_gap_filter*100:.0f}%) ",
              end=" ", flush=True)

        intraday = load_intraday(symbol)
        if not intraday:
            print("NO DATA — skipped")
            continue

        daily_df = _daily_from_intraday(intraday)

        # Each instrument maintains its own equity pool. Position sizing in
        # one instrument does not depend on the PnL history of another.
        # WHY isolation: in live trading, each ETF is allocated a fixed
        # capital tranche. A bad week in BOIL doesn't reduce UVXY's position
        # size. Shared equity would make early-instrument outcomes affect
        # late-instrument sizing in ways that don't reflect live operation.
        equity = config.initial_equity

        n_gap_days  = 0
        n_filtered  = 0
        n_breakouts = 0
        sym_pnl: list[float] = []

        _start = date.fromisoformat(config.start_date) if config.start_date else None
        _end   = date.fromisoformat(config.end_date)   if config.end_date   else None

        # ── RTG percentile pre-computation ────────────────────────────────────
        # Build rolling history of RTG values per date so the continuous
        # percentile rank is computed from prior-only data (no lookahead).
        # Skipped when neither RTG scaling nor gap exclusion is active.
        rtg_history_map: dict[date, list] = {}
        if config.use_rtg_scaling or config.rtg_gap_exclusion:
            _rtg_series: list[tuple[date, float]] = []
            for _d in sorted(intraday.keys()):
                if _start and _d < _start: continue
                if _end   and _d > _end:   continue
                _bars = intraday[_d]
                if _bars.empty: continue
                _fo = float(_bars.iloc[0]["close"])
                _gr = compute_gap(_d, daily_df, _fo)
                if _gr is None: continue
                _ga, _, _ = _gr
                if _ga < effective_gap_filter: continue
                _orb = compute_opening_range(_bars, config, symbol=symbol)
                if _orb is None: continue
                _gd = _ga * _fo
                if _gd > 0:
                    _rtg_series.append((_d, (_orb["high"] - _orb["low"]) / _gd))

            _WIN = 252
            for _i, (_d, _) in enumerate(_rtg_series):
                if _i >= config.rtg_min_history:
                    rtg_history_map[_d] = [
                        r for _, r in _rtg_series[max(0, _i - _WIN):_i]
                    ]

        for day_date in sorted(intraday.keys()):
            if _start and day_date < _start:
                continue
            if _end   and day_date > _end:
                continue
            day_bars = intraday[day_date]
            if day_bars.empty:
                continue

            # ── Gap ─────────────────────────────────────────────────────────
            # Use the close of the 9:30 bar as the "effective open" reference
            # for gap computation. The single-print open field can represent
            # a single odd-lot transaction; the 9:30 bar's close represents
            # where price settled after the full first minute of two-sided
            # trading and is the price all participants can reasonably act on.
            # This is consistent with the methodology used throughout the
            # event study pipeline.
            first_bar = day_bars.iloc[0]
            # The 9:30 bar is always the first bar in session-filtered data.
            first_open = float(first_bar["close"])

            gap_result = compute_gap(day_date, daily_df, first_open)  # first_open = close of 9:30 bar
            if gap_result is None:
                continue
            gap_abs, gap_direction, prior_close = gap_result
            if gap_abs < effective_gap_filter:
                continue

            # ── Day-of-week exclusion ────────────────────────────────────────
            _excluded_dow = config.day_of_week_exclusions.get(symbol)
            if _excluded_dow and day_date.weekday() in _excluded_dow:
                continue

            # ── Direction filter ─────────────────────────────────────────────
            _allowed_dir = config.direction_filters.get(symbol)
            if _allowed_dir is not None and gap_direction != _allowed_dir:
                continue

            n_gap_days += 1

            # ── Prior session filter ─────────────────────────────────────────
            if not check_prior_session_filter(
                    symbol, day_date, gap_direction, config, underlying_data):
                n_filtered += 1
                continue

            # ── Opening range ────────────────────────────────────────────────
            orb = compute_opening_range(day_bars, config, symbol=symbol)
            if orb is None:
                continue

            # ── RTG continuous scaling (post-ORB, pre-breakout, no lookahead) ─
            # RTG = orb_range_dollar / gap_dollar; knowable at 10:00am.
            # Continuous percentile rank in rolling prior history drives
            # linear interpolation of TP1/TP2 multipliers.
            rtg_val       = None
            rtg_pct       = np.nan
            tp1_mult_used = config.tp1_target_multiple  # default 1.0
            tp2_mult_used = config.tp2_target_multiple  # default 2.0

            if (config.use_rtg_scaling or config.rtg_gap_exclusion) and gap_abs > 0:
                _gap_dollar = gap_abs * first_open
                if _gap_dollar > 0:
                    rtg_val = (orb["high"] - orb["low"]) / _gap_dollar
                    _hist = rtg_history_map.get(day_date)
                    if _hist is not None and gap_abs < config.rtg_scale_gap_cap:
                        try:
                            from scipy import stats as _sc
                            rtg_pct = _sc.percentileofscore(
                                _hist, rtg_val, kind='weak'
                            ) / 100.0
                        except ImportError:
                            n_below = sum(1 for h in _hist if h <= rtg_val)
                            rtg_pct = n_below / len(_hist)
                        if config.use_rtg_scaling and rtg_pct >= config.rtg_scale_threshold:
                            _thr = config.rtg_scale_threshold
                            _sp  = (rtg_pct - _thr) / (1.0 - _thr)
                            tp1_mult_used = (config.rtg_tp1_max +
                                             _sp * (config.rtg_tp1_min
                                                    - config.rtg_tp1_max))
                            tp2_mult_used = (config.rtg_tp2_max +
                                             _sp * (config.rtg_tp2_min
                                                    - config.rtg_tp2_max))
                        # below threshold: tp1_mult_used/tp2_mult_used stay at defaults

            # ── RTG gap exclusion ─────────────────────────────────────────────
            # Skip high-RTG low-gap days: wide ORB on a small gap = no momentum.
            if (config.rtg_gap_exclusion
                    and rtg_pct is not None
                    and rtg_pct >= config.rtg_scale_threshold
                    and gap_abs < config.rtg_gap_exclusion_threshold
                    and (len(config.rtg_gap_exclusion_symbols) == 0
                         or symbol in config.rtg_gap_exclusion_symbols)):
                continue

            # ── RTG pair routing ──────────────────────────────────────────────
            # Routing decisions were pre-built before the main loop.
            # "skip"   → paired instrument won RTG comparison today; no trade.
            # "double" → this instrument won; trade at 2× daily_risk_pct.
            # "normal" → not a shared qualifying day; standard sizing.
            _routing_action = routing_decisions.get(symbol, {}).get(day_date, "normal")
            if _routing_action == "skip":
                continue
            _size_mult = 2.0 if _routing_action == "double" else 1.0

            # ── Full-session midpoint EMA for Flux exit decisions ────────────
            _hi = day_bars["high"] if "high" in day_bars.columns else day_bars["close"]
            _lo = day_bars["low"]  if "low"  in day_bars.columns else day_bars["close"]
            _mids = ((_hi + _lo) / 2.0).to_numpy()
            _k    = 2.0 / (config.ema_length + 1)
            _ev   = np.empty(len(_mids))
            _ev[0] = _mids[0]
            for _i in range(1, len(_mids)):
                _ev[_i] = _mids[_i] * _k + _ev[_i - 1] * (1.0 - _k)
            ema_map = dict(zip(day_bars.index, _ev))

            # ── ATR map for atr_trail TP3 mode ───────────────────────────────
            atr_map: dict = {}
            if config.tp3_mode == "atr_trail":
                _hi_s = day_bars["high"]  if "high"  in day_bars.columns else day_bars["close"]
                _lo_s = day_bars["low"]   if "low"   in day_bars.columns else day_bars["close"]
                _cl_s = day_bars["close"]
                _prev_cl = _cl_s.shift(1).fillna(_cl_s.iloc[0])
                _tr = pd.concat([
                    _hi_s - _lo_s,
                    (_hi_s - _prev_cl).abs(),
                    (_lo_s - _prev_cl).abs(),
                ], axis=1).max(axis=1)
                _atr_s = _tr.ewm(span=config.atr_length, min_periods=1, adjust=False).mean()
                atr_map = dict(zip(day_bars.index, _atr_s.to_numpy()))

            # ── Breakout scan ────────────────────────────────────────────────
            from datetime import datetime as _dt, timedelta as _td
            _mo = time(config.market_open_hour, config.market_open_minute)
            _orb_end = (_dt(2000, 1, 1, _mo.hour, _mo.minute)
                        + _td(minutes=config.orb_minutes)).time()
            post_orb = day_bars[day_bars.index.time >= _orb_end]
            if post_orb.empty:
                continue

            for bar_ts, bar in post_orb.iterrows():
                if bar_ts.time() >= time(config.eod_exit_hour, config.eod_exit_minute):
                    break  # No breakout before EOD

                if not check_breakout(bar, orb, gap_direction, config):
                    continue

                # Latest-entry cutoff: skip breakouts that fire too late
                if config.latest_entry_minute is not None:
                    open_ts = bar_ts.replace(
                        hour=config.market_open_hour,
                        minute=config.market_open_minute,
                        second=0, microsecond=0,
                    )
                    mins_after_open = (bar_ts - open_ts).seconds // 60
                    if mins_after_open >= config.latest_entry_minute:
                        break  # No more breakouts accepted today

                # Data quality gate: reject clearly impossible prices before
                # position sizing (un-split-adjusted historical data can produce
                # prices in the billions for instruments with many reverse splits).
                entry_price_raw = float(bar["close"])
                if entry_price_raw > config.max_entry_price:
                    break  # Data corruption — skip day, continue to next date

                n_breakouts += 1
                entry = compute_entry(
                    bar, orb, gap_direction, config, equity, symbol,
                    tp1_mult_override=tp1_mult_used,
                    tp2_mult_override=tp2_mult_used,
                    size_mult=_size_mult,
                )

                if entry["shares"] < 1:
                    break  # Equity too low for a meaningful position

                bars_after = post_orb.loc[post_orb.index > bar_ts]

                result = simulate_trade(bars_after, entry, config, ema_map, atr_map)

                all_trades.append({
                    "symbol":        symbol,
                    "date":          day_date,
                    "gap_abs_pct":   round(gap_abs * 100, 3),
                    "gap_direction": gap_direction,
                    "orb_high":      round(orb["high"], 4),
                    "orb_low":       round(orb["low"], 4),
                    "orb_size_pct":  round(orb["size_pct"] * 100, 3),
                    "entry_price":   round(entry["entry_price"], 4),
                    "stop_price":    round(entry["stop_price"], 4),
                    "tp1_price":     round(result["tp1_price"], 4),
                    "tp2_price":     round(result["tp2_price"], 4),
                    "shares":        entry["shares"],
                    "direction":     gap_direction,
                    "exit_reason":   result["exit_reason"],
                    "pnl_pct":       round(result["pnl_pct"], 6),
                    "pnl_dollar":    round(result["pnl_dollar"], 2),
                    "tp1_hit":       result["tp1_hit"],
                    "tp2_hit":       result["tp2_hit"],
                    "tp3_hit":       result["tp3_hit"],
                    "hold_minutes":  result["hold_minutes"],
                    "success":       result["success"],
                    "entry_time":    entry["entry_time"],
                    "exit_time":     result["exit_time"],
                    "prior_close":   round(prior_close, 4),
                    "orb_open":      round(first_open, 4),
                    "orb_range":     round(entry["orb_range"], 4),
                    "gap_abs_dollar": round(abs(first_open - prior_close), 4),
                    "mfe_dollar":          result["mfe_dollar"],
                    "mfe_pct":             result["mfe_pct"],
                    "post_tp2_mfe_dollar": result["post_tp2_mfe_dollar"],
                    "post_tp2_mfe_pct":    result["post_tp2_mfe_pct"],
                    "rtg":       round(rtg_val, 4) if rtg_val is not None else np.nan,
                    "rtg_pct":   round(rtg_pct, 4) if not np.isnan(rtg_pct) else np.nan,
                    "tp1_mult":  tp1_mult_used,
                    "tp2_mult":  tp2_mult_used,
                    "rtg_routing": _routing_action,
                })

                equity += result["pnl_dollar"]
                sym_pnl.append(result["pnl_pct"])
                break  # One trade per symbol per day

        n_trades = len(sym_pnl)
        wr_str = f"WR={sum(1 for p in sym_pnl if p > 0)/n_trades:.1%}" \
                 if n_trades > 0 else "WR=n/a"
        print(f"{n_gap_days} gap days | {n_filtered} filtered | "
              f"{n_breakouts} breakouts | {n_trades} trades | {wr_str}")

    if not all_trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    return trades_df.sort_values(["date", "symbol"]).reset_index(drop=True)


def compute_metrics(trades: pd.DataFrame, config: StrategyConfig) -> dict:
    """
    Compute the full suite of performance metrics from a trade log.

    Trade-level: n_trades, win_rate, avg_win_pct, avg_loss_pct,
                 win_loss_ratio, expected_value, trades_per_year,
                 tp1_rate, tp2_rate.

    Risk-adjusted: sharpe_ratio, annual_return_pct (CAGR), max_drawdown_pct,
                   calmar_ratio.
      Sharpe: filled daily PnL series (non-trade days = 0), excess over
      risk-free rate, annualised by sqrt(252).
      CAGR: (final_equity / initial_equity)^(1/years) - 1.
      Max drawdown: peak-to-trough decline in equity curve.
      Calmar: CAGR / abs(max_drawdown).

      WHY Calmar alongside Sharpe: Sharpe penalises all volatility equally,
      including upside volatility. Calmar focuses on drawdown — what actually
      tests a trader's conviction and capital adequacy. A strategy with
      Sharpe 1.5 and Calmar 3.0 is more robust than Sharpe 1.8 / Calmar 0.8
      because the former never produces a drawdown severe enough to cause
      abandonment at the worst moment.

    Robustness: min_viable_win_rate (break-even WR given observed avg win/loss),
                win_rate_buffer (cushion above break-even), pct_years_profitable.

    Per-instrument: sub-dict keyed by symbol with n_trades, win_rate,
                    avg_pnl_pct, sharpe.
    """
    if trades.empty:
        return {}

    n      = len(trades)
    wins   = trades[trades["pnl_pct"] > 0]
    losses = trades[trades["pnl_pct"] <= 0]

    win_rate = len(wins) / n
    avg_win  = float(wins["pnl_pct"].mean())   if len(wins) > 0   else 0.0
    avg_loss = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    ev       = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Account-level daily returns: fixed daily_risk_pct allocation split equally
    # among same-day trades, then normalised to initial_equity.  Matches the
    # reference orb_success_model.py methodology so Sharpe values are comparable.
    rf_daily    = config.risk_free_rate_annual / 252.0
    all_bdays   = pd.bdate_range(start=trades["date"].min(), end=trades["date"].max())
    daily_alloc = config.initial_equity * config.daily_risk_pct
    t2          = trades.copy()
    t2["n_day"]      = t2.groupby("date")["pnl_pct"].transform("count")
    t2["day_dollar"] = (daily_alloc / t2["n_day"]) * t2["pnl_pct"]
    daily_dollar  = t2.groupby("date")["day_dollar"].sum()
    daily_returns = daily_dollar / config.initial_equity
    filled       = daily_returns.reindex(all_bdays, fill_value=0.0)
    is_trade_day = filled.index.isin(daily_returns.index)
    excess       = filled.copy()
    excess[is_trade_day] -= rf_daily   # idle days earn rf → zero excess; only trade days charged
    sharpe  = (excess.mean() / excess.std() * math.sqrt(252)
               if excess.std() > 0 else np.nan)
    ann_ret = float(filled.mean() * 252)

    # Equity curve for drawdown (account-level, compounded).
    eq_curve   = [config.initial_equity]
    running_eq = config.initial_equity
    for d in sorted(daily_dollar.index):
        running_eq += float(daily_dollar.loc[d])
        eq_curve.append(running_eq)
    eq_series = pd.Series(eq_curve)
    peak   = eq_series.cummax()
    dd     = (eq_series - peak) / peak
    max_dd = float(dd.min())

    span_years = max((trades["date"].max() - trades["date"].min()).days / 365.25, 1e-6)
    calmar = (ann_ret / abs(max_dd)) if max_dd < 0 else np.nan

    # Robustness.
    denom_mvwr = avg_win + abs(avg_loss)
    min_viable_wr = abs(avg_loss) / denom_mvwr if denom_mvwr > 0 else 0.5
    wr_buffer     = win_rate - min_viable_wr

    trades_per_year = n / span_years
    tp1_rate = float(trades["tp1_hit"].mean())
    tp2_rate = float(trades["tp2_hit"].mean())
    tp3_rate = float(trades["tp3_hit"].mean()) if "tp3_hit" in trades.columns else 0.0

    # Flux WR: wins = TP1 hit; losses = stop before any TP; EOD trades excluded.
    tp1_wins   = trades["tp1_hit"].sum()
    flux_loss  = ((trades["tp1_hit"] == False) & (trades["exit_reason"] == "STOP")).sum()
    eod_count  = ((trades["tp1_hit"] == False) & (trades["exit_reason"] != "STOP")).sum()
    win_rate_flux = tp1_wins / (tp1_wins + flux_loss) if (tp1_wins + flux_loss) > 0 else 0.0
    eod_rate = eod_count / n

    trades = trades.copy()
    trades["_year"] = trades["date"].dt.year
    yearly_pnl = trades.groupby("_year")["pnl_dollar"].sum()
    pct_years_profitable = float((yearly_pnl > 0).mean()) if len(yearly_pnl) >= 3 \
                           else np.nan

    # Per-instrument breakdown.
    per_instrument: dict[str, dict] = {}
    for sym, grp in trades.groupby("symbol"):
        g_bdays = pd.bdate_range(start=grp["date"].min(), end=grp["date"].max())
        g2           = grp.copy()
        g2["n_day"]  = g2.groupby("date")["pnl_pct"].transform("count")
        g2["g_dollar"] = (daily_alloc / g2["n_day"]) * g2["pnl_pct"]
        g_daily_ret = g2.groupby("date")["g_dollar"].sum() / config.initial_equity
        g_ret       = g_daily_ret.reindex(g_bdays, fill_value=0.0)
        g_trade_day = g_ret.index.isin(g_daily_ret.index)
        g_exc       = g_ret.copy()
        g_exc[g_trade_day] -= rf_daily
        g_sharpe = (g_exc.mean() / g_exc.std() * math.sqrt(252)
                    if g_exc.std() > 0 else np.nan)
        g_tp1_wins  = grp["tp1_hit"].sum()
        g_flux_loss = ((grp["tp1_hit"] == False) & (grp["exit_reason"] == "STOP")).sum()
        g_wr_flux   = g_tp1_wins / (g_tp1_wins + g_flux_loss) \
                      if (g_tp1_wins + g_flux_loss) > 0 else 0.0
        per_instrument[sym] = {
            "n_trades":    len(grp),
            "win_rate":    g_wr_flux,
            "avg_pnl_pct": float(grp["pnl_pct"].mean()),
            "sharpe":      round(float(g_sharpe), 3),
        }

    return {
        "n_trades":             n,
        "win_rate":             win_rate,
        "avg_win_pct":          avg_win,
        "avg_loss_pct":         avg_loss,
        "win_loss_ratio":       wl_ratio,
        "expected_value":       ev,
        "trades_per_year":      trades_per_year,
        "tp1_rate":             tp1_rate,
        "tp2_rate":             tp2_rate,
        "tp3_rate":             tp3_rate,
        "win_rate_flux":        win_rate_flux,
        "eod_rate":             eod_rate,
        "annual_return_pct":    ann_ret * 100,
        "sharpe_ratio":         sharpe,
        "max_drawdown_pct":     max_dd * 100,
        "calmar_ratio":         calmar,
        "final_equity":         eq_curve[-1],
        "min_viable_win_rate":  min_viable_wr,
        "win_rate_buffer":      wr_buffer,
        "pct_years_profitable": pct_years_profitable,
        "first_trade_date":     str(trades["date"].min().date()),
        "last_trade_date":      str(trades["date"].max().date()),
        "per_instrument":       per_instrument,
    }


def print_report(metrics: dict, config: StrategyConfig) -> None:
    """Print a clean, human-readable performance report to stdout."""
    if not metrics:
        print("No metrics to report.")
        return

    pi   = metrics.get("per_instrument", {})
    sep  = "═" * 52
    mech = config.mechanism_bin.upper()

    def _fmt(val, fmt=""):
        return format(val, fmt) if (val is not None and not (isinstance(val, float) and np.isnan(val))) \
               else "n/a"

    print(f"\n{sep}")
    print(f"  ORB BACKTEST REPORT — {mech}")
    print(sep)
    print(f"  Configuration")
    print(f"    Symbols:           {', '.join(config.symbols)}")
    if config.instrument_gap_filters:
        filters_str = ", ".join(
            f"{s}:{v*100:.0f}%" for s, v in sorted(config.instrument_gap_filters.items())
        )
        print(f"    Gap filter:        per-instrument ({filters_str}; fallback {config.gap_filter_pct*100:.0f}%)")
    else:
        print(f"    Gap filter:        {config.gap_filter_pct*100:.1f}%")
    print(f"    ORB window:        {config.orb_minutes} minutes")
    print(f"    Date range:        "
          f"{metrics.get('first_trade_date','?')} — {metrics.get('last_trade_date','?')}")
    print(f"\n  Trade Statistics")
    print(f"    Total trades:      {metrics['n_trades']:,}")
    print(f"    Win rate (excl EOD): {metrics['win_rate_flux']*100:.1f}%")
    print(f"    Win rate (all):    {metrics['win_rate']*100:.1f}%")
    print(f"    EOD exit rate:     {metrics['eod_rate']*100:.1f}%")
    print(f"    Avg win:           +{metrics['avg_win_pct']*100:.3f}%")
    print(f"    Avg loss:          {metrics['avg_loss_pct']*100:.3f}%")
    print(f"    Win/loss ratio:    {_fmt(metrics['win_loss_ratio'],'.2f')}")
    print(f"    EV per trade:      {metrics['expected_value']*100:+.4f}%")
    print(f"    Trades per year:   {metrics['trades_per_year']:.1f}")
    print(f"    TP1 hit rate:      {metrics['tp1_rate']*100:.1f}%")
    print(f"    TP2 hit rate:      {metrics['tp2_rate']*100:.1f}%")
    print(f"    TP3 hit rate:      {metrics['tp3_rate']*100:.1f}%")
    print(f"\n  Risk-Adjusted Returns")
    ar = metrics.get("annual_return_pct", float("nan"))
    sh = metrics.get("sharpe_ratio",      float("nan"))
    md = metrics.get("max_drawdown_pct",  float("nan"))
    ca = metrics.get("calmar_ratio",      float("nan"))
    fe = metrics.get("final_equity",      float("nan"))
    print(f"    Annual return:     {_fmt(ar,'.1f')}%")
    print(f"    Sharpe ratio:      {_fmt(sh,'.3f')}")
    print(f"    Max drawdown:      {_fmt(md,'.1f')}%")
    print(f"    Calmar ratio:      {_fmt(ca,'.2f')}")
    print(f"    Final equity:      ${_fmt(fe,',.0f')}")
    print(f"\n  Robustness")
    mvwr = metrics.get("min_viable_win_rate", float("nan"))
    buf  = metrics.get("win_rate_buffer",     float("nan"))
    pyp  = metrics.get("pct_years_profitable",float("nan"))
    print(f"    Min viable WR:     {_fmt(mvwr*100 if not np.isnan(mvwr) else float('nan'),'.1f')}%")
    print(f"    WR buffer:         {_fmt(buf*100  if not np.isnan(buf)  else float('nan'),'.1f')}pp")
    print(f"    Profitable years:  {_fmt(pyp*100  if not np.isnan(pyp)  else float('nan'),'.0f')}%")
    print(f"\n  Per-Instrument Breakdown")
    print(f"    {'Symbol':<8}  {'N':>5}  {'WR':>6}  {'AvgPnL':>8}  {'Sharpe':>7}")
    print(f"    {'─'*46}")
    for sym, d in sorted(pi.items()):
        sh_s = f"{d['sharpe']:>7.3f}" if not np.isnan(d["sharpe"]) else "    n/a"
        print(f"    {sym:<8}  {d['n_trades']:>5}  "
              f"{d['win_rate']*100:>5.1f}%  "
              f"{d['avg_pnl_pct']*100:>+7.3f}%  {sh_s}")
    print(f"{sep}\n")


def save_results(
    trades: pd.DataFrame,
    metrics: dict,
    config: StrategyConfig,
) -> None:
    """
    Save the trade log, metrics, and config to the output directory.

    Files written (all timestamped):
      trades_{mechanism_bin}_{YYYYMMDD_HHMM}.csv
      metrics_{mechanism_bin}_{YYYYMMDD_HHMM}.json
      config_{mechanism_bin}_{YYYYMMDD_HHMM}.json

    WHY save the config alongside results: the config JSON makes every backtest
    run self-documenting. Six months from now, the exact parameters that
    produced a given result are unambiguous — no need to dig through code
    history or trust memory.
    """
    from datetime import datetime as dt
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts   = dt.now().strftime("%Y%m%d_%H%M")
    mech = config.mechanism_bin

    trades.to_csv(out_dir / f"trades_{mech}_{ts}.csv", index=False)

    def _json_safe(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
        if isinstance(obj, float):
            return None if (np.isnan(obj) or np.isinf(obj)) else obj
        if isinstance(obj, dict):        return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_json_safe(i) for i in obj]
        return obj

    with open(out_dir / f"metrics_{mech}_{ts}.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    with open(out_dir / f"config_{mech}_{ts}.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"  Saved: trades_{mech}_{ts}.csv")
    print(f"  Saved: metrics_{mech}_{ts}.json")
    print(f"  Saved: config_{mech}_{ts}.json")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ORB Strategy Backtester")
    parser.add_argument(
        "--mechanism", default="universal",
        choices=["universal", "double_reversal", "delayed_drift", "continuation"],
        help="Which mechanism bin configuration to run",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Override the symbol list (space-separated)",
    )
    parser.add_argument(
        "--gap-filter", type=float, default=None,
        help="Override gap filter as a fraction (e.g. 0.04 for 4%%)",
    )
    parser.add_argument(
        "--hypothesis3", action="store_true",
        help="Run min_entry_excess comparison: A=0.0 (baseline), B=0.08, C=0.10",
    )
    parser.add_argument(
        "--hypothesis-rtg-scaling", action="store_true",
        help=(
            "RTG threshold-gated scaling: A=off, B=th≈disc(0.999), "
            "C=th=0.67, D=th=0.60(default), E=th=0.50. "
            "Pre-registered 2026-05-08."
        ),
    )
    parser.add_argument(
        "--hypothesis-rtg-gap-exclusion", action="store_true",
        help=(
            "RTG gap exclusion: A=no exclusion (DEFAULT_CONFIG), "
            "B=excl<4%, C=excl<6%, D=excl<8%. "
            "Pre-registered 2026-05-08. Motivated by high-RTG low-gap "
            "cell: WR=44.7%%, AvgPnL=-0.184%%, STOP=42.1%%."
        ),
    )
    parser.add_argument(
        "--hypothesis-rtg-gap-excl-dr", action="store_true",
        help=(
            "RTG gap exclusion: A=no excl, B=DR<6%%, C=DR<8%%, D=univ<8%%. "
            "DR instruments: SOXL, SOXS, UVXY, JNUG, NUGT, FNGU. "
            "BOIL/ETHU/BITX excluded from exclusion list."
        ),
    )
    args = parser.parse_args()

    # ── RTG threshold-gated scaling comparison ───────────────────────────────
    if args.hypothesis_rtg_scaling:
        import dataclasses

        # A: baseline — no scaling
        cfg_a = dataclasses.replace(DEFAULT_CONFIG, use_rtg_scaling=False)
        # B: threshold≈discrete (0.999 means only the very top ~0.1% get compression)
        cfg_b = dataclasses.replace(DEFAULT_CONFIG, rtg_scale_threshold=0.999)
        # C: threshold=0.67 (matches old discrete T3 upper-third boundary)
        cfg_c = dataclasses.replace(DEFAULT_CONFIG, rtg_scale_threshold=0.67)
        # D: threshold=0.60 (pre-registered — DEFAULT_CONFIG value)
        cfg_d = DEFAULT_CONFIG
        # E: threshold=0.50
        cfg_e = dataclasses.replace(DEFAULT_CONFIG, rtg_scale_threshold=0.50)

        _lbl_a = "A(off)"
        _lbl_b = "B(≈disc)"
        _lbl_c = "C(th=0.67)"
        _lbl_d = "D(th=0.60)"
        _lbl_e = "E(th=0.50)"
        _lbls_tg   = [_lbl_a, _lbl_b, _lbl_c, _lbl_d, _lbl_e]
        configs_tg = [(_lbl_a, cfg_a), (_lbl_b, cfg_b), (_lbl_c, cfg_c),
                      (_lbl_d, cfg_d), (_lbl_e, cfg_e)]

        results_tg   = {}
        trade_dfs_tg = {}
        for _label, _cfg in configs_tg:
            print(f"\nRunning config {_label} ...")
            _df = run_backtest(_cfg)
            trade_dfs_tg[_label] = _df
            results_tg[_label]   = compute_metrics(_df, _cfg) if not _df.empty else {}

        _SEP = "-" * 82
        print(f"\n{'Metric':<18}  {_lbl_a:>8}  {_lbl_b:>8}  "
              f"{_lbl_c:>10}  {_lbl_d:>10}  {_lbl_e:>10}")
        print(_SEP)

        def _tgrow(label_str, key, fmt, suffix=""):
            vals = []
            for _lbl in _lbls_tg:
                v = results_tg.get(_lbl, {}).get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    vals.append("n/a")
                else:
                    vals.append(f"{v:{fmt}}{suffix}")
            print(f"  {label_str:<16}  {vals[0]:>8}  {vals[1]:>8}  "
                  f"{vals[2]:>10}  {vals[3]:>10}  {vals[4]:>10}")

        _tgrow("Total trades",    "n_trades",          ",.0f")
        _tgrow("Portfolio WR",    "win_rate",           ".1%")
        _tgrow("Portfolio EV",    "expected_value",     "+.4%")
        _tgrow("Portfolio Sharpe","sharpe_ratio",       ".3f")
        _tgrow("Max drawdown",    "max_drawdown_pct",   ".1f", suffix="%")
        _tgrow("Calmar",          "calmar_ratio",       ".3f")
        print()

        # Summarize Sharpe deltas
        _sharpes_tg = {lbl: results_tg.get(lbl, {}).get("sharpe_ratio", np.nan)
                       for lbl in _lbls_tg}
        for _lbl in _lbls_tg[1:]:
            _d = _sharpes_tg[_lbl] - _sharpes_tg[_lbl_a]
            print(f"  {_lbl} Sharpe={_sharpes_tg[_lbl]:.3f}  delta vs A = {_d:+.3f}")
        print()

        # Best non-baseline by Sharpe
        _best_lbl = max(_lbls_tg[1:], key=lambda l: _sharpes_tg[l])
        _best_cfg = dict(configs_tg)[_best_lbl]
        print(f"  Best: {_best_lbl}  Sharpe={_sharpes_tg[_best_lbl]:.3f}  "
              f"(delta vs A = {_sharpes_tg[_best_lbl]-_sharpes_tg[_lbl_a]:+.3f})")
        print()

        # RTG quartile breakdown for best config
        df_best = trade_dfs_tg.get(_best_lbl, pd.DataFrame())
        if not df_best.empty and "rtg_pct" in df_best.columns:
            valid = df_best[df_best["rtg_pct"].notna()]
            print(f"  RTG breakdown for best config ({_best_lbl}):")
            print(f"  {'RTG pct range':<14}  {'N':>5}  {'TP1_mult':>8}  "
                  f"{'TP2_mult':>8}  {'WR':>6}  {'AvgPnL':>8}")
            print("  " + "-" * 60)
            bins   = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
            labels = ["0.00 - 0.25", "0.25 - 0.50", "0.50 - 0.75", "0.75 - 1.00"]
            for (lo, hi), lbl in zip(bins, labels):
                mask = (valid["rtg_pct"] >= lo) & (valid["rtg_pct"] < hi)
                grp  = valid[mask]
                if grp.empty:
                    continue
                wr_g  = (grp["pnl_pct"] > 0).mean() * 100
                pnl_g = grp["pnl_pct"].mean() * 100
                tp1_m = grp["tp1_mult"].mean()
                tp2_m = grp["tp2_mult"].mean()
                print(f"  {lbl:<14}  {len(grp):>5}  {tp1_m:>7.3f}x  "
                      f"{tp2_m:>7.3f}x  {wr_g:>5.1f}%  {pnl_g:>+7.3f}%")
            print()

        # Walk-forward: best vs A
        print(f"  Walk-forward 2019–2026: A(off) vs {_best_lbl}")
        print(f"  {'Year':<6}  {'A_Sharpe':>9}  {'A_WR':>7}  {'A_EV':>8}  "
              f"  {'Best_Sharpe':>11}  {'Best_WR':>7}  {'Best_EV':>8}  {'Delta':>7}")
        print("  " + "-" * 78)

        def _yr_tg(df, cfg, yr):
            sub = df[df["date"].dt.year == yr]
            return compute_metrics(sub, cfg) if not sub.empty else {}

        def _wfvtg(m, key, fmt):
            v = m.get(key)
            if v is None or not isinstance(v, (int, float)) \
                    or (isinstance(v, float) and np.isnan(v)):
                return "n/a"
            return f"{v:{fmt}}"

        df_a_wf = trade_dfs_tg[_lbl_a]
        df_x_wf = trade_dfs_tg[_best_lbl]
        for _yr in range(2019, 2027):
            _ma = _yr_tg(df_a_wf, cfg_a, _yr)
            _mx = _yr_tg(df_x_wf, _best_cfg, _yr)
            if _ma.get("n_trades", 0) == 0 and _mx.get("n_trades", 0) == 0:
                continue
            _sha = _ma.get("sharpe_ratio", np.nan)
            _shx = _mx.get("sharpe_ratio", np.nan)
            _dlt = (_shx - _sha) if not (np.isnan(_sha) or np.isnan(_shx)) else np.nan
            print(
                f"  {_yr:<6}  "
                f"{_wfvtg(_ma,'sharpe_ratio','.3f'):>9}  "
                f"{_wfvtg(_ma,'win_rate','.1%'):>7}  "
                f"{_wfvtg(_ma,'expected_value','+.4%'):>8}  "
                f"  {_wfvtg(_mx,'sharpe_ratio','.3f'):>11}  "
                f"{_wfvtg(_mx,'win_rate','.1%'):>7}  "
                f"{_wfvtg(_mx,'expected_value','+.4%'):>8}  "
                f"{f'{_dlt:+.3f}' if not np.isnan(_dlt) else 'n/a':>7}"
            )

        print()
        sys.exit(0)

    # ── RTG gap exclusion comparison ──────────────────────────────────────────
    if args.hypothesis_rtg_gap_exclusion:
        import dataclasses

        # A: DEFAULT_CONFIG — no exclusion (RTG scaling on, no gap exclusion)
        cfg_ge_a = DEFAULT_CONFIG
        # B: exclude high-RTG days with gap < 4%
        cfg_ge_b = dataclasses.replace(
            DEFAULT_CONFIG,
            rtg_gap_exclusion=True,
            rtg_gap_exclusion_threshold=0.04,
        )
        # C: exclude high-RTG days with gap < 6% (pre-registered default)
        cfg_ge_c = dataclasses.replace(
            DEFAULT_CONFIG,
            rtg_gap_exclusion=True,
            rtg_gap_exclusion_threshold=0.06,
        )
        # D: exclude high-RTG days with gap < 8%
        cfg_ge_d = dataclasses.replace(
            DEFAULT_CONFIG,
            rtg_gap_exclusion=True,
            rtg_gap_exclusion_threshold=0.08,
        )

        _ge_lbls = ["A(no-excl)", "B(excl<4%)", "C(excl<6%)", "D(excl<8%)"]
        _ge_cfgs = [
            ("A(no-excl)", cfg_ge_a),
            ("B(excl<4%)", cfg_ge_b),
            ("C(excl<6%)", cfg_ge_c),
            ("D(excl<8%)", cfg_ge_d),
        ]

        _ge_results   = {}
        _ge_trade_dfs = {}
        for _lbl, _cfg in _ge_cfgs:
            print(f"\nRunning config {_lbl} ...")
            _df = run_backtest(_cfg)
            _ge_trade_dfs[_lbl] = _df
            _ge_results[_lbl]   = compute_metrics(_df, _cfg) if not _df.empty else {}

        _ge_SEP = "-" * 88
        print(f"\n{'Metric':<22}  {'A(no-excl)':>12}  {'B(excl<4%)':>12}  "
              f"{'C(excl<6%)':>12}  {'D(excl<8%)':>12}")
        print(_ge_SEP)

        _n_a_ge = _ge_results.get("A(no-excl)", {}).get("n_trades", 0)

        def _gerow(label_str, key, fmt, suffix=""):
            vals = []
            for _l in _ge_lbls:
                v = _ge_results.get(_l, {}).get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    vals.append("n/a")
                else:
                    vals.append(f"{v:{fmt}}{suffix}")
            print(f"  {label_str:<20}  {vals[0]:>12}  {vals[1]:>12}  "
                  f"{vals[2]:>12}  {vals[3]:>12}")

        def _ge_excl_row():
            vals = []
            for _l in _ge_lbls:
                n = _ge_results.get(_l, {}).get("n_trades", 0)
                vals.append(f"{_n_a_ge - n:,}" if _l != "A(no-excl)" else "0")
            print(f"  {'Trades excluded':<20}  {vals[0]:>12}  {vals[1]:>12}  "
                  f"{vals[2]:>12}  {vals[3]:>12}")

        _gerow("Total trades",      "n_trades",        ",.0f")
        _ge_excl_row()
        _gerow("Portfolio WR",      "win_rate",         ".1%")
        _gerow("Portfolio EV",      "expected_value",   "+.4%")
        _gerow("Portfolio Sharpe",  "sharpe_ratio",     ".3f")
        _gerow("Max drawdown",      "max_drawdown_pct", ".1f", suffix="%")
        _gerow("Calmar",            "calmar_ratio",     ".3f")
        print()

        # Sharpe deltas
        _ge_sharpes = {l: _ge_results.get(l, {}).get("sharpe_ratio", np.nan)
                       for l in _ge_lbls}
        for _l in _ge_lbls[1:]:
            _d = _ge_sharpes[_l] - _ge_sharpes["A(no-excl)"]
            print(f"  {_l}  Sharpe={_ge_sharpes[_l]:.3f}  "
                  f"delta vs A = {_d:+.3f}")
        print()

        # Best non-baseline by Sharpe
        _ge_best_lbl = max(_ge_lbls[1:], key=lambda l: _ge_sharpes[l])
        _ge_best_cfg = dict(_ge_cfgs)[_ge_best_lbl]
        print(f"  Best: {_ge_best_lbl}  Sharpe={_ge_sharpes[_ge_best_lbl]:.3f}  "
              f"(delta vs A = {_ge_sharpes[_ge_best_lbl]-_ge_sharpes['A(no-excl)']:+.3f})")
        print()

        # Per-instrument breakdown for Config C (pre-registered)
        _ge_df_c = _ge_trade_dfs["C(excl<6%)"]
        _ge_df_a = _ge_trade_dfs["A(no-excl)"]
        print("  Per-instrument breakdown — C(excl<6%) vs A(no-excl):")
        print(f"  {'Symbol':<8}  {'A_n':>6}  {'C_n':>6}  {'Excl':>6}  "
              f"{'A_WR':>7}  {'C_WR':>7}  {'A_EV':>8}  {'C_EV':>8}  {'Δ_Sharpe':>9}")
        print("  " + "-" * 72)
        _ge_syms = sorted(DEFAULT_CONFIG.symbols)
        for _sym in _ge_syms:
            _sub_a = _ge_df_a[_ge_df_a["symbol"] == _sym]
            _sub_c = _ge_df_c[_ge_df_c["symbol"] == _sym]
            _na  = len(_sub_a)
            _nc  = len(_sub_c)
            _excl = _na - _nc
            _wr_a = (_sub_a["pnl_pct"] > 0).mean() * 100 if _na > 0 else float("nan")
            _wr_c = (_sub_c["pnl_pct"] > 0).mean() * 100 if _nc > 0 else float("nan")
            _ev_a = _sub_a["pnl_pct"].mean() * 100 if _na > 0 else float("nan")
            _ev_c = _sub_c["pnl_pct"].mean() * 100 if _nc > 0 else float("nan")
            _cfg_sym = dataclasses.replace(DEFAULT_CONFIG,
                                           symbols=[_sym],
                                           instrument_gap_filters={})
            _cfg_sym_c = dataclasses.replace(cfg_ge_c,
                                             symbols=[_sym],
                                             instrument_gap_filters={})
            _sh_a = compute_metrics(_sub_a, _cfg_sym).get("sharpe_ratio", float("nan")) if _na > 5 else float("nan")
            _sh_c = compute_metrics(_sub_c, _cfg_sym_c).get("sharpe_ratio", float("nan")) if _nc > 5 else float("nan")
            _dsh  = (_sh_c - _sh_a) if not (np.isnan(_sh_a) or np.isnan(_sh_c)) else float("nan")
            print(
                f"  {_sym:<8}  {_na:>6}  {_nc:>6}  {_excl:>6}  "
                f"{f'{_wr_a:.1f}%':>7}  {f'{_wr_c:.1f}%' if _nc > 0 else 'n/a':>7}  "
                f"{f'{_ev_a:+.3f}%':>8}  {f'{_ev_c:+.3f}%' if _nc > 0 else 'n/a':>8}  "
                f"{f'{_dsh:+.3f}' if not np.isnan(_dsh) else 'n/a':>9}"
            )
        print()

        # Walk-forward: best config vs A
        print(f"  Walk-forward 2019–2026: A(no-excl) vs {_ge_best_lbl}")
        print(f"  {'Year':<6}  {'A_Sharpe':>9}  {'A_WR':>7}  {'A_EV':>8}  "
              f"  {f'{_ge_best_lbl}_Sharpe':>12}  {'Best_WR':>7}  {'Best_EV':>8}  "
              f"{'Delta':>7}")
        print("  " + "-" * 80)

        def _ge_yr(df, cfg, yr):
            sub = df[df["date"].dt.year == yr]
            return compute_metrics(sub, cfg) if not sub.empty else {}

        def _ge_fmt(m, key, fmt):
            v = m.get(key)
            if v is None or not isinstance(v, (int, float)) \
                    or (isinstance(v, float) and np.isnan(v)):
                return "n/a"
            return f"{v:{fmt}}"

        for _yr in range(2019, 2027):
            _ma = _ge_yr(_ge_df_a, cfg_ge_a, _yr)
            _mb = _ge_yr(_ge_trade_dfs[_ge_best_lbl], _ge_best_cfg, _yr)
            if _ma.get("n_trades", 0) == 0 and _mb.get("n_trades", 0) == 0:
                continue
            _sha = _ma.get("sharpe_ratio", np.nan)
            _shb = _mb.get("sharpe_ratio", np.nan)
            _dlt = (_shb - _sha) if not (np.isnan(_sha) or np.isnan(_shb)) else np.nan
            print(
                f"  {_yr:<6}  "
                f"{_ge_fmt(_ma,'sharpe_ratio','.3f'):>9}  "
                f"{_ge_fmt(_ma,'win_rate','.1%'):>7}  "
                f"{_ge_fmt(_ma,'expected_value','+.4%'):>8}  "
                f"  {_ge_fmt(_mb,'sharpe_ratio','.3f'):>12}  "
                f"{_ge_fmt(_mb,'win_rate','.1%'):>7}  "
                f"{_ge_fmt(_mb,'expected_value','+.4%'):>8}  "
                f"{f'{_dlt:+.3f}' if not np.isnan(_dlt) else 'n/a':>7}"
            )
        print()
        sys.exit(0)

    # ── RTG gap exclusion — DR instruments only ───────────────────────────────
    if args.hypothesis_rtg_gap_excl_dr:
        import dataclasses

        # DR instruments: BOIL/KOLD drift and ETHU continuation are excluded
        # from the exclusion list — RTG has different implications for those
        # mechanisms. BITX is also excluded: it already has a 4% gap filter
        # and its 4-6% gap days are legitimate continuation setups.
        _DR_SYMS = ("SOXL", "SOXS", "UVXY", "JNUG", "NUGT", "FNGD")

        # A: DEFAULT_CONFIG — no exclusion (baseline)
        cfg_dr_a = DEFAULT_CONFIG
        # B: DR-only exclusion, gap < 6%
        cfg_dr_b = dataclasses.replace(
            DEFAULT_CONFIG,
            rtg_gap_exclusion=True,
            rtg_gap_exclusion_threshold=0.06,
            rtg_gap_exclusion_symbols=_DR_SYMS,
        )
        # C: DR-only exclusion, gap < 8%
        cfg_dr_c = dataclasses.replace(
            DEFAULT_CONFIG,
            rtg_gap_exclusion=True,
            rtg_gap_exclusion_threshold=0.08,
            rtg_gap_exclusion_symbols=_DR_SYMS,
        )
        # D: Universal exclusion, gap < 8% (previous best from full test)
        cfg_dr_d = dataclasses.replace(
            DEFAULT_CONFIG,
            rtg_gap_exclusion=True,
            rtg_gap_exclusion_threshold=0.08,
            rtg_gap_exclusion_symbols=(),
        )

        _dr_lbls = ["A(none)", "B(DR<6%)", "C(DR<8%)", "D(univ<8%)"]
        _dr_cfgs = [
            ("A(none)",     cfg_dr_a),
            ("B(DR<6%)",    cfg_dr_b),
            ("C(DR<8%)",    cfg_dr_c),
            ("D(univ<8%)",  cfg_dr_d),
        ]

        _dr_results   = {}
        _dr_trade_dfs = {}
        for _lbl, _cfg in _dr_cfgs:
            print(f"\nRunning config {_lbl} ...")
            _df = run_backtest(_cfg)
            _dr_trade_dfs[_lbl] = _df
            _dr_results[_lbl]   = compute_metrics(_df, _cfg) if not _df.empty else {}

        _dr_SEP = "─" * 67
        print(f"\n{'Metric':<20}  {'A(none)':>10}  {'B(DR<6%)':>10}  "
              f"{'C(DR<8%)':>10}  {'D(univ<8%)':>11}")
        print(_dr_SEP)

        _n_a_dr = _dr_results.get("A(none)", {}).get("n_trades", 0)

        def _drrow(label_str, key, fmt, suffix=""):
            vals = []
            for _l in _dr_lbls:
                v = _dr_results.get(_l, {}).get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    vals.append("n/a")
                else:
                    vals.append(f"{v:{fmt}}{suffix}")
            print(f"  {label_str:<18}  {vals[0]:>10}  {vals[1]:>10}  "
                  f"{vals[2]:>10}  {vals[3]:>11}")

        def _dr_excl_row_total():
            vals = []
            for _l in _dr_lbls:
                n = _dr_results.get(_l, {}).get("n_trades", 0)
                vals.append(f"{_n_a_dr - n:,}" if _l != "A(none)" else "0")
            print(f"  {'Trades excluded':<18}  {vals[0]:>10}  {vals[1]:>10}  "
                  f"{vals[2]:>10}  {vals[3]:>11}")

        def _dr_excl_sym_row(label_str, sym):
            vals = []
            for _lbl_i, (_l, _cfg_i) in enumerate(zip(_dr_lbls, [cfg_dr_a, cfg_dr_b, cfg_dr_c, cfg_dr_d])):
                _df_a_sym = _dr_trade_dfs["A(none)"][_dr_trade_dfs["A(none)"]["symbol"] == sym]
                _df_x_sym = _dr_trade_dfs[_l][_dr_trade_dfs[_l]["symbol"] == sym]
                _na_s = len(_df_a_sym)
                _nx_s = len(_df_x_sym)
                vals.append(f"{_na_s - _nx_s:,}" if _l != "A(none)" else "0")
            print(f"  {label_str:<18}  {vals[0]:>10}  {vals[1]:>10}  "
                  f"{vals[2]:>10}  {vals[3]:>11}")

        _drrow("Total trades",      "n_trades",        ",.0f")
        _dr_excl_row_total()
        _dr_excl_sym_row("BOIL excluded",  "BOIL")
        _dr_excl_sym_row("JNUG excluded",  "JNUG")
        _drrow("Portfolio Sharpe",  "sharpe_ratio",     ".3f")
        _drrow("Max drawdown",      "max_drawdown_pct", ".1f", suffix="%")
        _drrow("Calmar",            "calmar_ratio",     ".3f")
        _drrow("EV per trade",      "expected_value",   "+.4%")
        print()

        # Sharpe deltas
        _dr_sharpes = {l: _dr_results.get(l, {}).get("sharpe_ratio", np.nan)
                       for l in _dr_lbls}
        for _l in _dr_lbls[1:]:
            _d = _dr_sharpes[_l] - _dr_sharpes["A(none)"]
            print(f"  {_l}  Sharpe={_dr_sharpes[_l]:.3f}  "
                  f"delta vs A = {_d:+.3f}")
        print()

        # Best non-baseline by Sharpe
        _dr_best_lbl = max(_dr_lbls[1:], key=lambda l: _dr_sharpes[l])
        _dr_best_cfg = dict(_dr_cfgs)[_dr_best_lbl]
        print(f"  Best: {_dr_best_lbl}  Sharpe={_dr_sharpes[_dr_best_lbl]:.3f}  "
              f"(delta vs A = {_dr_sharpes[_dr_best_lbl]-_dr_sharpes['A(none)']:+.3f})")
        print()

        # Per-instrument breakdown for Config B
        _dr_df_b = _dr_trade_dfs["B(DR<6%)"]
        _dr_df_a = _dr_trade_dfs["A(none)"]
        print("  Per-instrument breakdown — B(DR<6%) vs A(none):")
        print(f"  {'Symbol':<8}  {'A_n':>6}  {'B_n':>6}  {'Excl':>6}  "
              f"{'A_Sharpe':>9}  {'B_Sharpe':>9}  {'Delta':>7}")
        print("  " + "-" * 60)
        for _sym in sorted(DEFAULT_CONFIG.symbols):
            _sub_a = _dr_df_a[_dr_df_a["symbol"] == _sym]
            _sub_b = _dr_df_b[_dr_df_b["symbol"] == _sym]
            _na  = len(_sub_a)
            _nb  = len(_sub_b)
            _excl = _na - _nb
            _cfg_sym   = dataclasses.replace(DEFAULT_CONFIG, symbols=[_sym], instrument_gap_filters={})
            _cfg_sym_b = dataclasses.replace(cfg_dr_b,      symbols=[_sym], instrument_gap_filters={})
            _sh_a = compute_metrics(_sub_a, _cfg_sym).get("sharpe_ratio", float("nan")) if _na > 5 else float("nan")
            _sh_b = compute_metrics(_sub_b, _cfg_sym_b).get("sharpe_ratio", float("nan")) if _nb > 5 else float("nan")
            _dsh  = (_sh_b - _sh_a) if not (np.isnan(_sh_a) or np.isnan(_sh_b)) else float("nan")
            _excl_note = ""
            if _sym in ("BOIL", "ETHU", "BITX"):
                _excl_note = " ← should be 0"
            print(
                f"  {_sym:<8}  {_na:>6}  {_nb:>6}  {_excl:>6}  "
                f"{f'{_sh_a:.3f}' if not np.isnan(_sh_a) else 'n/a':>9}  "
                f"{f'{_sh_b:.3f}' if not np.isnan(_sh_b) else 'n/a':>9}  "
                f"{f'{_dsh:+.3f}' if not np.isnan(_dsh) else 'n/a':>7}"
                f"{_excl_note}"
            )
        print()

        # Walk-forward: best config vs A
        print(f"  Walk-forward 2019–2026: A(none) vs {_dr_best_lbl}")
        print(f"  {'Year':<6}  {'A_Sharpe':>9}  {'A_WR':>7}  {'A_EV':>8}  "
              f"  {'Best_Sharpe':>11}  {'Best_WR':>7}  {'Best_EV':>8}  {'Delta':>7}")
        print("  " + "-" * 78)

        def _dr_yr(df, cfg, yr):
            sub = df[df["date"].dt.year == yr]
            return compute_metrics(sub, cfg) if not sub.empty else {}

        def _dr_fmt(m, key, fmt):
            v = m.get(key)
            if v is None or not isinstance(v, (int, float)) \
                    or (isinstance(v, float) and np.isnan(v)):
                return "n/a"
            return f"{v:{fmt}}"

        for _yr in range(2019, 2027):
            _ma = _dr_yr(_dr_df_a,                    cfg_dr_a,     _yr)
            _mb = _dr_yr(_dr_trade_dfs[_dr_best_lbl], _dr_best_cfg, _yr)
            if _ma.get("n_trades", 0) == 0 and _mb.get("n_trades", 0) == 0:
                continue
            _sha = _ma.get("sharpe_ratio", np.nan)
            _shb = _mb.get("sharpe_ratio", np.nan)
            _dlt = (_shb - _sha) if not (np.isnan(_sha) or np.isnan(_shb)) else np.nan
            print(
                f"  {_yr:<6}  "
                f"{_dr_fmt(_ma,'sharpe_ratio','.3f'):>9}  "
                f"{_dr_fmt(_ma,'win_rate','.1%'):>7}  "
                f"{_dr_fmt(_ma,'expected_value','+.4%'):>8}  "
                f"  {_dr_fmt(_mb,'sharpe_ratio','.3f'):>11}  "
                f"{_dr_fmt(_mb,'win_rate','.1%'):>7}  "
                f"{_dr_fmt(_mb,'expected_value','+.4%'):>8}  "
                f"{f'{_dlt:+.3f}' if not np.isnan(_dlt) else 'n/a':>7}"
            )
        print()
        sys.exit(0)

    # ── Hypothesis 3: min_entry_excess comparison ─────────────────────────────
    if args.hypothesis3:
        import dataclasses

        cfg_a = DEFAULT_CONFIG
        cfg_b = dataclasses.replace(DEFAULT_CONFIG, min_entry_excess=0.08)
        cfg_c = dataclasses.replace(DEFAULT_CONFIG, min_entry_excess=0.10)
        configs = [("A (0.00)", cfg_a), ("B (0.08)", cfg_b), ("C (0.10)", cfg_c)]

        results = {}
        trade_dfs = {}
        for label, cfg in configs:
            print(f"\nRunning config {label} ...")
            df = run_backtest(cfg)
            trade_dfs[label] = df
            results[label] = compute_metrics(df, cfg) if not df.empty else {}

        # Reference trade count from config A
        n_a = results.get("A (0.00)", {}).get("n_trades", 0)

        SEP = "-" * 56
        print(f"\n{'Metric':<28}  {'A (0.00)':>10}  {'B (0.08)':>10}  {'C (0.10)':>10}")
        print(SEP)

        def row(label_str, key, fmt, scale=1.0, prefix="", suffix=""):
            vals = []
            for lbl in ["A (0.00)", "B (0.08)", "C (0.10)"]:
                v = results.get(lbl, {}).get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    vals.append("       n/a")
                else:
                    vals.append(f"{prefix}{v*scale:{fmt}}{suffix}")
            print(f"  {label_str:<26}  {vals[0]:>10}  {vals[1]:>10}  {vals[2]:>10}")

        def row_drop(label_str):
            vals = []
            for lbl in ["A (0.00)", "B (0.08)", "C (0.10)"]:
                n = results.get(lbl, {}).get("n_trades", 0)
                drop = n_a - n
                vals.append(f"{drop:>10,}" if lbl != "A (0.00)" else f"{'0':>10}")
            print(f"  {label_str:<26}  {vals[0]:>10}  {vals[1]:>10}  {vals[2]:>10}")

        row("Total trades",        "n_trades",       ",.0f")
        row_drop("Trades dropped")
        row("Win rate (excl EOD)", "win_rate_flux",  ".1%")
        row("Avg win",             "avg_win_pct",    "+.3%")
        row("Avg loss",            "avg_loss_pct",   ".3%")
        row("EV per trade",        "expected_value", "+.3%")
        row("Sharpe",              "sharpe_ratio",   ".3f")
        row("Max drawdown",        "max_drawdown_pct", ".1f", suffix="%")

        print(f"\n  Per-instrument trade counts:")
        syms_a = sorted(results.get("A (0.00)", {}).get("per_instrument", {}).keys())
        print(f"  {'Symbol':<8}  {'A_n':>6}  {'B_n':>6}  {'C_n':>6}  {'B_drop':>8}  {'C_drop':>8}")
        print("  " + "-" * 50)
        for sym in syms_a:
            pi_a = results["A (0.00)"]["per_instrument"].get(sym, {})
            pi_b = results["B (0.08)"].get("per_instrument", {}).get(sym, {})
            pi_c = results["C (0.10)"].get("per_instrument", {}).get(sym, {})
            na = pi_a.get("n_trades", 0)
            nb = pi_b.get("n_trades", 0)
            nc = pi_c.get("n_trades", 0)
            print(f"  {sym:<8}  {na:>6}  {nb:>6}  {nc:>6}  {na-nb:>8}  {na-nc:>8}")

        print()
        sys.exit(0)

    # ── Standard single-config run ────────────────────────────────────────────
    import dataclasses
    config = dataclasses.replace(DEFAULT_CONFIG, mechanism_bin=args.mechanism)
    if args.symbols:
        config.symbols = args.symbols
        # Symbol override resets instrument_gap_filters to avoid silently
        # applying filters for symbols that weren't calibrated.
        config.instrument_gap_filters = {}
    if args.gap_filter is not None:
        config.gap_filter_pct = args.gap_filter
        config.instrument_gap_filters = {}

    print(f"ORB Backtester — [{config.mechanism_bin}]")
    print(f"Symbols:    {config.symbols}")
    if config.instrument_gap_filters:
        print(f"Gap filter: per-instrument (fallback {config.gap_filter_pct*100:.0f}%)")
    else:
        print(f"Gap filter: {config.gap_filter_pct*100:.1f}%")
    print()

    trades = run_backtest(config)

    if trades.empty:
        print("No trades generated. Check data paths and configuration.")
    else:
        metrics = compute_metrics(trades, config)
        print_report(metrics, config)
        save_results(trades, metrics, config)
        print(f"Results saved to {config.output_dir}/")

        # ── Before/after comparison ───────────────────────────────────────────
        # "Before" = pre-RTG baseline (no scaling, no exclusion; Sharpe 1.194).
        # Hardcoded from validated pre-registration run 2026-05-08.
        _BEF = {
            "n_trades":          2622,
            "trades_per_year":   163.5,
            "win_rate":          0.556,
            "win_rate_flux":     0.531,
            "expected_value":    0.00578,
            "avg_win_pct":       0.02815,
            "avg_loss_pct":     -0.02175,
            "tp1_rate":          0.372,
            "tp2_rate":          0.134,
            "eod_rate":          0.299,
            "annual_return_pct": 8.1,
            "sharpe_ratio":      1.224,
            "max_drawdown_pct": -5.5,
            "calmar_ratio":      1.427,
            "pct_years_profitable": 0.94,
        }

        def _ba_fmt_b(key):
            v = _BEF.get(key)
            if v is None:
                return "n/a"
            if key in ("win_rate", "win_rate_flux", "tp1_rate", "tp2_rate",
                       "eod_rate", "pct_years_profitable"):
                return f"{v*100:.1f}%"
            if key in ("expected_value", "avg_win_pct", "avg_loss_pct"):
                return f"{v*100:+.3f}%"
            if key == "annual_return_pct":
                return f"{v:.1f}%"
            if key == "max_drawdown_pct":
                return f"{v:.1f}%"
            if key in ("sharpe_ratio", "calmar_ratio"):
                return f"{v:.3f}"
            if key == "trades_per_year":
                return f"{v:.1f}"
            return f"{v:,}"

        def _ba_fmt_a(key):
            v = metrics.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "n/a"
            if key in ("win_rate", "win_rate_flux", "tp1_rate", "tp2_rate",
                       "eod_rate", "pct_years_profitable"):
                return f"{v*100:.1f}%"
            if key in ("expected_value", "avg_win_pct", "avg_loss_pct"):
                return f"{v*100:+.3f}%"
            if key == "annual_return_pct":
                return f"{v:.1f}%"
            if key == "max_drawdown_pct":
                return f"{v:.1f}%"
            if key in ("sharpe_ratio", "calmar_ratio"):
                return f"{v:.3f}"
            if key == "trades_per_year":
                return f"{v:.1f}"
            return f"{v:,}"

        def _ba_delta(key):
            b = _BEF.get(key)
            a = metrics.get(key)
            if b is None or a is None or (isinstance(a, float) and np.isnan(a)):
                return "n/a"
            d = a - b
            if key in ("win_rate", "win_rate_flux", "tp1_rate", "tp2_rate",
                       "eod_rate", "pct_years_profitable"):
                return f"{d*100:+.1f}pp"
            if key in ("expected_value", "avg_win_pct", "avg_loss_pct"):
                return f"{d*100:+.3f}pp"
            if key in ("annual_return_pct", "max_drawdown_pct"):
                return f"{d:+.1f}pp"
            if key in ("sharpe_ratio", "calmar_ratio"):
                return f"{d:+.3f}"
            if key == "trades_per_year":
                return f"{d:+.1f}"
            return f"{int(a-b):+,}"

        _ba_rows = [
            ("Total trades",        "n_trades"),
            ("Trades/year",         "trades_per_year"),
            ("Win rate (all)",      "win_rate"),
            ("Win rate (excl EOD)", "win_rate_flux"),
            ("EV per trade",        "expected_value"),
            ("Avg win",             "avg_win_pct"),
            ("Avg loss",            "avg_loss_pct"),
            ("TP1 hit rate",        "tp1_rate"),
            ("TP2 hit rate",        "tp2_rate"),
            ("EOD exit rate",       "eod_rate"),
            ("Annual return",       "annual_return_pct"),
            ("Sharpe ratio",        "sharpe_ratio"),
            ("Max drawdown",        "max_drawdown_pct"),
            ("Calmar ratio",        "calmar_ratio"),
            ("Profitable years",    "pct_years_profitable"),
        ]

        _ba_SEP = "─" * 68
        print(f"\n  FINAL CONFIGURATION COMPARISON")
        print(f"  {_ba_SEP}")
        print(f"  {'Metric':<22}  {'Before RTG':>12}  {'After RTG':>12}  {'Delta':>12}")
        print(f"  {_ba_SEP}")
        for _lbl, _key in _ba_rows:
            print(f"  {_lbl:<22}  {_ba_fmt_b(_key):>12}  "
                  f"{_ba_fmt_a(_key):>12}  {_ba_delta(_key):>12}")
        print(f"  {_ba_SEP}")
        print()

        # ── 7-year results comparison ─────────────────────────────────────────
        _7yr_start = "2019-01-01"
        _trades_7 = trades[trades["date"] >= _7yr_start]
        _met_7 = compute_metrics(_trades_7, config) if not _trades_7.empty else {}

        def _7fmt(m, key):
            v = m.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "n/a"
            if key in ("win_rate", "tp1_rate", "tp2_rate", "eod_rate",
                       "pct_years_profitable"):
                return f"{v*100:.1f}%"
            if key == "expected_value":
                return f"{v*100:+.3f}%"
            if key == "annual_return_pct":
                return f"{v:.1f}%"
            if key == "max_drawdown_pct":
                return f"{v:.1f}%"
            if key in ("sharpe_ratio", "calmar_ratio"):
                return f"{v:.3f}"
            if key == "trades_per_year":
                return f"{v:.1f}"
            return f"{v:,.0f}"

        _7yr_rows = [
            ("Total trades",     "n_trades"),
            ("Trades/year",      "trades_per_year"),
            ("Win rate (all)",   "win_rate"),
            ("EV per trade",     "expected_value"),
            ("Annual return",    "annual_return_pct"),
            ("Sharpe ratio",     "sharpe_ratio"),
            ("Max drawdown",     "max_drawdown_pct"),
            ("Calmar",           "calmar_ratio"),
            ("Profitable years", "pct_years_profitable"),
        ]

        _7SEP = "─" * 54
        print(f"  7-YEAR RESULTS (2019–2026)")
        print(f"  {_7SEP}")
        print(f"  {'Metric':<20}  {'Full history':>14}  {'Last 7 years':>14}")
        print(f"  {_7SEP}")
        for _lbl, _key in _7yr_rows:
            print(f"  {_lbl:<20}  {_7fmt(metrics, _key):>14}  "
                  f"{_7fmt(_met_7, _key):>14}")
        print(f"  {_7SEP}")
        print()

        # ── Walk-forward summary ──────────────────────────────────────────────
        print("  Walk-forward 2019–2026:")
        print(f"  {'Year':<6}  {'Sharpe':>8}  {'WR':>7}  {'EV':>8}  "
              f"{'N':>6}  {'Prof':>5}")
        print("  " + "-" * 50)
        _wf_wins = 0
        _wf_total = 0
        for _yr in range(2019, 2027):
            _sub = trades[trades["date"].dt.year == _yr]
            if _sub.empty:
                continue
            _m = compute_metrics(_sub, config)
            _sh = _m.get("sharpe_ratio", float("nan"))
            _wr = _m.get("win_rate", float("nan"))
            _ev = _m.get("expected_value", float("nan"))
            _n  = _m.get("n_trades", 0)
            _prof = _sh > 0 if not np.isnan(_sh) else False
            _wf_wins  += int(_prof)
            _wf_total += 1
            print(
                f"  {_yr:<6}  "
                f"{f'{_sh:.3f}' if not np.isnan(_sh) else 'n/a':>8}  "
                f"{f'{_wr:.1%}' if not np.isnan(_wr) else 'n/a':>7}  "
                f"{f'{_ev:+.4%}' if not np.isnan(_ev) else 'n/a':>8}  "
                f"{_n:>6}  "
                f"{'YES' if _prof else 'NO':>5}"
            )
        print(f"\n  Profitable years: {_wf_wins}/{_wf_total}")
