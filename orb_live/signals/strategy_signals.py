"""
signals/strategy_signals.py — Verbatim port of the five core signal functions.

KEY PRINCIPLE: every function here must produce IDENTICAL outputs to its
backtest counterpart (reference/orb_backtester.py) on identical inputs.
Where the backtest has quirks, the live code reproduces them.

Functions ported:
    compute_gap                (backtester lines 946-983)
    check_prior_session_filter (backtester lines 984-1037)
    compute_opening_range      (backtester lines 1040-1104)
    check_breakout             (backtester lines 1107-1157)
    compute_entry              (backtester lines 1160-1236)

DATA-HICCUP HANDLING (check_prior_session_filter):
    When underlying data is missing or only one prior close is available,
    the backtest returns True (allow trade).  Live behaviour matches BUT
    also emits a WARN log when a logger is supplied.  The return value is
    unchanged — callers that check parity must pass logger=None.
"""

from __future__ import annotations

import math
from datetime import date, time as dtime, datetime as dt, timedelta
from typing import Optional

import pandas as pd


# ── compute_gap ───────────────────────────────────────────────────────────────

def compute_gap(
    date_: date,
    daily_df: pd.DataFrame,
    today_ref_price: float,
) -> Optional[tuple]:
    """
    Compute the overnight gap for an instrument on the given date.

    Returns (gap_abs_frac, gap_direction, prior_close):
        gap_abs_frac  — absolute gap as a fraction (e.g. 0.08 for 8%)
        gap_direction — +1 gap-up, -1 gap-down
        prior_close   — the prior session's closing price

    Returns None if the prior close is unavailable.
    """
    prior_rows = daily_df[daily_df["date"] < pd.Timestamp(date_)]
    if prior_rows.empty:
        return None

    prior_close = float(prior_rows.iloc[-1]["close"])
    if prior_close <= 0:
        return None

    gap_pct = (today_ref_price - prior_close) / prior_close
    return abs(gap_pct), (1 if gap_pct >= 0 else -1), prior_close


# ── check_prior_session_filter ────────────────────────────────────────────────

def check_prior_session_filter(
    symbol: str,
    date_: date,
    gap_direction: int,
    config,                          # LiveConfig or StrategyConfig (duck-typed)
    underlying_data: dict[str, pd.DataFrame],
    logger=None,
) -> bool:
    """
    Returns True if the trade is ALLOWED, False if BLOCKED.

    Identical logic to backtester lines 984-1037.  When underlying data is
    missing or insufficient, returns True (conservative default) and
    optionally logs a data-quality warning.
    """
    filter_cfg = config.prior_session_filters.get(symbol)
    if filter_cfg is None:
        return True

    is_inverse = len(filter_cfg) == 3 and filter_cfg[2] is True
    ul_sym, threshold = filter_cfg[0], filter_cfg[1]
    ul_df = underlying_data.get(ul_sym)

    if ul_df is None:
        if logger:
            logger.warning(
                "ps_filter_data_warning",
                symbol=symbol,
                ul_sym=ul_sym,
                message=f"Allowing {symbol}: no data for {ul_sym}; check yfinance health.",
                n_closes=0,
            )
        return True

    prior_rows = ul_df[ul_df["date"] < pd.Timestamp(date_)].tail(2)
    if len(prior_rows) < 2:
        if logger:
            logger.warning(
                "ps_filter_data_warning",
                symbol=symbol,
                ul_sym=ul_sym,
                message=(
                    f"Allowing {symbol}: only {len(prior_rows)} prior close(s) of "
                    f"{ul_sym} available (need 2); check yfinance health."
                ),
                n_closes=len(prior_rows),
            )
        return True

    close_t1 = float(prior_rows.iloc[-1]["close"])  # most recent prior close
    close_t2 = float(prior_rows.iloc[-2]["close"])  # one session earlier
    if close_t2 <= 0:
        return True

    prior_session_ret   = (close_t1 - close_t2) / close_t2
    effective_direction = -gap_direction if is_inverse else gap_direction
    dir_adj_ret         = prior_session_ret * effective_direction
    return dir_adj_ret <= threshold


# ── compute_opening_range ─────────────────────────────────────────────────────

def compute_opening_range(
    bars: pd.DataFrame,
    config,                  # StrategyConfig (accessed via live_cfg.strategy_config)
    symbol: str = "",
) -> Optional[dict]:
    """
    Compute the ORB from bars[timestamp index] using the first orb_minutes.

    bars must have a DatetimeIndex (tz-naive Eastern, or tz-aware is accepted
    and stripped).  Columns: close, and optionally high, low.

    Returns dict: {high, low, midpoint, size_pct, n_bars, ema}
    Returns None if insufficient bars.
    """
    _market_open = dtime(config.market_open_hour, config.market_open_minute)
    _orb_end = (
        dt(2000, 1, 1, _market_open.hour, _market_open.minute)
        + timedelta(minutes=config.orb_minutes)
    ).time()

    idx = bars.index
    # Strip tz if present.
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
        bars = bars.copy()
        bars.index = idx

    orb_bars = bars[
        (idx.time >= _market_open) &
        (idx.time < _orb_end)
    ]

    effective_min = (
        config.min_orb_bars_sparse
        if symbol in config.sparse_data_symbols
        else config.min_orb_bars
    )
    if len(orb_bars) < effective_min:
        return None

    orb_high = float(orb_bars["high"].max()) if "high" in orb_bars.columns \
               else float(orb_bars["close"].max())
    orb_low  = float(orb_bars["low"].min())  if "low"  in orb_bars.columns \
               else float(orb_bars["close"].min())

    if orb_low <= 0 or orb_high <= orb_low:
        return None

    midpoint  = (orb_high + orb_low) / 2.0
    size_pct  = (orb_high - orb_low) / midpoint

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


# ── check_breakout ────────────────────────────────────────────────────────────

def check_breakout(
    bar: pd.Series,
    orb: dict,
    gap_direction: int,
    config,
) -> bool:
    """
    Returns True if bar constitutes a valid ORB breakout in gap_direction.

    Identical to backtester lines 1107-1157.
    """
    close = float(bar["close"])

    if orb["size_pct"] < config.min_profit_pct:
        return False
    if orb["size_pct"] < config.min_increment_pct:
        return False

    orb_range = orb["high"] - orb["low"]
    if orb_range > 0 and config.min_entry_excess > 0:
        if gap_direction == 1:
            excess = (close - orb["high"]) / orb_range
        else:
            excess = (orb["low"] - close) / orb_range
        if excess < config.min_entry_excess:
            return False

    if getattr(config, "entry_at_boundary", False):
        bar_high = float(bar["high"]) if "high" in bar.index else close
        bar_low  = float(bar["low"])  if "low"  in bar.index else close
        if gap_direction == 1:
            price_touched = bar_high >= orb["high"]
        else:
            price_touched = bar_low  <= orb["low"]
        if not price_touched:
            return False
        if getattr(config, "require_ema_confirmation", True):
            if gap_direction == 1 and not (close > orb["ema"]):
                return False
            if gap_direction == -1 and not (close < orb["ema"]):
                return False
        return True

    if gap_direction == 1:
        if getattr(config, "require_ema_confirmation", True) and not (close > orb["ema"]):
            return False
        return close > orb["high"]
    else:
        if getattr(config, "require_ema_confirmation", True) and not (close < orb["ema"]):
            return False
        return close < orb["low"]


# ── compute_entry ─────────────────────────────────────────────────────────────

def compute_entry(
    bar: pd.Series,
    orb: dict,
    gap_direction: int,
    config,
    current_equity: float,
    symbol: str = "",
    tp1_mult_override: Optional[float] = None,
    tp2_mult_override: Optional[float] = None,
    size_mult: float = 1.0,
    v1_base_notional: Optional[float] = None,
) -> dict:
    """
    Compute entry price, stop, TP levels, and share counts.

    Identical to backtester lines 1160-1236.
    """
    if getattr(config, "entry_at_boundary", False):
        entry_price = float(orb["high"]) if gap_direction == 1 else float(orb["low"])
    else:
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

    if v1_base_notional is not None and v1_base_notional > 0:
        # v1 sizing: $1k-per-unit baseline × per-symbol multiplier (size_mult)
        shares = math.floor(v1_base_notional * size_mult / entry_price)
    elif config.use_risk_based_sizing:
        stop_dist = abs(entry_price - stop_price)
        if stop_dist > 0:
            shares = math.floor(
                current_equity * config.risk_pct_at_stop * size_mult / stop_dist
            )
        else:
            shares = 0
        cap    = math.floor(
            current_equity * config.max_position_pct * size_mult / entry_price
        )
        shares = min(shares, cap)
    else:
        shares = math.floor(
            current_equity * config.daily_risk_pct * size_mult / entry_price
        )

    override   = config.instrument_exit_overrides.get(symbol, {})
    r_tp1      = override.get("exit_ratio_tp1", config.exit_ratio_tp1)
    r_tp2      = override.get("exit_ratio_tp2", config.exit_ratio_tp2)
    tp1_shares = math.floor(shares * r_tp1)
    tp2_shares = math.floor(shares * r_tp2)
    tp3_shares = max(0, shares - tp1_shares - tp2_shares)

    return {
        "entry_price":  entry_price,
        "stop_price":   stop_price,
        "tp1_price":    tp1_price,
        "tp2_price":    tp2_price,
        "orb_range":    orb_range,
        "shares":       shares,
        "tp1_shares":   tp1_shares,
        "tp2_shares":   tp2_shares,
        "tp3_shares":   tp3_shares,
        "direction":    gap_direction,
        "exit_override": override,
        "entry_time":   bar.name,
    }
