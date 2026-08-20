"""signals/gap_scan.py — the pre-market gap scan, as a pure function.

This is the layer between raw market data and ``plan_session()``: it turns
per-ETF reference prices and daily bars into the UL-level input tuple the
strategy profile consumes. Everything the two repos can actually disagree
about lives here — the ETF gap, the ``leverage × GAP_THRESHOLD`` test, the
ETF→UL reconstruction, most-extreme-wins when several ETFs share an
underlying, and the prior-session filter.

Why it is separate from ``PreMarketJob``: the scan used to be inlined in
``run_phase1``, interleaved with ``save_gap_scan`` calls, structured logging
and broker fetches, so nothing could execute it without a database and a
broker connection. That made it untestable and, more importantly, made it
invisible to the golden-session replay gate — the fixture recorded this
function's *output* (``overnight_gaps``) as an input, so the whole layer
replayed as a given. Changing GDXU's leverage from 2 to 3, which moves its
gap threshold from 4% to 6%, left every fixture green.

``plan_session`` itself cannot drift between the repos: they share one
implementation via ``pip install -e``. This can. Keeping it pure is what
lets a replay start above the leverage conversion instead of below it.

No I/O, no logging, no persistence. Callers decide what to record; every
rejected symbol comes back as a ``SymbolScan`` carrying its reason rather
than being dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from orb_live.signals.strategy_signals import compute_gap, check_prior_session_filter
from orb_live.strategy.v1_strategy import GAP_THRESHOLD, Instrument


@dataclass
class SymbolScan:
    """One ETF's journey through the scan, including why it stopped.

    ``filter_reason`` is empty exactly when ``qualifies`` is True. The reason
    strings are the ones already persisted by ``StateStore.save_gap_scan`` and
    are part of that table's vocabulary — do not rename them casually.
    """
    symbol: str
    qualifies: bool = False
    filter_reason: str = ""

    # Populated once the corresponding stage is reached; None before that.
    ref_price:   Optional[float] = None
    prior_close: Optional[float] = None
    etf_gap_abs: Optional[float] = None
    gap_direction: Optional[int] = None
    ul_gap:      Optional[float] = None

    # Prior-session filter. ps_checked is False when the symbol never got
    # that far, which is a different thing from the filter passing.
    ps_checked:   bool = False
    ps_passed:    bool = False
    ps_underlying: Optional[str] = None
    ul_move_pct:  Optional[float] = None
    threshold_pct: Optional[float] = None
    ps_warned:    bool = False


@dataclass
class GapScanResult:
    """The ``plan_session()`` input tuple, plus a per-symbol audit trail."""
    overnight_gaps:   dict[str, float] = field(default_factory=dict)
    prior_two_closes: dict[str, tuple[float, float]] = field(default_factory=dict)
    prior_etf_close:  dict[str, float] = field(default_factory=dict)
    scans:            list[SymbolScan] = field(default_factory=list)

    @property
    def qualified_symbols(self) -> list[str]:
        """Symbols that cleared every filter, in scan order.

        Order matters: it becomes ``universe`` for ``plan_session``, which
        determines candidate order out of ``compute_candidates``. Do not sort.
        """
        return [s.symbol for s in self.scans if s.qualifies]


class _WarnCapture:
    """Records whether check_prior_session_filter emitted a data warning."""

    def __init__(self):
        self.warned = False

    def warning(self, event, **kw):  # noqa: ANN
        self.warned = True


def scan_gaps(
    trade_date: date,
    *,
    symbols: list[str],
    instruments: dict[str, Instrument],
    ref_prices: dict[str, Optional[float]],
    etf_daily: dict[str, Optional[pd.DataFrame]],
    ul_daily: dict[str, pd.DataFrame],
    config,                       # LiveConfig or StrategyConfig (duck-typed)
) -> GapScanResult:
    """Run the 09:31 gap scan over ``symbols``.

    ref_prices — {symbol: 9:30 bar close}. Missing or non-positive is a
        rejection, not an error; the caller has already decided how hard to
        try fetching.
    etf_daily  — {symbol: daily bars} for the prior-close lookup.
    ul_daily   — {underlying: daily bars} for the prior-session filter.

    Returns every symbol as a SymbolScan in the order given, so the caller can
    persist rejections without re-deriving why they were rejected.
    """
    result = GapScanResult()

    for symbol in symbols:
        inst = instruments.get(symbol)
        if inst is None:
            continue

        scan = SymbolScan(symbol=symbol)
        result.scans.append(scan)

        # 1. Reference price (9:30 bar close).
        ref_price = ref_prices.get(symbol)
        if ref_price is None or float(ref_price) <= 0:
            scan.filter_reason = "no_ref_price"
            continue
        scan.ref_price = float(ref_price)

        # 2. Daily bars, for the prior ETF close.
        d_bars = etf_daily.get(symbol)
        if d_bars is None or d_bars.empty:
            scan.filter_reason = "no_daily_bars"
            continue

        # 3. ETF gap and prior close.
        gap_result = compute_gap(trade_date, d_bars, scan.ref_price)
        if gap_result is None:
            scan.filter_reason = "no_prior_close"
            continue

        etf_gap_abs, gap_direction, prior_close = gap_result
        scan.etf_gap_abs   = etf_gap_abs
        scan.gap_direction = gap_direction
        scan.prior_close   = prior_close
        # Recorded even if the symbol is filtered below: plan_session prices
        # the whole group when it prunes cheap ETFs, not just the survivors.
        result.prior_etf_close[symbol] = prior_close

        # 4. Convert the ETF gap to a UL-equivalent gap.
        #    Non-inverse: ETF_gap =  UL_gap × leverage → UL_gap =  ETF_signed / leverage
        #    Inverse:     ETF_gap = -UL_gap × leverage → UL_gap = -ETF_signed / leverage
        etf_gap_signed = etf_gap_abs * gap_direction
        scan.ul_gap = (
            -etf_gap_signed if inst.inverse else etf_gap_signed
        ) / inst.leverage

        # 5. Gap size, tested on the ETF. Algebraically identical to
        #    |ul_gap| >= GAP_THRESHOLD, which is why plan_session's own 2%
        #    check can never fail on anything that gets past here.
        if etf_gap_abs < inst.leverage * GAP_THRESHOLD:
            scan.filter_reason = "gap_too_small"
            continue

        # 6. Direction filter (LABU/LABD).
        allowed_dir = config.direction_filters.get(symbol)
        if allowed_dir is not None and gap_direction != allowed_dir:
            scan.filter_reason = "direction_filtered"
            continue

        # 7. Prior-session filter, on real UL data rather than the
        #    reconstruction — this half already agrees with the backtest.
        warn = _WarnCapture()
        scan.ps_checked = True
        scan.ps_passed = check_prior_session_filter(
            symbol, trade_date, gap_direction, config, ul_daily, logger=warn,
        )
        scan.ps_warned = warn.warned

        ps_spec = config.prior_session_filters.get(symbol)
        if ps_spec:
            scan.ps_underlying = ps_spec[0]
            scan.threshold_pct = float(ps_spec[1])
            scan.ul_move_pct = _ps_ul_move(
                ul_daily.get(ps_spec[0]), trade_date, gap_direction, ps_spec,
            )

        if not scan.ps_passed:
            scan.filter_reason = "ps_filter"
            continue

        scan.qualifies = True

        # 8. Accumulate the UL-level inputs.
        ul = inst.underlying
        prior = result.overnight_gaps.get(ul)
        if prior is None or abs(scan.ul_gap) > abs(prior):
            # Most extreme wins when several ETFs map to one underlying.
            result.overnight_gaps[ul] = scan.ul_gap

        if ul not in result.prior_two_closes:
            closes = _prior_two_closes(ul_daily.get(ul), trade_date)
            if closes is not None:
                result.prior_two_closes[ul] = closes

    return result


def _prior_two_closes(
    ul_df: Optional[pd.DataFrame], trade_date: date,
) -> Optional[tuple[float, float]]:
    """(c_t-1, c_t-2) for the underlying, or None if fewer than two exist."""
    if ul_df is None:
        return None
    rows = ul_df[ul_df["date"] < pd.Timestamp(trade_date)].tail(2)
    if len(rows) < 2:
        return None
    return float(rows.iloc[-1]["close"]), float(rows.iloc[-2]["close"])


def _ps_ul_move(
    ul_df: Optional[pd.DataFrame],
    trade_date: date,
    gap_direction: int,
    ps_spec: tuple,
) -> Optional[float]:
    """The prior-session UL move, signed the way the filter sees it.

    Reporting only — check_prior_session_filter computes this itself. Kept in
    step with it so the state-store record matches the decision.
    """
    closes = _prior_two_closes(ul_df, trade_date)
    if closes is None:
        return None
    c_t1, c_t2 = closes
    if c_t2 <= 0:
        return None
    raw = (c_t1 - c_t2) / c_t2
    is_inverse = len(ps_spec) == 3 and ps_spec[2] is True
    return raw * (-gap_direction if is_inverse else gap_direction)
