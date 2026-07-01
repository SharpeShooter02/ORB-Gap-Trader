"""
execution/order_policy.py — Fill record and entry-price calculator.

Phase 3: MarketableLimitPolicy is now a price calculator only.
All time.sleep submit loops (_submit_entry/_submit_exit/_submit_stop)
and the buy()/sell() wrappers have been removed.

Entry orders are placed as marketable limits (one shot, no repeg) in
LivePositionManager.open_position().  Exits are handled by IB OCA
(TP1+stop bracket placed at entry) and direct broker.submit_market_order()
calls for TP3, EOD, and flatten_all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from orb_live.config.live_config import LiveConfig

UTC = timezone.utc


# ── Fill dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Fill:
    """Canonical record of a completed order leg."""
    symbol:            str
    side:              str           # 'buy' / 'sell'
    qty:               int           # actually filled shares (0 if unfilled)
    avg_price:         float         # average fill price (0.0 if unfilled)
    order_id:          str
    leg:               str           # entry / tp1 / tp2 / tp3 / stop / eod
    attempts:          int           # repeg attempts made (always 1 in Phase 3)
    reason:            str           # filled / partial_unfilled / unfilled / market
    submitted_at:      datetime
    filled_at:         Optional[datetime]
    raw_response_json: str = field(default="{}")


# ── MarketableLimitPolicy ──────────────────────────────────────────────────────

class MarketableLimitPolicy:
    """
    Entry-price calculator for IB-native execution.

    compute_entry_limit() returns the marketable limit price for an entry
    order without submitting anything.  The class is retained so callers
    that still instantiate it (e.g. legacy test fixtures) do not break;
    extra kwargs (state_store, _sleep, logger) are accepted and ignored.
    """

    def __init__(
        self,
        broker,
        live_cfg: "LiveConfig",
        state_store=None,
        logger=None,
        **_ignored,          # absorbs legacy kwargs (_sleep, etc.)
    ):
        self._broker = broker
        self._cfg    = live_cfg

    def compute_entry_limit(
        self,
        side: str,
        symbol: str,
        reference_price: Optional[float] = None,
        bps: Optional[int] = None,
    ) -> float:
        """Return the marketable limit price for an entry order (no submission)."""
        _bps = bps if bps is not None else self._cfg.entry_slippage_bps
        return self._compute_limit(side, symbol, _bps, reference_price)

    def _compute_limit(
        self,
        side: str,
        symbol: str,
        bps: int,
        reference_price: Optional[float],
    ) -> float:
        """Compute a marketable limit price from NBBO or reference_price."""
        if reference_price is None:
            q = self._broker.get_latest_quote(symbol)
            reference_price = (float(q.get("ask", 0)) if side == "buy"
                               else float(q.get("bid", 0)))
        factor = bps / 10_000.0
        if side == "buy":
            return reference_price * (1.0 + factor)
        return reference_price * (1.0 - factor)
