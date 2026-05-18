"""
signals/routing.py — RTG pair routing for the live session.

In the backtest, routing is pre-computed across the full history via a
qualification pre-scan for each pair.  In live, routing is resolved per-session
after the ORB window closes: given which symbols qualified and their RTG
percentile ranks, determine which symbol in each pair gets the trade.

ROUTING RULES (identical to backtester lines 1689-1713):
  - Both symbols in a pair must have qualified (gap + PS + ORB + RTG-excl).
  - Winner = symbol with higher RTG pct rank (or fixed override from config).
  - Winner routing action: "double" (2× size_mult).
  - Loser routing action: "skip" (no trade today).
  - Symbols not in any pair (or only one side qualified) default to "normal".

SIZE MULTIPLIERS:
  "double" → 2.0
  "normal" → 1.0
  "skip"   → no trade (executor must check is_skipped before placing orders)
"""

from __future__ import annotations

from datetime import date
from typing import Optional


SIZE_MULT: dict[str, float] = {"double": 2.0, "normal": 1.0, "skip": 0.0}


class PairRouter:
    """
    Determines per-session routing decisions for configured RTG pairs.

    Usage:
        router = PairRouter(live_cfg.strategy_config)
        # qualified_map populated after phase-2 ORB + RTG computation.
        decisions = router.decide(session_date, qualified_map)
        action    = decisions.get(symbol, "normal")  # "double" / "skip" / "normal"
        size_mult = SIZE_MULT[action]
    """

    def __init__(self, config):  # StrategyConfig
        self._config = config

    def decide(
        self,
        session_date: date,
        qualified_map: dict[str, dict],
    ) -> dict[str, str]:
        """
        Build routing decisions for session_date.

        qualified_map maps symbol → {
            "qualifies":   bool   — True after gap + PS + ORB + RTG-excl pass,
            "rtg_pct":     float|None,
            "rtg_val":     float|None,
            ...
        }

        Returns {symbol: "double" | "skip"}.
        Symbols absent from the result are implicitly "normal".
        Symbols where only one side of a pair qualifies are also "normal".
        """
        cfg = self._config
        if not cfg.rtg_pair_routing or not cfg.rtg_routing_pairs:
            return {}

        decisions: dict[str, str] = {}

        for pair in cfg.rtg_routing_pairs:
            sym_a, sym_b = pair[0], pair[1]

            qa = qualified_map.get(sym_a)
            qb = qualified_map.get(sym_b)

            # Only route when both symbols qualified today.
            if qa is None or qb is None:
                continue
            if not qa.get("qualifies", False) or not qb.get("qualifies", False):
                continue

            # Fixed winner override bypasses RTG comparison.
            fixed = (cfg.rtg_routing_fixed.get(f"{sym_a}_{sym_b}")
                     or cfg.rtg_routing_fixed.get(f"{sym_b}_{sym_a}"))

            if fixed:
                winner = fixed
            else:
                def _score(q: dict) -> float:
                    pct = q.get("rtg_pct")
                    if pct is not None:
                        return pct
                    val = q.get("rtg_val")
                    return val if val is not None else 0.0

                winner = sym_a if _score(qa) >= _score(qb) else sym_b

            loser = sym_b if winner == sym_a else sym_a
            decisions[winner] = "double"
            decisions[loser]  = "skip"

        return decisions

    def get_size_mult(self, symbol: str, decisions: dict[str, str]) -> float:
        """Return the position size multiplier for symbol given routing decisions."""
        return SIZE_MULT[decisions.get(symbol, "normal")]

    def is_skipped(self, symbol: str, decisions: dict[str, str]) -> bool:
        """Return True if this symbol lost its routing pair comparison today."""
        return decisions.get(symbol) == "skip"
