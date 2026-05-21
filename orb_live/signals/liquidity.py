"""
signals/liquidity.py — Pre-flight liquidity and eligibility checks.

Three sequential gates (A → B → C):
  A. Operator overrides — excluded_symbols, date_exclusions, force_long_only
  B. Asset eligibility  — Alpaca asset status, tradability, HTB short check
  C. ADV / dollar volume — 20-day ADV floor, max % of ADV, yesterday DV ratio

Gates are applied in order; the first failure short-circuits the rest.
Results are persisted to liquidity_metrics (B+C) and candidates via pre_market.

FAIL-OPEN POLICY:
  If Alpaca returns no data for a symbol, PreFlightCheck treats it as passed
  with a warning.  A transient API error must not block a valid trade setup.
  This mirrors the check_prior_session_filter data-hiccup policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from orb_live.config.live_config import LiveConfig
    from orb_live.core.state_store import StateStore
    from orb_live.data.alpaca_client import AlpacaClient


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CandidateDecision:
    symbol: str
    passed: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    adv_dollars: float = 0.0
    min_dv_20d: float = 0.0
    yesterday_dv: float = 0.0
    intended_dollars: float = 0.0
    intended_pct_of_adv: float = 0.0


# ── PreFlightCheck ────────────────────────────────────────────────────────────

class PreFlightCheck:
    """
    Per-symbol pre-flight eligibility gate.

    Usage:
        pfc = PreFlightCheck(live_cfg, state_store, alpaca_client)
        decision = pfc.check(symbol, session_date, gap_direction, current_equity)
        if not decision.passed:
            continue  # skip this symbol today
    """

    def __init__(
        self,
        config: "LiveConfig",
        store: "StateStore",
        client: "AlpacaClient",
        logger=None,
    ):
        self._cfg    = config
        self._store  = store
        self._client = client
        self._log    = logger

    # ── Gate A: operator overrides ────────────────────────────────────────────

    def _check_overrides(
        self, symbol: str, session_date: date, gap_direction: int
    ) -> Optional[str]:
        """Return failure reason or None if passes."""
        cfg = self._cfg

        if symbol in cfg.excluded_symbols:
            return "operator_excluded"

        date_key = str(session_date)
        if symbol in cfg.date_exclusions.get(date_key, []):
            return "date_excluded"

        if symbol in cfg.force_long_only and gap_direction == -1:
            return "force_long_only"

        return None

    # ── Gate B: asset eligibility ─────────────────────────────────────────────

    def _check_asset(
        self, symbol: str, gap_direction: int
    ) -> tuple[Optional[str], list[str]]:
        """Return (failure_reason_or_None, warnings_list)."""
        warnings: list[str] = []
        try:
            asset = self._client.get_asset(symbol)
        except Exception as exc:
            warnings.append(f"asset_fetch_error: {exc}")
            return None, warnings  # fail-open on transient errors

        if not asset.get("tradable", True):
            return "not_tradable", warnings

        status = asset.get("status", "active")
        # Normalize alpaca-py enum values (e.g. AssetStatus.ACTIVE → "active")
        status_str = (status.value if hasattr(status, "value") else str(status)).lower()
        if status_str != "active":
            return "asset_not_active", warnings

        if gap_direction == -1:
            if not self._cfg.allow_htb_shorts and not asset.get("easy_to_borrow", True):
                return "htb_short_not_allowed", warnings
            if not asset.get("shortable", True):
                return "not_shortable", warnings

        return None, warnings

    # ── Gate C: ADV / dollar volume ───────────────────────────────────────────

    def _check_liquidity(
        self, symbol: str, current_equity: float
    ) -> tuple[Optional[str], CandidateDecision]:
        """Fetch ADV data and verify dollar-volume thresholds."""
        cfg      = self._cfg
        decision = CandidateDecision(symbol=symbol, passed=False)

        try:
            bars = self._client.get_daily_bars(
                symbol, lookback_days=cfg.adv_lookback_days + 5
            )
        except Exception as exc:
            decision.warnings.append(f"bars_fetch_error: {exc}")
            decision.passed = True   # fail-open
            decision.reason = "liquidity_data_unavailable"
            return None, decision

        if bars.empty or len(bars) < 2:
            decision.warnings.append("insufficient_bars_for_adv")
            decision.passed = True   # fail-open
            decision.reason = "liquidity_data_unavailable"
            return None, decision

        bars = bars.copy()
        bars["dv"] = bars["close"] * bars["volume"]

        dv_window = bars["dv"].tail(cfg.adv_lookback_days)
        adv_dollars  = float(dv_window.mean()) if len(dv_window) > 0 else 0.0
        min_dv_20d   = float(dv_window.min())  if len(dv_window) > 0 else 0.0
        yesterday_dv = float(bars["dv"].iloc[-1])

        decision.adv_dollars  = adv_dollars
        decision.min_dv_20d   = min_dv_20d
        decision.yesterday_dv = yesterday_dv

        intended_dollars = current_equity * cfg.strategy_config.daily_risk_pct
        decision.intended_dollars = intended_dollars
        if adv_dollars > 0:
            decision.intended_pct_of_adv = intended_dollars / adv_dollars

        if adv_dollars < cfg.min_dollar_volume_floor:
            return "adv_below_floor", decision

        if decision.intended_pct_of_adv > cfg.max_pct_of_adv:
            return "exceeds_pct_of_adv", decision

        if adv_dollars > 0 and (yesterday_dv / adv_dollars) < cfg.min_yesterday_dv_ratio:
            return "yesterday_dv_ratio_low", decision

        return None, decision

    # ── Public interface ──────────────────────────────────────────────────────

    def check(
        self,
        symbol: str,
        session_date: date,
        gap_direction: int,
        current_equity: float,
        persist: bool = True,
    ) -> CandidateDecision:
        """
        Run all three gates in sequence.

        Returns CandidateDecision with passed=True if all gates pass.
        If persist=True, writes to state_store.liquidity_metrics.
        """
        # A: operator overrides
        override_fail = self._check_overrides(symbol, session_date, gap_direction)
        if override_fail:
            decision = CandidateDecision(
                symbol=symbol, passed=False, reason=override_fail,
            )
            if persist:
                self._store.save_liquidity_metrics(
                    session_date, symbol,
                    passed=False, reason=override_fail, warnings="",
                )
            return decision

        # B: asset eligibility
        asset_fail, asset_warnings = self._check_asset(symbol, gap_direction)
        if asset_fail:
            decision = CandidateDecision(
                symbol=symbol, passed=False, reason=asset_fail,
                warnings=asset_warnings,
            )
            if persist:
                self._store.save_liquidity_metrics(
                    session_date, symbol,
                    passed=False, reason=asset_fail,
                    warnings="; ".join(asset_warnings),
                )
            return decision

        # C: ADV / dollar volume
        liquidity_fail, decision = self._check_liquidity(symbol, current_equity)
        decision.warnings.extend(asset_warnings)

        if liquidity_fail:
            decision.passed = False
            decision.reason = liquidity_fail
        else:
            decision.passed = True

        if persist:
            self._store.save_liquidity_metrics(
                session_date, symbol,
                adv_dollars=decision.adv_dollars,
                min_dv_20d=decision.min_dv_20d,
                yesterday_dv=decision.yesterday_dv,
                intended_dollars=decision.intended_dollars,
                intended_pct_of_adv=decision.intended_pct_of_adv,
                passed=decision.passed,
                reason=decision.reason,
                warnings="; ".join(decision.warnings),
            )

        return decision
