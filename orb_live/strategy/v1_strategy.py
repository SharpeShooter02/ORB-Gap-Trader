"""
v1_strategy.py — self-contained v1 trading logic.
VENDORED from ../v1/v1_strategy.py on 2026-06-21. Do NOT import from ../v1/ at runtime.

Pure-function strategy module. Given market state at 9:30 ET, returns:
  - the set of symbols that qualified to trade today (candidates)
  - the day's regime (quiet/active/flood) and cap_factor
  - per-symbol position multipliers to apply when an ORB fires

INPUTS (CSV):
    master_universe.csv  — ETF catalog (class, underlying, etf, leverage, direction, binary_event)
    sigma_master.csv     — per-underlying volatility (columns: underlying, sigma, ...)

INPUTS (per-session, fed in by caller):
    overnight_gaps   : dict[UL, float]                — (open / prior_close - 1) per UL
    prior_two_closes : dict[UL, tuple[float, float]]  — (c_t-1, c_t-2) per UL for PS filter
    prior_etf_close  : dict[symbol, float]            — last-known ETF close, for skip-cheap ranking

OUTPUTS:
    SessionPlan(candidates, regime, cap_factor, multipliers)

The caller (live trader) is responsible for:
    - Maintaining a market data subscription
    - Computing gaps and prior closes pre-open
    - Watching for ORB breakouts on the candidates
    - Submitting orders sized by multipliers * base_notional

Locked config 2026-06-20:
    Sharpe 1.922, Sortino 4.39, MaxDD -12.96%, Calmar 3.77, $26,671 flat / $127,842 compounded
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS — frozen v1 config
# ───────────────────────────────────────────────────────────────────────────────

# Prior-session filter σ multiplier. Higher = looser filter.
K_SIGMA = 1.00

#: Siblings needing more than this fraction of notional in initial margin are
#: dropped before skip-cheap picks the most expensive. Measured rates cluster:
#: conventional funds sit at 0.5-0.96, crypto shorts at 1.12-1.21, then a gap
#: to the unfillable group (BTCZ 2.33, BTCL 3.88, ETU 3.99, ETHU 4.09). Any
#: value in 1.67-3.33 separates the clusters identically in backtest, so this
#: is not a tuned parameter. Measured +4.4% net P&L with partial fills on,
#: Sharpe 2.261 -> 2.309, MaxDD -7.28% -> -6.97%.
MAX_MARGIN_RATE: float = 2.0

# Position cap. Each unit of weight × leverage = 10% of equity, so 20 = 200% total.
CAP_UNITS = 20.0

# Regime cuts on n_uls (count of distinct underlyings with qualifying candidates)
ACTIVE_MIN = 5   # n_uls in [5, 8) -> active
FLOOD_MIN  = 8   # n_uls >= 8     -> flood; n_uls < 5 -> quiet

# Gap qualification — minimum |UL overnight gap| to consider the day
GAP_THRESHOLD = 0.02   # 2% on the underlying

# (class, regime) -> weight multiplier
WEIGHTS: dict[tuple[str, str], float] = {
    ("C1", "quiet"): 3.0, ("C1", "active"): 3.0, ("C1", "flood"): 3.0,
    ("C2", "quiet"): 0.5, ("C2", "active"): 3.0, ("C2", "flood"): 0.5,
    ("C3", "quiet"): 3.0, ("C3", "active"): 3.0, ("C3", "flood"): 3.0,
}

# 3-class taxonomy (everything not in C1 or C2 is C3).
CLASS_1_SYMS: frozenset[str] = frozenset({
    # Crypto (only)
    "BITX", "BITU", "SBIT", "BTCZ", "BTCL", "BTFX",
    "ETHU", "ETHT", "ETHD", "ETH", "ETU",
    "SOLT", "UXRP", "XRPT",
})
CLASS_2_SYMS: frozenset[str] = frozenset({
    # International leveraged ETFs
    "YINN", "YANG", "KORU", "BRZU", "INDL", "MEXX", "EDC", "EDZ",
    # Gold miners are NOT here: they are C3. They are the one C2 sub-group that
    # contradicts the flood notch -- +0.0265 per margin dollar on flood days
    # against international +0.0047 and single-stock +0.0014, and better than
    # C3 flood itself (+0.0181). Weighting them 0.5 in quiet and flood was
    # suppressing the strongest cell in the strategy. See OPEN_ISSUES R4; the
    # move only pays alongside SHARED_ALLOTMENT below.
    # Single-stock leveraged ETFs. Only the three FORCE_INCLUDE names are
    # listed: the rest are 5_SINGLE_STOCK with no binary_event flag, so the
    # class filter in build_universe drops them and they can never trade.
    # Listing them here only created symbols classify() called C2 while they
    # were structurally unreachable -- 12 of them, zero trades in 6.5 years.
    # Removed 2026-08-23; verified absent from the live universe first, since
    # dropping a name from this set silently reclassifies it to C3.
    "AMDL", "TSLL", "NVDU",
    # Volatility: none. UVIX sits in master_universe as 3_SECTOR_bull_only,
    # which run_v1_at_k never assembles, so listing it here only created a
    # symbol that classify() called C2 while it could never trade. Measured
    # standalone and net of costs the whole complex loses: UVIX -0.919 over
    # 189 trades (36% win, positive in 1 of 5 years), UVXY -5.530 over 358
    # (1 of 7), SVIX -0.354, SVXY -0.189. Volatility products gap and then
    # revert against the breakout, which is the opposite of what ORB needs.
    # Energy (XOP)
    "GUSH",
})

# Explicit drops — present in master_universe.csv but excluded from v1 universe.
# See V1_SYSTEM_REPORT §5 for rationale.
EXPLICIT_DROPS: frozenset[str] = frozenset({
    "CWEB", "CHAU", "BNKU", "DFEN", "DPST",
    "NAIL", "RETL", "DRN", "MIDU", "NRGU",
    "BTFX",  # untradable on IB (no security definition); BTC covered by 5 other C1 vehicles
    "UBR",   # ProShares Ultra MSCI Brazil — delisted; IB HMDS returns error 162 (no data)
})

# Force-include — added to universe even if missing from master (synthetic rows).
# Hardcoded metadata so this file has no external deps.
#: BOIL, KOLD and GUSH have no master_universe.csv row, so their margin can
#: only come from here. Without it they fell back to a conservative 1.0 until
#: prewarm_margin measured them each session -- roughly double the truth, in
#: the window that matters most (OPEN_ISSUES L6). Rates below measured against
#: IB 2026-08-24 via whatIfOrder. Live still pulls fresh rates every session
#: and those take precedence; these only make the fallback honest.
FORCE_INCLUDE: dict[str, dict] = {
    "AMDL": {"underlying": "AMD",  "leverage": 2, "inverse": False},
    "TSLL": {"underlying": "TSLA", "leverage": 2, "inverse": False},
    "NVDU": {"underlying": "NVDA", "leverage": 2, "inverse": False},
    "KOLD": {"underlying": "UNG",  "leverage": 2, "inverse": True,
             "margin_long": 0.521, "margin_short": 0.625},
    "BOIL": {"underlying": "UNG",  "leverage": 2, "inverse": False,
             "margin_long": 0.529, "margin_short": 0.635},
    "UXRP": {"underlying": "XRP",  "leverage": 2, "inverse": False},
    "XRPT": {"underlying": "XRP",  "leverage": 2, "inverse": False},
    "GUSH": {"underlying": "XOP",  "leverage": 2, "inverse": False,
             "margin_long": 0.526, "margin_short": 0.631},
}

# Per-instrument direction filter — only trade gap direction matching the sign.
# LABU only after gap-ups, LABD only after gap-downs. See memory:
# project_k1_ps_filters_2026_05_12 and downstream LABD verdict.
DIRECTION_FILTERS: dict[str, int] = {
    "LABU": +1,
    "LABD": -1,
}


# ───────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Instrument:
    symbol: str
    underlying: str
    leverage: int
    inverse: bool
    #: Measured IB initial-margin rate (init_margin / notional), per side.
    #: None where master_universe.csv carries no measurement. Optional so that
    #: golden-session fixtures written before these columns existed still
    #: deserialise through Instrument(**recorded).
    margin_long: Optional[float] = None
    margin_short: Optional[float] = None

    def margin_rate(self, etf_dir: int) -> Optional[float]:
        """Rate for the side this instrument would actually be traded on.

        Sides differ, and not by a constant: SQQQ is 0.79 long / 0.95 short
        while TSLL is 1.00 long / 0.72 short.
        """
        return self.margin_long if etf_dir >= 0 else self.margin_short


@dataclass(frozen=True)
class Candidate:
    symbol: str
    underlying: str
    etf_dir: int        # +1 = gap up, -1 = gap down
    prior_close: float  # for skip-cheap ranking


@dataclass
class SessionPlan:
    candidates: list[str]                  # symbols qualified to trade today
    regime: str                            # "quiet" | "active" | "flood"
    n_uls: int                             # n distinct underlyings in candidates
    cap_factor: float                      # 0 to 1, applied to every multiplier
    multipliers: dict[str, float]          # symbol -> final position multiplier
    diagnostics: dict = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────────
# LOADING
# ───────────────────────────────────────────────────────────────────────────────

def load_master(master_csv: str | Path) -> dict[str, Instrument]:
    """Load ETF catalog from master_universe.csv into {symbol: Instrument}.

    Filters out rows missing underlying. Adds FORCE_INCLUDE synthetic entries.
    """
    df = pd.read_csv(master_csv)
    instruments: dict[str, Instrument] = {}
    for _, row in df.iterrows():
        sym = str(row["etf"]).strip()
        ul  = str(row.get("underlying", "")).strip()
        if not sym or not ul or ul.lower() == "nan":
            continue
        def _rate(col):
            v = row.get(col)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if v > 0 else None

        instruments[sym] = Instrument(
            symbol=sym,
            underlying=ul,
            leverage=int(row.get("leverage", 2) or 2),
            inverse=str(row.get("direction", "bull")).strip().lower() == "bear",
            margin_long=_rate("margin_init_long"),
            margin_short=_rate("margin_init_short"),
        )
    # Add FORCE_INCLUDE entries if missing
    for sym, info in FORCE_INCLUDE.items():
        if sym in instruments:
            continue
        instruments[sym] = Instrument(
            symbol=sym, underlying=info["underlying"],
            leverage=info["leverage"], inverse=info["inverse"],
            margin_long=info.get("margin_long"),
            margin_short=info.get("margin_short"),
        )
    return instruments


def load_sigmas(sigma_csv: str | Path) -> dict[str, float]:
    """Load per-underlying σ from sigma_master.csv → {UL: sigma}."""
    df = pd.read_csv(sigma_csv)
    return dict(zip(df["underlying"].astype(str), df["sigma"].astype(float)))


# ───────────────────────────────────────────────────────────────────────────────
# UNIVERSE — applied once at startup (or on universe refresh)
# ───────────────────────────────────────────────────────────────────────────────

def build_universe(
    instruments: dict[str, Instrument],
    sigmas: dict[str, float],
    master_csv: str | Path,
) -> list[str]:
    """Apply class filter + EXPLICIT_DROPS + FORCE_INCLUDE + σ-coverage gate.

    Returns the sorted list of symbols that can be traded by v1.

    Class filter: keep 1_BROAD, 2_SECTOR_pair, 4_CRYPTO, 5_SINGLE_STOCK with
    binary_event=YES, force_include. Class 3_SECTOR_bull_only is dropped wholesale
    (paired ETFs cover the same exposures with better symmetry).
    """
    df = pd.read_csv(master_csv)
    keep_classes = {"1_BROAD", "2_SECTOR_pair", "4_CRYPTO", "force_include"}

    keep_syms: set[str] = set()
    for _, row in df.iterrows():
        sym = str(row["etf"]).strip()
        cls = str(row.get("class", "")).strip()
        be = str(row.get("binary_event", "")).strip().upper() == "YES"
        if cls in keep_classes:
            keep_syms.add(sym)
        elif be:
            keep_syms.add(sym)

    keep_syms |= set(FORCE_INCLUDE.keys())
    keep_syms -= set(EXPLICIT_DROPS)
    keep_syms &= set(instruments.keys())

    # σ-coverage gate: drop any sym whose UL has no σ
    kept = sorted(s for s in keep_syms if instruments[s].underlying in sigmas)
    return kept


# ───────────────────────────────────────────────────────────────────────────────
# CANDIDATE COMPUTATION — once per session, just before 9:30 ET
# ───────────────────────────────────────────────────────────────────────────────

def passes_ps_filter(
    ul_gap: float,
    prior_two_closes: tuple[float, float],
    sigma: float,
    k: float = K_SIGMA,
) -> bool:
    """Prior-session direction-adjusted filter.

    Block the trade if the prior session moved strongly *in the gap direction*
    (positive momentum into a positive gap → mean reversion edge erodes).
    Returns True if the trade should be KEPT.
    """
    c1, c2 = prior_two_closes
    if c2 <= 0:
        return True
    prior_ret = (c1 - c2) / c2
    gap_sign = 1 if ul_gap > 0 else -1
    dir_adj = prior_ret * gap_sign
    return dir_adj <= sigma * k


def compute_candidates(
    universe: list[str],
    instruments: dict[str, Instrument],
    sigmas: dict[str, float],
    overnight_gaps: dict[str, float],
    prior_two_closes: dict[str, tuple[float, float]],
    prior_etf_close: dict[str, float],
    max_margin_rate: Optional[float] = MAX_MARGIN_RATE,
) -> list[Candidate]:
    """Return today's qualified candidates after all filters and skip-cheap-top-2 pruning.

    Steps (per spec):
        1. Per UL: |gap| ≥ 2% AND passes_ps_filter(k=1.00 σ)
        2. Per instrument: drop if direction filter set and gap sign mismatches
        3. Per (date, UL) group: drop siblings the account cannot fill, then
           skip-cheap-top-2 — keep up to the 2 most-expensive ETFs by prior
           close. (Single-instrument groups are kept.)

    max_margin_rate — siblings whose measured rate for the side they would be
        traded on exceeds this are removed BEFORE the price sort, unless that
        would empty the group. Shorting ETHU costs 4.09x notional: at weight
        3.0 on a $32k account it needs ~$39k against ~$32k of available funds,
        so skip-cheap was selecting a trade that could never fill while a
        fillable sibling sat next to it. Pass None to disable.
    """
    # Step 1: which ULs qualify today
    ul_qualifying: dict[str, float] = {}
    for ul, gap in overnight_gaps.items():
        if abs(gap) < GAP_THRESHOLD:
            continue
        sigma = sigmas.get(ul)
        if sigma is not None:
            pc = prior_two_closes.get(ul)
            if pc is not None and not passes_ps_filter(gap, pc, sigma):
                continue
        ul_qualifying[ul] = gap

    # Step 2: direction filter per ETF
    cands_by_ul: dict[str, list[Candidate]] = defaultdict(list)
    for sym in universe:
        inst = instruments.get(sym)
        if inst is None:
            continue
        gap = ul_qualifying.get(inst.underlying)
        if gap is None:
            continue
        ul_sign = 1 if gap > 0 else -1
        etf_dir = -ul_sign if inst.inverse else ul_sign
        if sym in DIRECTION_FILTERS and DIRECTION_FILTERS[sym] != etf_dir:
            continue
        cands_by_ul[inst.underlying].append(Candidate(
            symbol=sym, underlying=inst.underlying,
            etf_dir=etf_dir, prior_close=prior_etf_close.get(sym, 0.0),
        ))

    # Step 3: skip-cheap per UL group — always keep exactly 1 (most expensive by prior close)
    final: list[Candidate] = []
    for ul, group in cands_by_ul.items():
        if max_margin_rate is not None and len(group) > 1:
            fillable = [
                c for c in group
                if (instruments[c.symbol].margin_rate(c.etf_dir) or 0.0) <= max_margin_rate
            ]
            # Only narrow the field when something survives. Two impossible
            # trades is not a reason to invent a preference between them.
            if fillable:
                group = fillable
        if len(group) <= 1:
            final.extend(group)
            continue
        group_sorted = sorted(group, key=lambda c: -c.prior_close)
        keep_n = 1
        final.extend(group_sorted[:keep_n])
    return final


# ───────────────────────────────────────────────────────────────────────────────
# REGIME, CLASSIFICATION, SIZING
# ───────────────────────────────────────────────────────────────────────────────

def assign_regime(n_uls: int) -> str:
    if n_uls >= FLOOD_MIN:  return "flood"
    if n_uls >= ACTIVE_MIN: return "active"
    return "quiet"


#: Underlyings that share ONE sizing allotment. GDX and GDXJ are a single
#: exposure -- 0.935 daily-P&L correlation over 115 shared days, 83% overlap,
#: and their five ETFs run 0.88-0.96 among themselves including bull against
#: bear, because this strategy trades the gap DIRECTION. Left as two
#: underlyings, one gold move opened two full-size positions.
#:
#: Deliberately not generalised. Correlation chains: at a 0.60 threshold with
#: single linkage, {AMD, SOXX, XLK, QQQ, XLC, FXI, EEM} collapses into one
#: seven-underlying cluster. Overlap matters as much as correlation, too --
#: AMD/SOXX correlate 0.812 but co-trade on only 32% of days, so bundling them
#: would almost never bind. GDX/GDXJ is an exception on every axis.
SHARED_ALLOTMENT: dict[str, str] = {"GDXJ": "GDX"}


def allotment_group(underlying: str) -> str:
    """The sizing group an underlying belongs to (itself, unless shared)."""
    return SHARED_ALLOTMENT.get(underlying, underlying)


def _group_counts(candidates) -> dict[str, int]:
    """How many candidates fall in each allotment group."""
    counts: dict[str, int] = {}
    for c in candidates or ():
        g = allotment_group(getattr(c, "underlying", getattr(c, "symbol", "")))
        counts[g] = counts.get(g, 0) + 1
    return counts


def _share_divisor(sym: str, candidates) -> int:
    """How many ways this symbol's allotment is split today."""
    if not candidates:
        return 1
    for c in candidates:
        if getattr(c, "symbol", None) == sym:
            g = allotment_group(getattr(c, "underlying", sym))
            return max(1, _group_counts(candidates).get(g, 1))
    return 1


def classify(sym: str) -> str:
    if sym in CLASS_1_SYMS: return "C1"
    if sym in CLASS_2_SYMS: return "C2"
    return "C3"


def compute_cap_factor(
    candidates: list[Candidate],
    regime: str,
    cap_units: float = CAP_UNITS,
) -> float:
    """Scale-down factor when expected total exposure exceeds the cap.

    For each candidate, look up its (class, regime) weight and sum. If the sum
    exceeds cap_units, scale everything by cap_units / sum.
    """
    # A shared allotment group contributes one position's worth, not one per
    # member -- otherwise exposure is overstated and the cap binds too early.
    counts = _group_counts(candidates)
    exp_mult = sum(
        WEIGHTS.get((classify(c.symbol), regime), 0.0)
        / max(1, counts.get(
            allotment_group(getattr(c, "underlying", c.symbol)), 1))
        for c in candidates
    )
    if exp_mult <= 0:
        return 0.0
    return min(1.0, cap_units / exp_mult)


def position_multiplier(sym: str, regime: str, cap_factor: float,
                        candidates=None) -> float:
    """Final per-symbol position multiplier (weight x cap_factor).

    When `candidates` is supplied and this symbol's underlying shares an
    allotment with another candidate that day (see SHARED_ALLOTMENT), the
    weight is divided among them, so the group consumes one position's worth of
    margin rather than one each. Omitting `candidates` preserves the old
    behaviour for callers that do not have the day's set to hand.
    """
    base = WEIGHTS.get((classify(sym), regime), 0.0) * cap_factor
    return base / _share_divisor(sym, candidates)


# ───────────────────────────────────────────────────────────────────────────────
# SESSION ENTRYPOINT — call once at ~9:30 ET
# ───────────────────────────────────────────────────────────────────────────────

def plan_session(
    universe: list[str],
    instruments: dict[str, Instrument],
    sigmas: dict[str, float],
    overnight_gaps: dict[str, float],
    prior_two_closes: dict[str, tuple[float, float]],
    prior_etf_close: dict[str, float],
) -> SessionPlan:
    """Build the day's trading plan from market-open state.

    Live trader calls this once at 9:30 ET, then watches for ORB breakouts
    among `plan.candidates` and sizes each fire by `plan.multipliers[sym]`.
    """
    cands = compute_candidates(
        universe, instruments, sigmas,
        overnight_gaps, prior_two_closes, prior_etf_close,
    )
    n_uls = len({c.underlying for c in cands})
    regime = assign_regime(n_uls)
    cap_factor = compute_cap_factor(cands, regime)

    # `cands` must be passed: it is what lets a shared allotment group split one
    # position's worth between its members (SHARED_ALLOTMENT). Without it the
    # rule silently does nothing here while still passing its unit tests.
    multipliers = {
        c.symbol: position_multiplier(c.symbol, regime, cap_factor, cands)
        for c in cands
    }
    # Drop candidates with zero multiplier (e.g. C2 on flood days)
    candidates = [c.symbol for c in cands if multipliers.get(c.symbol, 0.0) > 0.0]
    multipliers = {s: m for s, m in multipliers.items() if m > 0.0}

    return SessionPlan(
        candidates=candidates,
        regime=regime,
        n_uls=n_uls,
        cap_factor=cap_factor,
        multipliers=multipliers,
        diagnostics={
            "raw_candidate_count": len(cands),
            "exp_mult_pre_cap": sum(
                WEIGHTS.get((classify(c.symbol), regime), 0.0) for c in cands
            ),
        },
    )


# ───────────────────────────────────────────────────────────────────────────────
# Convenience CLI — verify the module loads and prints the universe
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    master_csv = root / "master_universe.csv"
    sigma_csv  = root / "sigma_master.csv"

    instruments = load_master(master_csv)
    sigmas = load_sigmas(sigma_csv)
    universe = build_universe(instruments, sigmas, master_csv)

    print(f"Loaded {len(instruments)} instruments from {master_csv}")
    print(f"Loaded {len(sigmas)} sigmas from {sigma_csv}")
    print(f"Working universe: {len(universe)} symbols\n")
    by_cls = defaultdict(list)
    for s in universe:
        by_cls[classify(s)].append(s)
    for cls in ("C1", "C2", "C3"):
        syms = by_cls.get(cls, [])
        print(f"  {cls} ({len(syms):>3}): {', '.join(sorted(syms))}")
