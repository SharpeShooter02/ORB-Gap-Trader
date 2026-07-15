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
    # Gold miners
    "NUGT", "DUST", "GDXU", "JNUG", "JDST",
    # Single-stock leveraged ETFs
    "AMDL", "AMDG", "AMUU",
    "TSLL", "TSL", "TSLG", "TSLI", "TSLR", "TSLT", "TSLW",
    "NVDU", "NVDG", "NVDL", "NVDW", "NVDX",
    # Volatility
    "UVIX",
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
FORCE_INCLUDE: dict[str, dict] = {
    "AMDL": {"underlying": "AMD",  "leverage": 2, "inverse": False},
    "TSLL": {"underlying": "TSLA", "leverage": 2, "inverse": False},
    "NVDU": {"underlying": "NVDA", "leverage": 2, "inverse": False},
    "KOLD": {"underlying": "UNG",  "leverage": 2, "inverse": True},
    "BOIL": {"underlying": "UNG",  "leverage": 2, "inverse": False},
    "UXRP": {"underlying": "XRP",  "leverage": 2, "inverse": False},
    "XRPT": {"underlying": "XRP",  "leverage": 2, "inverse": False},
    "GUSH": {"underlying": "XOP",  "leverage": 2, "inverse": False},
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
        instruments[sym] = Instrument(
            symbol=sym,
            underlying=ul,
            leverage=int(row.get("leverage", 2) or 2),
            inverse=str(row.get("direction", "bull")).strip().lower() == "bear",
        )
    # Add FORCE_INCLUDE entries if missing
    for sym, info in FORCE_INCLUDE.items():
        if sym in instruments:
            continue
        instruments[sym] = Instrument(
            symbol=sym, underlying=info["underlying"],
            leverage=info["leverage"], inverse=info["inverse"],
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
) -> list[Candidate]:
    """Return today's qualified candidates after all filters and skip-cheap-top-2 pruning.

    Steps (per spec):
        1. Per UL: |gap| ≥ 2% AND passes_ps_filter(k=1.00 σ)
        2. Per instrument: drop if direction filter set and gap sign mismatches
        3. Per (date, UL) group: skip-cheap-top-2 — keep up to the 2 most-expensive
           ETFs by prior close. (Single-instrument groups are kept.)
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

    # Step 3: skip-cheap-top-2 per UL group
    # Backtest rule (skip_cheap_by_class.py :: skip_cheap_then_top2_when_3plus):
    #   N==1 → keep 1; N==2 → drop cheaper, keep 1; N>=3 → keep top-2 by prior close
    final: list[Candidate] = []
    for ul, group in cands_by_ul.items():
        if len(group) <= 1:
            final.extend(group)
            continue
        group_sorted = sorted(group, key=lambda c: -c.prior_close)
        keep_n = 1 if len(group) == 2 else 2
        final.extend(group_sorted[:keep_n])
    return final


# ───────────────────────────────────────────────────────────────────────────────
# REGIME, CLASSIFICATION, SIZING
# ───────────────────────────────────────────────────────────────────────────────

def assign_regime(n_uls: int) -> str:
    if n_uls >= FLOOD_MIN:  return "flood"
    if n_uls >= ACTIVE_MIN: return "active"
    return "quiet"


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
    exp_mult = sum(WEIGHTS.get((classify(c.symbol), regime), 0.0) for c in candidates)
    if exp_mult <= 0:
        return 0.0
    return min(1.0, cap_units / exp_mult)


def position_multiplier(sym: str, regime: str, cap_factor: float) -> float:
    """Final per-symbol position multiplier (weight × cap_factor)."""
    return WEIGHTS.get((classify(sym), regime), 0.0) * cap_factor


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

    multipliers = {
        c.symbol: position_multiplier(c.symbol, regime, cap_factor)
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
