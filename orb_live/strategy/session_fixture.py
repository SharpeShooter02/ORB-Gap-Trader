"""Golden-session fixtures — the backtest/live synchronisation record.

Every live session writes one JSON file containing the complete input tuple
handed to ``plan_session()`` plus the ``SessionPlan`` that came back. The
backtest repo replays the recorded inputs through the *same* imported
function and asserts an identical plan.

Why this and not a code-level comparison: because backtest and live share
one implementation of ``plan_session`` (BacktestingGaps installs this
package with ``pip install -e ../orb-live-trading --no-deps``), the function
cannot drift. What *can* drift is the inputs — overnight gap reconstruction,
bar sources, prior-close selection. A fixture captures exactly that surface.

Writing a fixture must never be able to break a trading session: every entry
point here swallows its own exceptions and reports success as a bool.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from orb_live.strategy import v1_strategy as v1

SCHEMA_VERSION = 2

#: Schema 1 fixtures recorded only the plan_session() input tuple. Schema 2
#: adds the raw pre-scan inputs, so a replay can start above the ETF→UL
#: leverage conversion instead of below it. Both still load; only schema 2
#: can have its gap scan replayed.
SUPPORTED_SCHEMAS = (1, 2)

#: Daily rows kept per symbol. compute_gap needs the last close before the
#: trade date and the PS filter needs the last two, so three is one spare.
DAILY_TAIL_ROWS = 3

#: Default location, relative to the repo root. The backtest reads from here.
DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "sessions"


def _profile_snapshot() -> dict[str, Any]:
    """The profile constants in force when the plan was made.

    Recorded so a replay can tell "the inputs changed" apart from "somebody
    retuned the strategy" — the two produce identical symptoms otherwise.
    """
    return {
        "k_sigma": v1.K_SIGMA,
        "cap_units": v1.CAP_UNITS,
        "gap_threshold": v1.GAP_THRESHOLD,
        "active_min": v1.ACTIVE_MIN,
        "flood_min": v1.FLOOD_MIN,
        # dict keys must be strings in JSON; "C2|flood" round-trips cleanly.
        "weights": {f"{cls}|{reg}": w for (cls, reg), w in v1.WEIGHTS.items()},
        "explicit_drops": sorted(v1.EXPLICIT_DROPS),
        "direction_filters": dict(v1.DIRECTION_FILTERS),
        # Sizing inputs that are not in `weights` but change multipliers just
        # as surely. Without these a deliberate retune reads as unexplained
        # drift: moving the gold miners to C3 and giving GDX/GDXJ a shared
        # allotment made 2026-08-19 replay at 1.5 where it recorded 3.0, and
        # nothing in the snapshot could account for it.
        "shared_allotment": dict(v1.SHARED_ALLOTMENT),
        "class_1": sorted(v1.CLASS_1_SYMS),
        "class_2": sorted(v1.CLASS_2_SYMS),
    }


def build_fixture(
    trade_date: date,
    *,
    universe: list[str],
    instruments: Mapping[str, v1.Instrument],
    sigmas: Mapping[str, float],
    overnight_gaps: Mapping[str, float],
    prior_two_closes: Mapping[str, tuple[float, float]],
    prior_etf_close: Mapping[str, float],
    plan: v1.SessionPlan,
    raw: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the fixture payload. Pure — no I/O, safe to unit test.

    raw — the pre-scan inputs from ``serialise_raw_inputs``. Optional only so
    that a caller without them still writes a usable schema-1-shaped fixture;
    omitting it costs the replay its coverage of the gap scan.
    """
    return {
        "schema": SCHEMA_VERSION,
        "trade_date": trade_date.isoformat(),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "profile": _profile_snapshot(),
        "inputs": {
            # Insertion order is preserved deliberately, NOT sorted: universe
            # order determines candidate order out of compute_candidates, so
            # sorting here would make a replay disagree with the live plan.
            # Fixtures are a fidelity record first and a readable diff second.
            "universe": list(universe),
            "instruments": {s: asdict(inst) for s, inst in instruments.items()},
            "sigmas": {k: float(v) for k, v in sigmas.items()},
            "overnight_gaps": {k: float(v) for k, v in overnight_gaps.items()},
            "prior_two_closes": {
                k: [float(v[0]), float(v[1])] for k, v in prior_two_closes.items()
            },
            "prior_etf_close": {k: float(v) for k, v in prior_etf_close.items()},
            # Everything above is derived from this. Recorded so a replay can
            # re-derive it rather than trusting it.
            "raw": dict(raw) if raw is not None else None,
        },
        "plan": {
            "candidates": list(plan.candidates),
            "regime": plan.regime,
            "n_uls": plan.n_uls,
            "cap_factor": float(plan.cap_factor),
            "multipliers": {k: float(v) for k, v in plan.multipliers.items()},
        },
    }


def write_fixture(
    trade_date: date,
    *,
    universe: list[str],
    instruments: Mapping[str, v1.Instrument],
    sigmas: Mapping[str, float],
    overnight_gaps: Mapping[str, float],
    prior_two_closes: Mapping[str, tuple[float, float]],
    prior_etf_close: Mapping[str, float],
    plan: v1.SessionPlan,
    raw: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[Path] = None,
    log: Any = None,
) -> bool:
    """Write one session fixture. Returns True on success, never raises.

    The write is atomic (temp file + replace) so a crash mid-write cannot
    leave a truncated fixture that the replay test would read as a real
    disagreement.
    """
    try:
        payload = build_fixture(
            trade_date,
            universe=universe,
            instruments=instruments,
            sigmas=sigmas,
            overnight_gaps=overnight_gaps,
            prior_two_closes=prior_two_closes,
            prior_etf_close=prior_etf_close,
            plan=plan,
            raw=raw,
        )
        target_dir = Path(out_dir) if out_dir is not None else DEFAULT_FIXTURE_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{trade_date.isoformat()}.json"

        fd, tmp_name = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=False)
                fh.write("\n")
            Path(tmp_name).replace(target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

        if log:
            log.info("session_fixture_written", path=str(target),
                     n_candidates=len(plan.candidates), regime=plan.regime)
        return True
    except Exception as exc:  # never break a trading session over a fixture
        if log:
            log.warning("session_fixture_failed", error=repr(exc))
        return False


def load_fixture(path: str | Path) -> dict[str, Any]:
    """Read a fixture back. Used by the backtest replay test."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(
            f"{path}: fixture schema {payload.get('schema')} not in {SUPPORTED_SCHEMAS}"
        )
    return payload


def replay(payload: Mapping[str, Any]) -> v1.SessionPlan:
    """Re-run plan_session on a fixture's recorded inputs.

    Lives here rather than in the backtest so that the deserialisation of the
    input tuple stays next to its serialisation — the two must agree, and
    splitting them across repos is how they stop agreeing.
    """
    inp = payload["inputs"]
    instruments = {
        s: v1.Instrument(**d) for s, d in inp["instruments"].items()
    }
    return v1.plan_session(
        universe=list(inp["universe"]),
        instruments=instruments,
        sigmas=dict(inp["sigmas"]),
        overnight_gaps=dict(inp["overnight_gaps"]),
        prior_two_closes={k: (v[0], v[1]) for k, v in inp["prior_two_closes"].items()},
        prior_etf_close=dict(inp["prior_etf_close"]),
    )


def diff_profile(recorded: Mapping[str, Any]) -> list[str]:
    """Profile constants that have changed since a fixture was recorded.

    A plan can stop reproducing for two very different reasons: the code that
    derives it drifted, or somebody deliberately retuned the strategy. Only the
    first is a bug. Comparing the recorded snapshot against the current profile
    tells them apart, which is the whole reason the snapshot is stored.
    """
    cur = _profile_snapshot()
    out: list[str] = []
    for key in sorted(set(recorded) | set(cur)):
        r, c = recorded.get(key), cur.get(key)
        if r == c:
            continue
        if isinstance(r, list) and isinstance(c, list):
            gone, new = sorted(set(r) - set(c)), sorted(set(c) - set(r))
            bits = []
            if gone:
                bits.append(f"removed {gone}")
            if new:
                bits.append(f"added {new}")
            out.append(f"{key}: " + ", ".join(bits))
        else:
            out.append(f"{key}: recorded={r} current={c}")
    return out


def diff_plan(recorded: Mapping[str, Any], actual: v1.SessionPlan) -> list[str]:
    """Human-readable differences between a recorded plan and a replayed one.

    Empty list means the session reproduces exactly.
    """
    out: list[str] = []
    if list(recorded["candidates"]) != list(actual.candidates):
        missing = sorted(set(recorded["candidates"]) - set(actual.candidates))
        extra = sorted(set(actual.candidates) - set(recorded["candidates"]))
        if missing:
            out.append(f"candidates missing on replay: {missing}")
        if extra:
            out.append(f"candidates only on replay: {extra}")
        if not missing and not extra:
            out.append("candidate ordering differs")
    if recorded["regime"] != actual.regime:
        out.append(f"regime: recorded={recorded['regime']} replay={actual.regime}")
    if recorded["n_uls"] != actual.n_uls:
        out.append(f"n_uls: recorded={recorded['n_uls']} replay={actual.n_uls}")
    if abs(float(recorded["cap_factor"]) - actual.cap_factor) > 1e-9:
        out.append(
            f"cap_factor: recorded={recorded['cap_factor']} replay={actual.cap_factor}"
        )
    for sym in sorted(set(recorded["multipliers"]) | set(actual.multipliers)):
        r = recorded["multipliers"].get(sym)
        a = actual.multipliers.get(sym)
        if r is None or a is None or abs(float(r) - float(a)) > 1e-9:
            out.append(f"multiplier[{sym}]: recorded={r} replay={a}")
    return out


# ── Raw pre-scan inputs (schema 2) ────────────────────────────────────────────
#
# The plan_session() input tuple is a *derived* value: by the time it exists,
# the ETF gap has been computed, tested against leverage × GAP_THRESHOLD,
# divided by leverage, collided against sibling ETFs on the same underlying,
# and run through the PS filter. Recording only that tuple means a replay
# re-verifies arithmetic downstream of every decision worth checking.
#
# These helpers record what went *in* to that layer instead. Only the daily
# tail is kept — three rows per symbol, ~60 symbols — because compute_gap and
# the PS filter never look further back than two sessions.


def _daily_tail(df: Any, rows: int = DAILY_TAIL_ROWS) -> list[list[Any]]:
    """[[iso_date, close], ...] for the last `rows` bars. [] if unusable."""
    if df is None or getattr(df, "empty", True):
        return []
    tail = df.tail(rows)
    out: list[list[Any]] = []
    for _, r in tail.iterrows():
        try:
            out.append([str(r["date"])[:10], float(r["close"])])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def serialise_raw_inputs(
    *,
    symbols: list[str],
    ref_prices: Mapping[str, Any],
    etf_daily: Mapping[str, Any],
    ul_daily: Mapping[str, Any],
    prior_session_filters: Mapping[str, tuple],
) -> dict[str, Any]:
    """Pack the gap scan's inputs for the fixture. Pure.

    prior_session_filters is recorded because it is derived from sigma × k at
    config-build time, not read from the profile — a replay that rebuilt it
    from today's sigmas would silently test a different filter than the one
    that ran.
    """
    return {
        "symbols": list(symbols),
        "ref_prices": {
            s: (None if p is None else float(p)) for s, p in ref_prices.items()
        },
        "etf_daily": {s: _daily_tail(df) for s, df in etf_daily.items()},
        "ul_daily": {u: _daily_tail(df) for u, df in ul_daily.items()},
        "prior_session_filters": {
            s: list(spec) for s, spec in prior_session_filters.items()
        },
    }


def _frame(rows: list[list[Any]]):
    """Rebuild a minimal daily DataFrame from a recorded tail."""
    import pandas as pd

    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    return pd.DataFrame(
        {
            "date": pd.to_datetime([r[0] for r in rows]),
            "close": [float(r[1]) for r in rows],
        }
    )


class _ReplayConfig:
    """The two config attributes scan_gaps duck-types off."""

    def __init__(self, direction_filters: Mapping[str, int],
                 prior_session_filters: Mapping[str, tuple]):
        self.direction_filters = dict(direction_filters)
        self.prior_session_filters = {
            s: tuple(spec) for s, spec in prior_session_filters.items()
        }


def has_raw_inputs(payload: Mapping[str, Any]) -> bool:
    """True if this fixture can have its gap scan replayed."""
    return bool(payload.get("inputs", {}).get("raw"))


def replay_scan(payload: Mapping[str, Any]):
    """Re-run the gap scan from the recorded raw inputs.

    Returns a GapScanResult. Raises ValueError on a schema-1 fixture, which
    has nothing to replay — callers should check has_raw_inputs() first.
    """
    from datetime import date as _date

    from orb_live.signals.gap_scan import scan_gaps

    inp = payload["inputs"]
    raw = inp.get("raw")
    if not raw:
        raise ValueError("fixture has no raw inputs; nothing to replay")

    instruments = {s: v1.Instrument(**d) for s, d in inp["instruments"].items()}
    cfg = _ReplayConfig(
        direction_filters=payload["profile"]["direction_filters"],
        prior_session_filters=raw["prior_session_filters"],
    )
    return scan_gaps(
        _date.fromisoformat(payload["trade_date"]),
        symbols=list(raw["symbols"]),
        instruments=instruments,
        ref_prices=dict(raw["ref_prices"]),
        etf_daily={s: _frame(rows) for s, rows in raw["etf_daily"].items()},
        ul_daily={u: _frame(rows) for u, rows in raw["ul_daily"].items()},
        config=cfg,
    )


def diff_scan(payload: Mapping[str, Any], result: Any, tol: float = 1e-9) -> list[str]:
    """Differences between the recorded plan inputs and a re-derived scan.

    A failure here means the layer *below* plan_session moved: a bar source,
    the prior-close pick, the leverage table, or the PS filter. That is the
    class of drift the schema-1 fixture could not see.
    """
    inp = payload["inputs"]
    out: list[str] = []

    def _cmp_map(name: str, recorded: Mapping[str, Any], actual: Mapping[str, Any]):
        for key in sorted(set(recorded) | set(actual)):
            r, a = recorded.get(key), actual.get(key)
            if r is None or a is None:
                out.append(f"{name}[{key}]: recorded={r} replay={a}")
            elif abs(float(r) - float(a)) > tol:
                out.append(f"{name}[{key}]: recorded={r:.6f} replay={a:.6f}")

    _cmp_map("overnight_gaps", inp["overnight_gaps"], result.overnight_gaps)
    _cmp_map("prior_etf_close", inp["prior_etf_close"], result.prior_etf_close)

    rec_closes = {k: tuple(v) for k, v in inp["prior_two_closes"].items()}
    for key in sorted(set(rec_closes) | set(result.prior_two_closes)):
        r, a = rec_closes.get(key), result.prior_two_closes.get(key)
        if r is None or a is None or any(abs(x - y) > tol for x, y in zip(r, a)):
            out.append(f"prior_two_closes[{key}]: recorded={r} replay={a}")

    if list(inp["universe"]) != result.qualified_symbols:
        missing = sorted(set(inp["universe"]) - set(result.qualified_symbols))
        extra = sorted(set(result.qualified_symbols) - set(inp["universe"]))
        if missing:
            out.append(f"qualified missing on replay: {missing}")
        if extra:
            out.append(f"qualified only on replay: {extra}")
        if not missing and not extra:
            out.append("qualified-symbol ordering differs")
    return out
