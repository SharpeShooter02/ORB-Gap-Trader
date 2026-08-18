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

SCHEMA_VERSION = 1

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
) -> dict[str, Any]:
    """Assemble the fixture payload. Pure — no I/O, safe to unit test."""
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
    if payload.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: fixture schema {payload.get('schema')} != {SCHEMA_VERSION}"
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
