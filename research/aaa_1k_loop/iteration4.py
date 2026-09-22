"""Iteration 0004, second half: a challenger for M2, under the prospective rules.

Diagnosis rounds 1-4 (``diagnosis4*.py``) established on fresh diagnostic
blocks that M2 -- the champion's long-horizon runaway on quantized streams --
is a self-confirming target-unfolding frame lock (H32-H34), not a TBPTT
defect (H20-H23), a learned loop gain (H24, H27) or single-step overshoot
(H29). This module fixes, *before any development identity is observed*,
everything the rest of the iteration may do.

The pilot's declared gaps (``docs/loop_protocol.md``), closed here as rules
------------------------------------------------------------------------------
1. **Finite, precommitted candidate budget.** Exactly :data:`CANDIDATES`, in
   that order. No candidate may be added after development is observed.
2. **No screen-failed candidate advances.** A screen verdict other than PASS
   stops that candidate. There is no "tradeoff challenger".
3. **Uncertainty-aware screens.** Non-inferiority is judged on the upper end
   of a crossed initialization-by-stream bootstrap interval, never on a point
   estimate; an interval that straddles the margin is not a PASS.
4. **Prevalidated attack instruments.** Every attack criterion uses an
   instrument already validated in this iteration or phase: the divergence
   definition (diagnosis rounds 1-4), the lock tracer (round 4, bitwise the
   champion) and the AAA-1K crossed bootstrap. No attack criterion uses a
   probe known to be fragile.
5. **Scratch identities for every informal look.** Shakedown runs use
   ``aaa1k-loop-0004/diagnostic/scratch``; nothing else is looked at outside a
   declared stage.
6. **One attack.** At most one candidate is attacked. If the attack rejects
   it the iteration ends REJECTED; the other candidate is not tried.
7. **Long-horizon stability is a primary gate** of screen, attack and
   confirmation alike (audit R-01).

Selection among screen passes is fixed in advance: c7 is preferred to c8,
because c7 keeps unfolding at genuine wall crossings (the reason ``AAA-120``
introduced it) and removes only the self-reference.

Confirmation decision (frozen at freeze time; also re-derived independently)
---------------------------------------------------------------------------
Every criterion is three-valued and resolves uncertainty conservatively:
an interval crossing a decision boundary is INCONCLUSIVE, never the sign of
its mean.

K1 long coarse stability (primary)
    per-cell divergence indicator, champion minus challenger, crossed
    bootstrap over initializations x (both long coarse conditions' streams).
    PASS: interval low > 0 and challenger divergence fraction <= 5%.
    FAIL: interval high <= 0 or challenger fraction > 20%.
K2 long non-coarse non-regression (primary), for long bouncing and long occlusion each
    PASS: relative-MAE interval high <= +2% and challenger divergences <= champion's.
    FAIL: relative-MAE interval low > +2% or challenger divergences > champion's + 2.
K3 standard families, for each of the six round-3 plan entries
    PASS: relative-MAE interval high <= +2%. FAIL: interval low > +2%.

PROMOTE iff every criterion PASSes; REJECT if any FAILs; otherwise INCONCLUSIVE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.aaa_1k.agents import PersistenceAgent
from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed
from research.aaa_1k.stats import crossed_paired_difference
from research.aaa_1k.streams import coarse_speed_stream, motion_compat_stream, occlusion_stream

from .arms import gru
from .develop import plan_streams
from .harness import Cell, arm_errors, run_cells
from .identities import block_seeds, find_block, require_claimed, require_usable
from .unfolding import LOCK_RUN, DeadReckoningUnfoldAgent, UnfoldTraceAgent, longest_runs

ITERATION_ID = "aaa1k-loop-0004"

DEVELOPMENT_BLOCK = "aaa1k-loop-0004/development/all"
ATTACK_ENV_BLOCK = "aaa1k-loop-0004/attack/env"
ATTACK_INIT_BLOCK = "aaa1k-loop-0004/attack/init"
CONFIRMATION_ENV_BLOCK = "aaa1k-loop-0004/confirmation/env"
CONFIRMATION_INIT_BLOCK = "aaa1k-loop-0004/confirmation/init"

LONG_STEPS = 1120
INITIALIZATIONS = 5
DIVERGENCE_FACTOR = 2.0
MARGIN = 0.02
MINIMUM_CHAMPION_DIVERGENCES = 5

DEVELOPMENT_PER_ENTRY = 8
DEVELOPMENT_LONG = 16
ATTACK_PER_ENTRY = 8
ATTACK_LONG = 16
ATTACK_NEARBY = 16
CONFIRMATION_PER_ENTRY = 16
CONFIRMATION_LONG = 16

PLAN_ENTRY_COUNT = 6
DEVELOPMENT_BLOCK_SIZE = DEVELOPMENT_PER_ENTRY * PLAN_ENTRY_COUNT + DEVELOPMENT_LONG
ATTACK_ENV_BLOCK_SIZE = ATTACK_PER_ENTRY * PLAN_ENTRY_COUNT + ATTACK_LONG + ATTACK_NEARBY
CONFIRMATION_ENV_COUNT = CONFIRMATION_PER_ENTRY * PLAN_ENTRY_COUNT + CONFIRMATION_LONG
CONFIRMATION_INIT_COUNT = INITIALIZATIONS

CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "id": "c7",
        "arm": "c7_unfold_dr",
        "change": (
            "choose the target-unfolding branch nearest the public dead-reckoning reference (input position "
            "plus the tracker's velocity estimate) instead of the learner's own raw prediction"
        ),
        "sentence": (
            "this change should remove M2 because diagnosis round 4 showed the runaway is a frame lock created "
            "by choosing the branch with the learner's own prediction (H32-H34)"
        ),
    },
    {
        "id": "c8",
        "arm": "c8_folded",
        "change": "train on the folded (observed) displacement: unfold_target=False, an existing public option",
        "sentence": (
            "this change should remove M2 because without unfolding there is no branch for the prediction to "
            "confirm (H34), at the cost of the AAA-120 wall-contact correction"
        ),
    },
)
CANDIDATE_ARMS = tuple(candidate["arm"] for candidate in CANDIDATES)
LEARNERS = ("gru", *CANDIDATE_ARMS)

LONG_CONDITIONS = ("long:coarse_no_switch", "long:coarse_switching", "long:bouncing", "long:occlusion")
LONG_COARSE = ("long:coarse_no_switch", "long:coarse_switching")
LONG_OTHER = ("long:bouncing", "long:occlusion")
NEARBY_CONDITIONS = ("near:quantum_0.004", "near:quantum_0.006", "near:slow_0.08", "near:slow_0.15")


def init_seeds(block: Sequence[int] | None, count: int) -> list[tuple[int, int]]:
    """``(index, model seed)``: the champion's own AAA-1K initializations, or a fresh block's seeds."""

    if block is None:
        return [(index, derive_seed("model_init", index)) for index in range(count)]
    if len(block) < count:
        raise ValueError("the initialization block holds too few seeds")
    return list(enumerate(block[:count]))


# ----------------------------------------------------------------------
# arms and streams
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Iteration4Arms:
    """The champion (traced, bitwise the champion), both candidates, persistence."""

    arms: tuple[str, ...] = LEARNERS

    def __call__(self, seed: int) -> Sequence[Any]:
        builders = {
            "gru": lambda: UnfoldTraceAgent(gru(seed), name="gru"),
            "c7_unfold_dr": lambda: DeadReckoningUnfoldAgent(gru(seed), name="c7_unfold_dr"),
            "c8_folded": lambda: UnfoldTraceAgent(gru(seed), name="c8_folded", unfold_target=False),
        }
        return [builders[name]() for name in self.arms] + [PersistenceAgent()]


def long_stream(condition: str, seed: int) -> Any:
    if condition == "long:coarse_no_switch":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=100_000)
    if condition == "long:coarse_switching":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=60)
    if condition == "long:bouncing":
        return motion_compat_stream("bouncing", seed, steps=LONG_STEPS, change_step=None)
    if condition == "long:occlusion":
        return occlusion_stream(seed, steps=LONG_STEPS)
    if condition == "near:quantum_0.004":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=100_000, quantum=0.004)
    if condition == "near:quantum_0.006":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=100_000, quantum=0.006)
    if condition == "near:slow_0.08":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=100_000, slow_speed=0.08)
    if condition == "near:slow_0.15":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=100_000, slow_speed=0.15)
    raise ValueError(condition)


def _cells(
    plan_seeds: Sequence[int],
    per_entry: int,
    long_seeds: Sequence[int],
    long_conditions: Sequence[str],
    inits: Sequence[tuple[int, int]],
) -> list[Cell]:
    streams: list[tuple[str, Any]] = [
        (f"plan:{entry}", stream) for entry, stream in plan_streams(list(plan_seeds), per_entry)
    ]
    streams.extend(
        (condition, long_stream(condition, seed)) for condition in long_conditions for seed in long_seeds
    )
    return [Cell(condition, index, seed, stream) for condition, stream in streams for index, seed in inits]


def development_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    block = require_usable(ledger, DEVELOPMENT_BLOCK, purpose="selection")
    if block["role"] != "development":
        raise ValueError("development must run on a development block")
    seeds = block_seeds(find_block(ledger, DEVELOPMENT_BLOCK))
    plan = seeds[: DEVELOPMENT_PER_ENTRY * PLAN_ENTRY_COUNT]
    long = seeds[DEVELOPMENT_PER_ENTRY * PLAN_ENTRY_COUNT :][:DEVELOPMENT_LONG]
    return _cells(plan, DEVELOPMENT_PER_ENTRY, long, LONG_CONDITIONS, init_seeds(None, INITIALIZATIONS))


def attack_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    env = require_usable(ledger, ATTACK_ENV_BLOCK, purpose="selection")
    init = require_usable(ledger, ATTACK_INIT_BLOCK, purpose="selection")
    if env["role"] != "attack" or init["role"] != "attack":
        raise ValueError("the attack must run on attack blocks")
    seeds = block_seeds(find_block(ledger, ATTACK_ENV_BLOCK))
    plan = seeds[: ATTACK_PER_ENTRY * PLAN_ENTRY_COUNT]
    rest = seeds[ATTACK_PER_ENTRY * PLAN_ENTRY_COUNT :]
    long, nearby = rest[:ATTACK_LONG], rest[ATTACK_LONG : ATTACK_LONG + ATTACK_NEARBY]
    inits = init_seeds(block_seeds(find_block(ledger, ATTACK_INIT_BLOCK)), INITIALIZATIONS)
    cells = _cells(plan, ATTACK_PER_ENTRY, long, LONG_CONDITIONS, inits)
    cells.extend(
        Cell(condition, index, seed, long_stream(condition, stream_seed))
        for condition in NEARBY_CONDITIONS
        for stream_seed in nearby
        for index, seed in inits
    )
    return cells


def confirmation_cells(ledger: Mapping[str, Any], *, observer: str) -> list[Cell]:
    require_claimed(ledger, CONFIRMATION_ENV_BLOCK, observer=observer)
    require_claimed(ledger, CONFIRMATION_INIT_BLOCK, observer=observer)
    seeds = block_seeds(find_block(ledger, CONFIRMATION_ENV_BLOCK))
    plan = seeds[: CONFIRMATION_PER_ENTRY * PLAN_ENTRY_COUNT]
    long = seeds[CONFIRMATION_PER_ENTRY * PLAN_ENTRY_COUNT :][:CONFIRMATION_LONG]
    inits = init_seeds(block_seeds(find_block(ledger, CONFIRMATION_INIT_BLOCK)), INITIALIZATIONS)
    return _cells(plan, CONFIRMATION_PER_ENTRY, long, LONG_CONDITIONS, inits)


# ----------------------------------------------------------------------
# reduction: primitives only
# ----------------------------------------------------------------------
def reduce_cell(_cell: Cell, result: RunResult, agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    persistence = float(np.mean(errors["persistence"]))
    arms: dict[str, Any] = {}
    for agent in agents:
        if agent.name == "persistence":
            continue
        mae = float(np.mean(errors[agent.name]))
        runs = longest_runs(agent.mirror_trace)
        locks = [start for start, length in runs if length >= LOCK_RUN]
        arms[agent.name] = {
            "mae": mae,
            "diverged": bool(mae > DIVERGENCE_FACTOR * persistence),
            "mirror_steps": int(sum(agent.mirror_trace)),
            "longest_mirror_run": max((length for _s, length in runs), default=0),
            "first_lock": locks[0] if locks else None,
        }
    return {"persistence_mae": persistence, "arms": arms}


def run(
    cells: Sequence[Cell], *, arms: tuple[str, ...] = LEARNERS, workers: int | None = None
) -> list[dict[str, Any]]:
    return run_cells(cells, Iteration4Arms(arms), reduce_cell, isolate_failures=True, workers=workers)


# ----------------------------------------------------------------------
# statistics shared by screen, attack and decision
# ----------------------------------------------------------------------
def _failed(record: Mapping[str, Any], arm: str) -> bool:
    return arm in record.get("failures", {})


def grid(records: Sequence[Mapping[str, Any]], arm: str, field: str) -> np.ndarray:
    """``[initialization, stream]`` matrix of one per-cell primitive; refuses a ragged crossing."""

    inits = sorted({r["init_index"] for r in records})
    streams = sorted({(r["condition"], r["stream_id"]) for r in records})
    table = {}
    for r in records:
        if _failed(r, arm):
            raise ValueError(
                f"arm {arm} failed in cell {r['condition']}/{r['stream_id']}; no primitive exists"
            )
        value = r["arms"][arm][field]
        table[(r["init_index"], (r["condition"], r["stream_id"]))] = float(value)
    if len(table) != len(inits) * len(streams):
        raise ValueError("records do not form a complete crossing")
    return np.array([[table[(i, s)] for s in streams] for i in inits], dtype=float)


def relative_mae(records: Sequence[Mapping[str, Any]], arm: str, *, bootstrap_index: int) -> dict[str, Any]:
    first, second = grid(records, "gru", "mae"), grid(records, arm, "mae")
    summary = crossed_paired_difference(first, second, bootstrap_index=bootstrap_index)
    base = float(np.mean(first))
    return {
        "relative_difference": summary["mean_difference"] / base,
        "relative_ci_low": summary["ci_low"] / base,
        "relative_ci_high": summary["ci_high"] / base,
        "per_initialization_difference": summary["per_initialization_difference"],
        "interval_status": summary["interval_status"],
    }


def divergence_counts(records: Sequence[Mapping[str, Any]], arm: str) -> int:
    return sum(1 for r in records if _failed(r, arm) or r["arms"][arm]["diverged"])


def non_inferior(interval: Mapping[str, Any]) -> str:
    if interval["interval_status"] != "MEASURED":
        return "INCONCLUSIVE"
    if interval["relative_ci_high"] <= MARGIN:
        return "PASS"
    if interval["relative_ci_low"] > MARGIN:
        return "FAIL"
    return "INCONCLUSIVE"


def _select(records: Sequence[Mapping[str, Any]], conditions: Sequence[str]) -> list[Mapping[str, Any]]:
    return [r for r in records if r["condition"] in conditions]


def plan_conditions(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({r["condition"] for r in records if r["condition"].startswith("plan:")})


# ----------------------------------------------------------------------
# development screen
# ----------------------------------------------------------------------
def screen(records: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    coarse = _select(records, LONG_COARSE)
    champion_coarse = divergence_counts(coarse, "gru")
    candidate_coarse = divergence_counts(coarse, arm)
    if champion_coarse < MINIMUM_CHAMPION_DIVERGENCES:
        s1 = "INCONCLUSIVE"
    else:
        other = _select(records, LONG_OTHER)
        s1 = (
            "PASS"
            if candidate_coarse <= 0.2 * champion_coarse
            and divergence_counts(other, arm) <= divergence_counts(other, "gru")
            else "FAIL"
        )
    s2 = {}
    for index, condition in enumerate(plan_conditions(records)):
        interval = relative_mae(_select(records, [condition]), arm, bootstrap_index=400 + index)
        s2[condition] = {**interval, "status": non_inferior(interval)}
    s3 = {}
    for index, condition in enumerate(LONG_OTHER):
        interval = relative_mae(_select(records, [condition]), arm, bootstrap_index=420 + index)
        s3[condition] = {**interval, "status": non_inferior(interval)}
    statuses = [s1, *(v["status"] for v in s2.values()), *(v["status"] for v in s3.values())]
    return {
        "S1_long_horizon_stability": {
            "status": s1,
            "champion_coarse_divergences": champion_coarse,
            "candidate_coarse_divergences": candidate_coarse,
            "champion_other_divergences": divergence_counts(_select(records, LONG_OTHER), "gru"),
            "candidate_other_divergences": divergence_counts(_select(records, LONG_OTHER), arm),
        },
        "S2_standard_families": s2,
        "S3_long_non_coarse": s3,
        "passed": all(status == "PASS" for status in statuses),
    }


def select_for_attack(screens: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The first candidate, in precommitted order, whose screen passed."""

    for candidate in CANDIDATES:
        if screens[candidate["arm"]]["passed"]:
            return candidate["arm"]
    return None


# ----------------------------------------------------------------------
# attack
# ----------------------------------------------------------------------
def adjudicate_attack(records: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    coarse = _select(records, LONG_COARSE)
    champion_coarse, candidate_coarse = divergence_counts(coarse, "gru"), divergence_counts(coarse, arm)
    a1 = (
        "INCONCLUSIVE"
        if champion_coarse < MINIMUM_CHAMPION_DIVERGENCES
        else ("PASS" if candidate_coarse <= 0.2 * champion_coarse else "FAIL")
    )
    nearby = {
        condition: {
            "champion": divergence_counts(_select(records, [condition]), "gru"),
            "candidate": divergence_counts(_select(records, [condition]), arm),
        }
        for condition in NEARBY_CONDITIONS
    }
    champion_near = sum(v["champion"] for v in nearby.values())
    candidate_near = sum(v["candidate"] for v in nearby.values())
    every = all(v["candidate"] <= v["champion"] for v in nearby.values())
    if champion_near >= MINIMUM_CHAMPION_DIVERGENCES:
        a2 = "PASS" if every and candidate_near <= 0.2 * champion_near else "FAIL"
    else:
        a2 = "PASS" if every else "FAIL"
    a3 = {}
    for index, condition in enumerate(plan_conditions(records)):
        interval = relative_mae(_select(records, [condition]), arm, bootstrap_index=440 + index)
        a3[condition] = {**interval, "status": non_inferior(interval)}
    a4 = {}
    for index, condition in enumerate(LONG_OTHER):
        selected = _select(records, [condition])
        interval = relative_mae(selected, arm, bootstrap_index=460 + index)
        status = non_inferior(interval)
        if divergence_counts(selected, arm) > divergence_counts(selected, "gru"):
            status = "FAIL"
        a4[condition] = {**interval, "status": status}
    locked = sum(1 for r in records if not _failed(r, arm) and r["arms"][arm]["first_lock"] is not None)
    a5 = "PASS" if locked == 0 else "FAIL"
    outcome = {
        "A1_long_coarse_fresh_initializations": a1,
        "A2_nearby_quantized_conditions": a2,
        "A3_standard_families": "PASS" if all(v["status"] == "PASS" for v in a3.values()) else "FAIL",
        "A4_long_non_coarse": "PASS" if all(v["status"] == "PASS" for v in a4.values()) else "FAIL",
        "A5_no_frame_lock": a5,
    }
    return {
        "arm": arm,
        "long_coarse": {"champion": champion_coarse, "candidate": candidate_coarse},
        "nearby": nearby,
        "standard_families": a3,
        "long_non_coarse": a4,
        "candidate_locked_cells": locked,
        "outcome": outcome,
        "advance_to_freeze": all(status == "PASS" for status in outcome.values()),
    }


# ----------------------------------------------------------------------
# confirmation decision
# ----------------------------------------------------------------------
def decide4(primitives: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    coarse = _select(primitives, LONG_COARSE)
    champion_d, challenger_d = grid(coarse, "gru", "diverged"), grid(coarse, arm, "diverged")
    k1_interval = crossed_paired_difference(challenger_d, champion_d, bootstrap_index=480)
    fraction = float(np.mean(challenger_d))
    if k1_interval["interval_status"] != "MEASURED":
        k1 = "INCONCLUSIVE"
    elif k1_interval["ci_low"] > 0 and fraction <= 0.05:
        k1 = "PASS"
    elif k1_interval["ci_high"] <= 0 or fraction > 0.20:
        k1 = "FAIL"
    else:
        k1 = "INCONCLUSIVE"
    criteria: dict[str, Any] = {
        "K1_long_coarse_stability": {
            "status": k1,
            "champion_divergence_fraction": float(np.mean(champion_d)),
            "challenger_divergence_fraction": fraction,
            "difference": k1_interval["mean_difference"],
            "ci_low": k1_interval["ci_low"],
            "ci_high": k1_interval["ci_high"],
        }
    }
    for index, condition in enumerate(LONG_OTHER):
        selected = _select(primitives, [condition])
        interval = relative_mae(selected, arm, bootstrap_index=482 + index)
        champion_count, challenger_count = (
            divergence_counts(selected, "gru"),
            divergence_counts(selected, arm),
        )
        if interval["interval_status"] != "MEASURED":
            status = "INCONCLUSIVE"
        elif interval["relative_ci_low"] > MARGIN or challenger_count > champion_count + 2:
            status = "FAIL"
        elif interval["relative_ci_high"] <= MARGIN and challenger_count <= champion_count:
            status = "PASS"
        else:
            status = "INCONCLUSIVE"
        criteria[f"K2_{condition}"] = {
            **interval,
            "status": status,
            "champion_divergences": champion_count,
            "challenger_divergences": challenger_count,
        }
    for index, condition in enumerate(plan_conditions(primitives)):
        interval = relative_mae(_select(primitives, [condition]), arm, bootstrap_index=490 + index)
        criteria[f"K3_{condition}"] = {**interval, "status": non_inferior(interval)}
    statuses = [c["status"] for c in criteria.values()]
    if "FAIL" in statuses:
        outcome = "REJECT"
    elif all(s == "PASS" for s in statuses):
        outcome = "PROMOTE"
    else:
        outcome = "INCONCLUSIVE"
    return {
        "arm": arm,
        "criteria": criteria,
        "statuses": {name: c["status"] for name, c in criteria.items()},
        "outcome": outcome,
    }


FROZEN_RULES = {
    "margin": MARGIN,
    "divergence_factor": DIVERGENCE_FACTOR,
    "k1": {"pass": "ci_low > 0 and challenger fraction <= 0.05", "fail": "ci_high <= 0 or fraction > 0.20"},
    "k2": {
        "pass": "relative ci_high <= margin and challenger divergences <= champion's",
        "fail": "relative ci_low > margin or challenger divergences > champion's + 2",
    },
    "k3": {"pass": "relative ci_high <= margin", "fail": "relative ci_low > margin"},
    "outcome": "PROMOTE iff all PASS; REJECT if any FAIL; else INCONCLUSIVE",
    "bootstrap_indices": {"k1": 480, "k2": [482, 483], "k3_start": 490},
    "confirmation_design": {
        "per_entry": CONFIRMATION_PER_ENTRY,
        "long_seeds": CONFIRMATION_LONG,
        "long_conditions": list(LONG_CONDITIONS),
        "initializations": INITIALIZATIONS,
        "long_steps": LONG_STEPS,
    },
}
