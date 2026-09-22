"""Iteration 0004, diagnosis round 4: is M2 a self-confirming target-unfolding frame lock?

Declared after an exploratory, post-hoc look at round 3's already-observed
cells (disclosed in :mod:`research.aaa_1k_loop.unfolding`): in 34 of 39
diverged cells onset came 13-19 steps after the first wall bounce, 37 of 39
were slow-regime streams, and one traced cell showed the mirror-branch lock.
That look chose these hypotheses; it cannot count as evidence for them. They
are judged here, on a fresh block, committed before it was run.

Definitions: *mirror step*, *lock* (:data:`~research.aaa_1k_loop.unfolding.LOCK_RUN`
consecutive trained mirror steps) as in ``unfolding.py``; *divergence* and
*onset* as in rounds 2 and 3.

Arms
----
``gru``                 the champion, tracing its unfolding branch (bitwise the champion)
``gru_lr0.01``          the champion at lr 0.01, tracing
``gru_no_error_input``  the input-3 ablation, tracing (reported, not adjudicated)
``probe:gru_no_unfold`` the champion trained on the folded observation (``unfold_target=False``)
``probe:gru_unfold_dr`` the champion unfolding around a prediction-independent reference
``persistence``         the divergence reference
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.aaa_1k.agents import PersistenceAgent
from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed

from .arms import gru
from .diagnosis4b import build_stream, first_index, trailing
from .harness import Cell, arm_errors, run_cells
from .identities import block_seeds, find_block, require_usable
from .unfolding import LOCK_RUN, DeadReckoningUnfoldAgent, UnfoldTraceAgent, longest_runs

DIAGNOSIS4D_SCHEMA = "aaa.loop.diagnosis.v4d"
DIAGNOSTIC_BLOCK = "aaa1k-loop-0004/diagnostic/unfold"
SCRATCH_BLOCK = "aaa1k-loop-0004/diagnostic/scratch"

STREAMS = 16
INITIALIZATIONS = 5
STEPS = 1120
WINDOW = 40
DIVERGENCE_FACTOR = 2.0
MINIMUM_CELLS = 5

CONDITIONS = ("coarse_no_switch", "coarse_switching", "bouncing")
COARSE_CONDITIONS = ("coarse_no_switch", "coarse_switching")
TRACED = ("gru", "gru_lr0.01", "gru_no_error_input", "probe:gru_no_unfold", "probe:gru_unfold_dr")
LEARNERS = TRACED

HYPOTHESES: tuple[dict[str, str], ...] = (
    {
        "id": "H32",
        "name": "a frame lock precedes the runaway",
        "statement": (
            "divergence begins when the unfolding branch, chosen by the learner's own prediction, locks onto "
            "the mirrored frame"
        ),
        "prediction": f"in >= 80% of diverged coarse champion cells a lock (>= {LOCK_RUN} steps) starts at or before onset",
        "contradicted_by": "a fraction <= 50%",
    },
    {
        "id": "H33",
        "name": "stable learners do not lock",
        "statement": "cells that stay stable, and the lr-0.01 learner, never hold the mirrored frame",
        "prediction": "no lock in >= 80% of non-diverged coarse champion cells and in >= 80% of coarse lr-0.01 cells",
        "contradicted_by": "either fraction <= 50%",
    },
    {
        "id": "H34",
        "name": "removing the self-reference removes the runaway",
        "statement": (
            "training on the folded observation, or unfolding around a prediction-independent reference, "
            "prevents M2"
        ),
        "prediction": (
            "each of probe:gru_no_unfold and probe:gru_unfold_dr diverges in <= 20% as many coarse cells as "
            "the champion"
        ),
        "contradicted_by": "either probe diverging in >= 50% as many coarse cells",
    },
)


@dataclass(frozen=True)
class UnfoldArms:
    def __call__(self, seed: int) -> Sequence[Any]:
        return [
            UnfoldTraceAgent(gru(seed), name="gru"),
            UnfoldTraceAgent(gru(seed, learning_rate=0.01), name="gru_lr0.01"),
            UnfoldTraceAgent(gru(seed, zero_error_input=True), name="gru_no_error_input"),
            UnfoldTraceAgent(gru(seed), name="probe:gru_no_unfold", unfold_target=False),
            DeadReckoningUnfoldAgent(gru(seed), name="probe:gru_unfold_dr"),
            PersistenceAgent(),
        ]


def diagnostic_cells(
    ledger: Mapping[str, Any],
    *,
    block_id: str = DIAGNOSTIC_BLOCK,
    streams: int = STREAMS,
    initializations: int = INITIALIZATIONS,
    steps: int = STEPS,
) -> list[Cell]:
    block = require_usable(ledger, block_id, purpose="selection")
    if block["role"] != "diagnostic":
        raise ValueError("diagnosis round 4d must run on a diagnostic block")
    seeds = block_seeds(find_block(ledger, block_id))[:streams]
    return [
        Cell(condition, init, derive_seed("model_init", init), build_stream(condition, seed, steps))
        for condition in CONDITIONS
        for seed in seeds
        for init in range(initializations)
    ]


def reduce_cell(cell: Cell, result: RunResult, agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    persistence = errors["persistence"]
    persistence_trailing = trailing(persistence, WINDOW, np.mean)
    bounces = [step.index for step in cell.stream.steps if step.event == "bounce"]
    by_name = {agent.name: agent for agent in agents}
    arms: dict[str, Any] = {}
    for name in LEARNERS:
        if name not in by_name:
            continue
        agent = by_name[name]
        error = errors[name]
        offset = error.size - len(agent.mirror_trace)
        if offset not in (0, 1):
            raise RuntimeError(f"trained steps do not align with scored steps (offset {offset})")
        ratio = (trailing(error, WINDOW, np.mean) / persistence_trailing)[offset:]
        runs = longest_runs(agent.mirror_trace)
        locks = [(start, length) for start, length in runs if length >= LOCK_RUN]
        targets = np.abs(np.asarray(agent.target_trace, dtype=float))
        arms[name] = {
            "mae": float(np.mean(error)),
            "diverged": bool(np.mean(error) > DIVERGENCE_FACTOR * np.mean(persistence)),
            "onset": first_index(ratio > DIVERGENCE_FACTOR, start=WINDOW),
            "mirror_steps": int(sum(agent.mirror_trace)),
            "longest_mirror_run": max((length for _s, length in runs), default=0),
            "first_lock": locks[0][0] if locks else None,
            "locks": [[start, length] for start, length in locks[:8]],
            "max_abs_target": float(np.max(targets)) if targets.size else 0.0,
        }
    return {
        "persistence_mae": float(np.mean(persistence)),
        "first_bounce": bounces[0] if bounces else None,
        "initial_regime": cell.stream.steps[0].regime,
        "arms": arms,
    }


def run_diagnosis4d(
    ledger: Mapping[str, Any], *, workers: int | None = None, **design: Any
) -> list[dict[str, Any]]:
    return run_cells(
        diagnostic_cells(ledger, **design), UnfoldArms(), reduce_cell, isolate_failures=True, workers=workers
    )


def _fraction_verdict(hits: int, total: int) -> tuple[str, float | None]:
    if total < MINIMUM_CELLS:
        return "INSUFFICIENT_EVIDENCE", None
    fraction = hits / total
    return (
        "SUPPORTED" if fraction >= 0.8 else "CONTRADICTED" if fraction <= 0.5 else "INCONCLUSIVE"
    ), fraction


def _diverged(record: Mapping[str, Any], name: str) -> bool:
    if name in record.get("failures", {}):
        return True
    return bool(record["arms"][name]["diverged"])


def adjudicate4d(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coarse = [r for r in records if r["condition"] in COARSE_CONDITIONS]
    champion = [r["arms"]["gru"] for r in coarse if "gru" in r["arms"]]
    diverged = [a for a in champion if a["diverged"]]
    stable = [a for a in champion if not a["diverged"]]
    slow = [r["arms"]["gru_lr0.01"] for r in coarse if "gru_lr0.01" in r["arms"]]

    h32_hits = sum(
        1
        for a in diverged
        if a["first_lock"] is not None and a["onset"] is not None and a["first_lock"] <= a["onset"]
    )
    h32, f32 = _fraction_verdict(h32_hits, len(diverged))
    part_stable, fs = _fraction_verdict(sum(1 for a in stable if a["first_lock"] is None), len(stable))
    part_slow, fl = _fraction_verdict(sum(1 for a in slow if a["first_lock"] is None), len(slow))
    if "CONTRADICTED" in (part_stable, part_slow):
        h33 = "CONTRADICTED"
    elif part_stable == part_slow == "SUPPORTED":
        h33 = "SUPPORTED"
    elif "INSUFFICIENT_EVIDENCE" in (part_stable, part_slow):
        h33 = "INSUFFICIENT_EVIDENCE"
    else:
        h33 = "INCONCLUSIVE"

    counts = {name: sum(_diverged(r, name) for r in coarse) for name in LEARNERS}
    reference = counts["gru"]
    probes = {}
    for name in ("probe:gru_no_unfold", "probe:gru_unfold_dr"):
        if reference < MINIMUM_CELLS:
            probes[name] = ("INSUFFICIENT_EVIDENCE", None)
        else:
            ratio = counts[name] / reference
            probes[name] = (
                "SUPPORTED" if ratio <= 0.2 else "CONTRADICTED" if ratio >= 0.5 else "INCONCLUSIVE",
                ratio,
            )
    verdicts = [v for v, _r in probes.values()]
    if "CONTRADICTED" in verdicts:
        h34 = "CONTRADICTED"
    elif all(v == "SUPPORTED" for v in verdicts):
        h34 = "SUPPORTED"
    elif "INSUFFICIENT_EVIDENCE" in verdicts:
        h34 = "INSUFFICIENT_EVIDENCE"
    else:
        h34 = "INCONCLUSIVE"

    by_condition = {
        condition: {
            name: {
                "diverged": sum(_diverged(r, name) for r in records if r["condition"] == condition),
                "locked": sum(
                    1
                    for r in records
                    if r["condition"] == condition
                    and name in r["arms"]
                    and r["arms"][name]["first_lock"] is not None
                ),
                "cells": sum(1 for r in records if r["condition"] == condition),
            }
            for name in LEARNERS
        }
        for condition in CONDITIONS
    }
    return {
        "coarse_divergences": counts,
        "by_condition": by_condition,
        "lock_lead_steps": [
            a["onset"] - a["first_lock"]
            for a in diverged
            if a["first_lock"] is not None and a["onset"] is not None
        ],
        "verdicts": {
            "H32": {"verdict": h32, "fraction": f32},
            "H33": {
                "verdict": h33,
                "stable_champion": {"verdict": part_stable, "fraction": fs},
                "lr_0_01": {"verdict": part_slow, "fraction": fl},
            },
            "H34": {
                "verdict": h34,
                **{name: {"verdict": v, "ratio": r} for name, (v, r) in probes.items()},
            },
        },
    }
