"""Iteration 0004, diagnosis round 3: is M2 online-SGD overshoot? (audit R-01)

Declared after round 2 falsified the loop-gain hypothesis, and committed
before this round's diagnostic block was run. The mechanism and the
instrument are in :mod:`research.aaa_1k_loop.overshoot`.

Definitions
-----------
rolling amplification
    trailing median, over :data:`WINDOW` learning steps, of the exact
    same-sample amplification ``|r'| / |r|`` of each update.
overshoot crossing
    first learning step at which the rolling amplification exceeds one: most
    recent updates are making the sample they learned from worse.
divergence, onset
    as in round 2 (twice persistence; trailing :data:`WINDOW`-step mean).

Learning steps and scored steps coincide on these fully observed streams
after the first step; the reduction aligns them by index and checks it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.aaa_1k.agents import PersistenceAgent
from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed

from .arms import CHAMPION_CONFIGURATION
from .diagnosis4b import build_stream, first_index, trailing
from .harness import Cell, InstrumentedAgent, arm_errors, run_cells
from .identities import block_seeds, find_block, require_usable
from .overshoot import OvershootGRU

DIAGNOSIS4C_SCHEMA = "aaa.loop.diagnosis.v4c"
DIAGNOSTIC_BLOCK = "aaa1k-loop-0004/diagnostic/overshoot"
SCRATCH_BLOCK = "aaa1k-loop-0004/diagnostic/scratch"

STREAMS = 16
INITIALIZATIONS = 5
STEPS = 1120
WINDOW = 40
DIVERGENCE_FACTOR = 2.0
MINIMUM_CELLS = 5
DOWNSAMPLE = 20
BASELINE = (40, 120)
PRE_ONSET = 40

CONDITIONS = ("coarse_no_switch", "coarse_switching")
ARMS = ("gru", "gru_lr0.01", "gru_state_reset")

HYPOTHESES: tuple[dict[str, str], ...] = (
    {
        "id": "H29",
        "name": "overshoot precedes the runaway",
        "statement": (
            "divergence begins when single online updates start making the sample they learned from worse"
        ),
        "prediction": (
            "in >= 80% of diverged champion cells the rolling amplification exceeds one at or before onset"
        ),
        "contradicted_by": "a fraction <= 50%",
    },
    {
        "id": "H30",
        "name": "stable learners do not overshoot",
        "statement": "cells that stay stable, and the lr-0.01 learner, keep single-step amplification below one",
        "prediction": (
            "the rolling amplification never exceeds one in >= 80% of non-diverged champion cells and in "
            ">= 80% of lr-0.01 cells"
        ),
        "contradicted_by": "either fraction <= 50%",
    },
    {
        "id": "H31",
        "name": "the input-3 channel carries the curvature growth",
        "statement": (
            "overshoot arrives because the gate-weight curvature attached to input 3 grows with the error it feeds back"
        ),
        "prediction": (
            f"in >= 80% of diverged champion cells the median input-3 share of the displacement curvature over "
            f"the {PRE_ONSET} learning steps before onset exceeds its median over steps {BASELINE}"
        ),
        "contradicted_by": "a fraction <= 50%",
    },
)


@dataclass(frozen=True)
class OvershootArms:
    def __call__(self, seed: int) -> Sequence[Any]:
        config = dict(CHAMPION_CONFIGURATION)
        return [
            InstrumentedAgent(OvershootGRU(seed=seed, **config), name="gru"),
            InstrumentedAgent(
                OvershootGRU(seed=seed, **{**config, "learning_rate": 0.01}), name="gru_lr0.01"
            ),
            InstrumentedAgent(
                OvershootGRU(seed=seed, **{**config, "reset_state_every_step": True}), name="gru_state_reset"
            ),
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
        raise ValueError("diagnosis round 4c must run on a diagnostic block")
    seeds = block_seeds(find_block(ledger, block_id))[:streams]
    return [
        Cell(condition, init, derive_seed("model_init", init), build_stream(condition, seed, steps))
        for condition in CONDITIONS
        for seed in seeds
        for init in range(initializations)
    ]


def _series(model: OvershootGRU, field: str) -> np.ndarray:
    return np.asarray([record[field] for record in model.overshoot], dtype=float)


def _median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _down(values: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else float(v) for v in values[::DOWNSAMPLE]]


def reduce_cell(_cell: Cell, result: RunResult, agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    persistence = errors["persistence"]
    persistence_trailing = trailing(persistence, WINDOW, np.mean)
    by_name = {agent.name: agent for agent in agents}
    arms: dict[str, Any] = {}
    for name in ARMS:
        if name not in by_name:
            continue
        model = by_name[name].model
        error = errors[name]
        # Learning step k scores the prediction made at scored step k + offset.
        offset = error.size - len(model.overshoot)
        if offset not in (0, 1):
            raise RuntimeError(f"learning steps do not align with scored steps (offset {offset})")
        ratio = (trailing(error, WINDOW, np.mean) / persistence_trailing)[offset:]
        amplification = _series(model, "amplification")
        share = _series(model, "error_input_share")
        rolling = trailing(amplification, WINDOW, np.median)
        onset = first_index(ratio > DIVERGENCE_FACTOR, start=WINDOW)
        pre = share[max(0, onset - PRE_ONSET) : onset] if onset is not None else np.array([])
        arms[name] = {
            "mae": float(np.mean(error)),
            "diverged": bool(np.mean(error) > DIVERGENCE_FACTOR * np.mean(persistence)),
            "onset": onset,
            "overshoot_crossing": first_index(rolling > 1.0),
            "amplification_median": _median(amplification),
            "kappa_median": _median(_series(model, "kappa")),
            "share_baseline": _median(share[slice(*BASELINE)]),
            "share_pre_onset": _median(pre) if pre.size else None,
            "curvature_baseline": _median(_series(model, "curvature")[slice(*BASELINE)]),
            "rolling_amplification": _down(rolling),
            "rolling_kappa": _down(trailing(_series(model, "kappa"), WINDOW, np.median)),
            "rolling_curvature": _down(trailing(_series(model, "curvature"), WINDOW, np.median)),
            "rolling_error_input_share": _down(trailing(share, WINDOW, np.median)),
            "rolling_head_share": _down(trailing(_series(model, "head_share"), WINDOW, np.median)),
            "clip_active": _down(
                trailing((_series(model, "clip_scale") < 1.0).astype(float), WINDOW, np.mean)
            ),
            "error_ratio": _down(ratio),
        }
    return {"persistence_mae": float(np.mean(persistence)), "arms": arms}


def run_diagnosis4c(
    ledger: Mapping[str, Any], *, workers: int | None = None, **design: Any
) -> list[dict[str, Any]]:
    return run_cells(
        diagnostic_cells(ledger, **design),
        OvershootArms(),
        reduce_cell,
        isolate_failures=True,
        workers=workers,
    )


def _fraction_verdict(hits: int, total: int) -> tuple[str, float | None]:
    if total < MINIMUM_CELLS:
        return "INSUFFICIENT_EVIDENCE", None
    fraction = hits / total
    return (
        "SUPPORTED" if fraction >= 0.8 else "CONTRADICTED" if fraction <= 0.5 else "INCONCLUSIVE"
    ), fraction


def adjudicate4c(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    champion = [r["arms"]["gru"] for r in records if "gru" in r["arms"]]
    diverged = [a for a in champion if a["diverged"]]
    stable = [a for a in champion if not a["diverged"]]
    slow = [r["arms"]["gru_lr0.01"] for r in records if "gru_lr0.01" in r["arms"]]

    h29_hits = sum(
        1
        for a in diverged
        if a["overshoot_crossing"] is not None
        and a["onset"] is not None
        and a["overshoot_crossing"] <= a["onset"]
    )
    h29, f29 = _fraction_verdict(h29_hits, len(diverged))
    stable_clean = sum(1 for a in stable if a["overshoot_crossing"] is None)
    slow_clean = sum(1 for a in slow if a["overshoot_crossing"] is None)
    part_stable, fs = _fraction_verdict(stable_clean, len(stable))
    part_slow, fl = _fraction_verdict(slow_clean, len(slow))
    if "CONTRADICTED" in (part_stable, part_slow):
        h30 = "CONTRADICTED"
    elif part_stable == part_slow == "SUPPORTED":
        h30 = "SUPPORTED"
    elif "INSUFFICIENT_EVIDENCE" in (part_stable, part_slow):
        h30 = "INSUFFICIENT_EVIDENCE"
    else:
        h30 = "INCONCLUSIVE"
    rising = sum(
        1
        for a in diverged
        if a["share_pre_onset"] is not None
        and a["share_baseline"] is not None
        and a["share_pre_onset"] > a["share_baseline"]
    )
    h31, f31 = _fraction_verdict(rising, len(diverged))
    return {
        "counts": {
            "champion_cells": len(champion),
            "champion_diverged": len(diverged),
            "champion_failed": sum("gru" in r.get("failures", {}) for r in records),
            "slow_cells": len(slow),
            "slow_diverged": sum(1 for a in slow if a["diverged"]),
            "state_reset_diverged": sum(
                1
                for r in records
                if "gru_state_reset" in r["arms"] and r["arms"]["gru_state_reset"]["diverged"]
            ),
        },
        "crossing_lead_steps": [
            a["onset"] - a["overshoot_crossing"]
            for a in diverged
            if a["overshoot_crossing"] is not None and a["onset"] is not None
        ],
        "verdicts": {
            "H29": {"verdict": h29, "fraction": f29},
            "H30": {
                "verdict": h30,
                "stable_champion": {"verdict": part_stable, "fraction": fs},
                "lr_0_01": {"verdict": part_slow, "fraction": fl},
            },
            "H31": {"verdict": h31, "fraction": f31},
        },
    }
