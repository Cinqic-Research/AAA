"""Iteration 0004, diagnosis round 2: is M2 a learned closed-loop gain above one? (audit R-01)

Round 1 (:mod:`research.aaa_1k_loop.diagnosis4`) showed the runaway is the
same under every TBPTT rule including T=1, so it lives in the one-step update.
The pilot's refined hypothesis, made precise in :mod:`research.aaa_1k_loop.gain`,
is that the one-step gradient -- blind to the fact that input 3 is the model's
own previous error -- walks the direct error-feedback gain ``a_t`` upward on
quantized targets until the loop is unstable.

Everything below was committed before the diagnostic block was run.

Definitions
-----------
``a_t``
    the direct loop gain, measured after every prediction (``gain.py``).
rolling gain
    the trailing median of ``a_t`` over :data:`WINDOW` steps.
crossing
    the first step at which the rolling gain exceeds one.
divergence
    whole-stream mean error above twice persistence's (as in rounds 3 and 1).
onset
    for a diverged cell, the first step ``t >= WINDOW`` at which the trailing
    :data:`WINDOW`-step mean error exceeds twice persistence's trailing mean.

Conditions and arms
-------------------
``coarse_no_switch`` runs 3360 steps so that the slower learning rate has
three times as many updates; the other two conditions run 1120 steps. Arms:
the champion (lr 0.03), the champion at lr 0.01, the state-reset and
no-error-input ablations, and persistence. The state-reset and
no-error-input arms are reported but not adjudicated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.aaa_1k.agents import PersistenceAgent
from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed
from research.aaa_1k.streams import coarse_speed_stream, motion_compat_stream

from .arms import gru
from .gain import GainAgent
from .harness import Cell, arm_errors, run_cells
from .identities import block_seeds, find_block, require_usable

DIAGNOSIS4B_SCHEMA = "aaa.loop.diagnosis.v4b"
DIAGNOSTIC_BLOCK = "aaa1k-loop-0004/diagnostic/gain"
SCRATCH_BLOCK = "aaa1k-loop-0004/diagnostic/scratch"

STREAMS = 16
INITIALIZATIONS = 5
WINDOW = 40
DIVERGENCE_FACTOR = 2.0
MINIMUM_CELLS = 5
DOWNSAMPLE = 20
CHAMPION_PREFIX = 1120

CONDITIONS: dict[str, dict[str, Any]] = {
    "coarse_no_switch": {"steps": 3360},
    "coarse_switching": {"steps": 1120},
    "bouncing": {"steps": 1120},
}
COARSE_CONDITIONS = ("coarse_no_switch", "coarse_switching")
GAIN_ARMS = ("gru", "gru_lr0.01", "gru_state_reset", "gru_no_error_input")

EARLY = (40, 160)
LATE = (160, 280)
SMOOTH_GAIN_BOUND = 0.5
PACE_RANGE = (1.5, 6.0)
PACE_CONTRADICTED = (1.2, 10.0)

HYPOTHESES: tuple[dict[str, str], ...] = (
    {
        "id": "H24",
        "name": "the loop gain crosses one before the runaway",
        "statement": "divergence is preceded by a sustained direct error-feedback gain above one",
        "prediction": (
            "in >= 80% of diverged coarse champion cells the rolling gain crosses one at or before onset"
        ),
        "contradicted_by": "a fraction <= 50%",
    },
    {
        "id": "H25",
        "name": "stable cells stay below one",
        "statement": "a champion cell that does not diverge never sustains a loop gain above one",
        "prediction": "in >= 80% of non-diverged coarse champion cells the rolling gain never crosses one",
        "contradicted_by": "a fraction <= 50%",
    },
    {
        "id": "H26",
        "name": "the gain rises on quantized targets, not on smooth ones",
        "statement": (
            "the one-step gradient pushes the gain upward when the displacement deviations alternate, "
            "and has no such push on smooth motion"
        ),
        "prediction": (
            f"(a) the champion's median gain over steps {LATE} exceeds its median over {EARLY} in >= 80% of "
            f"coarse_no_switch cells, and (b) on bouncing cells the median gain over steps [40, 1120) has "
            f"magnitude < {SMOOTH_GAIN_BOUND} in >= 80% of cells"
        ),
        "contradicted_by": "either fraction <= 50%",
    },
    {
        "id": "H27",
        "name": "the learning rate sets the pace, not whether",
        "statement": "a smaller learning rate delays the runaway instead of preventing it",
        "prediction": (
            f"on coarse_no_switch, lr 0.01 over 3360 steps diverges in >= 50% as many cells as the champion "
            f"does over the first {CHAMPION_PREFIX} steps of the same cells"
        ),
        "contradicted_by": "a ratio <= 20%",
    },
    {
        "id": "H28",
        "name": "the drift rate is proportional to the learning rate",
        "statement": "time to cross one scales roughly inversely with the learning rate (0.03 / 0.01 = 3)",
        "prediction": (
            f"on coarse_no_switch cells where both arms cross, the median ratio of lr-0.01 to champion "
            f"crossing steps lies in [{PACE_RANGE[0]}, {PACE_RANGE[1]}]"
        ),
        "contradicted_by": f"a median ratio below {PACE_CONTRADICTED[0]} or above {PACE_CONTRADICTED[1]}",
    },
)


@dataclass(frozen=True)
class GainArms:
    """Picklable factory: the champion and its probes, each recording the loop gain."""

    def __call__(self, seed: int) -> Sequence[Any]:
        return [
            GainAgent(gru(seed), name="gru"),
            GainAgent(gru(seed, learning_rate=0.01), name="gru_lr0.01"),
            GainAgent(gru(seed, reset_state_every_step=True), name="gru_state_reset"),
            GainAgent(gru(seed, zero_error_input=True), name="gru_no_error_input"),
            PersistenceAgent(),
        ]


def build_stream(condition: str, seed: int, steps: int) -> Any:
    if condition == "coarse_no_switch":
        return coarse_speed_stream(seed, steps=steps, regime_length=100_000)
    if condition == "coarse_switching":
        return coarse_speed_stream(seed, steps=steps, regime_length=60)
    if condition == "bouncing":
        return motion_compat_stream("bouncing", seed, steps=steps, change_step=None)
    raise ValueError(condition)


def diagnostic_cells(
    ledger: Mapping[str, Any],
    *,
    block_id: str = DIAGNOSTIC_BLOCK,
    streams: int = STREAMS,
    initializations: int = INITIALIZATIONS,
    step_scale: float = 1.0,
) -> list[Cell]:
    block = require_usable(ledger, block_id, purpose="selection")
    if block["role"] != "diagnostic":
        raise ValueError("diagnosis round 4b must run on a diagnostic block")
    seeds = block_seeds(find_block(ledger, block_id))[:streams]
    cells: list[Cell] = []
    for condition, options in CONDITIONS.items():
        steps = max(WINDOW * 4, int(options["steps"] * step_scale))
        for seed in seeds:
            stream = build_stream(condition, seed, steps)
            cells.extend(
                Cell(condition, init, derive_seed("model_init", init), stream)
                for init in range(initializations)
            )
    return cells


# ----------------------------------------------------------------------
# reduction
# ----------------------------------------------------------------------
def trailing(values: np.ndarray, window: int, reducer: Any) -> np.ndarray:
    """``reducer`` over the trailing ``window`` values; NaN until the window is full."""

    out = np.full(values.shape, np.nan)
    if values.size >= window:
        view = np.lib.stride_tricks.sliding_window_view(values, window)
        out[window - 1 :] = reducer(view, axis=1)
    return out


def first_index(mask: np.ndarray, *, start: int = 0) -> int | None:
    hits = np.flatnonzero(mask[start:])
    return int(hits[0] + start) if hits.size else None


def _median(values: np.ndarray, lo: int, hi: int) -> float | None:
    segment = values[lo:hi]
    segment = segment[np.isfinite(segment)]
    return float(np.median(segment)) if segment.size else None


def reduce_cell(_cell: Cell, result: RunResult, agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    persistence = errors["persistence"]
    persistence_trailing = trailing(persistence, WINDOW, np.mean)
    by_name = {agent.name: agent for agent in agents}
    arms: dict[str, Any] = {}
    for name in GAIN_ARMS:
        if name not in by_name:
            continue
        agent = by_name[name]
        gain = np.asarray([m["direct_gain"] for m in agent.gain_trace], dtype=float)
        radius = np.asarray([m["closed_loop_radius"] for m in agent.gain_trace], dtype=float)
        rolling_gain = trailing(gain, WINDOW, np.median)
        rolling_radius = trailing(radius, WINDOW, np.median)
        error = errors[name]
        ratio = trailing(error, WINDOW, np.mean) / persistence_trailing
        prefix = min(CHAMPION_PREFIX, error.size)
        arms[name] = {
            "mae": float(np.mean(error)),
            "prefix_mae": float(np.mean(error[:prefix])),
            "diverged": bool(np.mean(error) > DIVERGENCE_FACTOR * np.mean(persistence)),
            "prefix_diverged": bool(
                np.mean(error[:prefix]) > DIVERGENCE_FACTOR * np.mean(persistence[:prefix])
            ),
            "onset": first_index(ratio > DIVERGENCE_FACTOR, start=WINDOW),
            "gain_crossing": first_index(rolling_gain > 1.0),
            "radius_crossing": first_index(rolling_radius > 1.0),
            "gain_early": _median(gain, *EARLY),
            "gain_late": _median(gain, *LATE),
            "gain_median": _median(gain, WINDOW, gain.size),
            "gain_max_rolling": float(np.nanmax(rolling_gain)) if np.any(np.isfinite(rolling_gain)) else None,
            "rolling_gain": [None if not np.isfinite(v) else float(v) for v in rolling_gain[::DOWNSAMPLE]],
            "rolling_radius": [
                None if not np.isfinite(v) else float(v) for v in rolling_radius[::DOWNSAMPLE]
            ],
            "error_ratio": [None if not np.isfinite(v) else float(v) for v in ratio[::DOWNSAMPLE]],
            "clip_fraction": (
                agent.model.clip_events / agent.model.update_count if agent.model.update_count else 0.0
            ),
        }
    return {
        "persistence_mae": float(np.mean(persistence)),
        "persistence_prefix_mae": float(np.mean(persistence[: min(CHAMPION_PREFIX, persistence.size)])),
        "steps": int(persistence.size),
        "arms": arms,
    }


def run_diagnosis4b(
    ledger: Mapping[str, Any], *, workers: int | None = None, **design: Any
) -> list[dict[str, Any]]:
    return run_cells(
        diagnostic_cells(ledger, **design), GainArms(), reduce_cell, isolate_failures=True, workers=workers
    )


# ----------------------------------------------------------------------
# adjudication
# ----------------------------------------------------------------------
def _fraction_verdict(hits: int, total: int) -> tuple[str, float | None]:
    if total < MINIMUM_CELLS:
        return "INSUFFICIENT_EVIDENCE", None
    fraction = hits / total
    return (
        "SUPPORTED" if fraction >= 0.8 else "CONTRADICTED" if fraction <= 0.5 else "INCONCLUSIVE"
    ), fraction


def _arm(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """The arm's reduction, or an empty (falsy) mapping when the arm failed in that cell."""

    return record["arms"].get(name, {})


def _failed(record: Mapping[str, Any], name: str) -> bool:
    return name in record.get("failures", {})


def adjudicate4b(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coarse = [r for r in records if r["condition"] in COARSE_CONDITIONS]
    champion = [(r, _arm(r, "gru")) for r in coarse if _arm(r, "gru")]
    diverged = [a for _r, a in champion if a["diverged"]]
    stable = [a for _r, a in champion if not a["diverged"]]
    failed_champion = sum(int(_failed(r, "gru")) for r in coarse)

    h24_hits = sum(
        1
        for a in diverged
        if a["gain_crossing"] is not None and a["onset"] is not None and a["gain_crossing"] <= a["onset"]
    )
    h24, f24 = _fraction_verdict(h24_hits, len(diverged))
    h25_hits = sum(1 for a in stable if a["gain_crossing"] is None)
    h25, f25 = _fraction_verdict(h25_hits, len(stable))

    no_switch = [_arm(r, "gru") for r in records if r["condition"] == "coarse_no_switch" and _arm(r, "gru")]
    rising = sum(
        1
        for a in no_switch
        if a["gain_early"] is not None and a["gain_late"] is not None and a["gain_late"] > a["gain_early"]
    )
    part_a, fa = _fraction_verdict(rising, len(no_switch))
    smooth = [_arm(r, "gru") for r in records if r["condition"] == "bouncing" and _arm(r, "gru")]
    flat = sum(
        1 for a in smooth if a["gain_median"] is not None and abs(a["gain_median"]) < SMOOTH_GAIN_BOUND
    )
    part_b, fb = _fraction_verdict(flat, len(smooth))
    if "CONTRADICTED" in (part_a, part_b):
        h26 = "CONTRADICTED"
    elif part_a == part_b == "SUPPORTED":
        h26 = "SUPPORTED"
    elif "INSUFFICIENT_EVIDENCE" in (part_a, part_b):
        h26 = "INSUFFICIENT_EVIDENCE"
    else:
        h26 = "INCONCLUSIVE"

    paired = [
        r
        for r in records
        if r["condition"] == "coarse_no_switch" and _arm(r, "gru") and _arm(r, "gru_lr0.01")
    ]
    champion_prefix = sum(1 for r in paired if _arm(r, "gru")["prefix_diverged"]) + sum(
        int(_failed(r, "gru")) for r in records if r["condition"] == "coarse_no_switch"
    )
    slow_full = sum(1 for r in paired if _arm(r, "gru_lr0.01")["diverged"]) + sum(
        int(_failed(r, "gru_lr0.01")) for r in records if r["condition"] == "coarse_no_switch"
    )
    if champion_prefix < MINIMUM_CELLS:
        h27, r27 = "INSUFFICIENT_EVIDENCE", None
    else:
        r27 = slow_full / champion_prefix
        h27 = "SUPPORTED" if r27 >= 0.5 else "CONTRADICTED" if r27 <= 0.2 else "INCONCLUSIVE"

    ratios = [
        _arm(r, "gru_lr0.01")["gain_crossing"] / _arm(r, "gru")["gain_crossing"]
        for r in paired
        if _arm(r, "gru")["gain_crossing"] and _arm(r, "gru_lr0.01")["gain_crossing"] is not None
    ]
    if len(ratios) < MINIMUM_CELLS:
        h28, m28 = "INSUFFICIENT_EVIDENCE", None
    else:
        m28 = float(np.median(ratios))
        if PACE_RANGE[0] <= m28 <= PACE_RANGE[1]:
            h28 = "SUPPORTED"
        elif m28 < PACE_CONTRADICTED[0] or m28 > PACE_CONTRADICTED[1]:
            h28 = "CONTRADICTED"
        else:
            h28 = "INCONCLUSIVE"

    by_condition = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        by_condition[condition] = {
            name: {
                "cells": len(rows),
                "diverged": sum(1 for r in rows if _arm(r, name) and _arm(r, name)["diverged"]),
                "prefix_diverged": sum(1 for r in rows if _arm(r, name) and _arm(r, name)["prefix_diverged"]),
                "failed": sum(int(_failed(r, name)) for r in rows),
                "crossed": sum(
                    1 for r in rows if _arm(r, name) and _arm(r, name)["gain_crossing"] is not None
                ),
                "median_gain": (
                    float(np.median([_arm(r, name)["gain_median"] for r in rows if _arm(r, name)]))
                    if any(_arm(r, name) for r in rows)
                    else None
                ),
            }
            for name in GAIN_ARMS
        }
    return {
        "by_condition": by_condition,
        "champion_coarse": {
            "diverged": len(diverged),
            "stable": len(stable),
            "failed": failed_champion,
            "crossed_before_onset": h24_hits,
            "stable_never_crossed": h25_hits,
            "crossing_lead_steps": [
                a["onset"] - a["gain_crossing"]
                for a in diverged
                if a["gain_crossing"] is not None and a["onset"] is not None
            ],
        },
        "pace_ratios": ratios,
        "verdicts": {
            "H24": {"verdict": h24, "fraction": f24},
            "H25": {"verdict": h25, "fraction": f25},
            "H26": {
                "verdict": h26,
                "part_a": {"verdict": part_a, "fraction": fa},
                "part_b": {"verdict": part_b, "fraction": fb},
            },
            "H27": {
                "verdict": h27,
                "ratio": r27,
                "champion_prefix_divergences": champion_prefix,
                "slow_divergences": slow_full,
            },
            "H28": {"verdict": h28, "median_ratio": m28, "paired_cells": len(ratios)},
        },
    }
