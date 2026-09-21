"""Diagnosis round 3: the long-horizon degradation round 2 stumbled on.

Round 2's learning-speed test ran the champion on 1120-step fixed-speed coarse
streams and found its error *growing*: 28 of 80 cells ended far worse than
persistence. One exploratory cell (a round-2 diagnostic identity, recorded as
such) showed saturated hidden units, gradient norms in the thousands, the clip
active on almost every update, an output head grown to norm 66 -- and the same
model with its previous-error input zeroed staying accurate. The previous-error
input is unbounded (error / 0.004), so a large error becomes a large input.

These hypotheses were declared after that observation and are judged on a
fresh block (``diagnostic/long-3``), across four long-horizon conditions, so
the diagnosis can say whether the failure is specific to the coarse family.

A cell *diverges* for an arm when its mean error exceeds twice the persistence
baseline on the same stream: worse than doing nothing, by a factor of two.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed
from research.aaa_1k.streams import coarse_speed_stream, motion_compat_stream, occlusion_stream

from .arms import ArmSet
from .harness import Cell, arm_errors, diagnostic_series, run_cells
from .identities import block_seeds, find_block

DIAGNOSIS3_SCHEMA = "aaa.loop.diagnosis.v3"
DIAGNOSTIC_BLOCK_3 = "aaa1k-loop-0001/diagnostic/long-3"
STREAMS = 16
INITIALIZATIONS = 5
LONG_STEPS = 1120
DIVERGENCE_FACTOR = 2.0
MINIMUM_CHAMPION_DIVERGENCES = 5

ARMS = ArmSet(
    (
        "gru",
        "gru_no_error_input",
        "gru_state_reset",
        "probe:gru_lr0.01",
        "probe:gru_keep_bias_-2",
        "rnn28",
        "mlp",
        "persistence",
        "linear_fit",
    )
)
LEARNERS = (
    "gru",
    "gru_no_error_input",
    "gru_state_reset",
    "probe:gru_lr0.01",
    "probe:gru_keep_bias_-2",
    "rnn28",
    "mlp",
)
COARSE_CONDITIONS = ("long_coarse_no_switch", "long_coarse_switching")
OTHER_CONDITIONS = ("long_bouncing", "long_occlusion")

HYPOTHESES: tuple[dict[str, str], ...] = (
    {
        "id": "H16",
        "name": "runaway previous-error feedback",
        "statement": (
            "input 3 feeds the model its own unbounded signed error; once errors grow the input grows, "
            "saturates the state and drives larger errors"
        ),
        "prediction": "the no-error-input ablation diverges in <= 20% as many long coarse cells as the champion",
        "contradicted_by": "the ablation diverging in >= 50% as many cells",
    },
    {
        "id": "H17",
        "name": "learning-rate instability",
        "statement": "the selected learning rate is simply too high for long online runs on quantized targets",
        "prediction": "the champion at lr 0.01 diverges in <= 20% as many long coarse cells",
        "contradicted_by": "lr 0.01 diverging in >= 50% as many cells",
    },
    {
        "id": "H18",
        "name": "recurrence is not required for the runaway",
        "statement": "the feedback loop runs through input 3 and the feed-forward path; hidden-state memory is not needed",
        "prediction": "the state-reset ablation diverges in >= 50% as many long coarse cells as the champion",
        "contradicted_by": "the state-reset ablation diverging in < 20% as many cells",
    },
    {
        "id": "H19",
        "name": "specific to quantized observations",
        "statement": "the runaway needs the unrealizable alternating targets of the coarse family",
        "prediction": (
            "on long smooth (bouncing) and long occlusion streams the champion diverges in <= 20% as many "
            "cells as on the long coarse conditions"
        ),
        "contradicted_by": "a divergence rate on the other families >= 50% of the coarse rate",
    },
)


def _stream(condition: str, seed: int) -> Any:
    if condition == "long_coarse_no_switch":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=10_000)
    if condition == "long_coarse_switching":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=60)
    if condition == "long_bouncing":
        return motion_compat_stream("bouncing", seed, steps=LONG_STEPS, change_step=None)
    if condition == "long_occlusion":
        return occlusion_stream(seed, steps=LONG_STEPS)
    raise ValueError(condition)


def diagnostic_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    block = find_block(ledger, DIAGNOSTIC_BLOCK_3)
    if block["role"] != "diagnostic":
        raise ValueError("diagnosis round 3 must run on a diagnostic block")
    seeds = block_seeds(block)[:STREAMS]
    cells: list[Cell] = []
    for condition in (*COARSE_CONDITIONS, *OTHER_CONDITIONS):
        streams = [_stream(condition, seed) for seed in seeds]
        for init in range(INITIALIZATIONS):
            cells.extend(Cell(condition, init, derive_seed("model_init", init), stream) for stream in streams)
    return cells


def reduce_cell(_cell: Cell, result: RunResult, agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    quarter = max(1, len(result.steps) // 4)
    by_name = {agent.name: agent for agent in agents}
    instrumentation = {}
    for name in ("gru", "probe:gru_keep_bias_-2", "gru_no_error_input"):
        gradient = diagnostic_series(result, name, "last_gradient_norm")
        error_input = np.abs(diagnostic_series(result, name, "previous_signed_error_input"))
        model = by_name[name].model
        instrumentation[name] = {
            "max_gradient_norm": float(np.nanmax(gradient)),
            "clip_fraction": model.clip_events / model.update_count if model.update_count else 0.0,
            "max_abs_error_input": float(np.max(error_input)),
            "final_output_head_norm": float(np.linalg.norm(model.parameters["W_o"])),
            "hidden_saturated_fraction_last_quarter": float(
                np.mean(diagnostic_series(result, name, "hidden_saturated_fraction")[-quarter:])
            ),
        }
    return {
        "mae": {name: float(np.mean(values)) for name, values in errors.items()},
        "last_quarter_mae": {name: float(np.mean(values[-quarter:])) for name, values in errors.items()},
        "max_step_error": {name: float(np.max(values)) for name, values in errors.items()},
        "instrumentation": instrumentation,
    }


def run_diagnosis3(ledger: Mapping[str, Any], *, workers: int | None = None) -> list[dict[str, Any]]:
    return run_cells(diagnostic_cells(ledger), ARMS, reduce_cell, collect_diagnostics=True, workers=workers)


def diverged(record: Mapping[str, Any], arm: str) -> bool:
    return bool(record["mae"][arm] > DIVERGENCE_FACTOR * record["mae"]["persistence"])


def _ratio_verdict(count: int, reference: int, *, supported_below: bool) -> tuple[str, float | None]:
    if reference < MINIMUM_CHAMPION_DIVERGENCES:
        return "NOT_RESOLVED", None
    ratio = count / reference
    if supported_below:
        return ("SUPPORTED" if ratio <= 0.2 else ("CONTRADICTED" if ratio >= 0.5 else "PARTIAL")), ratio
    return ("SUPPORTED" if ratio >= 0.5 else ("CONTRADICTED" if ratio < 0.2 else "PARTIAL")), ratio


def adjudicate3(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition = {
        condition: [r for r in records if r["condition"] == condition]
        for condition in (*COARSE_CONDITIONS, *OTHER_CONDITIONS)
    }
    counts = {
        condition: {arm: int(sum(diverged(r, arm) for r in rows)) for arm in LEARNERS}
        for condition, rows in by_condition.items()
    }
    coarse = {arm: sum(counts[c][arm] for c in COARSE_CONDITIONS) for arm in LEARNERS}
    other = {arm: sum(counts[c][arm] for c in OTHER_CONDITIONS) for arm in LEARNERS}
    champion = coarse["gru"]
    h16, r16 = _ratio_verdict(coarse["gru_no_error_input"], champion, supported_below=True)
    h17, r17 = _ratio_verdict(coarse["probe:gru_lr0.01"], champion, supported_below=True)
    h18, r18 = _ratio_verdict(coarse["gru_state_reset"], champion, supported_below=False)
    h19, r19 = _ratio_verdict(other["gru"], champion, supported_below=True)

    coarse_rows = [r for c in COARSE_CONDITIONS for r in by_condition[c]]
    split = {
        label: [r["instrumentation"]["gru"] for r in coarse_rows if diverged(r, "gru") == flag]
        for label, flag in (("diverged", True), ("stable", False))
    }
    signature = {
        label: {
            field: float(np.median([row[field] for row in rows])) if rows else None
            for field in (
                "max_gradient_norm",
                "clip_fraction",
                "max_abs_error_input",
                "final_output_head_norm",
                "hidden_saturated_fraction_last_quarter",
            )
        }
        for label, rows in split.items()
    }
    return {
        "divergence_rule": f"cell mean error > {DIVERGENCE_FACTOR} x persistence on the same stream",
        "cells_per_condition": {condition: len(rows) for condition, rows in by_condition.items()},
        "divergent_cells": counts,
        "divergent_cells_long_coarse": coarse,
        "divergent_cells_other_families": other,
        "champion_signature_by_outcome": signature,
        "mean_mae": {
            condition: {arm: float(np.mean([r["mae"][arm] for r in rows])) for arm in rows[0]["mae"]}
            for condition, rows in by_condition.items()
        },
        "verdicts": {
            "H16": {"verdict": h16, "ablation_over_champion_divergences": r16},
            "H17": {"verdict": h17, "low_lr_over_champion_divergences": r17},
            "H18": {"verdict": h18, "state_reset_over_champion_divergences": r18},
            "H19": {"verdict": h19, "other_families_over_coarse_divergences": r19},
        },
    }
