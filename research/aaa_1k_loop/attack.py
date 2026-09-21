"""Attack: try to make challenger c2 fail before any confirmation identity exists.

Every stream here is from the ``attack/env`` block and every initialization
from the ``attack/init`` block -- five model seeds the champion never trained
from -- except A6, which deliberately replays the champion's own stage-0
divergence probe on AAA-1K's *development* streams so the two stability
boundaries are measured on identical data.

The criteria and thresholds below were committed before the attack ran. The
challenger advances to a freeze only if every attack criterion passes. c2's
known development regression on ``occlusion_v1`` is not an attack criterion:
it is carried unchanged into the frozen promotion criteria (see
``docs/loop_pilot_report.md``, deviation L-4).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.agents import NeuralAgent
from research.aaa_1k.gradcheck import check_model
from research.aaa_1k.runner import RunResult, run_stream
from research.aaa_1k.selection import LEARNING_RATES, development_streams
from research.aaa_1k.stats import crossed_paired_difference
from research.aaa_1k.streams import coarse_speed_stream, motion_compat_stream, occlusion_stream

from .arms import ARMS, ArmSet, gru, gru_with_gate_biases
from .capabilities import adaptation_task, retention_task, trial_matrix
from .develop import plan_streams
from .harness import Cell, arm_errors, matrix, parallel_map, run_cells
from .identities import block_seeds, find_block, require_usable

ATTACK_SCHEMA = "aaa.loop.attack.v1"
ENV_BLOCK = "aaa1k-loop-0001/attack/env"
INIT_BLOCK = "aaa1k-loop-0001/attack/init"
CHALLENGER = "cand:keep_bias_-2"
CHAMPION = "gru"
PRACTICAL_MARGIN = 0.02
DIVERGENCE_FACTOR = 2.0
LONG_STEPS = 1120

NEARBY: dict[str, dict[str, Any]] = {
    "quantum_0.004": {"steps": 280, "quantum": 0.004},
    "quantum_0.006": {"steps": 280, "quantum": 0.006},
    "regime_40": {"steps": 280, "regime_length": 40},
    "regime_90": {"steps": 280, "regime_length": 90},
    "speeds_0.10_0.25": {"steps": 280, "slow_speed": 0.10, "fast_speed": 0.25},
    "speeds_0.15_0.35": {"steps": 280, "slow_speed": 0.15, "fast_speed": 0.35},
    "no_switch": {"steps": 280, "regime_length": 10_000},
}
LONG: tuple[str, ...] = ("long_coarse_no_switch", "long_coarse_switching", "long_bouncing", "long_occlusion")

CRITERIA: dict[str, str] = {
    "A1_target_fresh_initializations": (
        "coarse_speed_v1 (round-3 construction), fresh attack streams and initializations: champion-minus-"
        "challenger error has a 95% interval above zero, every initialization mean is positive, and at least "
        "75% of stream means favour the challenger"
    ),
    "A2_nearby_conditions": (
        "seven nearby coarse conditions: the point improvement is positive in at least 5 of 7 and no "
        "condition's interval lies entirely below zero"
    ),
    "A3_other_families": (
        "the five non-target plan entries: no entry other than occlusion_v1 has the lower bound of its "
        "relative regression above +2%; online-vs-frozen (Q1) for the challenger is POSITIVE; the "
        "challenger's hidden state is not NEGATIVE against its own state-reset ablation on occlusion"
    ),
    "A4_long_horizon_stability": (
        "1120-step coarse (no switch, switching), bouncing and occlusion streams: the challenger has no more "
        "divergent cells (mean error > 2x persistence) than the champion in any condition"
    ),
    "A5_protected_capabilities": (
        "Q2 difference-of-differences adaptation mean is positive for the challenger; Q5 forgetting interval "
        "does not lie entirely above zero"
    ),
    "A6_stability_margin": (
        "the unclipped divergence probe (stage 0 replayed, AAA-1K development streams, horizon 8) finds the "
        "challenger's lowest diverging learning rate no lower than the champion's, so the shared lr 0.03 "
        "still sits at least two grid steps below it"
    ),
    "A7_intervention_neighbourhood": (
        "keep biases -1.5 and -2.5 on the A1 cells: positive point improvement and no more divergent cells "
        "than the champion (c1 at -1 blew up; c2 must not sit on a cliff)"
    ),
    "A8_implementation": (
        "finite-difference gradient check passes; checkpoint round trip and mid-stream resume are bitwise "
        "exact; 994 parameters and the champion's state footprint; online/frozen clones share a complete "
        "interaction-state hash"
    ),
    "A9_causal_boundary": (
        "the challenger receives only float or None through accept_observation, predicts before every "
        "reveal, and is never offered a target it has not been scored on"
    ),
    "A10_compute": "challenger compute per transition within 10% of the champion's",
}


def _seeds(ledger: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    require_usable(ledger, ENV_BLOCK, purpose="selection")
    require_usable(ledger, INIT_BLOCK, purpose="selection")
    return block_seeds(find_block(ledger, ENV_BLOCK)), block_seeds(find_block(ledger, INIT_BLOCK))


def _long_stream(condition: str, seed: int) -> Any:
    if condition == "long_coarse_no_switch":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=10_000)
    if condition == "long_coarse_switching":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=60)
    if condition == "long_bouncing":
        return motion_compat_stream("bouncing", seed, steps=LONG_STEPS, change_step=None)
    return occlusion_stream(seed, steps=LONG_STEPS)


def attack_cells(ledger: Mapping[str, Any]) -> dict[str, list[Cell]]:
    env, inits = _seeds(ledger)
    cells: dict[str, list[Cell]] = {"A1": [], "A2": [], "A3": [], "A4": []}
    for stream_seed in env[0:16]:
        stream = coarse_speed_stream(stream_seed, steps=280)
        cells["A1"].extend(Cell("A1", i, seed, stream) for i, seed in enumerate(inits))
    for label, options in NEARBY.items():
        for stream_seed in env[16:24]:
            stream = coarse_speed_stream(stream_seed, **options)
            cells["A2"].extend(Cell(label, i, seed, stream) for i, seed in enumerate(inits))
    for entry, stream in plan_streams(env[24:72], 8):
        if stream.family == "coarse_speed_v1":
            continue
        cells["A3"].extend(Cell(entry, i, seed, stream) for i, seed in enumerate(inits))
    for condition in LONG:
        for stream_seed in env[72:80]:
            stream = _long_stream(condition, stream_seed)
            cells["A4"].extend(Cell(condition, i, seed, stream) for i, seed in enumerate(inits))
    return cells


ARMSETS = {
    "A1": ArmSet(
        (
            CHAMPION,
            CHALLENGER,
            f"{CHALLENGER}:frozen",
            "probe:gru_keep_bias_-1.5",
            "probe:gru_keep_bias_-2.5",
            "rnn28",
            "persistence",
        )
    ),
    "A2": ArmSet((CHAMPION, CHALLENGER, "persistence")),
    "A3": ArmSet(
        (
            CHAMPION,
            CHALLENGER,
            f"{CHALLENGER}:frozen",
            f"{CHALLENGER}:state_reset",
            "gru_state_reset",
            "mlp",
            "rnn28",
            "persistence",
        )
    ),
    "A4": ArmSet((CHAMPION, CHALLENGER, "persistence")),
}


def reduce_cell(cell: Cell, result: RunResult, _agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    record: dict[str, Any] = {
        "mae": {name: float(np.mean(values)) for name, values in errors.items()},
        "max_step_error": {name: float(np.max(values)) for name, values in errors.items()},
    }
    if cell.stream.family == "occlusion_v1":
        hidden = result.where(target_observed=False)
        record["occluded_mae"] = {name: hidden.mean_absolute_error(name) for name in result.agent_names}
    return record


# ----------------------------------------------------------------------
# A6: the champion's stage-0 divergence probe, replayed for both arms
# ----------------------------------------------------------------------
def _divergence_task(payload: tuple[str, float, int]) -> dict[str, Any]:
    which, learning_rate, stream_index = payload
    stream = development_streams()[stream_index]
    from research.aaa_1k.seeds import derive_seed

    seed = derive_seed("model_init", 0)
    options: dict[str, Any] = {"learning_rate": learning_rate, "tbptt_steps": 8, "gradient_clip": None}
    model = (
        gru(seed, **options) if which == CHAMPION else gru_with_gate_biases(seed, keep_bias=-2.0, **options)
    )
    agent = NeuralAgent(model, name="probe")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = run_stream(stream, [agent])
    except (FloatingPointError, ValueError, OverflowError) as error:
        return {
            "arm": which,
            "learning_rate": learning_rate,
            "stream": stream.stream_id,
            "diverged": True,
            "error": type(error).__name__,
        }
    return {
        "arm": which,
        "learning_rate": learning_rate,
        "stream": stream.stream_id,
        "diverged": False,
        "worst_normalized_error": float(np.max(result.errors("probe"))),
    }


def divergence_probe(*, workers: int | None = None) -> dict[str, Any]:
    count = len(development_streams())
    payloads = [
        (arm, lr, index) for arm in (CHAMPION, CHALLENGER) for lr in LEARNING_RATES for index in range(count)
    ]
    rows = parallel_map(_divergence_task, payloads, workers=workers)
    boundary = {}
    for arm in (CHAMPION, CHALLENGER):
        diverging = sorted({row["learning_rate"] for row in rows if row["arm"] == arm and row["diverged"]})
        boundary[arm] = diverging[0] if diverging else None
    return {"horizon": 8, "gradient_clip": None, "rows": rows, "lowest_diverging_learning_rate": boundary}


# ----------------------------------------------------------------------
# A8 / A9: implementation and causal-boundary checks on the challenger
# ----------------------------------------------------------------------
class _Recorder:
    """Wraps an agent and logs the order and type of everything it is given."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self.update_enabled = inner.update_enabled
        self.log: list[tuple[str, str]] = []

    def begin_episode(self) -> None:
        self.inner.begin_episode()

    def accept_observation(self, observation: float | None) -> None:
        self.log.append(("reveal", type(observation).__name__))
        self.inner.accept_observation(observation)

    def predict(self) -> float:
        self.log.append(("predict", ""))
        return float(self.inner.predict())


def implementation_checks(init_seed: int, env_seed: int) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    gradient = check_model(
        lambda: gru_with_gate_biases(1, keep_bias=-2.0, tbptt_steps=32), label="c2", seed=21, exhaustive=True
    )
    checks["gradient_check"] = {
        key: gradient[key] for key in ("checked", "violations", "max_absolute_error", "passed")
    }

    stream = occlusion_stream(env_seed, steps=160)
    whole = ARMS[CHALLENGER].build(init_seed)
    uninterrupted = run_stream(stream, [whole])
    first = ARMS[CHALLENGER].build(init_seed)
    run_stream(stream, [first], stop=60)
    saved = json.loads(json.dumps(first.state_dict(), allow_nan=False))
    resumed = ARMS[CHALLENGER].build(init_seed + 1)
    resumed.load_state(saved)
    resumed.name = CHALLENGER
    continued = run_stream(stream, [resumed], start=60, begin_episode=False)
    tail = [step.predictions[CHALLENGER] for step in uninterrupted.steps[60:]]
    checks["resume_bitwise"] = tail == [step.predictions[CHALLENGER] for step in continued.steps]
    checks["resume_final_state_hash_equal"] = whole.model.state_hash() == resumed.model.state_hash()
    roundtrip = type(first.model).from_state_dict(json.loads(json.dumps(first.model.state_dict())))
    checks["checkpoint_roundtrip_exact"] = roundtrip.state_hash() == first.model.state_hash()

    champion = ARMS[CHAMPION].build(init_seed).model
    challenger = ARMS[CHALLENGER].build(init_seed).model
    checks["parameter_count"] = challenger.parameter_count()
    checks["champion_parameter_count"] = champion.parameter_count()
    checks["state_footprint_equal"] = challenger.state_footprint() == champion.state_footprint()
    checks["state_footprint"] = challenger.state_footprint()
    checks["initial_parameters_differ_only_in_b_z"] = all(
        np.array_equal(array, challenger.parameters[name])
        for name, array in champion.parameters.items()
        if name != "b_z"
    ) and not np.array_equal(champion.parameters["b_z"], challenger.parameters["b_z"])
    online = first.branch(name="online", update_enabled=True)
    frozen = first.branch(name="frozen", update_enabled=False)
    checks["branch_interaction_hash_equal"] = (
        online.interaction_state_hash() == frozen.interaction_state_hash()
    )

    recorder = _Recorder(ARMS[CHALLENGER].build(init_seed))
    run_stream(stream, [recorder])
    types = {kind for event, kind in recorder.log if event == "reveal"}
    body = recorder.log[1:]
    alternates = all(
        body[index][0] == "predict" and body[index + 1][0] == "reveal" for index in range(0, len(body) - 1, 2)
    )
    trained = recorder.inner.trained_steps
    both_observed = sum(
        1
        for index in range(1, len(stream.steps))
        if stream.steps[index].observed and stream.steps[index - 1].observed
    )
    checks["reveal_types"] = sorted(types)
    checks["predict_precedes_every_reveal"] = alternates and recorder.log[0] == ("reveal", "float")
    checks["trained_only_on_observed_transitions"] = trained == both_observed
    return checks


def _compute(init_seed: int, env_seed: int) -> dict[str, Any]:
    stream = coarse_speed_stream(env_seed, steps=280)
    timings: dict[str, float] = {}
    for arm in (CHAMPION, CHALLENGER):
        best = math.inf
        for _ in range(3):
            agent = ARMS[arm].build(init_seed)
            started = time.perf_counter()
            run_stream(stream, [agent])
            best = min(best, time.perf_counter() - started)
        timings[arm] = best / (len(stream.steps) - 1)
    return {"seconds_per_transition": timings, "ratio": timings[CHALLENGER] / timings[CHAMPION]}


def run_attack(ledger: Mapping[str, Any], *, workers: int | None = None) -> dict[str, Any]:
    env, inits = _seeds(ledger)
    cells = attack_cells(ledger)
    records = {
        stage: run_cells(stage_cells, ARMSETS[stage], reduce_cell, workers=workers, isolate_failures=True)
        for stage, stage_cells in cells.items()
    }
    adaptation = parallel_map(
        adaptation_task,
        [
            (arm, env_seed, init)
            for arm in (CHAMPION, CHALLENGER)
            for env_seed in env[80:88]
            for init in inits
        ],
        workers=workers,
    )
    retention = parallel_map(
        retention_task,
        [
            (arm, env_seed, init)
            for arm in (CHAMPION, CHALLENGER)
            for env_seed in env[88:92]
            for init in inits
        ],
        workers=workers,
    )
    return {
        "records": records,
        "adaptation_trials": adaptation,
        "retention_trials": retention,
        "divergence_probe": divergence_probe(workers=workers),
        "implementation": implementation_checks(inits[0], env[92]),
        "compute": _compute(inits[0], env[93]),
    }


# ----------------------------------------------------------------------
# adjudication
# ----------------------------------------------------------------------
def _diverged(record: Mapping[str, Any], arm: str) -> bool:
    return bool(record["mae"][arm] > DIVERGENCE_FACTOR * record["mae"]["persistence"])


def _failed(records: Sequence[Mapping[str, Any]], arm: str) -> int:
    return int(sum(arm in record.get("failures", {}) for record in records))


def _improvement(
    records: Sequence[Mapping[str, Any]], arm: str, index: int, key: str = "mae"
) -> dict[str, Any]:
    return crossed_paired_difference(
        matrix(records, arm, key=key), matrix(records, CHAMPION, key=key), bootstrap_index=index
    )


def _verdict(record: Mapping[str, Any]) -> str:
    if record["interval_status"] != "MEASURED":
        return "INSUFFICIENT_EVIDENCE"
    if record["ci_low"] > 0:
        return "POSITIVE"
    if record["ci_high"] < 0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def adjudicate_attack(result: Mapping[str, Any]) -> dict[str, Any]:
    records = result["records"]
    outcome: dict[str, Any] = {}

    a1 = records["A1"]
    target = _improvement(a1, CHALLENGER, 400)
    outcome["A1_target_fresh_initializations"] = {
        "improvement": target,
        "passed": bool(
            _failed(a1, CHALLENGER) == 0
            and target["ci_low"] > 0
            and all(value > 0 for value in target["per_initialization_difference"])
            and target["favours_first"] >= 0.75 * target["streams"]
        ),
    }

    a2: dict[str, Any] = {}
    for offset, label in enumerate(NEARBY):
        rows = [record for record in records["A2"] if record["condition"] == label]
        a2[label] = _improvement(rows, CHALLENGER, 410 + offset)
    positive = sum(entry["mean_difference"] > 0 for entry in a2.values())
    outcome["A2_nearby_conditions"] = {
        "conditions": a2,
        "positive_conditions": positive,
        "passed": bool(
            _failed(records["A2"], CHALLENGER) == 0
            and positive >= 5
            and all(entry["ci_high"] >= 0 for entry in a2.values())
        ),
    }

    a3 = records["A3"]
    per_entry: dict[str, Any] = {}
    for offset, entry in enumerate(sorted({record["condition"] for record in a3})):
        rows = [record for record in a3 if record["condition"] == entry]
        champion_mean = float(np.mean([row["mae"][CHAMPION] for row in rows]))
        comparison = crossed_paired_difference(
            matrix(rows, CHAMPION), matrix(rows, CHALLENGER), bootstrap_index=430 + offset
        )
        per_entry[entry] = {
            "champion_mean": champion_mean,
            "relative_regression": comparison["mean_difference"] / champion_mean,
            "relative_regression_ci": [
                comparison["ci_low"] / champion_mean,
                comparison["ci_high"] / champion_mean,
            ],
        }
    new_regressions = [
        entry
        for entry, row in per_entry.items()
        if entry != "occlusion_v1" and row["relative_regression_ci"][0] > PRACTICAL_MARGIN
    ]
    q1_rows = a1 + a3
    q1 = crossed_paired_difference(
        matrix(q1_rows, CHALLENGER), matrix(q1_rows, f"{CHALLENGER}:frozen"), bootstrap_index=440
    )
    occlusion = [record for record in a3 if record["family"] == "occlusion_v1"]
    q3 = crossed_paired_difference(
        matrix(occlusion, CHALLENGER), matrix(occlusion, f"{CHALLENGER}:state_reset"), bootstrap_index=441
    )
    outcome["A3_other_families"] = {
        "per_plan_entry": per_entry,
        "new_regressions": new_regressions,
        "q1_challenger_online_vs_frozen": q1,
        "q3_challenger_vs_its_state_reset_occlusion": q3,
        "passed": bool(
            _failed(a3, CHALLENGER) == 0
            and not new_regressions
            and _verdict(q1) == "POSITIVE"
            and _verdict(q3) != "NEGATIVE"
        ),
    }

    a4: dict[str, Any] = {}
    for condition in LONG:
        rows = [record for record in records["A4"] if record["condition"] == condition]
        a4[condition] = {
            "cells": len(rows),
            "champion_divergent": sum(_diverged(row, CHAMPION) for row in rows if CHAMPION in row["mae"]),
            "challenger_divergent": sum(
                _diverged(row, CHALLENGER) for row in rows if CHALLENGER in row["mae"]
            ),
            "challenger_failures": _failed(rows, CHALLENGER),
            "champion_mean": float(np.mean([row["mae"][CHAMPION] for row in rows])),
            "challenger_mean": float(np.mean([row["mae"][CHALLENGER] for row in rows])),
        }
    outcome["A4_long_horizon_stability"] = {
        "conditions": a4,
        "passed": all(
            row["challenger_failures"] == 0 and row["challenger_divergent"] <= row["champion_divergent"]
            for row in a4.values()
        ),
    }

    adaptation = result["adaptation_trials"]
    retention = result["retention_trials"]
    q2 = {
        arm: float(np.mean(trial_matrix(adaptation, arm, "adaptation_effect")))
        for arm in (CHAMPION, CHALLENGER)
    }
    zeros = np.zeros_like(trial_matrix(retention, CHALLENGER, "forgetting"))
    q5 = crossed_paired_difference(
        zeros, trial_matrix(retention, CHALLENGER, "forgetting"), bootstrap_index=450
    )
    outcome["A5_protected_capabilities"] = {
        "q2_adaptation_mean": q2,
        "q5_challenger_forgetting": q5,
        "passed": bool(q2[CHALLENGER] > 0 and q5["ci_low"] <= 0),
    }

    boundary = result["divergence_probe"]["lowest_diverging_learning_rate"]

    def rank(value: float | None) -> float:
        return math.inf if value is None else float(value)

    outcome["A6_stability_margin"] = {
        "lowest_diverging_learning_rate": boundary,
        "passed": rank(boundary[CHALLENGER]) >= rank(boundary[CHAMPION]),
    }

    neighbourhood: dict[str, dict[str, Any]] = {}
    for offset, arm in enumerate(("probe:gru_keep_bias_-1.5", "probe:gru_keep_bias_-2.5")):
        improvement = _improvement(a1, arm, 460 + offset) if _failed(a1, arm) == 0 else None
        neighbourhood[arm] = {
            "failures": _failed(a1, arm),
            "improvement": improvement,
            "divergent": sum(_diverged(row, arm) for row in a1 if arm in row["mae"]),
            "champion_divergent": sum(_diverged(row, CHAMPION) for row in a1),
        }
    outcome["A7_intervention_neighbourhood"] = {
        "arms": neighbourhood,
        "passed": all(
            row["failures"] == 0
            and row["improvement"] is not None
            and row["improvement"]["mean_difference"] > 0
            and row["divergent"] <= row["champion_divergent"]
            for row in neighbourhood.values()
        ),
    }

    checks = result["implementation"]
    outcome["A8_implementation"] = {
        "checks": {
            key: checks[key]
            for key in checks
            if key
            not in ("reveal_types", "predict_precedes_every_reveal", "trained_only_on_observed_transitions")
        },
        "passed": bool(
            checks["gradient_check"]["passed"]
            and checks["resume_bitwise"]
            and checks["resume_final_state_hash_equal"]
            and checks["checkpoint_roundtrip_exact"]
            and checks["parameter_count"] == checks["champion_parameter_count"] == 994
            and checks["state_footprint_equal"]
            and checks["initial_parameters_differ_only_in_b_z"]
            and checks["branch_interaction_hash_equal"]
        ),
    }
    outcome["A9_causal_boundary"] = {
        "reveal_types": checks["reveal_types"],
        "passed": bool(
            set(checks["reveal_types"]) <= {"float", "NoneType"}
            and checks["predict_precedes_every_reveal"]
            and checks["trained_only_on_observed_transitions"]
        ),
    }
    outcome["A10_compute"] = {**result["compute"], "passed": bool(result["compute"]["ratio"] <= 1.1)}
    advance = all(entry["passed"] for entry in outcome.values())
    return {"criteria": CRITERIA, "outcome": outcome, "advance_to_freeze": advance}
