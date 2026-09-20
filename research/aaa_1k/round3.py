"""AAA-1K evaluation round 3: reviewer-corrected estimands and uncertainty.

Round 2 is retained as observed evidence.  This round uses fresh stream
identities because review changed Q1's estimand and Q2/Q5's sampling design.
"""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

from .experiments import BASELINES, CONTROLS, EVALUATION_PLAN, PRIMARY, build_agents
from .measurements import adaptation_difference_of_differences, probe_bank, retention_trial
from .round2 import _crossed, q3_hidden_state, q6_calibration, q7_baselines
from .round2 import q4_gating as round2_q4_gating
from .runner import run_stream
from .seeds import derive_seed
from .selection import Configuration
from .stats import (
    achieved_precision,
    calibration,
    capability_vector,
    crossed_paired_difference,
    summarize_values,
)
from .streams import Stream, build_stream

ROUND3_SCHEMA = "aaa.1k.experiments.v3"
ROUND3_OFFSET = 40_000
ADAPTATION_OFFSET = 50_000
RETENTION_OFFSET = 60_000
DEFAULT_INITIALIZATIONS = 5
DEFAULT_REPLICAS = 24
ADAPTATION_ENVIRONMENTS = 24
RETENTION_ENVIRONMENTS = 12
FROZEN_PRIMARY = "aaa1k_frozen"


def round3_streams(replicas: int) -> list[Stream]:
    streams: list[Stream] = []
    index = ROUND3_OFFSET
    for family, options in EVALUATION_PLAN:
        for _ in range(replicas):
            streams.append(build_stream(family, derive_seed("evaluation_env", index), **options))
            index += 1
    return streams


def _matrix(cells: list[dict[str, Any]], name: str, key: str = "mae") -> np.ndarray:
    initializations = sorted({cell["model_seed_index"] for cell in cells})
    stream_ids = sorted({cell["stream_id"] for cell in cells})
    lookup = {(cell["model_seed_index"], cell["stream_id"]): cell for cell in cells}
    if len(lookup) != len(initializations) * len(stream_ids):
        raise ValueError("round-3 cells are not a complete initialization-by-stream crossing")
    return np.asarray(
        [
            [lookup[(initialization, stream_id)][key][name] for stream_id in stream_ids]
            for initialization in initializations
        ],
        dtype=float,
    )


def _trial_matrix(trials: list[dict[str, Any]], key: str) -> np.ndarray:
    initializations = sorted({trial["model_seed_index"] for trial in trials})
    environments = sorted({trial["environment_index"] for trial in trials})
    lookup = {(trial["model_seed_index"], trial["environment_index"]): trial for trial in trials}
    if len(lookup) != len(initializations) * len(environments):
        raise ValueError("trials are not a complete initialization-by-environment crossing")
    return np.asarray(
        [
            [lookup[(initialization, environment)][key] for environment in environments]
            for initialization in initializations
        ],
        dtype=float,
    )


def run_round3_pass(
    configuration: Configuration,
    streams: list[Stream],
    *,
    initializations: int,
    architecture_configurations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_steps = 0
    for initialization in range(initializations):
        for stream in streams:
            agents = build_agents(
                configuration,
                model_seed_index=initialization,
                architecture_configurations=architecture_configurations,
            )
            frozen = agents[0].branch(name=FROZEN_PRIMARY, update_enabled=False)
            if agents[0].interaction_state_hash() != frozen.interaction_state_hash():
                raise RuntimeError("Q1 online and frozen arms did not start identically")
            agents.append(frozen)
            result = run_stream(stream, agents)
            total_steps += len(result.steps)
            errors = {name: result.errors(name) for name in result.agent_names}
            quarter = max(1, len(result.steps) // 4)
            scales = agents[0].scales
            estimates = np.asarray([step.error_estimates[PRIMARY] for step in result.steps], dtype=float)
            realized = np.asarray([abs(step.signed_errors[PRIMARY]) for step in result.steps], dtype=float)
            cell: dict[str, Any] = {
                "model_seed_index": initialization,
                "model_seed": derive_seed("model_init", initialization),
                "stream_id": stream.stream_id,
                "family": stream.family,
                "seed": stream.seed,
                "scored_steps": len(result.steps),
                "mae": {name: float(np.mean(values)) for name, values in errors.items()},
                "first_quarter_mae": {
                    name: float(np.mean(values[:quarter])) for name, values in errors.items()
                },
                "last_quarter_mae": {
                    name: float(np.mean(values[-quarter:])) for name, values in errors.items()
                },
                "clip_events": int(agents[0].model.clip_events),
                "clip_rate": agents[0].model.clip_events / agents[0].model.update_count
                if agents[0].model.update_count
                else float("nan"),
                "trained_steps": int(agents[0].trained_steps),
                "skipped_targets": int(agents[0].skipped_targets),
                "nonfinite_events": int(agents[0].model.nonfinite_events),
                "calibration": calibration(estimates, realized * scales.width / scales.displacement_scale),
                "q4_worst_steps": sorted(
                    (
                        {
                            "target_index": step.target_index,
                            "family": stream.family,
                            "regime": step.regime,
                            "event": step.event,
                            "target_observed": step.target_observed,
                            "ungated_minus_gated": step.normalized_absolute_errors["rnn_control"]
                            - step.normalized_absolute_errors[PRIMARY],
                        }
                        for step in result.steps
                    ),
                    key=lambda record: cast(float, record["ungated_minus_gated"]),
                )[:5],
            }
            if stream.family == "occlusion_v1":
                hidden = result.where(target_observed=False)
                cell["occluded_mae"] = {name: hidden.mean_absolute_error(name) for name in result.agent_names}
            cells.append(cell)
    elapsed = time.perf_counter() - started
    return {
        "cells": cells,
        "performance": {
            "wall_seconds": elapsed,
            "scored_transitions": total_steps,
            "transitions_per_second": total_steps / elapsed if elapsed > 0 else float("nan"),
            "arms": len(build_agents(configuration)) + 1,
            "initializations": initializations,
        },
    }


def q1_online_learning(cells: list[dict[str, Any]]) -> dict[str, Any]:
    online, frozen = _matrix(cells, PRIMARY), _matrix(cells, FROZEN_PRIMARY)
    overall = crossed_paired_difference(online, frozen, bootstrap_index=11)
    return {
        "question": "Does online weight updating improve prediction against a matched frozen copy?",
        "design": "online and frozen copies start from identical complete interaction state on every stream",
        "statistic": "frozen minus online mean normalized error",
        "positive_means": "weight updating helped under the same stream time structure",
        "overall": overall,
        "precision": achieved_precision(overall),
        "by_family": {
            family: crossed_paired_difference(
                _matrix([cell for cell in cells if cell["family"] == family], PRIMARY),
                _matrix([cell for cell in cells if cell["family"] == family], FROZEN_PRIMARY),
                bootstrap_index=12,
            )
            for family in sorted({cell["family"] for cell in cells})
        },
    }


def q2_adaptation(trials: list[dict[str, Any]]) -> dict[str, Any]:
    adaptation = _trial_matrix(trials, "adaptation_effect")
    continued = _trial_matrix(trials, "continued_learning_effect")
    zeros = np.zeros_like(adaptation)
    adaptation_record = crossed_paired_difference(zeros, adaptation, bootstrap_index=13)
    continued_record = crossed_paired_difference(zeros, continued, bootstrap_index=14)
    denominator = float(np.mean(adaptation) + np.mean(continued))
    return {
        "question": "Does continued learning help because the world changed?",
        "design": "difference-of-differences, fully crossed over initializations and environments",
        "statistic": "advantage(changed) minus advantage(control)",
        "positive_means": "part of the online advantage is caused by the change",
        "adaptation_effect": adaptation_record,
        "adaptation_precision": achieved_precision(adaptation_record),
        "continued_learning_effect": continued_record,
        "adaptation_share_of_total": float(np.mean(adaptation) / denominator)
        if denominator
        else float("nan"),
        "trunks_matched": all(trial["trunks_matched"] for trial in trials),
        "trials": trials,
    }


def q4_gating(cells: list[dict[str, Any]], fairness: dict[str, Any] | None = None) -> dict[str, Any]:
    record = round2_q4_gating(cells, fairness)
    record["by_family"] = {
        family: _crossed([cell for cell in cells if cell["family"] == family], PRIMARY, "rnn_control")
        for family in sorted({cell["family"] for cell in cells})
    }
    ranked = sorted(
        cells,
        key=lambda cell: cell["mae"]["rnn_control"] - cell["mae"][PRIMARY],
    )
    record["large_gated_loss_cells"] = [
        {
            "model_seed_index": cell["model_seed_index"],
            "stream_id": cell["stream_id"],
            "family": cell["family"],
            "mean_ungated_minus_gated": cell["mae"]["rnn_control"] - cell["mae"][PRIMARY],
            "worst_steps": cell["q4_worst_steps"],
        }
        for cell in ranked[:12]
    ]
    return record


def q5_retention(trials: list[dict[str, Any]]) -> dict[str, Any]:
    forgetting = _trial_matrix(trials, "forgetting")
    zeros = np.zeros_like(forgetting)
    record = crossed_paired_difference(zeros, forgetting, bootstrap_index=15)
    return {
        "question": "Does the model preserve prior capability after learning a new regime?",
        "design": "fixed held-out probe bank; fully crossed over initializations and training environments",
        "statistic": "probe error after B minus probe error after A1",
        "positive_means": "the model forgot regime A while learning B",
        "probe_error": {
            label: summarize_values([trial["probe_error"][label] for trial in trials])
            for label in ("after_A1", "after_B", "after_A2")
        },
        "forgetting": record,
        "reacquisition_gap": summarize_values([trial["reacquisition_gap"] for trial in trials]),
        "trials": trials,
    }


def run_round3(
    configuration: Configuration,
    *,
    replicas: int = DEFAULT_REPLICAS,
    initializations: int = DEFAULT_INITIALIZATIONS,
    adaptation_environments: int = ADAPTATION_ENVIRONMENTS,
    retention_environments: int = RETENTION_ENVIRONMENTS,
    fairness: dict[str, Any] | None = None,
    architecture_configurations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    streams = round3_streams(replicas)
    evaluation = run_round3_pass(
        configuration,
        streams,
        initializations=initializations,
        architecture_configurations=architecture_configurations,
    )
    cells = evaluation["cells"]
    adaptation = []
    for initialization in range(initializations):
        for environment in range(adaptation_environments):
            trial = adaptation_difference_of_differences(
                configuration,
                seed=derive_seed("evaluation_env", ADAPTATION_OFFSET + environment),
                model_seed_index=initialization,
            )
            trial["environment_index"] = environment
            adaptation.append(trial)
    bank = probe_bank()
    retention = []
    for initialization in range(initializations):
        for environment in range(retention_environments):
            trial = retention_trial(
                configuration,
                seed=derive_seed("evaluation_env", RETENTION_OFFSET + environment),
                model_seed_index=initialization,
                bank=bank,
            )
            trial["environment_index"] = environment
            retention.append(trial)
    reference = build_agents(configuration)[0]
    clip_rates = [cell["clip_rate"] for cell in cells if np.isfinite(cell["clip_rate"])]
    return {
        "schema": ROUND3_SCHEMA,
        "round": 3,
        "supersedes": "round 2, retained at docs/evidence/aaa_1k_evaluation_round2_superseded.json",
        "configuration": configuration.to_dict(),
        "architecture_configurations": architecture_configurations
        or {"note": "all arms shared primary configuration"},
        "design": {
            "initializations": initializations,
            "replicas_per_family": replicas,
            "adaptation_environments": adaptation_environments,
            "retention_environments": retention_environments,
            "adaptation_trials": len(adaptation),
            "retention_trials": len(retention),
            "evaluation_seed_offsets": {
                "streams": ROUND3_OFFSET,
                "adaptation": ADAPTATION_OFFSET,
                "retention": RETENTION_OFFSET,
            },
            "note": "fresh round-3 identities; complete crossed designs",
        },
        "streams": [stream.to_summary() for stream in streams],
        "arms": {
            "primary": PRIMARY,
            "frozen_primary": FROZEN_PRIMARY,
            "controls": list(CONTROLS),
            "baselines": list(BASELINES),
        },
        "model_accounting": {
            "parameter_count": reference.model.parameter_count(),
            "parameter_inventory": reference.model.parameter_inventory(),
            "state_footprint": reference.model.state_footprint(),
        },
        "cells": cells,
        "performance": evaluation["performance"],
        "capability_vector": capability_vector(
            {
                "prediction_accuracy": {
                    family: {
                        name: summarize_values(
                            [cell["mae"][name] for cell in cells if cell["family"] == family]
                        )
                        for name in cells[0]["mae"]
                    }
                    for family in sorted({cell["family"] for cell in cells})
                },
                "q1_online_learning": q1_online_learning(cells),
                "q2_adaptation": q2_adaptation(adaptation),
                "q3_hidden_state": q3_hidden_state(cells),
                "q4_gating": q4_gating(cells, fairness),
                "q5_retention": q5_retention(retention),
                "q6_error_calibration": q6_calibration(cells),
                "q7_baseline_competitiveness": q7_baselines(cells),
                "numerical_stability": {
                    "total_clip_events": int(sum(cell["clip_events"] for cell in cells)),
                    "mean_clip_rate": float(np.mean(clip_rates)) if clip_rates else float("nan"),
                    "max_clip_rate": float(np.max(clip_rates)) if clip_rates else float("nan"),
                    "total_nonfinite_events": int(sum(cell["nonfinite_events"] for cell in cells)),
                    "total_trained_steps": int(sum(cell["trained_steps"] for cell in cells)),
                    "total_skipped_targets": int(sum(cell["skipped_targets"] for cell in cells)),
                },
                "compute_cost": evaluation["performance"],
            }
        ),
    }
