"""AAA-1K evaluation round 2: the corrected designs, on fresh stream identities.

Round 1 is retained unchanged at `docs/evidence/aaa_1k_evaluation.json`. It is
superseded, not deleted, and its two design defects are exactly why this round
exists:

* `AAA-153` -- the online-versus-frozen comparison measured continued learning
  rather than adaptation. Replaced by a difference-of-differences against a
  bit-identical unchanged world.
* `AAA-154` -- the A/B/A comparison confounded retention with accumulated
  experience. Replaced by a fixed frozen probe bank asked the same questions at
  every checkpoint.

Two further corrections apply to every question in this round:

* **initializations are resampled.** Round 1 gave every arm one initialization
  seed so that an ablation differed from the primary in exactly one mechanism,
  then resampled only streams -- which treats the starting weights as fixed by
  nature. This round runs the whole design from several initializations and
  uses a crossed bootstrap over both factors.
* **precision is reported against the effect that was measured**, not only
  against a target sized from a pilot estimate of an effect nobody had seen.

Stream identities are drawn from a declared fresh offset in the
``evaluation_env`` namespace, so no round-1 stream is reused. Nothing in this
round changes the architecture, the hyperparameters or the selection, all of
which were frozen before round 1 and are unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .experiments import (
    BASELINES,
    CONTROLS,
    EVALUATION_PLAN,
    PRIMARY,
    build_agents,
)
from .measurements import (
    adaptation_difference_of_differences,
    probe_bank,
    retention_trial,
)
from .runner import run_stream
from .seeds import derive_seed
from .selection import Configuration
from .stats import (
    achieved_precision,
    calibration,
    capability_vector,
    crossed_paired_difference,
    paired_difference,
    summarize_values,
)
from .streams import Stream, build_stream

ROUND2_SCHEMA = "aaa.1k.experiments.v2"

# Fresh evaluation identities. Round 1 consumed indices 0..191; this round
# starts well clear of them so no stream is reused after its result was seen.
ROUND2_OFFSET = 10_000
ADAPTATION_OFFSET = 20_000
RETENTION_OFFSET = 30_000

DEFAULT_INITIALIZATIONS = 5
DEFAULT_REPLICAS = 24
ADAPTATION_TRIALS = 24
RETENTION_TRIALS = 12

MEMORY_FAMILIES = ("occlusion_v1", "coarse_speed_v1")


def round2_streams(replicas: int) -> list[Stream]:
    """The held-out bank for this round, from a fresh namespace offset."""

    streams: list[Stream] = []
    index = ROUND2_OFFSET
    for family, options in EVALUATION_PLAN:
        for _ in range(replicas):
            streams.append(build_stream(family, derive_seed("evaluation_env", index), **options))
            index += 1
    return streams


def _matrix(cells: list[dict[str, Any]], name: str, key: str = "mae") -> np.ndarray:
    """Reshape flat cells into ``[initialization, stream]`` for the crossed bootstrap."""

    initializations = sorted({cell["model_seed_index"] for cell in cells})
    stream_ids = sorted({cell["stream_id"] for cell in cells})
    lookup = {(cell["model_seed_index"], cell["stream_id"]): cell for cell in cells}
    return np.asarray(
        [
            [lookup[(initialization, stream_id)][key][name] for stream_id in stream_ids]
            for initialization in initializations
        ],
        dtype=float,
    )


def _crossed(cells: list[dict[str, Any]], first: str, second: str, key: str = "mae") -> dict[str, Any]:
    return crossed_paired_difference(_matrix(cells, first, key), _matrix(cells, second, key))


def _select(cells: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [cell for cell in cells if all(cell.get(field) == value for field, value in filters.items())]


def _in_families(cells: list[dict[str, Any]], families: tuple[str, ...]) -> list[dict[str, Any]]:
    return [cell for cell in cells if cell["family"] in families]


# ----------------------------------------------------------------------
# the evaluation pass
# ----------------------------------------------------------------------
def run_round2_pass(
    configuration: Configuration,
    streams: list[Stream],
    *,
    initializations: int,
    architecture_configurations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every arm, over every stream, from every initialization."""

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
                "clip_rate": (
                    agents[0].model.clip_events / agents[0].model.update_count
                    if agents[0].model.update_count
                    else float("nan")
                ),
                "trained_steps": int(agents[0].trained_steps),
                "skipped_targets": int(agents[0].skipped_targets),
                "nonfinite_events": int(agents[0].model.nonfinite_events),
                "calibration": calibration(estimates, realized * scales.width / scales.displacement_scale),
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
            "arms": len(build_agents(configuration)),
            "initializations": initializations,
        },
    }


# ----------------------------------------------------------------------
# questions
# ----------------------------------------------------------------------
def q1_online_learning(cells: list[dict[str, Any]]) -> dict[str, Any]:
    overall = crossed_paired_difference(
        _matrix(cells, PRIMARY, "last_quarter_mae"), _matrix(cells, PRIMARY, "first_quarter_mae")
    )
    return {
        "question": "Can AAA-1K learn online?",
        "statistic": "first-quarter minus last-quarter mean normalized error",
        "positive_means": "the model improved with experience",
        "overall": overall,
        "precision": achieved_precision(overall),
        "by_family": {
            family: crossed_paired_difference(
                _matrix(_select(cells, family=family), PRIMARY, "last_quarter_mae"),
                _matrix(_select(cells, family=family), PRIMARY, "first_quarter_mae"),
            )
            for family in sorted({cell["family"] for cell in cells})
        },
    }


def q2_adaptation(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Difference-of-differences: the part of the advantage caused by the change."""

    adaptation = np.asarray([trial["adaptation_effect"] for trial in trials], dtype=float)
    continued = np.asarray([trial["continued_learning_effect"] for trial in trials], dtype=float)
    zeros = [0.0] * len(trials)
    adaptation_record = paired_difference(zeros, adaptation.tolist(), bootstrap_index=2)
    continued_record = paired_difference(zeros, continued.tolist(), bootstrap_index=3)
    return {
        "question": "Does continued learning help *because the world changed*?",
        "design": (
            "two bit-identical worlds diverging at a declared step, one changed and one not; "
            "the online-minus-frozen advantage on the unchanged world is subtracted from the "
            "advantage on the changed one"
        ),
        "statistic": "advantage(changed) minus advantage(control), paired by trial",
        "positive_means": "part of the online advantage is genuinely caused by the change",
        "adaptation_effect": adaptation_record,
        "adaptation_precision": achieved_precision(adaptation_record),
        "continued_learning_effect": continued_record,
        "adaptation_share_of_total": (
            float(np.mean(adaptation) / (np.mean(adaptation) + np.mean(continued)))
            if (np.mean(adaptation) + np.mean(continued)) != 0
            else float("nan")
        ),
        "trunks_matched": all(trial["trunks_matched"] for trial in trials),
        "trials": trials,
        "supersedes": (
            "round 1's single-branch comparison, which an adversarial probe showed was 95% "
            "reproduced by branching where nothing happened (AAA-153)"
        ),
    }


def q3_hidden_state(cells: list[dict[str, Any]]) -> dict[str, Any]:
    memory = _in_families(cells, MEMORY_FAMILIES)
    occluded = _select(cells, family="occlusion_v1")
    versus_mlp = _crossed(memory, PRIMARY, "mlp_control")
    return {
        "question": "Does persistent recurrent state provide measurable value?",
        "statistic": "control minus AAA-1K mean normalized error",
        "positive_means": "the recurrent model was better",
        "memory_families": list(MEMORY_FAMILIES),
        "versus_state_reset": _crossed(memory, PRIMARY, "aaa1k_state_reset"),
        "versus_stateless_mlp": versus_mlp,
        "precision": achieved_precision(versus_mlp),
        "versus_state_reset_all_families": _crossed(cells, PRIMARY, "aaa1k_state_reset"),
        "versus_stateless_mlp_all_families": _crossed(cells, PRIMARY, "mlp_control"),
        "occluded_steps_only": {
            "versus_state_reset": _crossed(occluded, PRIMARY, "aaa1k_state_reset", "occluded_mae"),
            "versus_stateless_mlp": _crossed(occluded, PRIMARY, "mlp_control", "occluded_mae"),
        },
        "versus_no_error_input": _crossed(memory, PRIMARY, "aaa1k_no_error_input"),
        "versus_frozen_recurrent": _crossed(memory, PRIMARY, "aaa1k_frozen_recurrent"),
    }


def q4_gating(cells: list[dict[str, Any]], fairness: dict[str, Any] | None = None) -> dict[str, Any]:
    memory = _in_families(cells, MEMORY_FAMILIES)
    all_families = _crossed(cells, PRIMARY, "rnn_control")
    return {
        "question": "Does gated recurrence provide value beyond plain recurrence?",
        "statistic": "ungated RNN minus AAA-1K mean normalized error",
        "positive_means": "gating helped",
        "capacity_note": (
            "954 parameters against 994, but 28 hidden units against 16. Matching on "
            "parameters necessarily buys the ungated arm more state, because that is what a "
            "gate costs. This is the honest comparison at a fixed parameter budget and is not "
            "a comparison at matched hidden width."
        ),
        "all_families": all_families,
        "precision": achieved_precision(all_families),
        "memory_families": _crossed(memory, PRIMARY, "rnn_control"),
        "tuning_fairness": fairness
        or {"status": "NOT_VERIFIED", "reason": "per-architecture selection was not run"},
    }


def q5_retention(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Retention measured against a fixed frozen probe bank."""

    forgetting = np.asarray([trial["forgetting"] for trial in trials], dtype=float)
    reacquisition = np.asarray([trial["reacquisition_gap"] for trial in trials], dtype=float)
    forgetting_record = paired_difference([0.0] * len(trials), forgetting.tolist(), bootstrap_index=4)
    return {
        "question": "Does the model preserve prior capability after learning a new regime?",
        "design": (
            "a fixed bank of held-out regime-A episodes, never trained on, evaluated by a "
            "frozen clone with its hidden state reset, at the end of each of A1, B and A2"
        ),
        "statistic": "probe error after B minus probe error after A1",
        "positive_means": "the model forgot regime A while learning B",
        "probe_error": {
            label: summarize_values([trial["probe_error"][label] for trial in trials])
            for label in ("after_A1", "after_B", "after_A2")
        },
        "forgetting": forgetting_record,
        "reacquisition_gap": summarize_values(reacquisition.tolist()),
        "trials": trials,
        "supersedes": (
            "round 1's segment-tail comparison, which confounded retention with having had "
            "three times as much total experience by the final segment (AAA-154)"
        ),
    }


def q6_calibration(cells: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [cell["calibration"] for cell in cells if cell["calibration"]["status"] == "MEASURED"]
    if not measured:
        return {"question": "Can the model estimate its own likely error?", "status": "INSUFFICIENT_EVIDENCE"}
    return {
        "question": "Can the model estimate its own likely error?",
        "status": "MEASURED",
        "units": "normalized displacement, the units the error head is trained in",
        "spearman": summarize_values([entry["spearman"] for entry in measured]),
        "pearson": summarize_values([entry["pearson"] for entry in measured]),
        "slope": summarize_values([entry["slope"] for entry in measured]),
        "bias": summarize_values([entry["bias"] for entry in measured]),
        "cells_with_monotone_bins": int(sum(entry["bins_monotone"] for entry in measured)),
        "cells_measured": len(measured),
        "caveat": (
            "a learned error-magnitude estimate, not a calibrated predictive distribution and "
            "not a Bayesian posterior"
        ),
    }


def q7_baselines(cells: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({cell["family"] for cell in cells})
    table = {
        family: {
            baseline: _crossed(_select(cells, family=family), PRIMARY, baseline) for baseline in BASELINES
        }
        for family in families
    }
    wins = {
        family: sorted(name for name, record in row.items() if record["ci_low"] > 0)
        for family, row in table.items()
    }
    losses = {
        family: sorted(name for name, record in row.items() if record["ci_high"] < 0)
        for family, row in table.items()
    }
    return {
        "question": "Does the extra complexity beat simpler AAA baselines anywhere?",
        "statistic": "baseline minus AAA-1K mean normalized error",
        "positive_means": "AAA-1K was better",
        "decision_rule": "a win requires the whole 95% interval above zero, declared in advance",
        "by_family": table,
        "wins_by_family": wins,
        "losses_by_family": losses,
        "any_win": bool(any(wins.values())),
    }


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def run_round2(
    configuration: Configuration,
    *,
    replicas: int = DEFAULT_REPLICAS,
    initializations: int = DEFAULT_INITIALIZATIONS,
    adaptation_trials: int = ADAPTATION_TRIALS,
    retention_trials: int = RETENTION_TRIALS,
    fairness: dict[str, Any] | None = None,
    architecture_configurations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the corrected evaluation and answer all seven questions."""

    streams = round2_streams(replicas)
    evaluation = run_round2_pass(
        configuration,
        streams,
        initializations=initializations,
        architecture_configurations=architecture_configurations,
    )
    cells = evaluation["cells"]

    adaptation = [
        adaptation_difference_of_differences(
            configuration,
            seed=derive_seed("evaluation_env", ADAPTATION_OFFSET + index),
            model_seed_index=index % initializations,
        )
        for index in range(adaptation_trials)
    ]
    bank = probe_bank()
    retention = [
        retention_trial(
            configuration,
            seed=derive_seed("evaluation_env", RETENTION_OFFSET + index),
            model_seed_index=index % initializations,
            bank=bank,
        )
        for index in range(retention_trials)
    ]

    reference = build_agents(configuration)[0]
    clip_rates = [cell["clip_rate"] for cell in cells if np.isfinite(cell["clip_rate"])]
    return {
        "schema": ROUND2_SCHEMA,
        "round": 2,
        "supersedes": "aaa.1k.experiments.v1 at docs/evidence/aaa_1k_evaluation.json",
        "configuration": configuration.to_dict(),
        "architecture_configurations": architecture_configurations
        or {"note": "all arms shared the primary configuration"},
        "design": {
            "initializations": initializations,
            "replicas_per_family": replicas,
            "adaptation_trials": adaptation_trials,
            "retention_trials": retention_trials,
            "evaluation_seed_offsets": {
                "streams": ROUND2_OFFSET,
                "adaptation": ADAPTATION_OFFSET,
                "retention": RETENTION_OFFSET,
            },
            "note": "fresh evaluation identities; no round-1 stream is reused",
        },
        "streams": [stream.to_summary() for stream in streams],
        "arms": {
            "primary": PRIMARY,
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
                        name: summarize_values([cell["mae"][name] for cell in _select(cells, family=family)])
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
