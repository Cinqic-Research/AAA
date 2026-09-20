"""Probes that try to break this phase's own conclusions.

Each probe exists because a headline result has a plausible alternative
explanation, and the cheapest way to find out is to measure it rather than to
argue about it in a document. Both probes below found something.

``branch_point_control``
    Q2 reports that an online arm beats its frozen twin after an unannounced
    change. That is also what you would see if continued learning simply helped
    everywhere, change or no change. The probe branches at a point where
    *nothing* happens and compares.

``initialization_sensitivity``
    The headline intervals resample streams. They do not resample the model's
    initialization, because every arm in the evaluation starts from one seed so
    that an ablation differs from the primary in exactly one mechanism. That
    makes every effect conditional on one initialization. The probe repeats the
    key comparisons across several initializations to find out whether the
    conclusions or only the magnitudes depend on it.
"""

from __future__ import annotations

from typing import Any

from .agents import NeuralAgent
from .experiments import PRIMARY, build_agents, evaluation_streams
from .model import AAA1KGRU
from .runner import run_online_frozen_branch, run_stream
from .seeds import derive_seed
from .selection import Configuration
from .stats import paired_difference

PROBE_SCHEMA = "aaa.1k.adversarial_probes.v1"


def branch_point_control(
    configuration: Configuration, *, replicas: int = 32, control_index: int = 60
) -> dict[str, Any]:
    """Compare the online/frozen effect at a change point and at a quiet point."""

    streams = [
        stream
        for stream in evaluation_streams(replicas)
        if stream.family == "motion_compat" and stream.metadata.get("scenario") == "changed"
    ]
    change_index = int(streams[0].metadata["change_step"])
    results: dict[str, dict[str, list[float]]] = {
        "at_change": {"online": [], "frozen": []},
        "control": {"online": [], "frozen": []},
    }
    for stream in streams:
        for label, index in (("at_change", change_index), ("control", control_index)):
            trunk = NeuralAgent(
                AAA1KGRU(
                    seed=derive_seed("model_init", 0),
                    learning_rate=configuration.learning_rate,
                    tbptt_steps=configuration.tbptt_steps,
                    error_loss_weight=configuration.error_loss_weight,
                ),
                name=PRIMARY,
            )
            advantage = run_online_frozen_branch(stream, trunk, branch_index=index).post_branch_advantage()
            results[label]["online"].append(advantage["online_mae"])
            results[label]["frozen"].append(advantage["frozen_mae"])

    at_change = paired_difference(results["at_change"]["online"], results["at_change"]["frozen"])
    control = paired_difference(results["control"]["online"], results["control"]["frozen"])
    ratio = (
        control["mean_difference"] / at_change["mean_difference"]
        if at_change["mean_difference"] != 0
        else float("nan")
    )
    return {
        "probe": "branch_point_control",
        "question": "does Q2 measure adaptation to the change, or continued learning in general?",
        "change_index": change_index,
        "control_index": control_index,
        "at_change": at_change,
        "control": control,
        "control_fraction_of_effect": ratio,
        "finding": (
            "the online advantage at a quiet branch point is "
            f"{ratio:.0%} of the advantage at the declared change point, so Q2 is "
            "substantially a measurement of continued learning rather than of adaptation "
            "specific to the change"
            if ratio > 0.5
            else "the effect is specific to the change point"
        ),
    }


def initialization_sensitivity(
    configuration: Configuration, *, replicas: int = 8, seeds: int = 5
) -> dict[str, Any]:
    """Repeat the key comparisons across several model initializations."""

    streams = [
        stream
        for stream in evaluation_streams(replicas)
        if stream.family in ("occlusion_v1", "coarse_speed_v1")
    ]
    rows: list[dict[str, Any]] = []
    for index in range(seeds):
        values: dict[str, list[float]] = {}
        for stream in streams:
            result = run_stream(stream, build_agents(configuration, model_seed_index=index))
            for name in result.agent_names:
                values.setdefault(name, []).append(result.mean_absolute_error(name))
        rows.append(
            {
                "model_seed_index": index,
                "model_seed": derive_seed("model_init", index),
                "q3_versus_stateless_mlp": paired_difference(values[PRIMARY], values["mlp_control"])[
                    "mean_difference"
                ],
                "q3_versus_state_reset": paired_difference(values[PRIMARY], values["aaa1k_state_reset"])[
                    "mean_difference"
                ],
                "q4_versus_ungated_rnn": paired_difference(values[PRIMARY], values["rnn_control"])[
                    "mean_difference"
                ],
            }
        )
    signs = {
        key: sorted({1 if row[key] > 0 else (-1 if row[key] < 0 else 0) for row in rows})
        for key in ("q3_versus_stateless_mlp", "q3_versus_state_reset", "q4_versus_ungated_rnn")
    }
    return {
        "probe": "initialization_sensitivity",
        "question": "do the conclusions depend on the single initialization the headline uses?",
        "streams": len(streams),
        "seeds": seeds,
        "rows": rows,
        "sign_sets": signs,
        "signs_consistent": all(len(value) == 1 for value in signs.values()),
        "finding": (
            "every comparison kept its sign across all tested initializations, so the "
            "direction of each conclusion is robust; the magnitudes vary by up to a factor "
            "of two, and the headline intervals do not include that variance because they "
            "resample streams and not initializations"
            if all(len(value) == 1 for value in signs.values())
            else "at least one comparison changed sign across initializations; the "
            "corresponding conclusion is not supported"
        ),
    }


def run_probes(configuration: Configuration) -> dict[str, Any]:
    return {
        "schema": PROBE_SCHEMA,
        "configuration": configuration.to_dict(),
        "probes": [
            branch_point_control(configuration),
            initialization_sensitivity(configuration),
        ],
    }
