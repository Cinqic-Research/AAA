"""The seven AAA-1K research questions, frozen before any evaluation stream ran.

Q1  Can AAA-1K learn online?
Q2  Does continued learning help after an unannounced change?
Q3  Does persistent recurrent state provide measurable value?
Q4  Does *gated* recurrence provide value beyond plain recurrence?
Q5  Does the model preserve prior capability across A, B, A?
Q6  Can the model estimate its own likely error?
Q7  Does the extra complexity beat simpler AAA baselines anywhere that matters?

Results are reported as a capability vector. There is no single score, because
a single score is how a failure gets hidden inside a success.

Every arm in a family runs on the *same* stream realizations, so every
comparison is paired. Evaluation streams come from the ``evaluation_env``
namespace and were not touched by
:mod:`research.aaa_1k.selection`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from .agents import NeuralAgent, RLSAgent, baseline_suite
from .controls import StatelessMLPControl, VanillaRNNControl
from .model import AAA1KGRU
from .runner import run_online_frozen_branch, run_stream
from .seeds import derive_seed
from .selection import Configuration, development_streams
from .stats import calibration, capability_vector, paired_difference, summarize_values
from .streams import Stream, build_stream

EXPERIMENT_SCHEMA = "aaa.1k.experiments.v1"

PRIMARY = "aaa1k_gru"
ABLATIONS = ("aaa1k_state_reset", "aaa1k_no_error_input", "aaa1k_frozen_recurrent")
CONTROLS = ("mlp_control", "rnn_control")
BASELINES = (
    "rls_online",
    "persistence",
    "constant_motion",
    "constant_motion_reflected",
    "dead_reckoning",
    "linear_fit",
)

EVALUATION_PLAN: tuple[tuple[str, dict[str, Any]], ...] = (
    ("motion_compat", {"scenario": "bouncing", "steps": 200, "change_step": None}),
    ("motion_compat", {"scenario": "changed", "steps": 200, "change_step": 100}),
    ("motion_compat", {"scenario": "dynamics_change", "steps": 200, "change_step": 100}),
    ("occlusion_v1", {"steps": 240}),
    ("coarse_speed_v1", {"steps": 280}),
    ("aba_v1", {"segment_steps": 200}),
)

# Precision objective, declared before the evaluation streams were generated.
# The replication count is whatever it takes to resolve an effect a quarter of
# the size of the one the pilot measured, bounded so neither noise nor a
# fast computer decides the budget.
PRECISION_TARGET_FRACTION = 0.25
MIN_REPLICAS = 8
MAX_REPLICAS = 32


def build_agents(
    configuration: Configuration,
    *,
    model_seed_index: int = 0,
    architecture_configurations: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Any]:
    """Every arm, with each architecture on its own selected hyperparameters.

    The primary model and its three ablations share one initialization seed and
    one hyperparameter set, because an ablation is the same architecture with
    one mechanism removed and must differ in exactly that.

    The two neural **controls** are different architectures, and imposing the
    gated model's hyperparameters on them is how a comparison becomes a
    handicap. A clip threshold selected for the gated core destabilized the
    stateless control by two orders of magnitude on one family, which would
    have been reported as evidence that hidden state helps. Each control
    therefore runs on whatever the same declared rules select for it, from the
    same development streams; ``architecture_configurations`` carries that
    result, and omitting it falls back to the shared configuration.
    """

    seed = derive_seed("model_init", model_seed_index)
    primary = {"seed": seed, **configuration.model_kwargs()}

    def control_kwargs(name: str) -> dict[str, Any]:
        if not architecture_configurations or name not in architecture_configurations:
            return dict(primary)
        chosen = dict(architecture_configurations[name])
        chosen.pop("architecture", None)
        return {"seed": seed, **chosen}

    return [
        NeuralAgent(AAA1KGRU(**primary), name=PRIMARY),
        NeuralAgent(AAA1KGRU(**primary, reset_state_every_step=True), name="aaa1k_state_reset"),
        NeuralAgent(AAA1KGRU(**primary, zero_error_input=True), name="aaa1k_no_error_input"),
        NeuralAgent(AAA1KGRU(**primary, freeze_recurrent=True), name="aaa1k_frozen_recurrent"),
        NeuralAgent(StatelessMLPControl(**control_kwargs("StatelessMLPControl")), name="mlp_control"),
        NeuralAgent(VanillaRNNControl(**control_kwargs("VanillaRNNControl")), name="rnn_control"),
        RLSAgent(name="rls_online"),
        *baseline_suite(),
    ]


def evaluation_streams(replicas: int) -> list[Stream]:
    """The held-out stream bank. Generated once, from a namespace development never saw."""

    streams: list[Stream] = []
    index = 0
    for family, options in EVALUATION_PLAN:
        for _ in range(replicas):
            streams.append(build_stream(family, derive_seed("evaluation_env", index), **options))
            index += 1
    return streams


# ----------------------------------------------------------------------
# replication planning
# ----------------------------------------------------------------------
def plan_replication(configuration: Configuration) -> dict[str, Any]:
    """Choose the evaluation replication count from a development pilot.

    The pilot measures the per-stream spread of the primary paired effect --
    AAA-1K against the stateless control -- and the count is whatever resolves
    a quarter of that effect at 95%. The number is not chosen because the
    machine can afford more seeds.
    """

    streams = development_streams()
    differences: list[float] = []
    for stream in streams:
        agents = build_agents(configuration)
        result = run_stream(stream, agents)
        differences.append(result.mean_absolute_error("mlp_control") - result.mean_absolute_error(PRIMARY))
    values = np.asarray(differences, dtype=float)
    spread = float(np.std(values, ddof=1))
    effect = float(abs(np.mean(values)))
    target = PRECISION_TARGET_FRACTION * effect
    if target > 0 and spread > 0:
        required = math.ceil((1.96 * spread / target) ** 2)
    else:
        required = MAX_REPLICAS
    replicas = min(MAX_REPLICAS, max(MIN_REPLICAS, required))
    return {
        "pilot_streams": len(streams),
        "pilot_effect": effect,
        "pilot_spread": spread,
        "precision_target_fraction": PRECISION_TARGET_FRACTION,
        "precision_target_absolute": target,
        "required_replicas": required,
        "bounds": [MIN_REPLICAS, MAX_REPLICAS],
        "selected_replicas": replicas,
        "bounded": required > MAX_REPLICAS or required < MIN_REPLICAS,
        "note": (
            "the pilot uses development streams only; the bound was applied because an "
            "unbounded count would be decided by pilot noise rather than by the design"
        ),
    }


# ----------------------------------------------------------------------
# the main evaluation pass
# ----------------------------------------------------------------------
def run_evaluation_pass(
    configuration: Configuration, streams: list[Stream], *, collect_diagnostics: bool = False
) -> dict[str, Any]:
    """One fully paired pass: every arm over every evaluation stream."""

    per_stream: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_steps = 0
    for stream in streams:
        agents = build_agents(configuration)
        result = run_stream(stream, agents, collect_diagnostics=collect_diagnostics)
        total_steps += len(result.steps)
        errors = {name: result.errors(name) for name in result.agent_names}
        quarter = max(1, len(result.steps) // 4)
        primary_estimates = np.asarray([step.error_estimates[PRIMARY] for step in result.steps], dtype=float)
        primary_realized = np.asarray(
            [abs(step.signed_errors[PRIMARY]) for step in result.steps], dtype=float
        )
        scales = agents[0].scales
        entry: dict[str, Any] = {
            "stream_id": stream.stream_id,
            "family": stream.family,
            "seed": stream.seed,
            "scored_steps": len(result.steps),
            "mae": {name: float(np.mean(values)) for name, values in errors.items()},
            "first_quarter_mae": {name: float(np.mean(values[:quarter])) for name, values in errors.items()},
            "last_quarter_mae": {name: float(np.mean(values[-quarter:])) for name, values in errors.items()},
            "clip_events": int(agents[0].model.clip_events),
            "trained_steps": int(agents[0].trained_steps),
            "skipped_targets": int(agents[0].skipped_targets),
            "nonfinite_events": int(agents[0].model.nonfinite_events),
            # The error head works in normalized displacement units; the
            # realized error is converted into the same units so calibration
            # compares like with like.
            "calibration": calibration(
                primary_estimates, primary_realized * scales.width / scales.displacement_scale
            ),
        }
        if stream.family == "occlusion_v1":
            hidden = result.where(target_observed=False)
            visible = result.where(target_observed=True)
            entry["occluded_mae"] = {name: hidden.mean_absolute_error(name) for name in result.agent_names}
            entry["visible_mae"] = {name: visible.mean_absolute_error(name) for name in result.agent_names}
        if stream.family == "aba_v1":
            entry["segment_mae"] = {
                label: {
                    name: result.where(regime=label).mean_absolute_error(name) for name in result.agent_names
                }
                for label in ("A1", "B", "A2")
            }
            entry["segment_tail_mae"] = _segment_tails(result)
        per_stream.append(entry)
    elapsed = time.perf_counter() - started
    return {
        "per_stream": per_stream,
        "performance": {
            "wall_seconds": elapsed,
            "scored_transitions": total_steps,
            "transitions_per_second": total_steps / elapsed if elapsed > 0 else float("nan"),
            "arms": len(build_agents(configuration)),
        },
    }


def _segment_tails(result: Any) -> dict[str, dict[str, float]]:
    """Mean error over the last quarter of each labelled A/B/A segment."""

    tails: dict[str, dict[str, float]] = {}
    for label in ("A1", "B", "A2"):
        segment = result.where(regime=label)
        quarter = max(1, len(segment.steps) // 4)
        tail = segment.slice(len(segment.steps) - quarter)
        tails[label] = {name: tail.mean_absolute_error(name) for name in result.agent_names}
    return tails


def _by_family(per_stream: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [entry for entry in per_stream if entry["family"] == family]


def _paired(entries: list[dict[str, Any]], first: str, second: str, key: str = "mae") -> dict[str, Any]:
    return paired_difference(
        [entry[key][first] for entry in entries],
        [entry[key][second] for entry in entries],
    )


# ----------------------------------------------------------------------
# the questions
# ----------------------------------------------------------------------
def question_one(per_stream: list[dict[str, Any]]) -> dict[str, Any]:
    """Q1: does the model measurably improve from experience?"""

    improvement = paired_difference(
        [entry["last_quarter_mae"][PRIMARY] for entry in per_stream],
        [entry["first_quarter_mae"][PRIMARY] for entry in per_stream],
    )
    by_family = {
        family: paired_difference(
            [entry["last_quarter_mae"][PRIMARY] for entry in _by_family(per_stream, family)],
            [entry["first_quarter_mae"][PRIMARY] for entry in _by_family(per_stream, family)],
        )
        for family in sorted({entry["family"] for entry in per_stream})
    }
    return {
        "question": "Can AAA-1K learn online?",
        "statistic": "first-quarter minus last-quarter mean normalized error, paired by stream",
        "positive_means": "the model improved with experience",
        "overall": improvement,
        "by_family": by_family,
    }


def question_two(branches: list[dict[str, Any]]) -> dict[str, Any]:
    """Q2: does an online copy beat its identical frozen twin after a change?"""

    families = sorted({entry["family"] for entry in branches})
    return {
        "question": "Does continued learning help after an unannounced change?",
        "statistic": "frozen minus online mean normalized error after the branch, paired by stream",
        "positive_means": "continued updating helped",
        "overall": paired_difference(
            [entry["online_mae"] for entry in branches],
            [entry["frozen_mae"] for entry in branches],
        ),
        "by_family": {
            family: paired_difference(
                [e["online_mae"] for e in branches if e["family"] == family],
                [e["frozen_mae"] for e in branches if e["family"] == family],
            )
            for family in families
        },
        "clone_hashes_matched": all(entry["clone_matched"] for entry in branches),
        "branches": branches,
    }


def question_three(per_stream: list[dict[str, Any]]) -> dict[str, Any]:
    """Q3: is persistent recurrent state worth anything?"""

    memory_families = ("occlusion_v1", "coarse_speed_v1")
    memory = [entry for entry in per_stream if entry["family"] in memory_families]
    occluded = [entry for entry in per_stream if entry["family"] == "occlusion_v1"]
    return {
        "question": "Does persistent recurrent state provide measurable value?",
        "statistic": "control minus AAA-1K mean normalized error, paired by stream",
        "positive_means": "the recurrent model was better",
        "memory_families": list(memory_families),
        "versus_state_reset": _paired(memory, PRIMARY, "aaa1k_state_reset"),
        "versus_stateless_mlp": _paired(memory, PRIMARY, "mlp_control"),
        "versus_state_reset_all_families": _paired(per_stream, PRIMARY, "aaa1k_state_reset"),
        "versus_stateless_mlp_all_families": _paired(per_stream, PRIMARY, "mlp_control"),
        "occluded_steps_only": {
            "versus_state_reset": _paired(occluded, PRIMARY, "aaa1k_state_reset", key="occluded_mae"),
            "versus_stateless_mlp": _paired(occluded, PRIMARY, "mlp_control", key="occluded_mae"),
        }
        if occluded
        else {"status": "NOT_VERIFIED", "reason": "no occlusion streams in this run"},
    }


def question_four(per_stream: list[dict[str, Any]]) -> dict[str, Any]:
    """Q4: do the gates earn their parameters, against a plain RNN?"""

    memory = [entry for entry in per_stream if entry["family"] in ("occlusion_v1", "coarse_speed_v1")]
    return {
        "question": "Does gated recurrence provide value beyond plain recurrence?",
        "statistic": "ungated RNN minus AAA-1K mean normalized error, paired by stream",
        "positive_means": "gating helped",
        "note": (
            "the ungated control has 954 parameters against 994, so this compares mechanisms at "
            "close to matched capacity, not a large model against a small one"
        ),
        "all_families": _paired(per_stream, PRIMARY, "rnn_control"),
        "memory_families": _paired(memory, PRIMARY, "rnn_control"),
    }


def question_five(per_stream: list[dict[str, Any]]) -> dict[str, Any]:
    """Q5: what survives A, then B, then A again?"""

    aba = _by_family(per_stream, "aba_v1")
    if not aba:
        return {
            "question": "Does the model preserve prior capability?",
            "status": "NOT_VERIFIED",
            "reason": "no A-B-A streams in this run",
        }
    names = [PRIMARY, *ABLATIONS, *CONTROLS, "rls_online", "constant_motion_reflected"]
    forgetting = {
        name: summarize_values(
            [entry["segment_tail_mae"]["A2"][name] - entry["segment_tail_mae"]["A1"][name] for entry in aba]
        )
        for name in names
    }
    return {
        "question": "Does the model preserve prior capability across A, B, A?",
        "statistic": "mean normalized error over the last quarter of each labelled segment",
        "segments": {
            label: {
                name: summarize_values([entry["segment_tail_mae"][label][name] for entry in aba])
                for name in names
            }
            for label in ("A1", "B", "A2")
        },
        "reacquisition_gap": {
            "definition": "A2 tail minus A1 tail; positive means the model came back worse",
            "values": forgetting,
        },
        "primary_reacquired": paired_difference(
            [entry["segment_tail_mae"]["A2"][PRIMARY] for entry in aba],
            [entry["segment_tail_mae"]["A1"][PRIMARY] for entry in aba],
        ),
    }


def question_six(per_stream: list[dict[str, Any]]) -> dict[str, Any]:
    """Q6: is the error head informative about the error it is about to make?"""

    measured = [entry["calibration"] for entry in per_stream if entry["calibration"]["status"] == "MEASURED"]
    if not measured:
        return {
            "question": "Can the model estimate its own likely error?",
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "no stream produced enough finite pairs to calibrate",
        }
    return {
        "question": "Can the model estimate its own likely error?",
        "status": "MEASURED",
        "units": "normalized displacement, the same units the error head is trained in",
        "spearman": summarize_values([entry["spearman"] for entry in measured]),
        "pearson": summarize_values([entry["pearson"] for entry in measured]),
        "slope": summarize_values([entry["slope"] for entry in measured]),
        "bias": summarize_values([entry["bias"] for entry in measured]),
        "streams_with_monotone_bins": int(sum(entry["bins_monotone"] for entry in measured)),
        "streams_measured": len(measured),
        "caveat": (
            "this is a learned error-magnitude estimate, not a calibrated predictive "
            "distribution and not a Bayesian posterior"
        ),
    }


def question_seven(per_stream: list[dict[str, Any]]) -> dict[str, Any]:
    """Q7: does any of this beat the simple alternatives anywhere that matters?"""

    families = sorted({entry["family"] for entry in per_stream})
    table: dict[str, dict[str, Any]] = {}
    for family in families:
        entries = _by_family(per_stream, family)
        table[family] = {baseline: _paired(entries, PRIMARY, baseline) for baseline in BASELINES}
    wins = {
        family: [name for name, record in row.items() if record["ci_low"] > 0]
        for family, row in table.items()
    }
    return {
        "question": "Does the extra complexity beat simpler AAA baselines anywhere?",
        "statistic": "baseline minus AAA-1K mean normalized error, paired by stream",
        "positive_means": "AAA-1K was better",
        "decision_rule": "a win requires the whole 95% interval above zero, declared in advance",
        "by_family": table,
        "wins_by_family": wins,
        "any_win": bool(any(wins.values())),
    }


# ----------------------------------------------------------------------
# online / frozen branches
# ----------------------------------------------------------------------
def _branch_index(stream: Stream) -> int | None:
    """The declared branch point: the first labelled change, evaluator-side only."""

    for index, step in enumerate(stream.steps):
        if step.event in ("change", "regime_change"):
            return index
    return None


def run_branch_experiment(configuration: Configuration, streams: list[Stream]) -> list[dict[str, Any]]:
    """Q2's paired online/frozen comparison at each stream's declared change point."""

    records: list[dict[str, Any]] = []
    for stream in streams:
        index = _branch_index(stream)
        if index is None or not 0 < index < len(stream.steps) - 2:
            continue
        seed = derive_seed("model_init", 0)
        trunk = NeuralAgent(
            AAA1KGRU(
                seed=seed,
                learning_rate=configuration.learning_rate,
                tbptt_steps=configuration.tbptt_steps,
                error_loss_weight=configuration.error_loss_weight,
            ),
            name=PRIMARY,
        )
        outcome = run_online_frozen_branch(stream, trunk, branch_index=index)
        advantage = outcome.post_branch_advantage()
        records.append(
            {
                "stream_id": stream.stream_id,
                "family": stream.family,
                "branch_index": index,
                "clone_state_hash": outcome.clone_state_hash,
                "clone_matched": True,
                "trunk_mae": outcome.trunk.mean_absolute_error(PRIMARY),
                **advantage,
            }
        )
    return records


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def run_experiments(
    configuration: Configuration, *, replicas: int, collect_diagnostics: bool = False
) -> dict[str, Any]:
    """Run the whole held-out evaluation and answer all seven questions."""

    streams = evaluation_streams(replicas)
    evaluation = run_evaluation_pass(configuration, streams, collect_diagnostics=collect_diagnostics)
    per_stream = evaluation["per_stream"]
    branches = run_branch_experiment(configuration, streams)

    reference = build_agents(configuration)[0]
    return {
        "schema": EXPERIMENT_SCHEMA,
        "configuration": configuration.to_dict(),
        "replicas_per_family": replicas,
        "evaluation_plan": [{"family": family, "options": options} for family, options in EVALUATION_PLAN],
        "streams": [stream.to_summary() for stream in streams],
        "arms": {
            "primary": PRIMARY,
            "ablations": list(ABLATIONS),
            "controls": list(CONTROLS),
            "baselines": list(BASELINES),
        },
        "model_accounting": {
            "parameter_count": reference.model.parameter_count(),
            "parameter_inventory": reference.model.parameter_inventory(),
            "state_footprint": reference.model.state_footprint(),
        },
        "per_stream": per_stream,
        "performance": evaluation["performance"],
        "capability_vector": capability_vector(
            {
                "prediction_accuracy": {
                    family: {
                        name: summarize_values(
                            [entry["mae"][name] for entry in _by_family(per_stream, family)]
                        )
                        for name in per_stream[0]["mae"]
                    }
                    for family in sorted({entry["family"] for entry in per_stream})
                },
                "q1_online_learning": question_one(per_stream),
                "q2_adaptation": question_two(branches),
                "q3_hidden_state": question_three(per_stream),
                "q4_gating": question_four(per_stream),
                "q5_retention": question_five(per_stream),
                "q6_error_calibration": question_six(per_stream),
                "q7_baseline_competitiveness": question_seven(per_stream),
                "numerical_stability": {
                    "total_clip_events": int(sum(entry["clip_events"] for entry in per_stream)),
                    "total_nonfinite_events": int(sum(entry["nonfinite_events"] for entry in per_stream)),
                    "total_trained_steps": int(sum(entry["trained_steps"] for entry in per_stream)),
                    "total_skipped_targets": int(sum(entry["skipped_targets"] for entry in per_stream)),
                },
                "compute_cost": evaluation["performance"],
            }
        ),
    }
