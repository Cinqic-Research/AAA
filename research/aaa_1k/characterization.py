"""Development-only probes that settle the three concerns round 1 left open.

None of these touches an evaluation stream. They answer questions *about* the
benchmark and the optimizer, not about the result, and they are kept separate
from selection so that nothing here can influence a chosen hyperparameter.

``clipping_probe``
    Gradient clipping activated on 22% of round-1 updates. At that rate it is
    shaping the optimization rather than guarding it, and the honest question
    is whether the declared threshold is doing the learning. The probe reruns
    the selected configuration at several thresholds, including none at all.

``coarse_speed_decomposition``
    The `coarse_speed_v1` family hides a speed regime *and* quantizes the
    observation, so a win on it could be regime inference or could be nothing
    more than tolerance to a coarse grid. The probe holds the speed fixed --
    removing the regime entirely -- and asks whether the recurrent advantage
    survives.

``per_architecture_selection``
    The gated and ungated arms shared one learning rate, chosen by a rule run
    once. If that rate happened to suit the ungated arm better, Q4's negative
    result would be an artefact of tuning rather than of gating. The probe runs
    the same frozen selection rule independently for each architecture and
    compares each at its own best.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import NeuralAgent
from .controls import StatelessMLPControl, VanillaRNNControl
from .model import AAA1KGRU
from .runner import run_stream
from .seeds import derive_seed
from .selection import (
    LEARNING_RATES,
    Configuration,
    development_streams,
    eligible_learning_rates,
)
from .stats import paired_difference, summarize_values
from .streams import Stream, StreamStep, coarse_speed_stream, occlusion_stream

CHARACTERIZATION_SCHEMA = "aaa.1k.characterization.v1"

CLIP_THRESHOLDS: tuple[float | None, ...] = (0.3, 1.0, 3.0, 10.0, None)


def _agent(model: Any, name: str = "aaa1k") -> NeuralAgent:
    return NeuralAgent(model, name=name)


# ----------------------------------------------------------------------
# is the gradient clip doing the learning?
# ----------------------------------------------------------------------
def clipping_probe(configuration: Configuration, *, model_seed_index: int = 0) -> dict[str, Any]:
    """Rerun the selected configuration across clip thresholds, including none."""

    streams = development_streams()
    records: list[dict[str, Any]] = []
    for threshold in CLIP_THRESHOLDS:
        errors: list[float] = []
        clip_rates: list[float] = []
        diverged: list[str] = []
        for stream in streams:
            agent = _agent(
                AAA1KGRU(
                    seed=derive_seed("model_init", model_seed_index),
                    learning_rate=configuration.learning_rate,
                    tbptt_steps=configuration.tbptt_steps,
                    error_loss_weight=configuration.error_loss_weight,
                    gradient_clip=threshold,
                )
            )
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    result = run_stream(stream, [agent])
            except (FloatingPointError, ValueError, OverflowError) as error:
                diverged.append(f"{stream.stream_id}: {type(error).__name__}")
                continue
            errors.append(result.mean_absolute_error("aaa1k"))
            if agent.model.update_count:
                clip_rates.append(agent.model.clip_events / agent.model.update_count)
        records.append(
            {
                "gradient_clip": threshold,
                "diverged": bool(diverged),
                "diverged_streams": diverged,
                "mean_mae": float(np.mean(errors)) if errors else None,
                "mean_clip_rate": float(np.mean(clip_rates)) if clip_rates else 0.0,
                "streams_completed": len(errors),
            }
        )

    declared = next(record for record in records if record["gradient_clip"] == 1.0)
    unclipped = next(record for record in records if record["gradient_clip"] is None)
    stable = [record for record in records if not record["diverged"]]
    best = min(stable, key=lambda record: record["mean_mae"]) if stable else None
    if unclipped["diverged"]:
        finding = (
            "the model diverges on development data without a clip, so the declared threshold "
            "is load-bearing and is correctly described as a mechanism rather than a guard"
        )
    else:
        cost = declared["mean_mae"] / best["mean_mae"] - 1.0 if best else float("nan")
        if cost > 0.05:
            finding = (
                f"the declared threshold of 1.0 activates on {declared['mean_clip_rate']:.0%} of "
                f"updates and costs {cost:.0%} of development error against the best stable "
                f"threshold. At that rate it is a hyperparameter deciding what gets learned, "
                f"not a guard, and asserting it rather than selecting it was a defect. The "
                f"threshold is now chosen by stage 3 of the development selection."
            )
        else:
            finding = (
                f"the declared threshold activates on {declared['mean_clip_rate']:.0%} of updates "
                f"and costs {cost:.1%} of development error, so it is bounding transient "
                f"gradients without deciding what is learned"
            )
    return {
        "probe": "clipping_probe",
        "question": "is the declared gradient clip shaping the optimization or guarding it?",
        "configuration": configuration.to_dict(),
        "records": records,
        "best_threshold": None if best is None else best["gradient_clip"],
        "declared_threshold_rank": sorted(
            (record["mean_mae"], record["gradient_clip"]) for record in stable
        ).index((declared["mean_mae"], declared["gradient_clip"]))
        + 1
        if stable
        else None,
        "finding": finding,
    }


# ----------------------------------------------------------------------
# does coarse_speed_v1 measure memory or tolerance to a coarse grid?
# ----------------------------------------------------------------------
def coarse_speed_decomposition(
    configuration: Configuration, *, streams: int = 8, initializations: int = 3
) -> dict[str, Any]:
    """Compare the recurrent advantage with and without the hidden speed regime."""

    # a regime length longer than the stream means the speed never switches
    conditions: dict[str, int] = {"no_regime_switch": 10_000, "with_regime_switch": 60}
    results: dict[str, dict[str, list[float]]] = {}
    for label, regime_length in conditions.items():
        gru: list[float] = []
        mlp: list[float] = []
        reset: list[float] = []
        for initialization in range(initializations):
            for index in range(streams):
                stream = coarse_speed_stream(
                    derive_seed("development_env", 500 + index),
                    steps=240,
                    regime_length=regime_length,
                )
                seed = derive_seed("model_init", initialization)
                shared: dict[str, Any] = {"seed": seed, **configuration.model_kwargs()}
                agents = [
                    _agent(AAA1KGRU(**shared), "gru"),
                    _agent(AAA1KGRU(**shared, reset_state_every_step=True), "reset"),
                    _agent(StatelessMLPControl(**shared), "mlp"),
                ]
                result = run_stream(stream, agents)
                gru.append(result.mean_absolute_error("gru"))
                reset.append(result.mean_absolute_error("reset"))
                mlp.append(result.mean_absolute_error("mlp"))
        results[label] = {"gru": gru, "mlp": mlp, "reset": reset}

    comparisons = {
        label: {
            "versus_stateless_mlp": paired_difference(values["gru"], values["mlp"], bootstrap_index=5),
            "versus_state_reset": paired_difference(values["gru"], values["reset"], bootstrap_index=6),
        }
        for label, values in results.items()
    }
    without = comparisons["no_regime_switch"]["versus_stateless_mlp"]["mean_difference"]
    with_switch = comparisons["with_regime_switch"]["versus_stateless_mlp"]["mean_difference"]
    share = without / with_switch if with_switch != 0 else float("nan")
    if share <= 0.0:
        finding = (
            f"with the speed held fixed the recurrent model is {abs(without):.2e} *worse* than "
            "the stateless control, and it is better only once the speed starts switching. "
            "`coarse_speed_v1` is therefore measuring inference of a hidden regime, not "
            "tolerance to a coarse grid -- the quantizer alone hands the advantage to the "
            "stateless arm"
        )
    elif share > 0.7:
        finding = (
            f"the recurrent advantage survives at {share:.0%} of its size when the speed never "
            "switches, so `coarse_speed_v1` is mostly measuring integration through a coarse "
            "observation rather than inference of a hidden regime; the family name overstates "
            "what it isolates"
        )
    elif share < 0.3:
        finding = (
            f"the recurrent advantage falls to {share:.0%} without a regime switch, so the "
            "family is mostly measuring hidden-regime inference"
        )
    else:
        finding = (
            f"the advantage is {share:.0%} of its size without a regime switch, so the family "
            "measures a mixture of coarse-observation integration and regime inference and "
            "does not cleanly isolate either"
        )
    return {
        "probe": "coarse_speed_decomposition",
        "question": "does coarse_speed_v1 measure hidden-regime inference or coarse-observation integration?",
        "conditions": {label: {"regime_length": value} for label, value in conditions.items()},
        "comparisons": comparisons,
        "advantage_share_without_regime": share,
        "finding": finding,
    }


# ----------------------------------------------------------------------
# would the ungated control still win at its own best learning rate?
# ----------------------------------------------------------------------
def _architecture_sweep(
    factory: Any, configuration: Configuration, *, model_seed_index: int
) -> dict[str, Any]:
    streams = development_streams()
    records: list[dict[str, Any]] = []
    for learning_rate in LEARNING_RATES:
        errors: list[float] = []
        diverged = False
        for stream in streams:
            agent = _agent(
                factory(
                    seed=derive_seed("model_init", model_seed_index),
                    learning_rate=learning_rate,
                    tbptt_steps=configuration.tbptt_steps,
                    error_loss_weight=configuration.error_loss_weight,
                    gradient_clip=None,
                )
            )
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    result = run_stream(stream, [agent])
            except (FloatingPointError, ValueError, OverflowError):
                diverged = True
                break
            errors.append(result.mean_absolute_error("aaa1k"))
        records.append(
            {
                "learning_rate": learning_rate,
                "diverged_unclipped": diverged,
                "mean_mae": float(np.mean(errors)) if errors and not diverged else None,
            }
        )
    boundary = next((record["learning_rate"] for record in records if record["diverged_unclipped"]), None)
    allowed = eligible_learning_rates(boundary)
    eligible = [
        record
        for record in records
        if record["learning_rate"] in allowed and not record["diverged_unclipped"]
    ]
    selected = min(eligible, key=lambda record: record["mean_mae"]) if eligible else None
    return {
        "records": records,
        "unclipped_divergence_boundary": boundary,
        "eligible_learning_rates": list(allowed),
        "selected_learning_rate": None if selected is None else selected["learning_rate"],
        "selected_mean_mae": None if selected is None else selected["mean_mae"],
    }


def per_architecture_selection(configuration: Configuration, *, model_seed_index: int = 0) -> dict[str, Any]:
    """Run the frozen selection rule separately for the gated and ungated arms.

    Q4's negative result would be an artefact if the shared learning rate
    happened to suit the ungated arm. This gives each architecture the rate the
    same declared rule picks for it, on development data only, and then asks
    whether the conclusion survives.
    """

    gated = _architecture_sweep(AAA1KGRU, configuration, model_seed_index=model_seed_index)
    ungated = _architecture_sweep(VanillaRNNControl, configuration, model_seed_index=model_seed_index)

    head_to_head = None
    if gated["selected_learning_rate"] is not None and ungated["selected_learning_rate"] is not None:
        streams = development_streams()
        gated_errors: list[float] = []
        ungated_errors: list[float] = []
        for stream in streams:
            shared: dict[str, Any] = {
                "seed": derive_seed("model_init", model_seed_index),
                "tbptt_steps": configuration.tbptt_steps,
                "error_loss_weight": configuration.error_loss_weight,
                "gradient_clip": configuration.gradient_clip,
            }
            agents = [
                _agent(AAA1KGRU(**shared, learning_rate=gated["selected_learning_rate"]), "gru"),
                _agent(
                    VanillaRNNControl(**shared, learning_rate=ungated["selected_learning_rate"]),
                    "rnn",
                ),
            ]
            result = run_stream(stream, agents)
            gated_errors.append(result.mean_absolute_error("gru"))
            ungated_errors.append(result.mean_absolute_error("rnn"))
        head_to_head = paired_difference(gated_errors, ungated_errors, bootstrap_index=7)

    if head_to_head is None:
        finding = "no eligible learning rate for at least one architecture; comparison not made"
    elif head_to_head["ci_high"] < 0:
        finding = (
            "the ungated control still wins when each architecture is given the learning rate "
            "the same declared rule selects for it, so Q4's negative result is not an artefact "
            "of shared tuning"
        )
    elif head_to_head["ci_low"] > 0:
        finding = (
            "the gated model wins once each architecture is tuned by the same rule, so Q4's "
            "negative result WAS an artefact of the shared learning rate and the round-1 "
            "conclusion does not stand"
        )
    else:
        finding = (
            "the comparison is inconclusive once each architecture is tuned separately, so Q4's "
            "negative result is weaker than the shared-rate comparison suggested"
        )
    return {
        "probe": "per_architecture_selection",
        "question": "does Q4's negative result survive giving each architecture its own learning rate?",
        "gated": gated,
        "ungated": ungated,
        "head_to_head_at_own_best": head_to_head,
        "finding": finding,
    }


def width_matched_gating(
    configuration: Configuration, *, streams: int = 8, initializations: int = 3
) -> dict[str, Any]:
    """Development characterization at equal hidden width (16 units)."""

    gated: list[float] = []
    ungated: list[float] = []
    bank = development_streams()[:streams]
    for initialization in range(initializations):
        seed = derive_seed("model_init", initialization)
        shared: dict[str, Any] = {"seed": seed, **configuration.model_kwargs()}
        for stream in bank:
            agents = [
                _agent(AAA1KGRU(**shared), "gated_16"),
                _agent(VanillaRNNControl(**shared, hidden_size=16), "ungated_16"),
            ]
            result = run_stream(stream, agents)
            gated.append(result.mean_absolute_error("gated_16"))
            ungated.append(result.mean_absolute_error("ungated_16"))
    comparison = paired_difference(gated, ungated, bootstrap_index=8)
    return {
        "probe": "width_matched_gating",
        "question": "what happens when gated and ungated recurrence both have 16 hidden units?",
        "gated_parameters": 994,
        "ungated_parameters": int(VanillaRNNControl(hidden_size=16).parameter_count()),
        "comparison": comparison,
        "finding": (
            "at fixed hidden width, ungated minus gated error is "
            f"{comparison['mean_difference']:+.2e} with interval "
            f"[{comparison['ci_low']:+.2e}, {comparison['ci_high']:+.2e}]. This is a "
            "mechanism-at-fixed-state-width characterization, not a replacement for the "
            "frozen near-equal-parameter Q4 comparison"
        ),
    }


def occlusion_sensitivity(configuration: Configuration, *, initializations: int = 3) -> dict[str, Any]:
    """Reviewer-designed development characterization of occlusion mechanics."""

    conditions = [
        ("short_frequent", {"gap_period": 12, "gap_length": 2, "warmup": 8}),
        ("declared", {"gap_period": 20, "gap_length": 4, "warmup": 12}),
        ("long_sparse", {"gap_period": 32, "gap_length": 8, "warmup": 17}),
        ("phase_shifted", {"gap_period": 20, "gap_length": 4, "warmup": 18}),
    ]
    rows: dict[str, dict[str, Any]] = {}
    for label, options in conditions:
        gru: list[float] = []
        reset: list[float] = []
        for initialization in range(initializations):
            seed = derive_seed("model_init", initialization)
            shared: dict[str, Any] = {"seed": seed, **configuration.model_kwargs()}
            for index in range(6):
                stream = occlusion_stream(
                    derive_seed("development_env", 800 + index),
                    steps=180,
                    gap_period=options["gap_period"],
                    gap_length=options["gap_length"],
                    warmup=options["warmup"],
                )
                agents = [
                    _agent(AAA1KGRU(**shared), "gru"),
                    _agent(AAA1KGRU(**shared, reset_state_every_step=True), "reset"),
                ]
                result = run_stream(stream, agents).where(target_observed=False)
                gru.append(result.mean_absolute_error("gru"))
                reset.append(result.mean_absolute_error("reset"))
        rows[label] = paired_difference(gru, reset, bootstrap_index=9)

    stationary_steps = tuple(
        StreamStep(
            index=index, true_position=0.5, observed=(index == 0 or index % 5 != 0), regime="stationary"
        )
        for index in range(121)
    )
    stationary = Stream(
        family="occlusion_v1",
        stream_id="review_stationary_occlusion",
        seed=0,
        steps=stationary_steps,
        metadata={"purpose": "legitimate zero displacement and zero error ambiguity"},
    )
    stationary_errors: list[float] = []
    for initialization in range(initializations):
        agent = _agent(
            AAA1KGRU(seed=derive_seed("model_init", initialization), **configuration.model_kwargs()),
            "gru",
        )
        stationary_errors.append(run_stream(stationary, [agent]).mean_absolute_error("gru"))
    return {
        "probe": "occlusion_sensitivity",
        "question": "is the hidden-state effect robust to gap length, period, phase, initial conditions, and legitimate zeros?",
        "conditions": rows,
        "stationary_zero_displacement": summarize_values(stationary_errors),
        "finding": (
            "development-only gap-period, gap-length, phase, initialization and stationary-zero "
            "conditions are reported separately; the all-zero missingness code remains ambiguous "
            "with genuine stationarity and is not treated as an explicit missingness indicator"
        ),
    }


def run_characterization(configuration: Configuration) -> dict[str, Any]:
    return {
        "schema": CHARACTERIZATION_SCHEMA,
        "note": "development streams only; nothing here influences a selected hyperparameter",
        "configuration": configuration.to_dict(),
        "probes": [
            clipping_probe(configuration),
            coarse_speed_decomposition(configuration),
            per_architecture_selection(configuration),
            width_matched_gating(configuration),
            occlusion_sensitivity(configuration),
        ],
    }
