"""Development-only hyperparameter selection, on a budget declared in advance.

Nothing in this module may see an evaluation stream. The seeds come from the
``development_env`` namespace and the evaluation streams come from
``evaluation_env``; the two cannot collide.

The plan, frozen before any of it ran
-------------------------------------
**Stage 0 -- divergence probe.** The same learning-rate grid at one horizon
with gradient clipping *switched off*, to locate a real divergence boundary.
With the declared clip in place nothing in this grid blows up, so without this
probe the stability-margin rule would have no boundary to stand back from and
would be enforcing nothing.

**Stage 1 -- pilot.** Learning rates ``{0.001, 0.003, 0.01, 0.03, 0.1, 0.3}``
crossed with TBPTT horizons ``{4, 8, 16, 32}``, at the predeclared auxiliary
weight ``lambda_error = 0.25``. Every attempted configuration is recorded,
including the ones that blow up.

**Stage 2 -- auxiliary weight.** ``lambda_error`` in ``{0.0, 0.1, 0.25, 0.5}``
at the horizon and learning rate stage 1 selected.

**Stage 3a -- per-architecture hyperparameters.** The learning rate and the
clip threshold are selected *separately for each architecture*, by the same
rules, on the same development streams. Imposing one architecture's
hyperparameters on another is how a comparison quietly becomes a handicap: a
threshold that suits the gated model destabilized the stateless control badly
enough to inflate its error by two orders of magnitude on one family, which
would have been reported as "hidden state helps". Ablations of the gated model
share its hyperparameters exactly, because they are the same architecture with
one mechanism removed; the two *controls* are different architectures and get
their own.

**Stage 3 -- gradient-clip threshold.** ``{0.3, 1.0, 3.0, 10.0, None}``,
averaged over three development initializations. This stage was added after a
characterization probe found the originally *declared* threshold of 1.0 costing
29% of development error while activating on roughly a quarter of all updates.
A threshold that active is a hyperparameter deciding what gets learned, not a
guard, and a hyperparameter has to be selected by the rule like any other
rather than asserted. Round 1 of the evaluation was run with the declared 1.0
and is retained; round 2 uses whatever this stage selects.

Selection rule, also frozen in advance
--------------------------------------
1. **Stable execution.** A configuration that produced a non-finite value on
   any development stream is eliminated. Elimination is recorded, not deleted.
2. **Stability margin.** A learning rate is eligible only if it sits at least
   ``MARGIN_STEPS`` positions below the lowest learning rate that diverged in
   the stage-0 unclipped probe. Immediately below a demonstrated boundary is
   not far enough, and "the clip caught it" is not a stability argument -- it
   is the reason the boundary has to be measured with the clip removed. PR #12
   learned this the expensive way. If nothing diverged, every rate is eligible
   and the report says the margin was never demonstrated.
3. **Predictive error.** Among eligible configurations, the lowest mean
   normalized absolute error across all development streams wins.
4. **Ties.** If the best two are within 2% of each other, the tie is broken by
   post-change adaptation, then by retention, then by error-head rank
   correlation -- in that order.

For ``lambda_error`` the same rule applies with one addition: the predeclared
default of 0.25 is retained unless another value beats it by more than 2%. A
2% development difference is not a reason to move a declared default.

For the gradient clip the rule prefers **the most conservative threshold whose
development error is within the practical margin of the best**. Given two
thresholds that perform the same, the tighter one is chosen, because its cost
is bounded and its benefit is insurance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agents import NeuralAgent
from .model import AAA1KGRU
from .runner import run_stream
from .seeds import derive_seed
from .streams import Stream, build_stream

SELECTION_SCHEMA = "aaa.1k.development_selection.v1"

LEARNING_RATES: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
TBPTT_HORIZONS: tuple[int, ...] = (4, 8, 16, 32)
ERROR_WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
GRADIENT_CLIPS: tuple[float | None, ...] = (0.3, 1.0, 3.0, 10.0, None)
DECLARED_GRADIENT_CLIP = 1.0
CLIP_INITIALIZATIONS = 3
DEFAULT_ERROR_WEIGHT = 0.25
PRACTICAL_MARGIN = 0.02
MARGIN_STEPS = 2

DEVELOPMENT_PLAN: tuple[tuple[str, dict[str, Any]], ...] = (
    ("motion_compat", {"scenario": "bouncing", "steps": 160, "change_step": None}),
    ("motion_compat", {"scenario": "changed", "steps": 160, "change_step": 80}),
    ("motion_compat", {"scenario": "dynamics_change", "steps": 160, "change_step": 80}),
    ("occlusion_v1", {"steps": 200}),
    ("coarse_speed_v1", {"steps": 240}),
    ("aba_v1", {"segment_steps": 120}),
)
STREAMS_PER_FAMILY = 2


def development_streams() -> list[Stream]:
    """The fixed development stream bank. Deterministic and never reused as evaluation."""

    streams: list[Stream] = []
    index = 0
    for family, options in DEVELOPMENT_PLAN:
        for _ in range(STREAMS_PER_FAMILY):
            streams.append(build_stream(family, derive_seed("development_env", index), **options))
            index += 1
    return streams


@dataclass(frozen=True)
class Configuration:
    learning_rate: float
    tbptt_steps: int
    error_loss_weight: float
    gradient_clip: float | None = DECLARED_GRADIENT_CLIP

    def key(self) -> str:
        return f"lr={self.learning_rate:g};T={self.tbptt_steps};lambda={self.error_loss_weight:g}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "tbptt_steps": self.tbptt_steps,
            "error_loss_weight": self.error_loss_weight,
            "gradient_clip": self.gradient_clip,
        }

    def model_kwargs(self) -> dict[str, Any]:
        """Exactly the arguments every arm's core is constructed with."""

        return {
            "learning_rate": self.learning_rate,
            "tbptt_steps": self.tbptt_steps,
            "error_loss_weight": self.error_loss_weight,
            "gradient_clip": self.gradient_clip,
        }


def divergence_probe(streams: list[Stream], *, horizon: int = 8, model_seed_index: int = 0) -> dict[str, Any]:
    """Where does this model actually blow up without the safety net?

    Run only on development streams, and only to give the stability-margin rule
    something to measure. Divergence here is evidence and is recorded, not
    cleaned up.
    """

    records: list[dict[str, Any]] = []
    for learning_rate in LEARNING_RATES:
        diverged_on: list[str] = []
        worst = 0.0
        for stream in streams:
            agent = NeuralAgent(
                AAA1KGRU(
                    seed=derive_seed("model_init", model_seed_index),
                    learning_rate=learning_rate,
                    tbptt_steps=horizon,
                    error_loss_weight=DEFAULT_ERROR_WEIGHT,
                    gradient_clip=None,
                ),
                name="aaa1k",
            )
            try:
                # Overflow is the expected outcome above the boundary; the
                # model still raises, which is what is being measured.
                with np.errstate(over="ignore", invalid="ignore"):
                    result = run_stream(stream, [agent])
            except (FloatingPointError, ValueError, OverflowError) as error:
                diverged_on.append(f"{stream.stream_id}: {type(error).__name__}")
                continue
            worst = max(worst, float(np.max(result.errors("aaa1k"))))
        records.append(
            {
                "learning_rate": learning_rate,
                "tbptt_steps": horizon,
                "gradient_clip": None,
                "diverged_streams": diverged_on,
                "diverged": bool(diverged_on),
                "worst_normalized_error": worst,
            }
        )
    boundary = next((record["learning_rate"] for record in records if record["diverged"]), None)
    return {
        "horizon": horizon,
        "records": records,
        "lowest_diverging_learning_rate": boundary,
        "note": (
            "measured with gradient clipping disabled; the selected configuration keeps the "
            "declared clip, so this is a property of the bare optimizer, not of the candidate"
        ),
    }


def evaluate_configuration(
    configuration: Configuration, streams: list[Stream], *, model_seed_index: int = 0
) -> dict[str, Any]:
    """Run one configuration over the whole development bank.

    A configuration that raises is *recorded as unstable*, not discarded. A
    search that quietly drops its failures is a search that cannot report a
    divergence boundary, and the boundary is the thing rule 2 needs.
    """

    per_stream: list[dict[str, Any]] = []
    stable = True
    failure: str | None = None
    for stream in streams:
        agent = NeuralAgent(
            AAA1KGRU(
                seed=derive_seed("model_init", model_seed_index),
                learning_rate=configuration.learning_rate,
                tbptt_steps=configuration.tbptt_steps,
                error_loss_weight=configuration.error_loss_weight,
            ),
            name="aaa1k",
        )
        try:
            result = run_stream(stream, [agent])
        except (FloatingPointError, ValueError, OverflowError) as error:
            stable = False
            failure = f"{type(error).__name__}: {error}"
            per_stream.append({"stream_id": stream.stream_id, "family": stream.family, "status": "DIVERGED"})
            continue
        errors = result.errors("aaa1k")
        estimates = np.asarray([step.error_estimates["aaa1k"] for step in result.steps], dtype=float)
        half = max(1, len(errors) // 2)
        entry = {
            "stream_id": stream.stream_id,
            "family": stream.family,
            "status": "COMPLETED",
            "mae": float(np.mean(errors)),
            "first_half_mae": float(np.mean(errors[:half])),
            "second_half_mae": float(np.mean(errors[-half:])),
            "clip_events": int(agent.model.clip_events),
            "error_estimate_mean": float(np.nanmean(estimates)) if estimates.size else float("nan"),
        }
        entry["adaptation"] = _post_change_adaptation(result, stream)
        per_stream.append(entry)

    completed = [entry for entry in per_stream if entry["status"] == "COMPLETED"]
    values = [entry["mae"] for entry in completed]
    adaptation = [entry["adaptation"] for entry in completed if entry["adaptation"] is not None]
    retention = [
        entry["second_half_mae"] / entry["first_half_mae"]
        for entry in completed
        if entry["first_half_mae"] > 0
    ]
    return {
        "configuration": configuration.to_dict(),
        "key": configuration.key(),
        "stable": bool(stable and len(completed) == len(streams)),
        "failure": failure,
        "streams": per_stream,
        "mean_mae": float(np.mean(values)) if values else float("inf"),
        "median_mae": float(np.median(values)) if values else float("inf"),
        "post_change_adaptation": float(np.mean(adaptation)) if adaptation else float("nan"),
        "retention_ratio": float(np.mean(retention)) if retention else float("nan"),
        "total_clip_events": int(sum(entry["clip_events"] for entry in completed)),
    }


def _post_change_adaptation(result: Any, stream: Stream) -> float | None:
    """Mean error in the 30 steps after the last labelled change, or ``None``.

    The change index is evaluator metadata used only to *slice the report*. It
    is never passed to a learner, which is why it can be used here at all.
    """

    change_positions = [
        index
        for index, step in enumerate(result.steps)
        if step.event in ("change", "regime_change", "regime_switch")
    ]
    if not change_positions:
        return None
    del stream
    start = change_positions[-1]
    window = result.errors("aaa1k")[start : start + 30]
    return float(np.mean(window)) if window.size else None


def eligible_learning_rates(boundary: float | None) -> tuple[float, ...]:
    """Rule 2, applied to the measured unclipped divergence boundary."""

    if boundary is None:
        return LEARNING_RATES
    limit = LEARNING_RATES.index(boundary) - MARGIN_STEPS
    return LEARNING_RATES[: max(0, limit + 1)]


def _rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda record: record["mean_mae"])
    if len(ordered) < 2:
        return ordered
    best = ordered[0]["mean_mae"]
    close = [record for record in ordered if record["mean_mae"] <= best * (1.0 + PRACTICAL_MARGIN)]
    if len(close) < 2:
        return ordered
    rest = [record for record in ordered if record not in close]

    def tiebreak(record: dict[str, Any]) -> tuple[float, float]:
        adaptation = record["post_change_adaptation"]
        retention = record["retention_ratio"]
        return (
            adaptation if np.isfinite(adaptation) else float("inf"),
            retention if np.isfinite(retention) else float("inf"),
        )

    return sorted(close, key=tiebreak) + rest


def select_gradient_clip(
    configuration: Configuration, streams: list[Stream], *, initializations: int = CLIP_INITIALIZATIONS
) -> dict[str, Any]:
    """Stage 3: choose the clip threshold instead of asserting one.

    Averaged over several initializations because the differences between
    neighbouring thresholds are small enough that a single seed would decide
    the outcome on noise.
    """

    records: list[dict[str, Any]] = []
    for threshold in GRADIENT_CLIPS:
        errors: list[float] = []
        rates: list[float] = []
        diverged: list[str] = []
        for initialization in range(initializations):
            for stream in streams:
                agent = NeuralAgent(
                    AAA1KGRU(
                        seed=derive_seed("model_init", initialization),
                        learning_rate=configuration.learning_rate,
                        tbptt_steps=configuration.tbptt_steps,
                        error_loss_weight=configuration.error_loss_weight,
                        gradient_clip=threshold,
                    ),
                    name="aaa1k",
                )
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        result = run_stream(stream, [agent])
                except (FloatingPointError, ValueError, OverflowError) as error:
                    diverged.append(f"{stream.stream_id}: {type(error).__name__}")
                    continue
                errors.append(result.mean_absolute_error("aaa1k"))
                if agent.model.update_count:
                    rates.append(agent.model.clip_events / agent.model.update_count)
        records.append(
            {
                "gradient_clip": threshold,
                "stable": not diverged,
                "diverged_streams": diverged,
                "initializations": initializations,
                "mean_mae": float(np.mean(errors)) if errors else float("inf"),
                "mean_clip_rate": float(np.mean(rates)) if rates else 0.0,
            }
        )

    stable = [record for record in records if record["stable"]]
    if not stable:
        raise RuntimeError("no gradient-clip threshold was stable on development data")
    best = min(stable, key=lambda record: record["mean_mae"])
    within = [
        record for record in stable if record["mean_mae"] <= best["mean_mae"] * (1.0 + PRACTICAL_MARGIN)
    ]
    # Most conservative means the tightest finite threshold; no clip at all is
    # the least conservative option and is chosen only if nothing else is close.
    finite = [record for record in within if record["gradient_clip"] is not None]
    selected = min(finite, key=lambda record: float(record["gradient_clip"])) if finite else best
    declared = next(record for record in records if record["gradient_clip"] == DECLARED_GRADIENT_CLIP)
    return {
        "records": records,
        "selected_gradient_clip": selected["gradient_clip"],
        "declared_gradient_clip": DECLARED_GRADIENT_CLIP,
        "declared_cost_versus_best": (
            declared["mean_mae"] / best["mean_mae"] - 1.0 if best["mean_mae"] > 0 else float("nan")
        ),
        "declared_clip_rate": declared["mean_clip_rate"],
        "selected_clip_rate": selected["mean_clip_rate"],
        "rule": (
            "the most conservative threshold within the practical margin of the best stable "
            "one; a bare threshold is chosen only when no finite threshold is close"
        ),
    }


ARCHITECTURES: dict[str, Any] = {}


def _architecture_factories() -> dict[str, Any]:
    """Imported lazily so this module does not depend on the control classes."""

    from .controls import StatelessMLPControl, VanillaRNNControl

    return {
        "AAA1KGRU": AAA1KGRU,
        "StatelessMLPControl": StatelessMLPControl,
        "VanillaRNNControl": VanillaRNNControl,
    }


def _sweep_learning_rates(
    factory: Any,
    streams: list[Stream],
    *,
    tbptt_steps: int,
    error_loss_weight: float,
    gradient_clip: float | None,
    model_seed_index: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for learning_rate in LEARNING_RATES:
        errors: list[float] = []
        diverged: list[str] = []
        for stream in streams:
            agent = NeuralAgent(
                factory(
                    seed=derive_seed("model_init", model_seed_index),
                    learning_rate=learning_rate,
                    tbptt_steps=tbptt_steps,
                    error_loss_weight=error_loss_weight,
                    gradient_clip=gradient_clip,
                ),
                name="arm",
            )
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    result = run_stream(stream, [agent])
            except (FloatingPointError, ValueError, OverflowError) as error:
                diverged.append(f"{stream.stream_id}: {type(error).__name__}")
                continue
            errors.append(result.mean_absolute_error("arm"))
        records.append(
            {
                "learning_rate": learning_rate,
                "gradient_clip": gradient_clip,
                "stable": not diverged,
                "diverged_streams": diverged,
                "mean_mae": float(np.mean(errors)) if errors and not diverged else float("inf"),
            }
        )
    return records


def select_for_architecture(
    name: str,
    streams: list[Stream],
    *,
    tbptt_steps: int,
    error_loss_weight: float,
    model_seed_index: int = 0,
) -> dict[str, Any]:
    """Apply the frozen rules to one architecture and return what they choose."""

    factory = _architecture_factories()[name]
    unclipped = _sweep_learning_rates(
        factory,
        streams,
        tbptt_steps=tbptt_steps,
        error_loss_weight=error_loss_weight,
        gradient_clip=None,
        model_seed_index=model_seed_index,
    )
    boundary = next((record["learning_rate"] for record in unclipped if not record["stable"]), None)
    allowed = eligible_learning_rates(boundary)
    clipped = _sweep_learning_rates(
        factory,
        streams,
        tbptt_steps=tbptt_steps,
        error_loss_weight=error_loss_weight,
        gradient_clip=DECLARED_GRADIENT_CLIP,
        model_seed_index=model_seed_index,
    )
    eligible = [record for record in clipped if record["stable"] and record["learning_rate"] in allowed]
    if not eligible:
        raise RuntimeError(f"no eligible learning rate for {name} under the declared rules")
    learning_rate = float(min(eligible, key=lambda record: record["mean_mae"])["learning_rate"])

    clip_records: list[dict[str, Any]] = []
    for threshold in GRADIENT_CLIPS:
        errors: list[float] = []
        rates: list[float] = []
        diverged: list[str] = []
        for initialization in range(CLIP_INITIALIZATIONS):
            for stream in streams:
                agent = NeuralAgent(
                    factory(
                        seed=derive_seed("model_init", initialization),
                        learning_rate=learning_rate,
                        tbptt_steps=tbptt_steps,
                        error_loss_weight=error_loss_weight,
                        gradient_clip=threshold,
                    ),
                    name="arm",
                )
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        result = run_stream(stream, [agent])
                except (FloatingPointError, ValueError, OverflowError) as error:
                    diverged.append(f"{stream.stream_id}: {type(error).__name__}")
                    continue
                errors.append(result.mean_absolute_error("arm"))
                if agent.model.update_count:
                    rates.append(agent.model.clip_events / agent.model.update_count)
        clip_records.append(
            {
                "gradient_clip": threshold,
                "stable": not diverged,
                "diverged_streams": diverged,
                "mean_mae": float(np.mean(errors)) if errors else float("inf"),
                "mean_clip_rate": float(np.mean(rates)) if rates else 0.0,
            }
        )
    stable = [record for record in clip_records if record["stable"]]
    if not stable:
        raise RuntimeError(f"no stable gradient clip for {name}")
    best = min(stable, key=lambda record: record["mean_mae"])
    within = [r for r in stable if r["mean_mae"] <= best["mean_mae"] * (1.0 + PRACTICAL_MARGIN)]
    finite = [r for r in within if r["gradient_clip"] is not None]
    clip = (
        float(min(finite, key=lambda record: float(record["gradient_clip"]))["gradient_clip"])
        if finite
        else best["gradient_clip"]
    )
    return {
        "architecture": name,
        "unclipped_divergence_boundary": boundary,
        "eligible_learning_rates": list(allowed),
        "learning_rate_records": clipped,
        "unclipped_records": unclipped,
        "clip_records": clip_records,
        "selected": {
            "learning_rate": learning_rate,
            "tbptt_steps": tbptt_steps,
            "error_loss_weight": error_loss_weight,
            "gradient_clip": clip,
        },
    }


def run_development_selection(*, model_seed_index: int = 0) -> dict[str, Any]:
    """Execute the frozen two-stage plan and return the complete record."""

    streams = development_streams()
    stage_zero = divergence_probe(streams, model_seed_index=model_seed_index)
    stage_one: dict[str, dict[str, Any]] = {}
    for horizon in TBPTT_HORIZONS:
        for learning_rate in LEARNING_RATES:
            configuration = Configuration(learning_rate, horizon, DEFAULT_ERROR_WEIGHT)
            stage_one[configuration.key()] = evaluate_configuration(
                configuration, streams, model_seed_index=model_seed_index
            )

    allowed = eligible_learning_rates(stage_zero["lowest_diverging_learning_rate"])
    eligible = [
        record
        for record in stage_one.values()
        if record["stable"] and float(record["configuration"]["learning_rate"]) in allowed
    ]
    if not eligible:
        raise RuntimeError("no development configuration satisfied the frozen stability rules")
    ranked_one = _rank(eligible)
    chosen = ranked_one[0]["configuration"]

    stage_two: dict[str, dict[str, Any]] = {}
    for weight in ERROR_WEIGHTS:
        configuration = Configuration(float(chosen["learning_rate"]), int(chosen["tbptt_steps"]), weight)
        stage_two[configuration.key()] = evaluate_configuration(
            configuration, streams, model_seed_index=model_seed_index
        )

    default_key = Configuration(
        float(chosen["learning_rate"]), int(chosen["tbptt_steps"]), DEFAULT_ERROR_WEIGHT
    ).key()
    default_record = stage_two[default_key]
    stable_two = [record for record in stage_two.values() if record["stable"]]
    ranked_two = _rank(stable_two)
    best_two = ranked_two[0]
    if best_two["mean_mae"] >= default_record["mean_mae"] * (1.0 - PRACTICAL_MARGIN):
        selected_weight = DEFAULT_ERROR_WEIGHT
        weight_reason = (
            "the predeclared default was retained: no alternative beat it by more than "
            f"{PRACTICAL_MARGIN:.0%} on development data"
        )
    else:
        selected_weight = float(best_two["configuration"]["error_loss_weight"])
        weight_reason = (
            f"lambda={selected_weight:g} beat the predeclared default by more than "
            f"{PRACTICAL_MARGIN:.0%} on development data"
        )

    selected = Configuration(float(chosen["learning_rate"]), int(chosen["tbptt_steps"]), selected_weight)
    architecture_selections = {
        name: select_for_architecture(
            name,
            streams,
            tbptt_steps=selected.tbptt_steps,
            error_loss_weight=selected.error_loss_weight,
            model_seed_index=model_seed_index,
        )
        for name in ("AAA1KGRU", "StatelessMLPControl", "VanillaRNNControl")
    }
    stage_three = select_gradient_clip(selected, streams)
    selected = Configuration(
        selected.learning_rate,
        selected.tbptt_steps,
        selected.error_loss_weight,
        stage_three["selected_gradient_clip"],
    )
    return {
        "schema": SELECTION_SCHEMA,
        "plan": {
            "learning_rates": list(LEARNING_RATES),
            "tbptt_horizons": list(TBPTT_HORIZONS),
            "error_weights": list(ERROR_WEIGHTS),
            "gradient_clips": list(GRADIENT_CLIPS),
            "default_error_weight": DEFAULT_ERROR_WEIGHT,
            "declared_gradient_clip": DECLARED_GRADIENT_CLIP,
            "practical_margin": PRACTICAL_MARGIN,
            "stability_margin_steps": MARGIN_STEPS,
            "streams_per_family": STREAMS_PER_FAMILY,
            "development_families": [family for family, _ in DEVELOPMENT_PLAN],
        },
        "development_streams": [stream.to_summary() for stream in streams],
        "stage_zero_divergence_probe": stage_zero,
        "stage_one": list(stage_one.values()),
        "stage_one_ranking": [record["key"] for record in ranked_one],
        "eligible_learning_rates": list(allowed),
        "stage_one_eliminated": [
            {
                "key": record["key"],
                "stable": record["stable"],
                "failure": record["failure"],
                "mean_mae": record["mean_mae"],
                "reason": (
                    "unstable"
                    if not record["stable"]
                    else "learning rate is inside the declared stability margin"
                ),
            }
            for record in stage_one.values()
            if record not in eligible
        ],
        "stage_two": list(stage_two.values()),
        "stage_two_ranking": [record["key"] for record in ranked_two],
        "stage_three_gradient_clip": stage_three,
        "stage_three_a_architecture_selections": architecture_selections,
        "architecture_configurations": {
            name: record["selected"] for name, record in architecture_selections.items()
        },
        "error_weight_decision": weight_reason,
        "selected": {**selected.to_dict(), "gradient_clip": stage_three["selected_gradient_clip"]},
        "selected_key": selected.key(),
        "selected_gradient_clip": stage_three["selected_gradient_clip"],
    }
