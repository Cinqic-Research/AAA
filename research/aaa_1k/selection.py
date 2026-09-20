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

    def key(self) -> str:
        return f"lr={self.learning_rate:g};T={self.tbptt_steps};lambda={self.error_loss_weight:g}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "tbptt_steps": self.tbptt_steps,
            "error_loss_weight": self.error_loss_weight,
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
    return {
        "schema": SELECTION_SCHEMA,
        "plan": {
            "learning_rates": list(LEARNING_RATES),
            "tbptt_horizons": list(TBPTT_HORIZONS),
            "error_weights": list(ERROR_WEIGHTS),
            "default_error_weight": DEFAULT_ERROR_WEIGHT,
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
        "error_weight_decision": weight_reason,
        "selected": selected.to_dict(),
        "selected_key": selected.key(),
    }
