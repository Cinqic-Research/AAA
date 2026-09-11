"""Development-only candidate selection evidence.

The selected candidate must be auditable: why *this* feature set, *this*
forgetting factor, *this* ridge, *this* training budget and *this* boundary
policy. Every configuration below is scored on development streams only, and
unsuccessful alternatives are retained rather than discarded.

Selection rule
--------------
Among configurations that

1. survive the numerical stress probes without an invalid state,
2. reach development straight-motion normalized MAE at or below ``1e-5``, and
3. satisfy the benchmark's declared always-online non-regression requirement on
   development streams (continuous updating across regimes must stay within
   ``1.10 x`` the reflected constant-motion baseline plus ``1e-5``),

choose the one with the largest development changed-law improvement of the
updating copy over its identical frozen copy. Improvements within
``TIE_TOLERANCE`` of the maximum count as tied, because a difference of a
percent or two across ten development episodes is sampling noise and picking
the exact argmax would be selecting on noise. Ties are broken, in order,
toward the smaller feature set, the larger forgetting factor, the smaller
detector multiplier, and finally the lexicographically first label, so the
choice is fully deterministic and no remaining tie is resolved by a numeric
difference that is itself sampling noise.

Criterion 3 was added after a first development pass showed that the variant
maximizing changed-law adaptation under criteria 1-2 alone (``lambda = 0.90``)
degraded by roughly thirty-fold under continuous operation. No gate threshold
was changed: the always-online requirement was already declared in the frozen
benchmark specification, and the selection rule was extended on **development**
evidence, before freezing, to stop proposing a candidate that the declared
protocol would rightly reject. The first-pass table is retained in
``docs/evidence/candidate_selection_first_pass.json``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .config import WorldConfig
from .environment import DampedOscillatorEnvironment, MovingDotEnvironment, as_scenario
from .experiment import TrialIdentity, continue_episode, run_episode
from .metrics import normalized_errors
from .predictors import (
    InvalidLearnerState,
    OnlineRLSPredictor,
    ReflectedConstantMotionPredictor,
    validate_covariance,
)

SELECTION_SCHEMA = "aaa.candidate_selection.v1"

STRAIGHT_MAE_LIMIT = 1e-5
ALWAYS_ONLINE_MAX_RATIO = 1.10
ALWAYS_ONLINE_ABSOLUTE_FLOOR = 1e-5
ALWAYS_ONLINE_ROTATION = ("straight", "bouncing", "changed")
TIE_TOLERANCE = 0.02


@dataclass(frozen=True)
class Variant:
    feature_set: str
    forgetting: float
    ridge: float
    reflect: bool
    trace_bound: float
    training_episodes: int
    dead_zone: float = 0.0
    forgetting_mode: str = "exponential"
    unfold_target: bool = False
    detector_multiplier: float = 0.0

    @property
    def label(self) -> str:
        return (
            f"{self.feature_set}|lambda={self.forgetting}|{self.forgetting_mode}"
            f"|ridge={self.ridge:g}|reflect={int(self.reflect)}|trace={self.trace_bound:g}"
            f"|deadzone={self.dead_zone:g}|unfold={int(self.unfold_target)}"
            f"|detector={self.detector_multiplier:g}|episodes={self.training_episodes}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "forgetting": self.forgetting,
            "ridge": self.ridge,
            "reflect": self.reflect,
            "trace_bound": self.trace_bound,
            "training_episodes": self.training_episodes,
            "dead_zone": self.dead_zone,
            "forgetting_mode": self.forgetting_mode,
            "unfold_target": self.unfold_target,
            "detector_multiplier": self.detector_multiplier,
        }


def _identity(name: str, seed: int, episode: int, scenario: str, *, mode: str = "frozen") -> TrialIdentity:
    return TrialIdentity(
        trial_id=f"selection:{name}:s{seed}:e{episode}",
        role="development",
        family="selection",
        scenario=scenario,
        environment_seed=seed,
        replica_id=0,
        episode=episode,
        stratum="selection",
        update_mode=mode,
    )


def _make(variant: Variant, world: WorldConfig, *, name: str, update_enabled: bool) -> OnlineRLSPredictor:
    return OnlineRLSPredictor(
        lower_bound=world.lower_bound,
        upper_bound=world.upper_bound,
        displacement_scale=world.dt * world.speed_max,
        forgetting=variant.forgetting,
        forgetting_mode=variant.forgetting_mode,
        ridge=variant.ridge,
        feature_set=variant.feature_set,
        reflect=variant.reflect,
        trace_bound=variant.trace_bound,
        dead_zone=variant.dead_zone,
        unfold_target=variant.unfold_target,
        detector_multiplier=variant.detector_multiplier,
        name=name,
        update_enabled=update_enabled,
    )


def _stress(variant: Variant, world: WorldConfig, *, updates: int = 20_000) -> dict[str, Any]:
    """Weak-excitation stress probes; a numerical failure disqualifies a variant."""

    failures: list[str] = []
    for label, history, target in (
        ("repeated_input", (0.5, 0.5, 0.5, 0.5), 0.5),
        ("zero_displacement", (0.2, 0.2, 0.2, 0.2), 0.2),
    ):
        model = _make(variant, world, name="stress", update_enabled=True)
        try:
            for _ in range(updates):
                model.update(history, target)
            validate_covariance(model.covariance)
        except (InvalidLearnerState, FloatingPointError, ValueError) as error:
            failures.append(f"{label}: {type(error).__name__}: {error}")
    model = _make(variant, world, name="stress", update_enabled=True)
    position = 0.2
    try:
        for _ in range(updates):
            velocity = 1e-4
            history = (position - 3 * velocity, position - 2 * velocity, position - velocity, position)
            model.update(history, position + velocity)
            position = position + velocity if position < 0.85 else 0.2
        diagnostics = validate_covariance(model.covariance)
    except (InvalidLearnerState, FloatingPointError, ValueError) as error:
        failures.append(f"slow_constant_velocity: {type(error).__name__}: {error}")
        diagnostics = {}
    return {"stable": not failures, "failures": failures, "diagnostics": diagnostics}


def _train(variant: Variant, world: WorldConfig, seeds: Sequence[int]) -> OnlineRLSPredictor:
    model = _make(variant, world, name="candidate", update_enabled=True)
    for episode, seed in enumerate(seeds[: variant.training_episodes]):
        environment = MovingDotEnvironment("straight", seed, world)
        run_episode(
            environment, [model], _identity("train", seed, episode, "straight", mode="online"), learn=True
        )
    return model


def _straight_score(model: OnlineRLSPredictor, world: WorldConfig, seeds: Sequence[int]) -> float:
    values: list[float] = []
    for episode, seed in enumerate(seeds):
        frozen = model.clone(name="candidate_frozen", update_enabled=False)
        environment = MovingDotEnvironment("straight", seed, world)
        records = run_episode(
            environment, [frozen], _identity("straight", seed, episode, "straight"), learn=False
        )
        values.append(mean(normalized_errors(records, "candidate_frozen")))
    return float(np.mean(values))


def _bouncing_score(model: OnlineRLSPredictor, world: WorldConfig, seeds: Sequence[int]) -> float:
    values: list[float] = []
    bouncing_world = replace(world, steps_per_episode=400, change_step=None)
    for episode, seed in enumerate(seeds):
        frozen = model.clone(name="candidate_frozen", update_enabled=False)
        environment = MovingDotEnvironment("bouncing", seed, bouncing_world)
        records = run_episode(
            environment, [frozen], _identity("bounce", seed, episode, "bouncing"), learn=False
        )
        values.append(mean(normalized_errors(records, "candidate_frozen")))
    return float(np.mean(values))


def _always_online_score(
    model: OnlineRLSPredictor, world: WorldConfig, seeds: Sequence[int]
) -> dict[str, float]:
    """Continuous deployment probe: one instance keeps learning across regimes.

    Mirrors the benchmark's always-online family: no evaluator mode switching,
    no reset between segments, no scenario label reaching the model.
    """

    online = model.clone(name="candidate_online", update_enabled=True)
    online_values: list[float] = []
    baseline_values: list[float] = []
    for episode, seed in enumerate(seeds):
        scenario = ALWAYS_ONLINE_ROTATION[episode % len(ALWAYS_ONLINE_ROTATION)]
        if scenario == "straight":
            segment = replace(world, steps_per_episode=40, change_step=None)
        elif scenario == "bouncing":
            segment = replace(world, steps_per_episode=200, change_step=None)
        else:
            segment = replace(world, steps_per_episode=200, change_step=100, event_margin=0.18)
        baseline = ReflectedConstantMotionPredictor(
            lower_bound=world.lower_bound, upper_bound=world.upper_bound
        )
        environment = MovingDotEnvironment(as_scenario(scenario), seed, segment)
        records = run_episode(
            environment,
            [baseline, online],
            _identity("always_online", seed, episode, scenario, mode="mixed"),
            learn=True,
        )
        online_values.append(mean(normalized_errors(records, "candidate_online")))
        baseline_values.append(mean(normalized_errors(records, "constant_motion_reflected")))
    online_mean = float(np.mean(online_values))
    baseline_mean = float(np.mean(baseline_values))
    margin = online_mean - ALWAYS_ONLINE_MAX_RATIO * baseline_mean - ALWAYS_ONLINE_ABSOLUTE_FLOOR
    return {
        "always_online_mae": online_mean,
        "always_online_baseline_mae": baseline_mean,
        "always_online_margin": margin,
        "always_online_stable": bool(margin <= 0.0),
        "always_online_forgetting_suspensions": online.forgetting_suspensions,
    }


def _changed_law_score(
    model: OnlineRLSPredictor, world: WorldConfig, seeds: Sequence[int]
) -> dict[str, float]:
    prefix_world = replace(world, steps_per_episode=300, change_step=None)
    branch_world = replace(world, steps_per_episode=100, change_step=0)
    online_values: list[float] = []
    frozen_values: list[float] = []
    for episode, seed in enumerate(seeds):
        prefix_model = model.clone(name="prefix_model", update_enabled=True)
        prefix_env = DampedOscillatorEnvironment(
            seed,
            prefix_world,
            omega=1.5,
            damping=0.10,
            changed_omega=1.5,
            changed_damping=0.10,
            change_step=None,
        )
        prefix_records = run_episode(
            prefix_env,
            [prefix_model],
            _identity("prefix", seed, episode, "dynamics_change", mode="online"),
            learn=True,
        )
        history = [*prefix_records[-1].history[1:], prefix_records[-1].actual_next_position]
        state = prefix_model.state_dict()
        rng = np.random.default_rng(seed ^ 0x5EED)
        omega = float(rng.uniform(5.0, 11.0))
        damping = float(rng.uniform(0.05, 0.30))
        environment = DampedOscillatorEnvironment(
            seed,
            branch_world,
            omega=omega,
            damping=damping,
            changed_omega=omega,
            changed_damping=damping,
            change_step=0,
            initial_position=prefix_env.position,
            initial_velocity=prefix_env.velocity,
        )
        frozen = OnlineRLSPredictor.from_state_dict(state, name="frozen", update_enabled=False)
        online = OnlineRLSPredictor.from_state_dict(state, name="online", update_enabled=True)
        records = continue_episode(
            environment,
            [frozen, online],
            history,
            _identity("changed", seed, episode, "dynamics_change", mode="mixed"),
            learn=True,
            step_offset=300,
        )
        window = records[:50]
        online_values.append(float(sum(normalized_errors(window, "online"))))
        frozen_values.append(float(sum(normalized_errors(window, "frozen"))))
    online_mean = float(np.mean(online_values))
    frozen_mean = float(np.mean(frozen_values))
    return {
        "online_cumulative": online_mean,
        "frozen_cumulative": frozen_mean,
        "improvement_vs_frozen": 1.0 - online_mean / frozen_mean if frozen_mean else float("nan"),
    }


def variant_grid(*, quick: bool) -> list[Variant]:
    """The development grid. Unsuccessful alternatives are retained on purpose."""

    def base(**overrides: Any) -> Variant:
        settings: dict[str, Any] = {
            "feature_set": "displacement_position",
            "forgetting": 0.5,
            "ridge": 1e-4,
            "reflect": True,
            "trace_bound": 1e5,
            "training_episodes": 24,
            "dead_zone": 0.0,
            "forgetting_mode": "exponential",
            "unfold_target": True,
            "detector_multiplier": 8.0,
        }
        settings.update(overrides)
        return Variant(**settings)

    if quick:
        return [base(), base(detector_multiplier=0.0), base(forgetting=1.0, detector_multiplier=0.0)]

    variants: list[Variant] = []
    # forgetting sweep with the self-triggered detector and boundary-consistent targets
    for forgetting in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        variants.append(base(forgetting=forgetting))
    # detector sensitivity
    for multiplier in (2.0, 4.0, 16.0, 32.0):
        variants.append(base(detector_multiplier=multiplier))
    # mechanism ablations, one factor at a time
    variants.append(base(unfold_target=False))
    variants.append(base(detector_multiplier=0.0))
    variants.append(base(unfold_target=False, detector_multiplier=0.0))
    variants.append(base(reflect=False))
    variants.append(base(feature_set="displacement_only"))
    variants.append(base(forgetting_mode="directional"))
    variants.append(base(forgetting_mode="directional", detector_multiplier=0.0))
    variants.append(base(dead_zone=1e-7))
    # regularization, covariance bound and training budget
    for ridge in (1e-2, 1e-6):
        variants.append(base(ridge=ridge))
    for trace_bound in (1e3, 1e4, 1e6):
        variants.append(base(trace_bound=trace_bound))
    for episodes in (4, 8, 16):
        variants.append(base(training_episodes=episodes))
    # historical family: plain exponential forgetting with no detector and no
    # boundary-consistent target, i.e. the pre-repair candidate shape
    for forgetting in (1.0, 0.99, 0.98, 0.95, 0.90):
        variants.append(base(forgetting=forgetting, unfold_target=False, detector_multiplier=0.0))
    for forgetting in (0.5, 0.9):
        variants.append(
            base(
                forgetting=forgetting,
                forgetting_mode="directional",
                unfold_target=False,
                detector_multiplier=0.0,
            )
        )
    seen: dict[str, Variant] = {}
    for variant in variants:
        seen.setdefault(variant.label, variant)
    return list(seen.values())


def run_candidate_selection(
    output: str | Path = "docs/evidence/candidate_selection.json", *, quick: bool = False
) -> Path:
    world = WorldConfig(steps_per_episode=180, speed_min=0.08, speed_max=0.20, change_step=None)
    training_seeds = [7_000_000 + index * 131 for index in range(24)]
    straight_seeds = [7_100_000 + index * 137 for index in range(8)]
    bounce_seeds = [7_200_000 + index * 139 for index in range(4 if quick else 8)]
    changed_seeds = [7_300_000 + index * 149 for index in range(4 if quick else 10)]
    always_online_seeds = [7_400_000 + index * 151 for index in range(6 if quick else 18)]

    rows: list[dict[str, Any]] = []
    for variant in variant_grid(quick=quick):
        stress = _stress(variant, world, updates=2_000 if quick else 20_000)
        model = _train(variant, world, training_seeds)
        straight = _straight_score(model, world, straight_seeds)
        bouncing = _bouncing_score(model, world, bounce_seeds)
        changed = _changed_law_score(model, world, changed_seeds)
        always_online = _always_online_score(model, world, always_online_seeds)
        rows.append(
            {
                "label": variant.label,
                **variant.to_dict(),
                "numerically_stable": stress["stable"],
                "stress_failures": stress["failures"],
                "covariance_diagnostics": stress["diagnostics"],
                "development_straight_mae": straight,
                "development_bouncing_mae": bouncing,
                **changed,
                **always_online,
                "weights": model.weights.tolist(),
                "forgetting_suspensions": model.forgetting_suspensions,
                "training_dead_zone_skips": model.dead_zone_skips,
            }
        )

    eligible = [
        row
        for row in rows
        if row["numerically_stable"]
        and row["development_straight_mae"] <= STRAIGHT_MAE_LIMIT
        and row["always_online_stable"]
    ]
    selected = None
    tied: list[dict[str, Any]] = []
    if eligible:
        best_improvement = max(float(row["improvement_vs_frozen"]) for row in eligible)
        tied = [
            row for row in eligible if float(row["improvement_vs_frozen"]) >= best_improvement - TIE_TOLERANCE
        ]
        selected = sorted(
            tied,
            key=lambda row: (
                0 if row["feature_set"] == "displacement_only" else 1,
                -float(row["forgetting"]),
                float(row["detector_multiplier"]),
                float(row["dead_zone"]),
                0 if row["forgetting_mode"] == "exponential" else 1,
                str(row["label"]),
            ),
        )[0]
    result = {
        "format_version": SELECTION_SCHEMA,
        "selection_rule": (
            "among numerically stable variants with development straight-motion normalized MAE <= 1e-5 that "
            "also satisfy the declared always-online non-regression requirement on development streams, "
            "maximize development changed-law improvement over the identical frozen copy; ties break toward "
            "the smaller feature set and then toward less forgetting"
        ),
        "data_discipline": "development streams only; no confirmation batch was inspected",
        "straight_mae_limit": STRAIGHT_MAE_LIMIT,
        "always_online_criterion": {
            "max_ratio": ALWAYS_ONLINE_MAX_RATIO,
            "absolute_floor": ALWAYS_ONLINE_ABSOLUTE_FLOOR,
            "baseline": "constant_motion_reflected",
            "note": "identical in form to the frozen benchmark's always_online_stability gate",
        },
        "world": {
            "lower_bound": world.lower_bound,
            "upper_bound": world.upper_bound,
            "dt": world.dt,
            "speed_min": world.speed_min,
            "speed_max": world.speed_max,
            "training_steps_per_episode": world.steps_per_episode,
        },
        "seeds": {
            "training": training_seeds,
            "straight": straight_seeds,
            "bouncing": bounce_seeds,
            "changed_law": changed_seeds,
            "always_online": always_online_seeds,
        },
        "variants": rows,
        "tie_tolerance": TIE_TOLERANCE,
        "eligible_count": len(eligible),
        "tied_count": len(tied),
        "tied_labels": [row["label"] for row in tied],
        "selected": selected,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
