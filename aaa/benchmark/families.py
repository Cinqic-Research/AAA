"""Episode planning, deliberate stratum allocation, and family execution.

Required strata are *allocated*, not hoped for. The exact Cartesian product of
direction x position band x speed band is enumerated up front and every
confirmation episode is assigned to one cell before any result exists, so
coverage is a property of the plan rather than a lucky draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import mean
from typing import Callable, Iterable, Sequence

import numpy as np

from ..config import WorldConfig
from ..environment import DampedOscillatorEnvironment, Environment, MovingDotEnvironment
from ..experiment import StepRecord, TrialIdentity, continue_episode, run_episode
from ..metrics import (
    RecoveryConfig,
    episode_metrics,
    normalized_errors,
    signed_normalized_errors,
)
from ..predictors import (
    ConstantMotionPredictor,
    OnlineLinearPredictor,
    OnlineRLSPredictor,
    PersistencePredictor,
    Predictor,
    ReflectedConstantMotionPredictor,
)
from .spec import BenchmarkSpec, MotionFamilySpec
from .seeds import trial_seed

SAMPLE_MARGIN_FRACTION = 0.08


@dataclass(frozen=True)
class EpisodePlan:
    """A fully determined episode, fixed before any measurement is taken."""

    family: str
    branch: str
    replica: int
    episode: int
    stratum: str
    environment_seed: int
    world: WorldConfig
    scenario: str
    initial_position: float | None = None
    initial_velocity: float | None = None

    @property
    def trial_id(self) -> str:
        return f"{self.family}:{self.branch}:r{self.replica:03d}:e{self.episode:04d}"


def stratum_name(direction: str, position_band: int, speed_band: int) -> str:
    return f"{direction}-position{position_band}-speed{speed_band}"


def classify_stratum(
    position: float, velocity: float, world: WorldConfig, *, position_bands: int, speed_bands: int
) -> str:
    """Classify a realized initial state. Used to verify the plan was honoured."""

    width = world.upper_bound - world.lower_bound
    margin = SAMPLE_MARGIN_FRACTION * width
    low = world.lower_bound + margin
    span = max(width - 2 * margin, np.finfo(float).eps)
    band = min(position_bands - 1, max(0, int((position - low) / span * position_bands)))
    speed_span = max(world.speed_max - world.speed_min, np.finfo(float).eps)
    speed = min(speed_bands - 1, max(0, int((abs(velocity) - world.speed_min) / speed_span * speed_bands)))
    direction = "positive" if velocity > 0 else "negative"
    return stratum_name(direction, band, speed)


def _band_interval(index: int, count: int, low: float, high: float) -> tuple[float, float]:
    step = (high - low) / count
    return low + index * step, low + (index + 1) * step


def sample_stratified_state(
    rng: np.random.Generator,
    world: WorldConfig,
    *,
    direction: str,
    position_band: int,
    speed_band: int,
    position_bands: int,
    speed_bands: int,
    must_stay_in_bounds: bool,
) -> tuple[float, float]:
    """Realize one required stratum exactly, or fail loudly."""

    width = world.upper_bound - world.lower_bound
    margin = SAMPLE_MARGIN_FRACTION * width
    position_low, position_high = _band_interval(
        position_band, position_bands, world.lower_bound + margin, world.upper_bound - margin
    )
    speed_low, speed_high = _band_interval(speed_band, speed_bands, world.speed_min, world.speed_max)
    # Keep the sampled speed strictly inside the band so the realized stratum
    # classification cannot land on a neighbouring band through a boundary hit.
    speed_span = speed_high - speed_low
    speed = float(rng.uniform(speed_low + 0.05 * speed_span, speed_high - 0.05 * speed_span))
    sign = 1.0 if direction == "positive" else -1.0
    velocity = sign * speed
    if must_stay_in_bounds:
        travel = speed * world.dt * world.steps_per_episode
        if sign > 0:
            position_high = min(position_high, world.upper_bound - travel - 1e-9)
        else:
            position_low = max(position_low, world.lower_bound + travel + 1e-9)
        if position_high <= position_low:
            raise ValueError(
                f"stratum {stratum_name(direction, position_band, speed_band)} cannot be realized without "
                f"leaving the interval: travel {travel:.4f} exceeds the available band"
            )
    span = position_high - position_low
    position = float(rng.uniform(position_low + 0.02 * span, position_high - 0.02 * span))
    return position, velocity


def plan_motion_family(
    spec: BenchmarkSpec,
    family_name: str,
    purpose: str,
    *,
    replicas: int,
    episodes: int,
) -> list[EpisodePlan]:
    """Deterministic plan for one motion family."""

    family = spec.motion_families[family_name]
    world = motion_world(spec, family)
    strata = spec.required_strata()
    plans: list[EpisodePlan] = []
    for replica in range(replicas):
        for episode in range(episodes):
            seed = trial_seed(spec.randomness.root_seed, purpose, family_name, replica, episode)
            if not family.stratified:
                plans.append(
                    EpisodePlan(
                        family=family_name,
                        branch="main",
                        replica=replica,
                        episode=episode,
                        stratum="unstratified",
                        environment_seed=seed,
                        world=world,
                        scenario=family.scenario,
                    )
                )
                continue
            # Deliberate allocation: rotate through the exact required product
            # and offset by replica so coverage is spread across replicas.
            index = (episode + replica) % len(strata)
            target = strata[index]
            direction, position_token, speed_token = target.split("-")
            position_band = int(position_token.removeprefix("position"))
            speed_band = int(speed_token.removeprefix("speed"))
            rng = np.random.default_rng(seed)
            position, velocity = sample_stratified_state(
                rng,
                world,
                direction=direction,
                position_band=position_band,
                speed_band=speed_band,
                position_bands=spec.stratification.position_bands,
                speed_bands=spec.stratification.speed_bands,
                must_stay_in_bounds=family.scenario == "straight",
            )
            realized = classify_stratum(
                position,
                velocity,
                world,
                position_bands=spec.stratification.position_bands,
                speed_bands=spec.stratification.speed_bands,
            )
            if realized != target:
                raise AssertionError(
                    f"stratum allocation mismatch: planned {target}, realized {realized}"
                )
            plans.append(
                EpisodePlan(
                    family=family_name,
                    branch="main",
                    replica=replica,
                    episode=episode,
                    stratum=target,
                    environment_seed=seed,
                    world=world,
                    scenario=family.scenario,
                    initial_position=position,
                    initial_velocity=velocity,
                )
            )
    return plans


def motion_world(spec: BenchmarkSpec, family: MotionFamilySpec) -> WorldConfig:
    return WorldConfig(
        lower_bound=spec.world.lower_bound,
        upper_bound=spec.world.upper_bound,
        dt=spec.world.dt,
        steps_per_episode=family.steps_per_episode,
        history_length=spec.world.history_length,
        speed_min=family.speed_min,
        speed_max=family.speed_max,
        change_step=family.change_step,
        change_factor_low=family.change_factor_low if family.change_factor_low is not None else 0.55,
        change_factor_high=family.change_factor_high if family.change_factor_high is not None else 1.65,
        event_margin=family.event_margin if family.event_margin is not None else 0.14,
    )


def changed_law_worlds(spec: BenchmarkSpec) -> tuple[WorldConfig, WorldConfig]:
    base = WorldConfig(
        lower_bound=spec.world.lower_bound,
        upper_bound=spec.world.upper_bound,
        dt=spec.world.dt,
        steps_per_episode=spec.changed_law.prefix_steps,
        history_length=spec.world.history_length,
        speed_min=0.08,
        speed_max=0.2,
        change_step=None,
    )
    branch = replace(base, steps_per_episode=spec.changed_law.branch_steps, change_step=0)
    return base, branch


# ---------------------------------------------------------------------------
# predictor construction
# ---------------------------------------------------------------------------


def make_candidate(spec: BenchmarkSpec, *, name: str, update_enabled: bool, reflect: bool | None = None) -> OnlineRLSPredictor:
    return OnlineRLSPredictor(
        lower_bound=spec.world.lower_bound,
        upper_bound=spec.world.upper_bound,
        displacement_scale=spec.displacement_scale,
        forgetting=spec.candidate.forgetting,
        forgetting_mode=spec.candidate.forgetting_mode,
        ridge=spec.candidate.ridge,
        feature_set=spec.candidate.feature_set,
        reflect=spec.candidate.reflect if reflect is None else reflect,
        unfold_target=spec.candidate.unfold_target,
        trace_bound=spec.candidate.trace_bound,
        dead_zone=spec.candidate.dead_zone,
        detector_multiplier=spec.candidate.detector_multiplier,
        detector_floor=spec.candidate.detector_floor,
        detector_decay=spec.candidate.detector_decay,
        symmetry_tolerance=spec.tolerances.covariance_symmetry,
        psd_tolerance=spec.tolerances.covariance_psd,
        max_condition_number=spec.tolerances.max_condition_number,
        name=name,
        update_enabled=update_enabled,
    )


def clone_candidate(
    model: OnlineRLSPredictor, *, name: str, update_enabled: bool, reflect: bool | None = None
) -> OnlineRLSPredictor:
    """Clone a trained model, carrying its declared validation tolerances."""

    state = model.state_dict()
    if reflect is not None:
        state["reflect"] = bool(reflect)
    return OnlineRLSPredictor.from_state_dict(
        state,
        name=name,
        update_enabled=update_enabled,
        symmetry_tolerance=model.symmetry_tolerance,
        psd_tolerance=model.psd_tolerance,
        max_condition_number=model.max_condition_number,
    )


def baseline_predictors(spec: BenchmarkSpec) -> list[Predictor]:
    return [
        PersistencePredictor(),
        ConstantMotionPredictor(),
        ReflectedConstantMotionPredictor(
            lower_bound=spec.world.lower_bound, upper_bound=spec.world.upper_bound
        ),
    ]


MOTION_PREDICTOR_NAMES = (
    "persistence",
    "constant_motion",
    "constant_motion_reflected",
    "zero_control",
    "legacy_linear_sgd",
    "candidate_frozen",
    "candidate_no_reflect",
)

ONLINE_PREDICTOR_NAMES = MOTION_PREDICTOR_NAMES + ("candidate_online",)

CHANGED_LAW_PREDICTOR_NAMES = (
    "persistence",
    "constant_motion",
    "constant_motion_reflected",
    "frozen",
    "online",
)


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


class FamilyCollector:
    """Accumulates one family's evidence in normalized units only."""

    def __init__(
        self,
        name: str,
        predictor_names: Sequence[str],
        *,
        recovery: RecoveryConfig,
        branch: str = "main",
    ) -> None:
        self.name = name
        self.branch = branch
        self.predictor_names = list(predictor_names)
        self.recovery = recovery
        self.episode_summaries: list[dict[str, object]] = []
        self.replica_episode_mae: dict[str, dict[str, list[float]]] = {}
        self.replica_event_mae: dict[str, dict[str, list[float]]] = {}
        self.replica_post_change_mae: dict[str, dict[str, list[float]]] = {}
        self.replica_post_change_cumulative: dict[str, dict[str, list[float]]] = {}
        self.pooled_errors: dict[str, list[float]] = {name: [] for name in predictor_names}
        self.pooled_event_errors: dict[str, list[float]] = {name: [] for name in predictor_names}
        self.pooled_non_event_errors: dict[str, list[float]] = {name: [] for name in predictor_names}
        self.signed_sums: dict[str, float] = {name: 0.0 for name in predictor_names}
        self.signed_counts: dict[str, int] = {name: 0 for name in predictor_names}
        self.stratum_errors: dict[str, dict[str, list[float]]] = {name: {} for name in predictor_names}
        self.stratum_episodes: dict[str, int] = {}
        self.stratum_replicas: dict[str, set[int]] = {}
        self.wall_counts: dict[str, int] = {"lower": 0, "upper": 0}
        self.bounce_events = 0
        self.bounce_transitions = 0
        self.event_episode_count = 0
        self.no_event_episode_count = 0
        self.completed = 0
        self.recovery_status_counts: dict[str, dict[str, int]] = {name: {} for name in predictor_names}
        self.recovery_times: dict[str, list[float]] = {name: [] for name in predictor_names}

    def add(
        self,
        records: Sequence[StepRecord],
        *,
        pre_event_errors: dict[str, Sequence[float]] | None = None,
        post_change_window: int | None = None,
    ) -> dict[str, object]:
        if not records:
            raise ValueError("cannot collect an empty episode")
        summary = episode_metrics(
            records,
            self.predictor_names,
            recovery=self.recovery,
            pre_event_errors=pre_event_errors,
            post_change_window=post_change_window,
        )
        self.episode_summaries.append(summary)
        replica = str(records[0].replica_id)
        stratum = records[0].stratum
        self.stratum_episodes[stratum] = self.stratum_episodes.get(stratum, 0) + 1
        self.stratum_replicas.setdefault(stratum, set()).add(records[0].replica_id)
        episode_values = self.replica_episode_mae.setdefault(replica, {n: [] for n in self.predictor_names})
        event_values = self.replica_event_mae.setdefault(replica, {n: [] for n in self.predictor_names})
        post_values = self.replica_post_change_mae.setdefault(replica, {n: [] for n in self.predictor_names})
        post_cumulative = self.replica_post_change_cumulative.setdefault(
            replica, {n: [] for n in self.predictor_names}
        )

        event_records = [record for record in records if record.bounced]
        has_events = bool(event_records)
        if has_events:
            self.event_episode_count += 1
        else:
            self.no_event_episode_count += 1
        self.bounce_events += sum(record.bounce_count for record in records)
        self.bounce_transitions += len(event_records)
        for record in records:
            for wall in record.bounce_walls:
                if wall in self.wall_counts:
                    self.wall_counts[wall] += 1

        for name in self.predictor_names:
            errors = normalized_errors(records, name)
            episode_values[name].append(mean(errors))
            self.pooled_errors[name].extend(errors)
            if has_events:
                # An episode with zero events contributes nothing to the event
                # resample. Substituting its whole-episode error would silently
                # inject non-event evidence into an event comparison.
                event_values[name].append(mean(normalized_errors(event_records, name)))
            self.pooled_event_errors[name].extend(normalized_errors(event_records, name))
            self.pooled_non_event_errors[name].extend(
                normalized_errors([record for record in records if not record.bounced], name)
            )
            signed = signed_normalized_errors(records, name)
            self.signed_sums[name] += float(sum(signed))
            self.signed_counts[name] += len(signed)
            self.stratum_errors[name].setdefault(stratum, []).extend(errors)
            predictor_summary = summary["predictors"][name]  # type: ignore[index]
            if predictor_summary["post_change_window_mae"] is not None:
                post_values[name].append(float(predictor_summary["post_change_window_mae"]))
                post_cumulative[name].append(float(predictor_summary["post_change_window_cumulative"]))
            status = str(predictor_summary["recovery"]["status"])  # type: ignore[index]
            counts = self.recovery_status_counts[name]
            counts[status] = counts.get(status, 0) + 1
            elapsed = predictor_summary["recovery"].get("recovery_time_steps")  # type: ignore[index]
            if elapsed is not None:
                self.recovery_times[name].append(float(elapsed))
        self.completed += 1
        return summary

    # -- summarizing -----------------------------------------------------
    def finish(self) -> dict[str, object]:
        predictors: dict[str, object] = {}
        for name in self.predictor_names:
            replica_means = [mean(values[name]) for values in self.replica_episode_mae.values() if values[name]]
            pooled = self.pooled_errors[name]
            event_pooled = self.pooled_event_errors[name]
            predictors[name] = {
                "episode_balanced_mae": _flat_mean(self.replica_episode_mae, name),
                "pooled_transition_mae": float(np.mean(pooled)) if pooled else None,
                "replica_mae": {replica: mean(values[name]) for replica, values in self.replica_episode_mae.items() if values[name]},
                "replica_mae_mean": mean(replica_means) if replica_means else None,
                "worst_replica_mae": max(replica_means) if replica_means else None,
                "best_replica_mae": min(replica_means) if replica_means else None,
                "median": float(np.median(pooled)) if pooled else None,
                "p95": float(np.percentile(pooled, 95)) if pooled else None,
                "p99": float(np.percentile(pooled, 99)) if pooled else None,
                "event_episode_balanced_mae": _flat_mean(self.replica_event_mae, name),
                "event_pooled_mae": float(np.mean(event_pooled)) if event_pooled else None,
                "event_p95": float(np.percentile(event_pooled, 95)) if event_pooled else None,
                "event_p99": float(np.percentile(event_pooled, 99)) if event_pooled else None,
                "non_event_pooled_mae": (
                    float(np.mean(self.pooled_non_event_errors[name])) if self.pooled_non_event_errors[name] else None
                ),
                "post_change_window_mae": _flat_mean(self.replica_post_change_mae, name),
                "post_change_cumulative_mean": _flat_mean(self.replica_post_change_cumulative, name),
                "signed_bias": (
                    self.signed_sums[name] / self.signed_counts[name] if self.signed_counts[name] else None
                ),
                "strata": {
                    stratum: {
                        "mae": float(np.mean(values)),
                        "p95": float(np.percentile(values, 95)),
                        "transitions": len(values),
                        "episodes": self.stratum_episodes.get(stratum, 0),
                        "replicas": len(self.stratum_replicas.get(stratum, set())),
                    }
                    for stratum, values in sorted(self.stratum_errors[name].items())
                },
                "recovery_status_counts": dict(sorted(self.recovery_status_counts[name].items())),
                "recovery_time_steps": {
                    "count": len(self.recovery_times[name]),
                    "median": float(np.median(self.recovery_times[name])) if self.recovery_times[name] else None,
                    "p90": float(np.percentile(self.recovery_times[name], 90)) if self.recovery_times[name] else None,
                    "p95": float(np.percentile(self.recovery_times[name], 95)) if self.recovery_times[name] else None,
                    "max": max(self.recovery_times[name]) if self.recovery_times[name] else None,
                },
            }
        return {
            "family": self.name,
            "branch": self.branch,
            "episodes": self.completed,
            "replicas": len(self.replica_episode_mae),
            "bounce_events": self.bounce_events,
            "bounce_transitions": self.bounce_transitions,
            "bounce_walls": dict(self.wall_counts),
            "event_episodes": self.event_episode_count,
            "no_event_episodes": self.no_event_episode_count,
            "stratum_episodes": dict(sorted(self.stratum_episodes.items())),
            "stratum_replicas": {key: sorted(value) for key, value in sorted(self.stratum_replicas.items())},
            "predictors": predictors,
        }


def _flat_mean(container: dict[str, dict[str, list[float]]], name: str) -> float | None:
    values = [value for replica in container.values() for value in replica.get(name, [])]
    return float(np.mean(values)) if values else None


RecordSink = Callable[[EpisodePlan, Sequence[StepRecord]], None]


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def build_environment(plan: EpisodePlan) -> Environment:
    return MovingDotEnvironment(
        plan.scenario,  # type: ignore[arg-type]
        plan.environment_seed,
        plan.world,
        initial_position=plan.initial_position,
        initial_velocity=plan.initial_velocity,
    )


def run_motion_family(
    spec: BenchmarkSpec,
    family_name: str,
    plans: Sequence[EpisodePlan],
    trained: Sequence[OnlineRLSPredictor],
    *,
    role: str,
    batch_id: str | None,
    training_lineage: Sequence[int],
    checkpoint_hashes: Sequence[str],
    recovery: RecoveryConfig,
    legacy: Sequence[dict[str, object]],
    sink: RecordSink | None = None,
) -> FamilyCollector:
    """Run one motion family with frozen candidate arms (plus an online arm).

    ``legacy`` carries one frozen historical-v1 SGD state per replica. It is a
    reported diagnostic arm only; no gate depends on it.
    """

    family = spec.motion_families[family_name]
    include_online = family.update_mode == "online" or family_name == "speed_change"
    names = ONLINE_PREDICTOR_NAMES if include_online else MOTION_PREDICTOR_NAMES
    collector = FamilyCollector(family_name, names, recovery=recovery)
    post_window = spec.recovery.post_event_horizon
    for plan in plans:
        model = trained[plan.replica]
        predictors: list[Predictor] = list(baseline_predictors(spec))
        predictors.append(make_candidate(spec, name="zero_control", update_enabled=False))
        predictors.append(
            OnlineLinearPredictor.from_state_dict(
                legacy[plan.replica], name="legacy_linear_sgd", update_enabled=False
            )
        )
        predictors.append(clone_candidate(model, name="candidate_frozen", update_enabled=False))
        predictors.append(clone_candidate(model, name="candidate_no_reflect", update_enabled=False, reflect=False))
        if include_online:
            predictors.append(clone_candidate(model, name="candidate_online", update_enabled=True))
        identity = TrialIdentity(
            trial_id=plan.trial_id,
            role=role,
            family=family_name,
            scenario=plan.scenario,
            environment_seed=plan.environment_seed,
            replica_id=plan.replica,
            episode=plan.episode,
            confirmation_batch=batch_id,
            training_seed_lineage=tuple(training_lineage),
            stratum=plan.stratum,
            checkpoint_hash=checkpoint_hashes[plan.replica],
            update_mode="mixed" if include_online else "frozen",
        )
        records = run_episode(build_environment(plan), predictors, identity, learn=include_online)
        if not records:
            raise RuntimeError(f"family {family_name} episode {plan.trial_id} produced no scored transitions")
        collector.add(records, post_change_window=post_window)
        if sink is not None:
            sink(plan, records)
    return collector


def run_always_online_family(
    spec: BenchmarkSpec,
    plans: Sequence[EpisodePlan],
    trained: Sequence[OnlineRLSPredictor],
    *,
    role: str,
    batch_id: str | None,
    training_lineage: Sequence[int],
    checkpoint_hashes: Sequence[str],
    recovery: RecoveryConfig,
    legacy: Sequence[dict[str, object]],
    purpose: str,
    sink: RecordSink | None = None,
) -> FamilyCollector:
    """Continuous deployment track.

    One candidate instance per replica keeps predicting and updating across a
    rotation of regimes. The evaluator never tells it the scenario, never
    freezes it, never resets it and never signals an event. The only thing it
    ever receives is the observation history and, after scoring, the revealed
    target.
    """

    family = spec.motion_families["always_online"]
    rotation = ("straight", "bouncing", "changed")
    names = ONLINE_PREDICTOR_NAMES
    collector = FamilyCollector("always_online", names, recovery=recovery)
    persistent: dict[int, OnlineRLSPredictor] = {}
    for plan in plans:
        if plan.replica not in persistent:
            persistent[plan.replica] = clone_candidate(
                trained[plan.replica], name="candidate_online", update_enabled=True
            )
        online = persistent[plan.replica]
        scenario = rotation[plan.episode % len(rotation)]
        world = motion_world(spec, family)
        if scenario == "changed":
            world = replace(world, change_step=world.steps_per_episode // 2, event_margin=0.18)
        elif scenario == "straight":
            # Straight motion must not leave the interval; shorten this segment.
            world = replace(world, steps_per_episode=min(family.steps_per_episode, 40), change_step=None)
        else:
            world = replace(world, change_step=None)
        segment = replace(plan, world=world, scenario=scenario, stratum=f"segment:{scenario}")
        predictors: list[Predictor] = list(baseline_predictors(spec))
        predictors.append(make_candidate(spec, name="zero_control", update_enabled=False))
        predictors.append(
            OnlineLinearPredictor.from_state_dict(
                legacy[plan.replica], name="legacy_linear_sgd", update_enabled=False
            )
        )
        predictors.append(
            clone_candidate(trained[plan.replica], name="candidate_frozen", update_enabled=False)
        )
        predictors.append(
            clone_candidate(trained[plan.replica], name="candidate_no_reflect", update_enabled=False, reflect=False)
        )
        predictors.append(online)
        identity = TrialIdentity(
            trial_id=segment.trial_id,
            role=role,
            family="always_online",
            scenario=scenario,
            environment_seed=segment.environment_seed,
            replica_id=segment.replica,
            episode=segment.episode,
            confirmation_batch=batch_id,
            training_seed_lineage=tuple(training_lineage),
            stratum=segment.stratum,
            checkpoint_hash=checkpoint_hashes[segment.replica],
            update_mode="mixed",
        )
        records = run_episode(build_environment(segment), predictors, identity, learn=True)
        if not records:
            raise RuntimeError(f"always_online episode {segment.trial_id} produced no scored transitions")
        collector.add(records, post_change_window=spec.recovery.post_event_horizon)
        if sink is not None:
            sink(segment, records)
    return collector


@dataclass
class ChangedLawOutcome:
    changed: FamilyCollector
    unchanged: FamilyCollector
    interventions: list[dict[str, object]]


def run_changed_law_family(
    spec: BenchmarkSpec,
    trained: Sequence[OnlineRLSPredictor],
    *,
    purpose: str,
    role: str,
    batch_id: str | None,
    training_lineage: Sequence[int],
    checkpoint_hashes: Sequence[str],
    recovery: RecoveryConfig,
    replicas: int,
    episodes: int,
    sink: RecordSink | None = None,
) -> ChangedLawOutcome:
    """Matched frozen/updating branch experiment at a common intervention point.

    This structure is the soundest part of the design and is preserved exactly:
    a common pre-change trajectory, a common history, an identical learner
    state at the intervention, a frozen copy, an updating copy, the same
    changed world, and a paired unchanged control.
    """

    law = spec.changed_law
    prefix_world, branch_world = changed_law_worlds(spec)
    names = CHANGED_LAW_PREDICTOR_NAMES
    changed = FamilyCollector("changed_law", names, recovery=recovery, branch="changed-law")
    unchanged = FamilyCollector("changed_law", names, recovery=recovery, branch="unchanged-control")
    interventions: list[dict[str, object]] = []
    for replica in range(replicas):
        for episode in range(episodes):
            seed = trial_seed(spec.randomness.root_seed, purpose, "changed_law", replica, episode)
            coefficient_rng = np.random.default_rng(seed ^ 0x5EED)
            post_omega = float(coefficient_rng.uniform(law.post_omega_low, law.post_omega_high))
            post_damping = float(coefficient_rng.uniform(law.post_damping_low, law.post_damping_high))

            prefix_env = DampedOscillatorEnvironment(
                seed,
                prefix_world,
                omega=law.pre_omega,
                damping=law.pre_damping,
                changed_omega=law.pre_omega,
                changed_damping=law.pre_damping,
                change_step=None,
                initial_position_low=law.initial_position_low,
                initial_position_high=law.initial_position_high,
                initial_velocity_scale=law.initial_velocity_scale,
            )
            prefix_model = clone_candidate(trained[replica], name="prefix_model", update_enabled=True)
            prefix_baselines: list[Predictor] = list(baseline_predictors(spec))
            prefix_identity = TrialIdentity(
                trial_id=f"changed_law:prefix:r{replica:03d}:e{episode:04d}",
                role=role,
                family="changed_law",
                scenario="dynamics_change",
                environment_seed=seed,
                replica_id=replica,
                episode=episode,
                branch="prefix",
                confirmation_batch=batch_id,
                training_seed_lineage=tuple(training_lineage),
                stratum="prefix",
                checkpoint_hash=checkpoint_hashes[replica],
                update_mode="mixed",
            )
            prefix_records = run_episode(
                prefix_env, [*prefix_baselines, prefix_model], prefix_identity, learn=True
            )
            if not prefix_records:
                raise RuntimeError("changed-law prefix produced no records")
            if sink is not None:
                sink(
                    EpisodePlan(
                        family="changed_law",
                        branch="prefix",
                        replica=replica,
                        episode=episode,
                        stratum="prefix",
                        environment_seed=seed,
                        world=prefix_world,
                        scenario="dynamics_change",
                    ),
                    prefix_records,
                )

            history = list(prefix_records[-1].history[1:]) + [prefix_records[-1].actual_next_position]
            start_position = prefix_env.position
            start_velocity = prefix_env.velocity
            pre_event_state = prefix_model.state_dict()
            # Predictor-specific pre-event references. Persistence and constant
            # motion get their own error sequences; the learner's copies get the
            # learner's. Reusing one sequence for all of them would describe the
            # wrong model's recovery.
            pre_event_errors = {
                "persistence": normalized_errors(prefix_records, "persistence"),
                "constant_motion": normalized_errors(prefix_records, "constant_motion"),
                "constant_motion_reflected": normalized_errors(prefix_records, "constant_motion_reflected"),
                "frozen": normalized_errors(prefix_records, "prefix_model"),
                "online": normalized_errors(prefix_records, "prefix_model"),
            }

            for branch_name, collector, omega, damping in (
                ("changed-law", changed, post_omega, post_damping),
                ("unchanged-control", unchanged, law.pre_omega, law.pre_damping),
            ):
                environment = DampedOscillatorEnvironment(
                    seed,
                    branch_world,
                    omega=omega,
                    damping=damping,
                    changed_omega=omega,
                    changed_damping=damping,
                    change_step=0,
                    initial_position=start_position,
                    initial_velocity=start_velocity,
                    initial_position_low=law.initial_position_low,
                    initial_position_high=law.initial_position_high,
                    initial_velocity_scale=law.initial_velocity_scale,
                )
                frozen = OnlineRLSPredictor.from_state_dict(
                    pre_event_state, name="frozen", update_enabled=False
                )
                online = OnlineRLSPredictor.from_state_dict(
                    pre_event_state, name="online", update_enabled=True
                )
                identity = TrialIdentity(
                    trial_id=f"changed_law:{branch_name}:r{replica:03d}:e{episode:04d}",
                    role=role,
                    family="changed_law",
                    scenario="dynamics_change",
                    environment_seed=seed,
                    replica_id=replica,
                    episode=episode,
                    branch=branch_name,
                    confirmation_batch=batch_id,
                    training_seed_lineage=tuple(training_lineage),
                    stratum=branch_name,
                    checkpoint_hash=checkpoint_hashes[replica],
                    update_mode="mixed",
                )
                records = continue_episode(
                    environment,
                    [*baseline_predictors(spec), frozen, online],
                    history,
                    identity,
                    learn=True,
                    step_offset=law.prefix_steps,
                )
                collector.add(
                    records,
                    pre_event_errors=pre_event_errors,
                    post_change_window=spec.recovery.post_event_horizon,
                )
                if sink is not None:
                    sink(
                        EpisodePlan(
                            family="changed_law",
                            branch=branch_name,
                            replica=replica,
                            episode=episode,
                            stratum=branch_name,
                            environment_seed=seed,
                            world=branch_world,
                            scenario="dynamics_change",
                        ),
                        records,
                    )
                if branch_name == "changed-law":
                    frozen_after = frozen.state_dict()
                    interventions.append(
                        {
                            "replica": replica,
                            "episode": episode,
                            "environment_seed": seed,
                            "post_omega": post_omega,
                            "post_damping": post_damping,
                            "start_position": start_position,
                            "start_velocity": start_velocity,
                            "history": list(history),
                            "prefix_updates": int(pre_event_state["update_count"]),
                            "frozen_update_count_before": int(pre_event_state["update_count"]),
                            "frozen_update_count_after": int(frozen_after["update_count"]),
                            "frozen_weights_before": list(pre_event_state["weights"]),  # type: ignore[arg-type]
                            "frozen_weights_after": list(frozen_after["weights"]),  # type: ignore[arg-type]
                            "online_update_count_after": int(online.state_dict()["update_count"]),
                        }
                    )
    return ChangedLawOutcome(changed=changed, unchanged=unchanged, interventions=interventions)


# ---------------------------------------------------------------------------
# training and learning probes
# ---------------------------------------------------------------------------


def training_world(spec: BenchmarkSpec) -> WorldConfig:
    return WorldConfig(
        lower_bound=spec.world.lower_bound,
        upper_bound=spec.world.upper_bound,
        dt=spec.world.dt,
        steps_per_episode=spec.training.steps_per_episode,
        history_length=spec.world.history_length,
        speed_min=spec.training.speed_min,
        speed_max=spec.training.speed_max,
        change_step=None,
    )


def straight_training_state(rng: np.random.Generator, world: WorldConfig) -> tuple[float, float]:
    """Sample a straight training episode that stays inside the interval."""

    width = world.upper_bound - world.lower_bound
    margin = SAMPLE_MARGIN_FRACTION * width
    for _ in range(10_000):
        speed = float(rng.uniform(world.speed_min, world.speed_max))
        velocity = speed if rng.integers(0, 2) else -speed
        travel = speed * world.dt * world.steps_per_episode
        low = world.lower_bound + margin + (travel if velocity < 0 else 0.0)
        high = world.upper_bound - margin - (travel if velocity > 0 else 0.0)
        if high <= low:
            continue
        return float(rng.uniform(low, high)), velocity
    raise RuntimeError("could not sample a valid straight training state")
