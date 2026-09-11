"""Canonical, strictly validated benchmark specification.

There is exactly one authoritative executable specification. Everything the
runner needs — world, candidate identity, hyperparameters, baselines, family
definitions, stratification, event distributions, confirmation minimums,
recovery rules, statistics, gate thresholds, tolerances, latency procedure and
artifact schema versions — is parsed here into typed objects and consumed from
those objects.

Two properties are enforced deliberately:

``strict parsing``
    An unknown or misspelled key raises. A JSON edit cannot be silently
    ignored, which is what previously allowed the documentation and the code
    to disagree.
``content addressing``
    :func:`spec_hash` hashes the *resolved* specification, so confirmation can
    assert the exact protocol identity rather than a filename.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# The schema validates against the model's own declared option sets so the two
# can never drift apart.
from ..predictors import OnlineRLSPredictor

SPEC_VERSION = "aaa.benchmark.v2.1"
SUPERSEDED_VERSIONS = ("aaa.benchmark.v2",)


class SpecError(ValueError):
    """Raised when a benchmark specification is malformed or inconsistent."""


def _strict(cls: type, value: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    """Return constructor kwargs, rejecting unknown and missing-required keys."""

    if not isinstance(value, Mapping):
        raise SpecError(f"{path}: expected an object, got {type(value).__name__}")
    declared = fields(cls)
    known = {item.name for item in declared}
    unknown = sorted(set(value) - known)
    if unknown:
        raise SpecError(f"{path}: unknown specification keys {unknown}")
    required = {item.name for item in declared if item.default is MISSING and item.default_factory is MISSING}
    absent = sorted(required - set(value))
    if absent:
        raise SpecError(f"{path}: missing required specification keys {absent}")
    return dict(value)


def _positive_int(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{path}: expected an integer, got {value!r}")
    if value < minimum:
        raise SpecError(f"{path}: must be >= {minimum}, got {value}")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{path}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SpecError(f"{path}: must be finite")
    return number


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomnessSpec:
    root_seed: int
    stream_algorithm: str
    bootstrap_seed: int
    description: str

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> RandomnessSpec:
        data = _strict(cls, value, path=path)
        if data.get("stream_algorithm") != "numpy.random.SeedSequence":
            raise SpecError(f"{path}.stream_algorithm: only numpy.random.SeedSequence is implemented")
        return cls(
            root_seed=_positive_int(data["root_seed"], f"{path}.root_seed", minimum=0),
            stream_algorithm=str(data["stream_algorithm"]),
            bootstrap_seed=_positive_int(data["bootstrap_seed"], f"{path}.bootstrap_seed", minimum=0),
            description=str(data["description"]),
        )


@dataclass(frozen=True)
class WorldSpec:
    lower_bound: float
    upper_bound: float
    dt: float
    history_length: int

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> WorldSpec:
        data = _strict(cls, value, path=path)
        lower = _finite(data["lower_bound"], f"{path}.lower_bound")
        upper = _finite(data["upper_bound"], f"{path}.upper_bound")
        if upper <= lower:
            raise SpecError(f"{path}: upper_bound must exceed lower_bound")
        dt = _finite(data["dt"], f"{path}.dt")
        if dt <= 0:
            raise SpecError(f"{path}.dt: must be positive")
        return cls(
            lower, upper, dt, _positive_int(data["history_length"], f"{path}.history_length", minimum=2)
        )

    @property
    def width(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def midpoint(self) -> float:
        return (self.lower_bound + self.upper_bound) / 2


@dataclass(frozen=True)
class CandidateSpec:
    model: str
    feature_set: str
    features: tuple[str, ...]
    forgetting: float
    forgetting_mode: str
    ridge: float
    trace_bound: float
    reflect: bool
    unfold_target: bool
    dead_zone: float
    detector_multiplier: float
    detector_floor: float
    detector_decay: float
    displacement_scale_speed: float
    boundary_policy: str
    self_triggered_forgetting: str
    selection_evidence: str

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> CandidateSpec:
        data = _strict(cls, value, path=path)
        if data["model"] != "OnlineRLSPredictor":
            raise SpecError(f"{path}.model: only OnlineRLSPredictor is implemented")
        forgetting = _finite(data["forgetting"], f"{path}.forgetting")
        if not 0 < forgetting <= 1:
            raise SpecError(f"{path}.forgetting: must satisfy 0 < lambda <= 1")
        if data["forgetting_mode"] not in OnlineRLSPredictor.FORGETTING_MODES:
            raise SpecError(f"{path}.forgetting_mode: must be one of {OnlineRLSPredictor.FORGETTING_MODES}")
        if data["feature_set"] not in OnlineRLSPredictor.FEATURE_SETS:
            raise SpecError(f"{path}.feature_set: must be one of {OnlineRLSPredictor.FEATURE_SETS}")
        return cls(
            model=str(data["model"]),
            feature_set=str(data["feature_set"]),
            features=tuple(str(item) for item in data["features"]),
            forgetting=forgetting,
            forgetting_mode=str(data["forgetting_mode"]),
            ridge=_finite(data["ridge"], f"{path}.ridge"),
            trace_bound=_finite(data["trace_bound"], f"{path}.trace_bound"),
            reflect=bool(data["reflect"]),
            unfold_target=bool(data["unfold_target"]),
            dead_zone=_finite(data["dead_zone"], f"{path}.dead_zone"),
            detector_multiplier=_finite(data["detector_multiplier"], f"{path}.detector_multiplier"),
            detector_floor=_finite(data["detector_floor"], f"{path}.detector_floor"),
            detector_decay=_finite(data["detector_decay"], f"{path}.detector_decay"),
            displacement_scale_speed=_finite(
                data["displacement_scale_speed"], f"{path}.displacement_scale_speed"
            ),
            boundary_policy=str(data["boundary_policy"]),
            self_triggered_forgetting=str(data["self_triggered_forgetting"]),
            selection_evidence=str(data["selection_evidence"]),
        )


@dataclass(frozen=True)
class TrainingSpec:
    family: str
    episodes_per_replica: int
    steps_per_episode: int
    speed_min: float
    speed_max: float
    distribution_relationship: str
    learning_probe_budgets: tuple[int, ...]
    learning_probe_episodes: int
    learning_probe_seed_offset: int

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> TrainingSpec:
        data = _strict(cls, value, path=path)
        budgets = tuple(
            _positive_int(item, f"{path}.learning_probe_budgets", minimum=0)
            for item in data["learning_probe_budgets"]
        )
        if not budgets or sorted(budgets) != list(budgets) or len(set(budgets)) != len(budgets):
            raise SpecError(f"{path}.learning_probe_budgets: must be strictly increasing and non-empty")
        episodes = _positive_int(data["episodes_per_replica"], f"{path}.episodes_per_replica")
        if budgets[-1] != episodes:
            raise SpecError(f"{path}.learning_probe_budgets: final budget must equal episodes_per_replica")
        speed_min = _finite(data["speed_min"], f"{path}.speed_min")
        speed_max = _finite(data["speed_max"], f"{path}.speed_max")
        if not 0 < speed_min <= speed_max:
            raise SpecError(f"{path}: speed range must satisfy 0 < speed_min <= speed_max")
        return cls(
            family=str(data["family"]),
            episodes_per_replica=episodes,
            steps_per_episode=_positive_int(data["steps_per_episode"], f"{path}.steps_per_episode"),
            speed_min=speed_min,
            speed_max=speed_max,
            distribution_relationship=str(data["distribution_relationship"]),
            learning_probe_budgets=budgets,
            learning_probe_episodes=_positive_int(
                data["learning_probe_episodes"], f"{path}.learning_probe_episodes"
            ),
            learning_probe_seed_offset=_positive_int(
                data["learning_probe_seed_offset"], f"{path}.learning_probe_seed_offset", minimum=0
            ),
        )


@dataclass(frozen=True)
class StratificationSpec:
    directions: tuple[str, ...]
    position_bands: int
    speed_bands: int
    minimum_episodes_per_stratum: int
    minimum_replicas_per_stratum: int

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> StratificationSpec:
        data = _strict(cls, value, path=path)
        directions = tuple(str(item) for item in data["directions"])
        if sorted(directions) != ["negative", "positive"]:
            raise SpecError(f"{path}.directions: must be exactly ['positive', 'negative']")
        return cls(
            directions=directions,
            position_bands=_positive_int(data["position_bands"], f"{path}.position_bands"),
            speed_bands=_positive_int(data["speed_bands"], f"{path}.speed_bands"),
            minimum_episodes_per_stratum=_positive_int(
                data["minimum_episodes_per_stratum"], f"{path}.minimum_episodes_per_stratum"
            ),
            minimum_replicas_per_stratum=_positive_int(
                data["minimum_replicas_per_stratum"], f"{path}.minimum_replicas_per_stratum"
            ),
        )

    def strata(self) -> tuple[str, ...]:
        """Exact required Cartesian product of confirmation strata."""

        return tuple(
            f"{direction}-position{position}-speed{speed}"
            for direction in ("positive", "negative")
            for position in range(self.position_bands)
            for speed in range(self.speed_bands)
        )


@dataclass(frozen=True)
class MotionFamilySpec:
    scenario: str
    steps_per_episode: int
    speed_min: float
    speed_max: float
    stratified: bool
    description: str
    change_step: int | None = None
    change_factor_low: float | None = None
    change_factor_high: float | None = None
    event_margin: float | None = None
    minimum_bounce_events: int | None = None
    required_walls: tuple[str, ...] = ()
    update_mode: str = "frozen"

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> MotionFamilySpec:
        data = _strict(cls, value, path=path)
        speed_min = _finite(data["speed_min"], f"{path}.speed_min")
        speed_max = _finite(data["speed_max"], f"{path}.speed_max")
        if not 0 < speed_min <= speed_max:
            raise SpecError(f"{path}: speed range must satisfy 0 < speed_min <= speed_max")
        steps = _positive_int(data["steps_per_episode"], f"{path}.steps_per_episode")
        change_step = data.get("change_step")
        if change_step is not None:
            change_step = _positive_int(change_step, f"{path}.change_step", minimum=0)
            if change_step >= steps:
                raise SpecError(f"{path}.change_step: must satisfy 0 <= change_step < steps_per_episode")
        update_mode = str(data.get("update_mode", "frozen"))
        if update_mode not in ("frozen", "online"):
            raise SpecError(f"{path}.update_mode: must be 'frozen' or 'online'")
        return cls(
            scenario=str(data["scenario"]),
            steps_per_episode=steps,
            speed_min=speed_min,
            speed_max=speed_max,
            stratified=bool(data["stratified"]),
            description=str(data["description"]),
            change_step=change_step,
            change_factor_low=None
            if data.get("change_factor_low") is None
            else _finite(data["change_factor_low"], f"{path}.change_factor_low"),
            change_factor_high=None
            if data.get("change_factor_high") is None
            else _finite(data["change_factor_high"], f"{path}.change_factor_high"),
            event_margin=None
            if data.get("event_margin") is None
            else _finite(data["event_margin"], f"{path}.event_margin"),
            minimum_bounce_events=None
            if data.get("minimum_bounce_events") is None
            else _positive_int(data["minimum_bounce_events"], f"{path}.minimum_bounce_events"),
            required_walls=tuple(str(item) for item in data.get("required_walls", ())),
            update_mode=update_mode,
        )


@dataclass(frozen=True)
class ChangedLawSpec:
    prefix_steps: int
    branch_steps: int
    pre_omega: float
    pre_damping: float
    post_omega_low: float
    post_omega_high: float
    post_damping_low: float
    post_damping_high: float
    initial_position_low: float
    initial_position_high: float
    initial_velocity_scale: float
    equations: str
    description: str

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> ChangedLawSpec:
        data = _strict(cls, value, path=path)
        numbers = {
            name: _finite(data[name], f"{path}.{name}")
            for name in (
                "pre_omega",
                "pre_damping",
                "post_omega_low",
                "post_omega_high",
                "post_damping_low",
                "post_damping_high",
                "initial_position_low",
                "initial_position_high",
                "initial_velocity_scale",
            )
        }
        if numbers["post_omega_low"] > numbers["post_omega_high"]:
            raise SpecError(f"{path}: post_omega_low must not exceed post_omega_high")
        if numbers["post_damping_low"] > numbers["post_damping_high"]:
            raise SpecError(f"{path}: post_damping_low must not exceed post_damping_high")
        if not 0.0 <= numbers["initial_position_low"] <= numbers["initial_position_high"] <= 1.0:
            raise SpecError(f"{path}: initial position fractions must satisfy 0 <= low <= high <= 1")
        return cls(
            prefix_steps=_positive_int(data["prefix_steps"], f"{path}.prefix_steps"),
            branch_steps=_positive_int(data["branch_steps"], f"{path}.branch_steps"),
            equations=str(data["equations"]),
            description=str(data["description"]),
            **numbers,
        )


@dataclass(frozen=True)
class ConfirmationSpec:
    replicas: int
    episodes_per_family: int
    high_replication_replicas: int
    minimum_bounce_events: int
    minimum_eligible_change_events: int
    ab_relationship: str
    require_clean_source_tree: bool
    batch_registry: str
    freeze_manifest: str

    AB_RELATIONSHIPS = ("shared_frozen_checkpoints", "independent_replication")

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> ConfirmationSpec:
        data = _strict(cls, value, path=path)
        relationship = str(data["ab_relationship"])
        if relationship not in cls.AB_RELATIONSHIPS:
            raise SpecError(f"{path}.ab_relationship: must be one of {cls.AB_RELATIONSHIPS}")
        return cls(
            replicas=_positive_int(data["replicas"], f"{path}.replicas", minimum=2),
            episodes_per_family=_positive_int(data["episodes_per_family"], f"{path}.episodes_per_family"),
            high_replication_replicas=_positive_int(
                data["high_replication_replicas"], f"{path}.high_replication_replicas", minimum=2
            ),
            minimum_bounce_events=_positive_int(
                data["minimum_bounce_events"], f"{path}.minimum_bounce_events"
            ),
            minimum_eligible_change_events=_positive_int(
                data["minimum_eligible_change_events"], f"{path}.minimum_eligible_change_events"
            ),
            ab_relationship=relationship,
            require_clean_source_tree=bool(data["require_clean_source_tree"]),
            batch_registry=str(data["batch_registry"]),
            freeze_manifest=str(data["freeze_manifest"]),
        )


@dataclass(frozen=True)
class RecoverySpec:
    pre_event_reference_length: int
    post_event_horizon: int
    shock_window: int
    shock_multiplier: float
    shock_floor: float
    tolerance_multiplier: float
    tolerance_floor: float
    rolling_window: int
    sustain_windows: int

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> RecoverySpec:
        data = _strict(cls, value, path=path)
        return cls(
            pre_event_reference_length=_positive_int(
                data["pre_event_reference_length"], f"{path}.pre_event_reference_length"
            ),
            post_event_horizon=_positive_int(data["post_event_horizon"], f"{path}.post_event_horizon"),
            shock_window=_positive_int(data["shock_window"], f"{path}.shock_window"),
            shock_multiplier=_finite(data["shock_multiplier"], f"{path}.shock_multiplier"),
            shock_floor=_finite(data["shock_floor"], f"{path}.shock_floor"),
            tolerance_multiplier=_finite(data["tolerance_multiplier"], f"{path}.tolerance_multiplier"),
            tolerance_floor=_finite(data["tolerance_floor"], f"{path}.tolerance_floor"),
            rolling_window=_positive_int(data["rolling_window"], f"{path}.rolling_window"),
            sustain_windows=_positive_int(data["sustain_windows"], f"{path}.sustain_windows"),
        )


@dataclass(frozen=True)
class StatisticsSpec:
    method: str
    draws: int
    interval: float
    multiplicity: str
    family_wise_alpha: float
    estimands: dict[str, str]
    event_weighting: str
    exclude_no_event_episodes: bool

    METHODS = ("paired_hierarchical_bootstrap",)
    MULTIPLICITY = ("holm_bonferroni", "none")
    WEIGHTINGS = ("episode_balanced", "event_weighted", "replica_balanced")

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> StatisticsSpec:
        data = _strict(cls, value, path=path)
        if data["method"] not in cls.METHODS:
            raise SpecError(f"{path}.method: must be one of {cls.METHODS}")
        if data["multiplicity"] not in cls.MULTIPLICITY:
            raise SpecError(f"{path}.multiplicity: must be one of {cls.MULTIPLICITY}")
        if data["event_weighting"] not in cls.WEIGHTINGS:
            raise SpecError(f"{path}.event_weighting: must be one of {cls.WEIGHTINGS}")
        interval = _finite(data["interval"], f"{path}.interval")
        if not 0 < interval < 1:
            raise SpecError(f"{path}.interval: must be strictly between 0 and 1")
        return cls(
            method=str(data["method"]),
            draws=_positive_int(data["draws"], f"{path}.draws", minimum=200),
            interval=interval,
            multiplicity=str(data["multiplicity"]),
            family_wise_alpha=_finite(data["family_wise_alpha"], f"{path}.family_wise_alpha"),
            estimands={str(k): str(v) for k, v in data["estimands"].items()},
            event_weighting=str(data["event_weighting"]),
            exclude_no_event_episodes=bool(data["exclude_no_event_episodes"]),
        )


@dataclass(frozen=True)
class GateSpec:
    name: str
    evaluator: str
    required: bool
    description: str
    threshold: dict[str, Any] = field(default_factory=dict)

    EVALUATORS = (
        "coverage",
        "absolute_accuracy",
        "absolute_accuracy_by_stratum",
        "learning_progress",
        "event_accuracy_and_parity",
        "non_regression",
        "changed_law_adaptation",
        "unchanged_control",
        "recovery",
        "correctness",
        "reproducibility",
        "latency",
    )

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> GateSpec:
        data = _strict(cls, value, path=path)
        if str(data["evaluator"]) not in cls.EVALUATORS:
            raise SpecError(f"{path}.evaluator: must be one of {cls.EVALUATORS}")
        return cls(
            name=str(data["name"]),
            evaluator=str(data["evaluator"]),
            required=bool(data["required"]),
            description=str(data["description"]),
            threshold=dict(data.get("threshold", {})),
        )


@dataclass(frozen=True)
class LatencySpec:
    samples: int
    warmup_samples: int
    p95_limit_ms: float
    use_selected_candidate: bool
    description: str

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> LatencySpec:
        data = _strict(cls, value, path=path)
        if not bool(data["use_selected_candidate"]):
            raise SpecError(f"{path}.use_selected_candidate: latency must measure the selected candidate")
        return cls(
            samples=_positive_int(data["samples"], f"{path}.samples", minimum=100),
            warmup_samples=_positive_int(data["warmup_samples"], f"{path}.warmup_samples", minimum=0),
            p95_limit_ms=_finite(data["p95_limit_ms"], f"{path}.p95_limit_ms"),
            use_selected_candidate=True,
            description=str(data["description"]),
        )


@dataclass(frozen=True)
class TolerancesSpec:
    reproducibility_absolute: float
    reproducibility_relative: float
    recompute_absolute: float
    covariance_symmetry: float
    covariance_psd: float
    max_condition_number: float

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> TolerancesSpec:
        data = _strict(cls, value, path=path)
        return cls(**{name: _finite(data[name], f"{path}.{name}") for name in _field_names(cls)})


@dataclass(frozen=True)
class ArtifactsSpec:
    step_record_schema: str
    checkpoint_schema: str
    summary_schema: str
    manifest_schema: str
    registry_schema: str
    retain_raw_steps: bool
    compress_raw_steps: bool

    @classmethod
    def parse(cls, value: Mapping[str, Any], path: str) -> ArtifactsSpec:
        data = _strict(cls, value, path=path)
        return cls(
            step_record_schema=str(data["step_record_schema"]),
            checkpoint_schema=str(data["checkpoint_schema"]),
            summary_schema=str(data["summary_schema"]),
            manifest_schema=str(data["manifest_schema"]),
            registry_schema=str(data["registry_schema"]),
            retain_raw_steps=bool(data["retain_raw_steps"]),
            compress_raw_steps=bool(data["compress_raw_steps"]),
        )


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(item.name for item in fields(cls))


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSpec:
    spec_version: str
    status: str
    supersedes: tuple[str, ...]
    title: str
    randomness: RandomnessSpec
    world: WorldSpec
    candidate: CandidateSpec
    training: TrainingSpec
    stratification: StratificationSpec
    motion_families: dict[str, MotionFamilySpec]
    changed_law: ChangedLawSpec
    confirmation: ConfirmationSpec
    recovery: RecoverySpec
    statistics: StatisticsSpec
    gates: tuple[GateSpec, ...]
    latency: LatencySpec
    tolerances: TolerancesSpec
    artifacts: ArtifactsSpec
    baselines: dict[str, str]
    notes: dict[str, str]

    REQUIRED_MOTION_FAMILIES = ("constant_velocity", "bouncing", "speed_change", "always_online")

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> BenchmarkSpec:
        data = _strict(cls, value, path="spec")
        version = str(data["spec_version"])
        if version != SPEC_VERSION:
            raise SpecError(f"spec.spec_version: expected {SPEC_VERSION!r}, got {version!r}")
        families = {
            name: MotionFamilySpec.parse(item, f"spec.motion_families.{name}")
            for name, item in data["motion_families"].items()
        }
        missing = [name for name in cls.REQUIRED_MOTION_FAMILIES if name not in families]
        if missing:
            raise SpecError(f"spec.motion_families: missing required families {missing}")
        gates = tuple(
            GateSpec.parse(item, f"spec.gates[{index}]") for index, item in enumerate(data["gates"])
        )
        names = [gate.name for gate in gates]
        if len(set(names)) != len(names):
            raise SpecError("spec.gates: gate names must be unique")
        if not any(gate.required for gate in gates):
            raise SpecError("spec.gates: at least one gate must be required")
        spec = cls(
            spec_version=version,
            status=str(data["status"]),
            supersedes=tuple(str(item) for item in data["supersedes"]),
            title=str(data["title"]),
            randomness=RandomnessSpec.parse(data["randomness"], "spec.randomness"),
            world=WorldSpec.parse(data["world"], "spec.world"),
            candidate=CandidateSpec.parse(data["candidate"], "spec.candidate"),
            training=TrainingSpec.parse(data["training"], "spec.training"),
            stratification=StratificationSpec.parse(data["stratification"], "spec.stratification"),
            motion_families=families,
            changed_law=ChangedLawSpec.parse(data["changed_law"], "spec.changed_law"),
            confirmation=ConfirmationSpec.parse(data["confirmation"], "spec.confirmation"),
            recovery=RecoverySpec.parse(data["recovery"], "spec.recovery"),
            statistics=StatisticsSpec.parse(data["statistics"], "spec.statistics"),
            gates=gates,
            latency=LatencySpec.parse(data["latency"], "spec.latency"),
            tolerances=TolerancesSpec.parse(data["tolerances"], "spec.tolerances"),
            artifacts=ArtifactsSpec.parse(data["artifacts"], "spec.artifacts"),
            baselines={str(k): str(v) for k, v in data["baselines"].items()},
            notes={str(k): str(v) for k, v in data["notes"].items()},
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        """Cross-section consistency checks."""

        recovery = self.recovery
        if recovery.shock_window > recovery.post_event_horizon:
            raise SpecError("spec.recovery: shock_window must fit inside post_event_horizon")
        if recovery.rolling_window + recovery.sustain_windows - 1 > recovery.post_event_horizon:
            raise SpecError("spec.recovery: rolling and sustain windows must fit inside post_event_horizon")
        if self.changed_law.branch_steps < recovery.post_event_horizon:
            raise SpecError("spec.changed_law.branch_steps must cover the full post-event recovery horizon")
        if self.changed_law.prefix_steps < recovery.pre_event_reference_length + self.world.history_length:
            raise SpecError("spec.changed_law.prefix_steps must cover the pre-event reference window")
        speed_change = self.motion_families["speed_change"]
        if speed_change.change_step is None:
            raise SpecError("spec.motion_families.speed_change requires an explicit change_step")
        if speed_change.change_step + recovery.post_event_horizon > speed_change.steps_per_episode:
            raise SpecError(
                "spec.motion_families.speed_change: post-event horizon must fit inside the episode"
            )
        constant_velocity = self.motion_families["constant_velocity"]
        if not constant_velocity.stratified:
            raise SpecError("spec.motion_families.constant_velocity must be stratified")
        evaluators = {gate.evaluator for gate in self.gates}
        if "correctness" not in evaluators or "reproducibility" not in evaluators:
            raise SpecError("spec.gates: correctness and reproducibility gates are mandatory")

    # -- derived helpers -------------------------------------------------
    @property
    def displacement_scale(self) -> float:
        return self.world.dt * self.candidate.displacement_scale_speed

    def required_strata(self) -> tuple[str, ...]:
        return self.stratification.strata()

    def gate(self, name: str) -> GateSpec:
        for gate in self.gates:
            if gate.name == name:
                return gate
        raise SpecError(f"unknown gate {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    return value


def canonical_spec_path(project_root: Path | None = None) -> Path:
    """Locate the canonical spec, working from an installed package too."""

    packaged = Path(__file__).resolve().parent / "data" / "benchmark_v2_1.json"
    if packaged.exists():
        return packaged
    root = project_root or Path(__file__).resolve().parents[2]
    candidate = root / "benchmarks" / "benchmark_v2_1.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "canonical benchmark specification not found; expected aaa/benchmark/data/benchmark_v2_1.json"
    )


def load_spec(path: str | Path | None = None) -> BenchmarkSpec:
    """Load and validate a specification. ``None`` loads the canonical one."""

    source = Path(path) if path is not None else canonical_spec_path()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SpecError(f"{source}: malformed JSON: {error}") from error
    return BenchmarkSpec.parse(raw)


def spec_hash(spec: BenchmarkSpec) -> str:
    """SHA-256 of the resolved specification, independent of file formatting."""

    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_spec_hash() -> str:
    return spec_hash(load_spec())


def leaf_paths(value: Any, prefix: str = "") -> list[str]:
    """Every leaf key path in a resolved spec; used by drift tests."""

    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(leaf_paths(item, f"{prefix}{key}."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(leaf_paths(item, f"{prefix}{index}."))
    else:
        paths.append(prefix.rstrip("."))
    return paths


def describe_sections() -> Sequence[str]:
    return tuple(_field_names(BenchmarkSpec))
