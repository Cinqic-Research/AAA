"""The strict observation, prediction, scoring, update loop.

This module owns AAA's causal boundary. For every scored transition:

1. the learner receives only causally available observation history;
2. all predictors produce their predictions;
3. predictions are recorded;
4. the environment advances;
5. the actual target is revealed;
6. the prediction error is scored;
7. only then may an enabled learner update;
8. the newly revealed observation enters history.

No scenario name, event flag, velocity, change schedule, hidden coefficient or
future observation ever crosses into a predictor.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .environment import Environment
from .predictors import Predictor

STEP_RECORD_SCHEMA = "aaa.step_record.v2"

UPDATE_MODES = ("frozen", "online", "mixed")


@dataclass(frozen=True)
class TrialIdentity:
    """Immutable provenance for one evaluated trajectory.

    Previously ``StepRecord.seed`` was overloaded to carry a replica ID while
    ``environment_seed`` carried the trajectory seed. Every distinct concept
    now has its own named field.
    """

    trial_id: str
    role: str
    family: str
    scenario: str
    environment_seed: int
    replica_id: int
    episode: int
    branch: str = "main"
    confirmation_batch: str | None = None
    training_seed_lineage: tuple[int, ...] = ()
    stratum: str = "unstratified"
    checkpoint_hash: str | None = None
    update_mode: str = "frozen"
    schema_version: str = STEP_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.update_mode not in UPDATE_MODES:
            raise ValueError(f"update_mode must be one of {UPDATE_MODES}")
        for name in ("replica_id", "episode"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["training_seed_lineage"] = list(self.training_seed_lineage)
        return value


@dataclass(frozen=True)
class StepRecord:
    """One scored transition and every predictor's outcome on it."""

    identity: TrialIdentity
    step: int
    target_step: int
    history: tuple[float, ...]
    current_observation: float
    actual_next_position: float
    bounced: bool
    changed: bool
    predictions: dict[str, dict[str, float]]
    updates_enabled: dict[str, bool]
    bounce_walls: tuple[str, ...] = field(default=())
    interval_width: float = 1.0

    # -- convenience accessors -------------------------------------------
    @property
    def replica_id(self) -> int:
        return self.identity.replica_id

    @property
    def episode(self) -> int:
        return self.identity.episode

    @property
    def scenario(self) -> str:
        return self.identity.scenario

    @property
    def stratum(self) -> str:
        return self.identity.stratum

    @property
    def environment_seed(self) -> int:
        return self.identity.environment_seed

    @property
    def bounce_count(self) -> int:
        return len(self.bounce_walls)

    @property
    def bounce_wall(self) -> str | None:
        return self.bounce_walls[0] if self.bounce_walls else None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity.to_dict(),
            "step": self.step,
            "target_step": self.target_step,
            "history": list(self.history),
            "current_observation": self.current_observation,
            "actual_next_position": self.actual_next_position,
            "bounced": self.bounced,
            "bounce_walls": list(self.bounce_walls),
            "bounce_count": self.bounce_count,
            "changed": self.changed,
            "interval_width": self.interval_width,
            "predictions": self.predictions,
            "updates_enabled": self.updates_enabled,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StepRecord:
        allowed = {
            "schema_version",
            "trial_id",
            "role",
            "family",
            "scenario",
            "environment_seed",
            "replica_id",
            "episode",
            "branch",
            "confirmation_batch",
            "training_seed_lineage",
            "stratum",
            "checkpoint_hash",
            "update_mode",
            "step",
            "target_step",
            "history",
            "current_observation",
            "actual_next_position",
            "bounced",
            "bounce_walls",
            "bounce_count",
            "changed",
            "interval_width",
            "predictions",
            "updates_enabled",
        }
        required = allowed - {
            "branch",
            "confirmation_batch",
            "training_seed_lineage",
            "stratum",
            "checkpoint_hash",
            "update_mode",
            "bounce_walls",
            "bounce_count",
            "interval_width",
        }
        unknown = sorted(set(value) - allowed)
        missing = sorted(required - set(value))
        if unknown or missing:
            raise ValueError(f"invalid step record fields: unknown={unknown}, missing={missing}")
        if value.get("schema_version") != STEP_RECORD_SCHEMA:
            raise ValueError(
                f"unsupported step record schema {value.get('schema_version')!r}; expected {STEP_RECORD_SCHEMA!r}"
            )
        integer_fields = ("environment_seed", "replica_id", "episode", "step", "target_step")
        for name in integer_fields:
            if isinstance(value[name], bool) or not isinstance(value[name], int):
                raise ValueError(f"step record {name} must be an integer")
        for name in ("bounced", "changed"):
            if not isinstance(value[name], bool):
                raise ValueError(f"step record {name} must be a boolean")
        history = value["history"]
        if not isinstance(history, list) or not history:
            raise ValueError("step record history must be a non-empty list")
        numeric_values = [
            *history,
            value["current_observation"],
            value["actual_next_position"],
            value.get("interval_width", 1.0),
        ]
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
            for item in numeric_values
        ):
            raise ValueError("step record numeric values must be finite numbers")
        predictions = value["predictions"]
        updates = value["updates_enabled"]
        if not isinstance(predictions, dict) or not predictions:
            raise ValueError("step record predictions must be a non-empty object")
        if not isinstance(updates, dict) or set(updates) != set(predictions):
            raise ValueError("step record update flags must exactly match predictor names")
        metric_fields = {
            "raw",
            "scored",
            "absolute_error",
            "normalized_absolute_error",
            "signed_error",
        }
        for name, metrics in predictions.items():
            if not isinstance(name, str) or not isinstance(metrics, dict) or set(metrics) != metric_fields:
                raise ValueError(f"predictor {name!r} has an invalid metric shape")
            if any(
                isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
                for item in metrics.values()
            ):
                raise ValueError(f"predictor {name!r} has a non-finite or non-numeric metric")
        if any(not isinstance(flag, bool) for flag in updates.values()):
            raise ValueError("step record update flags must be booleans")
        lineage = value.get("training_seed_lineage", [])
        if not isinstance(lineage, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in lineage
        ):
            raise ValueError("training_seed_lineage must be a list of integers")
        walls = value.get("bounce_walls", [])
        if not isinstance(walls, list) or any(not isinstance(item, str) for item in walls):
            raise ValueError("bounce_walls must be a list of strings")
        if "bounce_count" in value:
            count = value["bounce_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count != len(walls):
                raise ValueError("bounce_count must equal the number of bounce_walls")

        identity = TrialIdentity(
            trial_id=str(value["trial_id"]),
            role=str(value["role"]),
            family=str(value["family"]),
            scenario=str(value["scenario"]),
            environment_seed=int(value["environment_seed"]),
            replica_id=int(value["replica_id"]),
            episode=int(value["episode"]),
            branch=str(value.get("branch", "main")),
            confirmation_batch=(
                None if value.get("confirmation_batch") is None else str(value["confirmation_batch"])
            ),
            training_seed_lineage=tuple(lineage),
            stratum=str(value.get("stratum", "unstratified")),
            checkpoint_hash=(None if value.get("checkpoint_hash") is None else str(value["checkpoint_hash"])),
            update_mode=str(value.get("update_mode", "frozen")),
        )
        return cls(
            identity=identity,
            step=int(value["step"]),
            target_step=int(value["target_step"]),
            history=tuple(float(item) for item in history),
            current_observation=float(value["current_observation"]),
            actual_next_position=float(value["actual_next_position"]),
            bounced=value["bounced"],
            changed=value["changed"],
            predictions={
                str(name): {str(k): float(v) for k, v in metrics.items()}
                for name, metrics in predictions.items()
            },
            updates_enabled=dict(updates),
            bounce_walls=tuple(walls),
            interval_width=float(value.get("interval_width", 1.0)),
        )


ScoreHook = Callable[[StepRecord], None]
UpdateHook = Callable[[StepRecord, Sequence[Predictor]], None]


def run_episode(
    environment: Environment,
    predictors: Sequence[Predictor],
    identity: TrialIdentity,
    *,
    learn: bool = False,
    clip_predictions: bool = False,
    score_hook: ScoreHook | None = None,
    update_hook: UpdateHook | None = None,
    initial_history: Sequence[float] | None = None,
    reset_environment: bool = True,
    step_offset: int = 0,
) -> list[StepRecord]:
    """Run one episode with prediction before outcome reveal and learning."""

    if reset_environment:
        current = environment.reset()
        history: list[float] = [current]
    else:
        if initial_history is None:
            raise ValueError("initial_history is required when reset_environment is false")
        history = [float(value) for value in initial_history]
        if not history:
            raise ValueError("initial_history must not be empty")
        if not all(math.isfinite(value) for value in history):
            raise ValueError("initial_history must be finite")
    records: list[StepRecord] = []
    history_length = environment.config.history_length
    lower = environment.config.lower_bound
    upper = environment.config.upper_bound
    width = upper - lower
    if len({predictor.name for predictor in predictors}) != len(predictors):
        raise ValueError("predictor names must be unique within an episode")

    for local_step in range(environment.config.steps_per_episode):
        step = step_offset + local_step
        if len(history) < history_length:
            # Warm-up observations are not scored so all predictors start on
            # precisely the same observation window.
            transition = environment.advance()
            history.append(transition.position)
            continue

        input_history = tuple(history[-history_length:])
        raw_predictions: dict[str, float] = {}
        scored_predictions: dict[str, float] = {}
        for predictor in predictors:
            raw = float(predictor.predict(input_history))
            if not math.isfinite(raw):
                raise FloatingPointError(f"predictor {predictor.name} returned a non-finite value")
            raw_predictions[predictor.name] = raw
            scored_predictions[predictor.name] = min(max(raw, lower), upper) if clip_predictions else raw

        # The environment is advanced only after every prediction has been
        # recorded. This is the temporal boundary that prevents leakage.
        transition = environment.advance()
        predictions = {
            predictor.name: {
                "raw": raw_predictions[predictor.name],
                "scored": scored_predictions[predictor.name],
                "absolute_error": abs(scored_predictions[predictor.name] - transition.position),
                "normalized_absolute_error": abs(scored_predictions[predictor.name] - transition.position)
                / width,
                "signed_error": scored_predictions[predictor.name] - transition.position,
            }
            for predictor in predictors
        }
        record = StepRecord(
            identity=identity,
            step=step,
            target_step=step + 1,
            history=input_history,
            current_observation=float(input_history[-1]),
            actual_next_position=float(transition.position),
            bounced=transition.bounced,
            changed=transition.changed,
            bounce_walls=transition.bounce_walls,
            predictions=predictions,
            updates_enabled={
                predictor.name: bool(learn and predictor.update_enabled) for predictor in predictors
            },
            interval_width=width,
        )
        records.append(record)
        if score_hook is not None:
            score_hook(record)

        # Scoring has completed. Only now may an enabled predictor learn from
        # the revealed target. Baselines implement a no-op update.
        if learn:
            for predictor in predictors:
                if predictor.update_enabled:
                    predictor.update(input_history, transition.position)
            if update_hook is not None:
                update_hook(record, predictors)
        history.append(transition.position)

    return records


def continue_episode(
    environment: Environment,
    predictors: Sequence[Predictor],
    history: Sequence[float],
    identity: TrialIdentity,
    *,
    learn: bool = False,
    clip_predictions: bool = False,
    score_hook: ScoreHook | None = None,
    update_hook: UpdateHook | None = None,
    step_offset: int = 0,
) -> list[StepRecord]:
    """Continue an already-realized world from a supplied observation history.

    This is used for matched frozen/online branch experiments. The caller
    creates the branch at the intervention boundary, clones learner state,
    and supplies the last observations from the common pre-event prefix.
    """

    if not hasattr(environment, "advance") or not hasattr(environment, "config"):
        raise TypeError("environment must provide advance() and config")
    return run_episode(
        environment,
        predictors,
        identity,
        learn=learn,
        clip_predictions=clip_predictions,
        score_hook=score_hook,
        update_hook=update_hook,
        initial_history=history,
        reset_environment=False,
        step_offset=step_offset,
    )


def with_stratum(identity: TrialIdentity, stratum: str) -> TrialIdentity:
    return replace(identity, stratum=stratum)


def write_step_records(
    records: Sequence[StepRecord], jsonl_path: str | Path, csv_path: str | Path | None = None
) -> None:
    """Write complete nested JSONL and an optional flat CSV view."""

    jsonl_destination = Path(jsonl_path)
    jsonl_destination.parent.mkdir(parents=True, exist_ok=True)
    jsonl_temporary = jsonl_destination.with_suffix(jsonl_destination.suffix + ".tmp")
    with jsonl_temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    jsonl_temporary.replace(jsonl_destination)

    if csv_path is None:
        return
    csv_destination = Path(csv_path)
    predictor_names = sorted({name for record in records for name in record.predictions})
    fieldnames = [
        "trial_id",
        "role",
        "family",
        "branch",
        "replica_id",
        "environment_seed",
        "episode",
        "scenario",
        "stratum",
        "update_mode",
        "step",
        "target_step",
        "history",
        "current_observation",
        "actual_next_position",
        "bounced",
        "bounce_count",
        "changed",
    ]
    for name in predictor_names:
        fieldnames.extend(
            [f"{name}_raw", f"{name}_scored", f"{name}_absolute_error", f"{name}_normalized_absolute_error"]
        )
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = csv_destination.with_suffix(csv_destination.suffix + ".tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "trial_id": record.identity.trial_id,
                "role": record.identity.role,
                "family": record.identity.family,
                "branch": record.identity.branch,
                "replica_id": record.replica_id,
                "environment_seed": record.environment_seed,
                "episode": record.episode,
                "scenario": record.scenario,
                "stratum": record.stratum,
                "update_mode": record.identity.update_mode,
                "step": record.step,
                "target_step": record.target_step,
                "history": json.dumps(list(record.history)),
                "current_observation": record.current_observation,
                "actual_next_position": record.actual_next_position,
                "bounced": record.bounced,
                "bounce_count": record.bounce_count,
                "changed": record.changed,
            }
            for name in predictor_names:
                prediction = record.predictions.get(name, {})
                row[f"{name}_raw"] = prediction.get("raw", "")
                row[f"{name}_scored"] = prediction.get("scored", "")
                row[f"{name}_absolute_error"] = prediction.get("absolute_error", "")
                row[f"{name}_normalized_absolute_error"] = prediction.get("normalized_absolute_error", "")
            writer.writerow(row)
    csv_temporary.replace(csv_destination)


def read_step_records(path: str | Path) -> list[StepRecord]:
    """Read a JSONL (optionally gzipped) step log back into typed records."""

    source = Path(path)
    if source.suffix == ".gz":
        import gzip

        with gzip.open(source, "rt", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    else:
        lines = source.read_text(encoding="utf-8").splitlines()
    return [StepRecord.from_dict(json.loads(line)) for line in lines if line.strip()]
