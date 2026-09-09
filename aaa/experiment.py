"""The strict observation, prediction, scoring, update loop."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .environment import MovingDotEnvironment
from .predictors import Predictor


@dataclass(frozen=True)
class StepRecord:
    seed: int
    environment_seed: int
    episode: int
    scenario: str
    step: int
    target_step: int
    history: tuple[float, ...]
    current_observation: float
    actual_next_position: float
    bounced: bool
    changed: bool
    predictions: dict[str, dict[str, float]]
    updates_enabled: dict[str, bool]
    bounce_wall: str | None = None
    stratum: str = "unstratified"

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "environment_seed": self.environment_seed,
            "episode": self.episode,
            "scenario": self.scenario,
            "step": self.step,
            "target_step": self.target_step,
            "history": list(self.history),
            "current_observation": self.current_observation,
            "actual_next_position": self.actual_next_position,
            "bounced": self.bounced,
            "changed": self.changed,
            "bounce_wall": self.bounce_wall,
            "predictions": self.predictions,
            "updates_enabled": self.updates_enabled,
            "stratum": self.stratum,
        }


ScoreHook = Callable[[StepRecord], None]


def run_episode(
    environment: MovingDotEnvironment,
    predictors: Sequence[Predictor],
    *,
    episode: int = 0,
    record_seed: int | None = None,
    learn: bool = False,
    clip_predictions: bool = False,
    score_hook: ScoreHook | None = None,
    initial_history: Sequence[float] | None = None,
    reset_environment: bool = True,
    step_offset: int = 0,
    stratum: str = "unstratified",
) -> list[StepRecord]:
    """Run one episode with prediction before outcome reveal and learning.

    The runner is the only code that sees evaluator event labels. A predictor
    receives a tuple of positions and, after scoring, the target position.
    No velocity, scenario, event flag, or schedule is passed into the model.
    """

    if reset_environment:
        current = environment.reset()
        history: list[float] = [current]
    else:
        if initial_history is None:
            raise ValueError("initial_history is required when reset_environment is false")
        history = [float(value) for value in initial_history]
        if not history:
            raise ValueError("initial_history must not be empty")
    records: list[StepRecord] = []
    history_length = environment.config.history_length
    bounds = (environment.config.lower_bound, environment.config.upper_bound)
    if len({predictor.name for predictor in predictors}) != len(predictors):
        raise ValueError("predictor names must be unique within an episode")

    for local_step in range(environment.config.steps_per_episode):
        step = step_offset + local_step
        if len(history) < history_length:
            # Warm-up observations are not scored so all predictors start on
            # precisely the same four-observation input window.
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
            scored_predictions[predictor.name] = (
                min(max(raw, bounds[0]), bounds[1]) if clip_predictions else raw
            )

        # The environment is advanced only after every prediction has been
        # recorded. This is the temporal boundary that prevents leakage.
        transition = environment.advance()
        predictions = {
            predictor.name: {
                "raw": raw_predictions[predictor.name],
                "scored": scored_predictions[predictor.name],
                "absolute_error": abs(scored_predictions[predictor.name] - transition.position),
                "normalized_absolute_error": abs(scored_predictions[predictor.name] - transition.position)
                / (bounds[1] - bounds[0]),
            }
            for predictor in predictors
        }
        record = StepRecord(
            seed=environment.seed if record_seed is None else int(record_seed),
            environment_seed=environment.seed,
            episode=episode,
            scenario=environment.scenario,
            step=step,
            target_step=step + 1,
            history=input_history,
            current_observation=float(input_history[-1]),
            actual_next_position=float(transition.position),
            bounced=transition.bounced,
            changed=transition.changed,
            bounce_wall=transition.bounce_wall,
            predictions=predictions,
            updates_enabled={
                predictor.name: bool(learn and predictor.update_enabled) for predictor in predictors
            },
            stratum=stratum,
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
        history.append(transition.position)

    return records


def continue_episode(
    environment: object,
    predictors: Sequence[Predictor],
    history: Sequence[float],
    *,
    episode: int = 0,
    record_seed: int | None = None,
    learn: bool = False,
    clip_predictions: bool = False,
    score_hook: ScoreHook | None = None,
    step_offset: int = 0,
    stratum: str = "unstratified",
) -> list[StepRecord]:
    """Continue an already-realized world from a supplied observation history.

    This is used for matched frozen/online branch experiments. The caller
    creates the branch at the intervention boundary, clones learner state,
    and supplies the last observations from the common pre-event prefix.
    """

    if not hasattr(environment, "advance") or not hasattr(environment, "config"):
        raise TypeError("environment must provide advance() and config")
    return run_episode(
        environment,  # type: ignore[arg-type]
        predictors,
        episode=episode,
        record_seed=record_seed,
        learn=learn,
        clip_predictions=clip_predictions,
        score_hook=score_hook,
        initial_history=history,
        reset_environment=False,
        step_offset=step_offset,
        stratum=stratum,
    )


def write_step_records(records: Sequence[StepRecord], jsonl_path: str | Path, csv_path: str | Path | None = None) -> None:
    """Write complete nested JSONL and an optional flat CSV view."""

    jsonl_destination = Path(jsonl_path)
    jsonl_destination.parent.mkdir(parents=True, exist_ok=True)
    jsonl_temporary = jsonl_destination.with_suffix(jsonl_destination.suffix + ".tmp")
    with jsonl_temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    os.replace(jsonl_temporary, jsonl_destination)

    if csv_path is None:
        return
    csv_destination = Path(csv_path)
    predictor_names = sorted({name for record in records for name in record.predictions})
    fieldnames = [
        "seed",
        "environment_seed",
        "episode",
        "scenario",
        "step",
        "target_step",
        "history",
        "current_observation",
        "actual_next_position",
        "bounced",
        "changed",
    ]
    for name in predictor_names:
        fieldnames.extend([f"{name}_raw", f"{name}_scored", f"{name}_absolute_error", f"{name}_normalized_absolute_error"])
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = csv_destination.with_suffix(csv_destination.suffix + ".tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {
                "seed": record.seed,
                "environment_seed": record.environment_seed,
                "episode": record.episode,
                "scenario": record.scenario,
                "step": record.step,
                "target_step": record.target_step,
                "history": json.dumps(record.history),
                "current_observation": record.current_observation,
                "actual_next_position": record.actual_next_position,
                "bounced": record.bounced,
                "changed": record.changed,
            }
            for name in predictor_names:
                prediction = record.predictions.get(name, {})
                row[f"{name}_raw"] = prediction.get("raw", "")
                row[f"{name}_scored"] = prediction.get("scored", "")
                row[f"{name}_absolute_error"] = prediction.get("absolute_error", "")
                row[f"{name}_normalized_absolute_error"] = prediction.get("normalized_absolute_error", "")
            writer.writerow(row)
    os.replace(csv_temporary, csv_destination)
