"""The strict observation, prediction, scoring, update loop."""

from __future__ import annotations

import csv
import json
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
            "predictions": self.predictions,
            "updates_enabled": self.updates_enabled,
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
) -> list[StepRecord]:
    """Run one episode with prediction before outcome reveal and learning.

    The runner is the only code that sees evaluator event labels. A predictor
    receives a tuple of positions and, after scoring, the target position.
    No velocity, scenario, event flag, or schedule is passed into the model.
    """

    current = environment.reset()
    history: list[float] = [current]
    records: list[StepRecord] = []
    history_length = environment.config.history_length
    bounds = (environment.config.lower_bound, environment.config.upper_bound)

    for step in range(environment.config.steps_per_episode):
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
            predictions=predictions,
            updates_enabled={
                predictor.name: bool(learn and predictor.update_enabled) for predictor in predictors
            },
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


def write_step_records(records: Sequence[StepRecord], jsonl_path: str | Path, csv_path: str | Path | None = None) -> None:
    """Write complete nested JSONL and an optional flat CSV view."""

    jsonl_destination = Path(jsonl_path)
    jsonl_destination.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

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
        fieldnames.extend([f"{name}_raw", f"{name}_scored", f"{name}_absolute_error"])
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    with csv_destination.open("w", newline="", encoding="utf-8") as handle:
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
            writer.writerow(row)
