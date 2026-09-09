"""Small, explicit configuration objects used by the AAA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class WorldConfig:
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    dt: float = 0.02
    steps_per_episode: int = 120
    history_length: int = 4
    speed_min: float = 0.12
    speed_max: float = 0.32
    change_step: int = 60
    change_factor_low: float = 0.55
    change_factor_high: float = 1.65
    event_margin: float = 0.14


@dataclass(frozen=True)
class ExperimentConfig:
    world: WorldConfig = WorldConfig()
    learning_rate: float = 0.08
    learning_rate_candidates: tuple[float, ...] = (0.02, 0.04, 0.08, 0.16)
    dev_seeds: tuple[int, ...] = (101, 102, 103)
    training_seeds: tuple[int, ...] = (11, 12, 13, 14, 15)
    final_seeds: tuple[int, ...] = (201, 202, 203, 204, 205, 206, 207, 208, 209, 210)
    dev_training_episodes: int = 8
    dev_validation_episodes: int = 3
    training_episodes: int = 14
    generalization_episodes_per_scenario: int = 2
    training_scenarios: tuple[str, ...] = ("straight", "bouncing")
    final_generalization_scenarios: tuple[str, ...] = ("straight", "bouncing")
    post_change_window: int = 20
    rolling_window: int = 5
    recovery_sustain_windows: int = 3
    recovery_multiplier: float = 1.5
    recovery_floor: float = 0.01
    meaningful_change_fraction: float = 0.25
    clip_predictions: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly resolved configuration."""

        return _jsonable(asdict(self))

    def quick(self) -> "ExperimentConfig":
        """Return a small configuration suitable for a smoke run."""

        return replace(
            self,
            dev_seeds=(101,),
            training_seeds=(11,),
            final_seeds=(201, 202),
            dev_training_episodes=3,
            dev_validation_episodes=1,
            training_episodes=4,
            generalization_episodes_per_scenario=1,
            world=replace(self.world, steps_per_episode=70, change_step=35),
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
