"""Small, explicit configuration objects used by the AAA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
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

    def __post_init__(self) -> None:
        values = {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "dt": self.dt,
            "speed_min": self.speed_min,
            "speed_max": self.speed_max,
            "change_factor_low": self.change_factor_low,
            "change_factor_high": self.change_factor_high,
            "event_margin": self.event_margin,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("world configuration values must be finite")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be positive")
        # A short diagnostic episode may intentionally contain only warm-up
        # observations. The runner will then produce zero scored transitions.
        if self.history_length < 2:
            raise ValueError("history_length must be at least 2")
        if self.speed_min <= 0 or self.speed_max < self.speed_min:
            raise ValueError("speed range must satisfy 0 < speed_min <= speed_max")
        if self.change_step < 0:
            raise ValueError("change_step must be non-negative")
        if self.change_factor_low <= 0 or self.change_factor_high <= 0:
            raise ValueError("change factors must be positive")
        if self.event_margin < 0 or self.event_margin > (self.upper_bound - self.lower_bound) / 2:
            raise ValueError("event_margin must fit inside half the configured interval")


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

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or any(rate <= 0 for rate in self.learning_rate_candidates):
            raise ValueError("learning rates must be positive")
        if not self.dev_seeds or not self.training_seeds or not self.final_seeds:
            raise ValueError("development, training, and final seed sets must be non-empty")
        if self.post_change_window <= 0 or self.rolling_window <= 0:
            raise ValueError("post-change and rolling windows must be positive")
        if self.recovery_sustain_windows <= 0:
            raise ValueError("recovery_sustain_windows must be positive")
        if self.recovery_multiplier <= 0 or self.recovery_floor < 0:
            raise ValueError("recovery thresholds must be non-negative and meaningful")
        if self.meaningful_change_fraction < 0:
            raise ValueError("meaningful_change_fraction must be non-negative")
        if not self.training_scenarios or not self.final_generalization_scenarios:
            raise ValueError("training and evaluation scenario sets must be non-empty")

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
