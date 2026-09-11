"""Small, explicit configuration objects used by the AAA experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    """Reject booleans, floats and non-integral values used as counts."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if not isinstance(value, (int, float, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {number}")
    return number


@dataclass(frozen=True)
class WorldConfig:
    """Resolved world parameters.

    ``change_step`` uses explicit semantics: ``None`` means *no change event
    in this episode*; an integer must index a transition that is actually
    simulated, i.e. ``0 <= change_step < steps_per_episode``.
    """

    lower_bound: float = 0.0
    upper_bound: float = 1.0
    dt: float = 0.02
    steps_per_episode: int = 120
    history_length: int = 4
    speed_min: float = 0.12
    speed_max: float = 0.32
    change_step: int | None = None
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
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
                raise ValueError(f"world configuration value {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"world configuration value {name} must be finite")
        object.__setattr__(self, "steps_per_episode", _require_int(self.steps_per_episode, "steps_per_episode", minimum=1))
        object.__setattr__(self, "history_length", _require_int(self.history_length, "history_length", minimum=2))
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.speed_min <= 0 or self.speed_max < self.speed_min:
            raise ValueError("speed range must satisfy 0 < speed_min <= speed_max")
        if self.change_step is not None:
            step = _require_int(self.change_step, "change_step", minimum=0)
            if step >= self.steps_per_episode:
                raise ValueError(
                    "change_step must identify a simulated transition: 0 <= change_step < steps_per_episode"
                )
            object.__setattr__(self, "change_step", step)
        if self.change_factor_low <= 0 or self.change_factor_high <= 0:
            raise ValueError("change factors must be positive")
        if self.event_margin < 0 or self.event_margin > (self.upper_bound - self.lower_bound) / 2:
            raise ValueError("event_margin must fit inside half the configured interval")

    @property
    def width(self) -> float:
        """Interval width ``L``; the canonical normalization constant."""

        return float(self.upper_bound - self.lower_bound)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the historical v1 evaluation track."""

    world: WorldConfig = WorldConfig(change_step=60)
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
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rates must be positive and finite")
        if any(not math.isfinite(rate) or rate <= 0 for rate in self.learning_rate_candidates):
            raise ValueError("learning rates must be positive and finite")
        if not self.dev_seeds or not self.training_seeds or not self.final_seeds:
            raise ValueError("development, training, and final seed sets must be non-empty")
        for name in (
            "dev_training_episodes",
            "dev_validation_episodes",
            "training_episodes",
            "generalization_episodes_per_scenario",
            "post_change_window",
            "rolling_window",
            "recovery_sustain_windows",
        ):
            _require_int(getattr(self, name), name, minimum=1)
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
