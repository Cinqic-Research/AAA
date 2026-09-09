"""The one-dimensional moving-dot world used by AAA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import WorldConfig

Scenario = Literal["straight", "bouncing", "changed"]
SCENARIOS: tuple[Scenario, ...] = ("straight", "bouncing", "changed")


@dataclass(frozen=True)
class EnvironmentStep:
    """Evaluator-side information for one transition.

    ``bounced`` and ``changed`` are deliberately evaluator metadata. The
    experiment runner passes only ``position`` to a learner.
    """

    step_index: int
    position: float
    bounced: bool
    changed: bool


class MovingDotEnvironment:
    """Deterministic, seeded one-dimensional motion with optional events."""

    def __init__(
        self,
        scenario: Scenario,
        seed: int,
        config: WorldConfig | None = None,
        *,
        initial_position: float | None = None,
        initial_velocity: float | None = None,
        change_step: int | None = None,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.seed = int(seed)
        self.config = config or WorldConfig()
        self._initial_position = initial_position
        self._initial_velocity = initial_velocity
        self._requested_change_step = change_step
        self._rng = np.random.default_rng(self.seed)
        self._position = 0.0
        self._velocity = 0.0
        self._step_index = 0
        self._change_factor = 1.0
        self._change_step = None
        self.reset()

    @property
    def position(self) -> float:
        """True position, for evaluator/debugging use only."""

        return self._position

    @property
    def velocity(self) -> float:
        """True velocity, for evaluator/debugging use only.

        The AAA runner never passes this value to a predictor.
        """

        return self._velocity

    @property
    def change_step(self) -> int | None:
        return self._change_step

    def observe(self) -> float:
        """Return the only observation available to the learner."""

        return float(self._position)

    def reset(self) -> float:
        """Reset to a deterministic episode state and return its position."""

        self._rng = np.random.default_rng(self.seed)
        self._step_index = 0
        self._position, self._velocity = self._sample_initial_state()
        self._change_factor = self._sample_change_factor()
        self._change_step = self._resolve_change_step()
        return self.observe()

    def advance(self) -> EnvironmentStep:
        """Advance exactly one step after a prediction has been recorded."""

        changed = self._change_step is not None and self._step_index == self._change_step
        if changed:
            self._velocity *= self._change_factor

        bounced = False
        next_position = self._position + self._velocity * self.config.dt
        if self.scenario in ("bouncing", "changed"):
            next_position, self._velocity, bounced = self._reflect(
                next_position,
                self._velocity,
                self.config.lower_bound,
                self.config.upper_bound,
            )
        else:
            if not self.config.lower_bound <= next_position <= self.config.upper_bound:
                raise RuntimeError("straight scenario crossed a boundary; sampling is invalid")

        self._position = float(next_position)
        transition = EnvironmentStep(
            step_index=self._step_index,
            position=self._position,
            bounced=bounced,
            changed=changed,
        )
        self._step_index += 1
        return transition

    def _sample_initial_state(self) -> tuple[float, float]:
        if self._initial_position is not None or self._initial_velocity is not None:
            if self._initial_position is None or self._initial_velocity is None:
                raise ValueError("initial_position and initial_velocity must be provided together")
            if not self.config.lower_bound <= self._initial_position <= self.config.upper_bound:
                raise ValueError("initial_position is outside the configured interval")
            return float(self._initial_position), float(self._initial_velocity)

        for _ in range(10_000):
            position = float(self._rng.uniform(0.08, 0.92))
            speed = float(self._rng.uniform(self.config.speed_min, self.config.speed_max))
            velocity = speed if self._rng.integers(0, 2) else -speed
            if self.scenario == "straight":
                final_position = position + velocity * self.config.dt * self.config.steps_per_episode
                if self.config.lower_bound < final_position < self.config.upper_bound:
                    return position, velocity
            elif self.scenario in ("bouncing", "changed"):
                if self.scenario == "changed":
                    event_step = self._requested_change_step
                    if event_step is None:
                        event_step = self.config.change_step
                    event_position, _, _ = self._simulate_reflection(
                        position,
                        velocity,
                        event_step,
                    )
                    if not (
                        self.config.lower_bound + self.config.event_margin
                        <= event_position
                        <= self.config.upper_bound - self.config.event_margin
                    ):
                        continue
                return position, velocity
        raise RuntimeError("could not sample a valid initial state")

    def _sample_change_factor(self) -> float:
        if self.scenario != "changed":
            return 1.0
        if self._rng.integers(0, 2) == 0:
            return self.config.change_factor_low
        return self.config.change_factor_high

    def _resolve_change_step(self) -> int | None:
        if self.scenario != "changed":
            return None
        step = self._requested_change_step
        if step is None:
            step = self.config.change_step
        if not 0 <= step < self.config.steps_per_episode:
            raise ValueError("change_step must identify a transition inside the episode")
        return int(step)

    @staticmethod
    def _reflect(
        position: float,
        velocity: float,
        lower: float,
        upper: float,
    ) -> tuple[float, float, bool]:
        """Reflect overshoot until the position is inside the interval."""

        bounced = False
        # Speeds used by this prototype cross at most a few boundaries per
        # step, but the loop keeps the boundary operation correct for tests
        # and future configurations with larger steps.
        for _ in range(10_000):
            if position > upper:
                position = upper - (position - upper)
                velocity = -abs(velocity)
                bounced = True
            elif position < lower:
                position = lower + (lower - position)
                velocity = abs(velocity)
                bounced = True
            else:
                return float(position), float(velocity), bounced
        raise RuntimeError("reflection exceeded safety iteration limit")

    def _simulate_reflection(
        self,
        position: float,
        velocity: float,
        steps: int,
    ) -> tuple[float, float, bool]:
        bounced = False
        for _ in range(steps):
            position, velocity, this_bounced = self._reflect(
                position + velocity * self.config.dt,
                velocity,
                self.config.lower_bound,
                self.config.upper_bound,
            )
            bounced = bounced or this_bounced
        return position, velocity, bounced
