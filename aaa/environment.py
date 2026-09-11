"""The one-dimensional moving-dot world used by AAA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .config import WorldConfig

Scenario = Literal["straight", "bouncing", "changed", "dynamics_change"]
SCENARIOS: tuple[Scenario, ...] = ("straight", "bouncing", "changed", "dynamics_change")


def as_scenario(value: str) -> Scenario:
    """Narrow a string to :data:`Scenario`, rejecting anything else."""

    if value not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {value}")
    return value


@dataclass(frozen=True)
class ReflectionResult:
    """Complete outcome of resolving one raw position against the bounds."""

    position: float
    velocity: float
    walls: tuple[str, ...]

    @property
    def bounced(self) -> bool:
        return bool(self.walls)

    @property
    def bounce_count(self) -> int:
        return len(self.walls)

    @property
    def first_wall(self) -> str | None:
        return self.walls[0] if self.walls else None


def resolve_reflection(
    position: float, velocity: float, lower: float, upper: float, *, iteration_limit: int = 10_000
) -> ReflectionResult:
    """Reflect an overshooting position back into ``[lower, upper]``.

    Every wall contact is recorded in crossing order, so a transition that
    crosses two walls is reported as two walls rather than collapsed into a
    single boolean plus one inferred wall. Exact contact counts as a bounce
    only when the velocity points out of the interval.
    """

    if upper <= lower:
        raise ValueError("upper bound must exceed lower bound")
    if not all(math.isfinite(float(value)) for value in (position, velocity, lower, upper)):
        raise ValueError("reflection inputs must be finite")
    walls: list[str] = []
    for _ in range(iteration_limit):
        if position > upper or (position == upper and velocity > 0):
            position = upper - (position - upper)
            velocity = -abs(velocity)
            walls.append("upper")
        elif position < lower or (position == lower and velocity < 0):
            position = lower + (lower - position)
            velocity = abs(velocity)
            walls.append("lower")
        else:
            return ReflectionResult(float(position), float(velocity), tuple(walls))
    raise RuntimeError("reflection exceeded safety iteration limit")


@dataclass(frozen=True)
class EnvironmentStep:
    """Evaluator-side information for one transition.

    ``bounced``, ``bounce_walls`` and ``changed`` are deliberately evaluator
    metadata. The experiment runner passes only ``position`` to a learner.
    """

    step_index: int
    position: float
    bounced: bool
    changed: bool
    bounce_walls: tuple[str, ...] = field(default=())

    @property
    def bounce_count(self) -> int:
        return len(self.bounce_walls)

    @property
    def bounce_wall(self) -> str | None:
        """First wall contacted in this transition, if any."""

        return self.bounce_walls[0] if self.bounce_walls else None


@runtime_checkable
class Environment(Protocol):
    """Structural interface every AAA world implements.

    Declaring this explicitly removes the ``object`` plus ``type: ignore``
    pattern that previously hid interface mistakes from the type checker.
    """

    scenario: str
    seed: int
    config: WorldConfig

    def observe(self) -> float: ...

    def reset(self) -> float: ...

    def advance(self) -> EnvironmentStep: ...


def _validate_change_step(change_step: int | None, steps_per_episode: int) -> int | None:
    """Validate explicit change-event semantics.

    ``None`` means *this episode has no change event*. Any integer must index
    a transition that is actually simulated and scored, so the end-of-horizon
    sentinel that previously slipped through (``change_step == steps``) is
    rejected instead of silently describing an event that never happens.
    """

    if change_step is None:
        return None
    if isinstance(change_step, bool) or not isinstance(change_step, (int, np.integer)):
        raise ValueError("change_step must be an integer or None")
    step = int(change_step)
    if not 0 <= step < steps_per_episode:
        raise ValueError(
            f"change_step must satisfy 0 <= change_step < steps_per_episode ({steps_per_episode}); got {step}"
        )
    return step


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
        if scenario == "dynamics_change":
            raise ValueError("use DampedOscillatorEnvironment for dynamics_change")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.scenario: str = scenario
        self.seed = int(seed)
        self.config = config or WorldConfig()
        for label, value in (("initial_position", initial_position), ("initial_velocity", initial_velocity)):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        self._initial_position = initial_position
        self._initial_velocity = initial_velocity
        self._requested_change_step = change_step
        self._rng = np.random.default_rng(self.seed)
        self._position = 0.0
        self._velocity = 0.0
        self._step_index = 0
        self._change_factor = 1.0
        self._change_step: int | None = None
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
        self._change_step = self._resolve_change_step()
        self._position, self._velocity = self._sample_initial_state()
        self._change_factor = self._sample_change_factor()
        return self.observe()

    def advance(self) -> EnvironmentStep:
        """Advance exactly one step after a prediction has been recorded."""

        changed = self._change_step is not None and self._step_index == self._change_step
        if changed:
            self._velocity *= self._change_factor

        next_position = self._position + self._velocity * self.config.dt
        if self.scenario in ("bouncing", "changed"):
            reflection = resolve_reflection(
                next_position, self._velocity, self.config.lower_bound, self.config.upper_bound
            )
            self._position = reflection.position
            self._velocity = reflection.velocity
            walls = reflection.walls
        else:
            if not self.config.lower_bound <= next_position <= self.config.upper_bound:
                raise RuntimeError("straight scenario crossed a boundary; sampling is invalid")
            self._position = float(next_position)
            walls = ()

        transition = EnvironmentStep(
            step_index=self._step_index,
            position=self._position,
            bounced=bool(walls),
            changed=changed,
            bounce_walls=walls,
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

        width = self.config.upper_bound - self.config.lower_bound
        sample_margin = 0.08 * width
        sample_low = self.config.lower_bound + sample_margin
        sample_high = self.config.upper_bound - sample_margin
        for _ in range(10_000):
            position = float(self._rng.uniform(sample_low, sample_high))
            speed = float(self._rng.uniform(self.config.speed_min, self.config.speed_max))
            velocity = speed if self._rng.integers(0, 2) else -speed
            if self.scenario == "straight":
                final_position = position + velocity * self.config.dt * self.config.steps_per_episode
                if self.config.lower_bound < final_position < self.config.upper_bound:
                    return position, velocity
            elif self.scenario in ("bouncing", "changed"):
                if self.scenario == "changed" and self._change_step is not None:
                    event_position, _, _ = self._simulate_reflection(position, velocity, self._change_step)
                    if not (
                        self.config.lower_bound + self.config.event_margin
                        <= event_position
                        <= self.config.upper_bound - self.config.event_margin
                    ):
                        continue
                return position, velocity
        raise RuntimeError("could not sample a valid initial state")

    def _sample_change_factor(self) -> float:
        if self.scenario != "changed" or self._change_step is None:
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
        return _validate_change_step(step, self.config.steps_per_episode)

    def _simulate_reflection(self, position: float, velocity: float, steps: int) -> tuple[float, float, bool]:
        bounced = False
        for _ in range(steps):
            reflection = resolve_reflection(
                position + velocity * self.config.dt,
                velocity,
                self.config.lower_bound,
                self.config.upper_bound,
            )
            position, velocity = reflection.position, reflection.velocity
            bounced = bounced or reflection.bounced
        return position, velocity, bounced


class DampedOscillatorEnvironment:
    """Bounded second-order world used by the changed-law family.

    The evaluator exposes only ``position`` through the common runner. The
    oscillator is integrated with a semi-implicit Euler step around the
    interval midpoint. At the configured change transition its natural
    frequency and damping coefficient change without a learner notification.
    """

    def __init__(
        self,
        seed: int,
        config: WorldConfig | None = None,
        *,
        omega: float = 1.5,
        damping: float = 0.10,
        changed_omega: float = 8.0,
        changed_damping: float = 0.15,
        initial_position: float | None = None,
        initial_velocity: float | None = None,
        change_step: int | None = None,
        initial_position_low: float = 0.35,
        initial_position_high: float = 0.65,
        initial_velocity_scale: float = 0.25,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.scenario: str = "dynamics_change"
        self.seed = int(seed)
        self.config = config or WorldConfig()
        for name, value in {
            "omega": omega,
            "damping": damping,
            "changed_omega": changed_omega,
            "changed_damping": changed_damping,
        }.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= initial_position_low <= initial_position_high <= 1.0:
            raise ValueError("initial position fractions must satisfy 0 <= low <= high <= 1")
        if not math.isfinite(initial_velocity_scale) or initial_velocity_scale < 0:
            raise ValueError("initial_velocity_scale must be finite and non-negative")
        self.omega = float(omega)
        self.damping = float(damping)
        self.changed_omega = float(changed_omega)
        self.changed_damping = float(changed_damping)
        self.initial_position_low = float(initial_position_low)
        self.initial_position_high = float(initial_position_high)
        self.initial_velocity_scale = float(initial_velocity_scale)
        self._initial_position = initial_position
        self._initial_velocity = initial_velocity
        self._requested_change_step = change_step
        self._rng = np.random.default_rng(self.seed)
        self._position = 0.0
        self._velocity = 0.0
        self._step_index = 0
        self._change_step: int | None = None
        self.reset()

    @property
    def position(self) -> float:
        return self._position

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def change_step(self) -> int | None:
        return self._change_step

    def observe(self) -> float:
        return float(self._position)

    def reset(self) -> float:
        self._rng = np.random.default_rng(self.seed)
        self._step_index = 0
        width = self.config.upper_bound - self.config.lower_bound
        span = self.initial_position_high - self.initial_position_low
        self._position = (
            float(self._initial_position)
            if self._initial_position is not None
            else float(
                self.config.lower_bound + width * (self.initial_position_low + span * self._rng.random())
            )
        )
        self._velocity = (
            float(self._initial_velocity)
            if self._initial_velocity is not None
            else float((self._rng.random() - 0.5) * width * self.initial_velocity_scale)
        )
        if not self.config.lower_bound <= self._position <= self.config.upper_bound:
            raise ValueError("initial_position is outside the configured interval")
        if not np.isfinite(self._velocity):
            raise ValueError("initial_velocity must be finite")
        step = self._requested_change_step
        if step is None:
            step = self.config.change_step
        self._change_step = _validate_change_step(step, self.config.steps_per_episode)
        return self.observe()

    def advance(self) -> EnvironmentStep:
        after_change = self._change_step is not None and self._step_index >= self._change_step
        changed = self._change_step is not None and self._step_index == self._change_step
        omega = self.changed_omega if after_change else self.omega
        damping = self.changed_damping if after_change else self.damping
        midpoint = (self.config.lower_bound + self.config.upper_bound) / 2
        acceleration = -2.0 * damping * omega * self._velocity - omega * omega * (self._position - midpoint)
        self._velocity += acceleration * self.config.dt
        reflection = resolve_reflection(
            self._position + self._velocity * self.config.dt,
            self._velocity,
            self.config.lower_bound,
            self.config.upper_bound,
        )
        self._position = reflection.position
        self._velocity = reflection.velocity
        transition = EnvironmentStep(
            step_index=self._step_index,
            position=self._position,
            bounced=reflection.bounced,
            changed=changed,
            bounce_walls=reflection.walls,
        )
        self._step_index += 1
        return transition
