"""AAA-1K observation streams: what the evaluator knows, and what the agent sees.

A stream is a list of :class:`StreamStep`. Each step carries the latent true
position, which belongs to the evaluator alone, and an ``observed`` flag saying
whether the agent is given that position. Regime and event labels are evaluator
metadata; :mod:`research.aaa_1k.runner` never passes them to an agent, and the
test suite fails if one does.

Four families
-------------
``motion_compat``
    The existing AAA worlds, unchanged and fully observed: constant velocity,
    reflecting motion, an unannounced speed change, and the damped oscillator
    whose coefficients change mid-episode. This is the compatibility family; it
    exists so AAA-1K is measured on the benchmark the project already has.

``occlusion_v1``
    Reflecting motion with periodic observation gaps. During a gap the agent
    receives nothing and the feature adapter holds the last known position, so
    its displacement and error inputs are exactly zero. Anything the agent knows
    about motion across a gap has to be in its recurrent state. This is the
    family that can actually distinguish a model with memory from one without.

``coarse_speed_v1``
    Reflecting motion whose speed switches between two hidden values at
    unannounced times, observed through a coarse deterministic quantizer whose
    step is comparable to one step of motion. A single transition therefore
    does not identify the current speed; integrating several of them does. The
    hidden speed, the switch times and the exact position are never revealed.
    This is a memory task, not an observation-noise study: the quantizer is
    deterministic and public, and `aaa.observation_noise` is a separate phase.

``aba_v1``
    One long unlabelled stream whose law goes A (reflecting constant velocity)
    to B (damped oscillation) and back to A. Transitions are unannounced. This
    is the continual-adaptation probe.

``paired_change_v1``
    Two streams that are *bit-identical* up to a declared step, after which one
    changes its speed and the other does not. This family exists because the
    single-stream online-versus-frozen comparison could not distinguish
    "continued learning helps after a change" from "continued learning helps",
    and a probe showed the difference was 5%. Running both members of a pair
    from the same model and differencing removes everything the two have in
    common, including the ordinary benefit of continuing to learn.

Every family is fully deterministic given its seed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaa.config import WorldConfig
from aaa.environment import DampedOscillatorEnvironment, MovingDotEnvironment, resolve_reflection

BENCHMARK_VERSION = "aaa.1k.benchmarks.v1"

FAMILIES = ("motion_compat", "occlusion_v1", "coarse_speed_v1", "aba_v1", "paired_change_v1")
COMPAT_SCENARIOS = ("straight", "bouncing", "changed", "dynamics_change")


@dataclass(frozen=True)
class StreamStep:
    """One simulated step.

    ``true_position`` is latent truth. It is used for scoring and never passed
    to an agent. ``observed`` decides whether the agent is given the value.
    ``regime`` and ``event`` are evaluator labels for reporting only.
    """

    index: int
    true_position: float
    observed: bool
    regime: str
    event: str | None = None


@dataclass(frozen=True)
class Stream:
    """A complete realized trajectory plus its evaluator-only description."""

    family: str
    stream_id: str
    seed: int
    steps: tuple[StreamStep, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        if len(self.steps) < 3:
            raise ValueError("a stream needs at least three steps")
        if not self.steps[0].observed:
            raise ValueError("the seed observation must always be given to the agent")
        if any(not math.isfinite(step.true_position) for step in self.steps):
            raise ValueError("stream contains a non-finite position")

    @property
    def scored_steps(self) -> int:
        """Transitions that are scored: every step after the seed."""

        return len(self.steps) - 1

    def observed_fraction(self) -> float:
        return float(np.mean([step.observed for step in self.steps]))

    def regime_at(self, index: int) -> str:
        return self.steps[index].regime

    def to_summary(self) -> dict[str, Any]:
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "family": self.family,
            "stream_id": self.stream_id,
            "seed": self.seed,
            "steps": len(self.steps),
            "scored_steps": self.scored_steps,
            "observed_fraction": self.observed_fraction(),
            "metadata": dict(self.metadata),
        }


# ----------------------------------------------------------------------
# compatibility family: the existing AAA worlds, untouched
# ----------------------------------------------------------------------
def motion_compat_stream(
    scenario: str,
    seed: int,
    *,
    steps: int = 120,
    change_step: int | None = 60,
    config: WorldConfig | None = None,
) -> Stream:
    """Drive an existing AAA environment and record its realized trajectory."""

    if scenario not in COMPAT_SCENARIOS:
        raise ValueError(f"unknown compatibility scenario {scenario!r}")
    world = config or WorldConfig(steps_per_episode=steps, change_step=change_step)
    if scenario == "dynamics_change":
        environment: Any = DampedOscillatorEnvironment(seed=seed, config=world)
    else:
        environment = MovingDotEnvironment(
            scenario,  # type: ignore[arg-type]
            seed=seed,
            config=world,
            change_step=change_step if scenario == "changed" else None,
        )
    start = environment.reset()
    records = [StreamStep(index=0, true_position=float(start), observed=True, regime=scenario)]
    for index in range(world.steps_per_episode):
        transition = environment.advance()
        records.append(
            StreamStep(
                index=index + 1,
                true_position=float(transition.position),
                observed=True,
                regime=scenario,
                event="change" if transition.changed else ("bounce" if transition.bounced else None),
            )
        )
    return Stream(
        family="motion_compat",
        stream_id=f"motion_compat:{scenario}:{seed}",
        seed=int(seed),
        steps=tuple(records),
        metadata={
            "scenario": scenario,
            "change_step": change_step,
            "steps_per_episode": world.steps_per_episode,
        },
    )


# ----------------------------------------------------------------------
# shared reflecting-motion integrator for the new families
# ----------------------------------------------------------------------
def _reflect_advance(
    position: float, velocity: float, dt: float, lower: float, upper: float
) -> tuple[float, float, bool]:
    result = resolve_reflection(position + velocity * dt, velocity, lower, upper)
    return result.position, result.velocity, result.bounced


# ----------------------------------------------------------------------
# occlusion_v1
# ----------------------------------------------------------------------
def occlusion_stream(
    seed: int,
    *,
    steps: int = 200,
    gap_period: int = 20,
    gap_length: int = 4,
    warmup: int = 12,
    config: WorldConfig | None = None,
) -> Stream:
    """Reflecting motion with deterministic, periodic observation gaps.

    The gap schedule is generated from the stream seed and is fixed before the
    run. Gaps never hide the seed observation and never start inside the
    warm-up prefix, so every agent begins on the same visible window.
    """

    world = config or WorldConfig(steps_per_episode=steps)
    if gap_length < 1 or gap_period <= gap_length:
        raise ValueError("gap_period must exceed gap_length, and gap_length must be positive")
    if warmup < 2:
        raise ValueError("warmup must leave at least two visible steps")
    rng = np.random.default_rng(seed)
    speed = float(rng.uniform(world.speed_min, world.speed_max))
    velocity = speed if rng.integers(0, 2) else -speed
    position = float(rng.uniform(world.lower_bound + 0.15, world.upper_bound - 0.15))

    hidden: set[int] = set()
    cursor = warmup + int(rng.integers(0, gap_period - gap_length))
    while cursor + gap_length <= steps:
        hidden.update(range(cursor, cursor + gap_length))
        cursor += gap_period

    records = [StreamStep(index=0, true_position=position, observed=True, regime="reflecting")]
    for index in range(steps):
        position, velocity, bounced = _reflect_advance(
            position, velocity, world.dt, world.lower_bound, world.upper_bound
        )
        step_index = index + 1
        records.append(
            StreamStep(
                index=step_index,
                true_position=position,
                observed=step_index not in hidden,
                regime="reflecting",
                event="bounce" if bounced else None,
            )
        )
    return Stream(
        family="occlusion_v1",
        stream_id=f"occlusion_v1:{seed}",
        seed=int(seed),
        steps=tuple(records),
        metadata={
            "gap_period": gap_period,
            "gap_length": gap_length,
            "warmup": warmup,
            "hidden_steps": sorted(hidden),
            "initial_speed": abs(velocity),
        },
    )


# ----------------------------------------------------------------------
# coarse_speed_v1
# ----------------------------------------------------------------------
def quantize(value: float, quantum: float) -> float:
    """Deterministic public quantizer: round to the nearest multiple of ``quantum``."""

    if quantum <= 0:
        raise ValueError("quantum must be positive")
    return float(round(float(value) / quantum) * quantum)


def coarse_speed_stream(
    seed: int,
    *,
    steps: int = 240,
    quantum: float = 0.005,
    regime_length: int = 60,
    slow_speed: float = 0.12,
    fast_speed: float = 0.30,
    config: WorldConfig | None = None,
) -> Stream:
    """Reflecting motion with a hidden speed regime, seen through a coarse grid.

    One step of motion moves the dot by between ``slow_speed * dt`` and
    ``fast_speed * dt`` -- here 0.0024 to 0.0060 -- against a quantum of 0.005.
    A single observed transition therefore pins the speed down very poorly; a
    model that integrates several of them can do much better. Every step is
    observed, so every step is a legitimate training target, and the thing that
    is hidden is the *precision* of the observation rather than its presence.

    The quantizer is deterministic, public and applied identically to every
    arm. The agent is scored on predicting the next *quantized* observation --
    the thing that is actually revealed -- while the continuous position and
    velocity that generate the staircase stay latent. That is what makes this a
    hidden-state problem rather than an irreducible-noise floor.
    """

    world = config or WorldConfig(steps_per_episode=steps)
    if regime_length < 2:
        raise ValueError("regime_length must be at least 2")
    if not (0 < slow_speed < fast_speed):
        raise ValueError("speeds must satisfy 0 < slow < fast")
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    rng = np.random.default_rng(seed)
    fast = bool(rng.integers(0, 2))
    position = float(rng.uniform(world.lower_bound + 0.2, world.upper_bound - 0.2))
    velocity = (1.0 if rng.integers(0, 2) else -1.0) * (fast_speed if fast else slow_speed)

    records = [
        StreamStep(
            index=0,
            true_position=quantize(position, quantum),
            observed=True,
            regime="fast" if fast else "slow",
        )
    ]
    for index in range(steps):
        step_index = index + 1
        event = None
        if step_index % regime_length == 0:
            fast = not fast
            velocity = math.copysign(fast_speed if fast else slow_speed, velocity)
            event = "regime_switch"
        position, velocity, bounced = _reflect_advance(
            position, velocity, world.dt, world.lower_bound, world.upper_bound
        )
        records.append(
            StreamStep(
                index=step_index,
                true_position=quantize(position, quantum),
                observed=True,
                regime="fast" if fast else "slow",
                event=event or ("bounce" if bounced else None),
            )
        )
    return Stream(
        family="coarse_speed_v1",
        stream_id=f"coarse_speed_v1:{seed}",
        seed=int(seed),
        steps=tuple(records),
        metadata={
            "quantum": quantum,
            "regime_length": regime_length,
            "slow_speed": slow_speed,
            "fast_speed": fast_speed,
            "scoring": "the quantized observable; the latent continuous position is never a target",
        },
    )


# ----------------------------------------------------------------------
# aba_v1
# ----------------------------------------------------------------------
def aba_stream(
    seed: int,
    *,
    segment_steps: int = 400,
    omega: float = 6.0,
    damping: float = 0.08,
    config: WorldConfig | None = None,
) -> Stream:
    """One unlabelled stream whose law runs A, then B, then A again.

    Regime A is reflecting constant velocity. Regime B is a damped harmonic
    oscillator about the interval midpoint. The agent is never told that a
    transition happened, and the velocity carries across the boundary, so the
    only signal is that predictions suddenly stop working.
    """

    world = config or WorldConfig(steps_per_episode=3 * segment_steps)
    if segment_steps < 20:
        raise ValueError("segment_steps must be at least 20")
    rng = np.random.default_rng(seed)
    speed = float(rng.uniform(world.speed_min, world.speed_max))
    velocity = speed if rng.integers(0, 2) else -speed
    position = float(rng.uniform(world.lower_bound + 0.2, world.upper_bound - 0.2))
    midpoint = (world.lower_bound + world.upper_bound) / 2.0
    boundaries = (segment_steps, 2 * segment_steps)

    records = [StreamStep(index=0, true_position=position, observed=True, regime="A1")]
    for index in range(3 * segment_steps):
        step_index = index + 1
        if index < boundaries[0]:
            regime, label = "A", "A1"
        elif index < boundaries[1]:
            regime, label = "B", "B"
        else:
            regime, label = "A", "A2"
        if regime == "B":
            acceleration = -2.0 * damping * omega * velocity - omega * omega * (position - midpoint)
            velocity += acceleration * world.dt
        position, velocity, bounced = _reflect_advance(
            position, velocity, world.dt, world.lower_bound, world.upper_bound
        )
        event = None
        if index in boundaries:
            event = "regime_change"
        elif bounced:
            event = "bounce"
        records.append(
            StreamStep(index=step_index, true_position=position, observed=True, regime=label, event=event)
        )
    return Stream(
        family="aba_v1",
        stream_id=f"aba_v1:{seed}",
        seed=int(seed),
        steps=tuple(records),
        metadata={
            "segment_steps": segment_steps,
            "omega": omega,
            "damping": damping,
            "change_steps": [boundaries[0] + 1, boundaries[1] + 1],
        },
    )


def paired_change_streams(
    seed: int,
    *,
    steps: int = 240,
    change_step: int = 120,
    config: WorldConfig | None = None,
) -> tuple[Stream, Stream]:
    """Return ``(changed, control)``: identical until ``change_step``, then not.

    Both members share one initial state and one velocity, so every scored
    transition before ``change_step`` is bit-identical and a model driven
    through either prefix arrives at exactly the same state. After that step
    the changed member's speed is multiplied by a factor drawn from the public
    configuration and the control member carries on unchanged.

    This is what makes a difference-of-differences possible: the online-minus-
    frozen advantage measured on the control is everything that is *not* about
    the change, and subtracting it leaves the part that is.
    """

    world = config or WorldConfig(steps_per_episode=steps)
    if not 0 < change_step < steps - 1:
        raise ValueError("change_step must fall strictly inside the stream")
    rng = np.random.default_rng(seed)
    speed = float(rng.uniform(world.speed_min, world.speed_max))
    velocity0 = speed if rng.integers(0, 2) else -speed
    position0 = float(rng.uniform(world.lower_bound + 0.2, world.upper_bound - 0.2))
    factor = world.change_factor_low if rng.integers(0, 2) == 0 else world.change_factor_high

    def integrate(apply_change: bool) -> tuple[StreamStep, ...]:
        position, velocity = position0, velocity0
        records = [StreamStep(index=0, true_position=position, observed=True, regime="pre")]
        for index in range(steps):
            step_index = index + 1
            event = None
            if index == change_step and apply_change:
                velocity *= factor
                event = "change"
            position, velocity, bounced = _reflect_advance(
                position, velocity, world.dt, world.lower_bound, world.upper_bound
            )
            records.append(
                StreamStep(
                    index=step_index,
                    true_position=position,
                    observed=True,
                    regime="pre" if index < change_step else "post",
                    event=event or ("bounce" if bounced else None),
                )
            )
        return tuple(records)

    metadata = {
        "change_step": change_step,
        "change_factor": factor,
        "steps": steps,
        "pairing": "changed and control share one initial state; prefixes are bit-identical",
    }
    changed = Stream(
        family="paired_change_v1",
        stream_id=f"paired_change_v1:{seed}:changed",
        seed=int(seed),
        steps=integrate(True),
        metadata={**metadata, "variant": "changed"},
    )
    control = Stream(
        family="paired_change_v1",
        stream_id=f"paired_change_v1:{seed}:control",
        seed=int(seed),
        steps=integrate(False),
        metadata={**metadata, "variant": "control", "change_factor": 1.0},
    )
    return changed, control


def build_stream(family: str, seed: int, **options: Any) -> Stream:
    """Dispatch to a family generator by name."""

    if family == "motion_compat":
        scenario = options.pop("scenario", "bouncing")
        return motion_compat_stream(scenario, seed, **options)
    if family == "occlusion_v1":
        return occlusion_stream(seed, **options)
    if family == "coarse_speed_v1":
        return coarse_speed_stream(seed, **options)
    if family == "aba_v1":
        return aba_stream(seed, **options)
    if family == "paired_change_v1":
        variant = options.pop("variant", "changed")
        changed, control = paired_change_streams(seed, **options)
        return changed if variant == "changed" else control
    raise ValueError(f"unknown family {family!r}")


def observable_view(stream: Stream) -> list[float | None]:
    """Exactly what any agent may be given, in order. Nothing else is permitted."""

    return [step.true_position if step.observed else None for step in stream.steps]


def latent_positions(stream: Stream) -> list[float]:
    """Evaluator-only truth, for scoring."""

    return [step.true_position for step in stream.steps]


def regime_labels(stream: Stream) -> Sequence[str]:
    """Evaluator-only regime labels, for reporting."""

    return [step.regime for step in stream.steps]
