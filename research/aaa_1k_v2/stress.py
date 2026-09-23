"""``aaa.1k.stress.v1``: the final-1K stress suite for the moving dot.

Separately versioned from round 3's ``aaa.1k.benchmarks.v1``, which remains
the regression/reference suite and is regenerated here with this phase's own
seeds through the unmodified historical generators.

Every family is deterministic given its seed. Latent truth, the observed mask
and evaluator labels are evaluator data; only the observed values reach a cell.
The public observation format is AAA-1K's: interval ``[0, 1]``, step
``dt = 0.02``, displacement scale ``0.004``. Some families deliberately break
the assumptions behind the public boundary map (soft and inelastic walls,
gravity): the agents still apply the map, because it is public knowledge of
the *usual* format, and the families measure what that costs.

Roles (declared before any v2 evidence existed; see
``docs/aaa_1k_v2_benchmark_protocol.md``):

``development``  may inform selection;
``attack``       appears first in the attack stage;
``held_out``     appears first in confirmation (out-of-family generalization).

Families
--------
Regression (historical generators, fresh seeds): ``v1_motion_straight``,
``v1_motion_bouncing``, ``v1_motion_changed``, ``v1_motion_dynamics``,
``v1_occlusion``, ``v1_coarse_speed``, ``v1_aba``.

Stress: ``long_bouncing``, ``long_coarse``, ``random_gaps``, ``wall_gaps``,
``long_gap_recall``, ``long_gap_recall_extended``, ``stationary_mixed``,
``noise``, ``noise_gaps``, ``change_gaps``, ``change_noise``,
``accel_switch``, ``quantized``, ``coarse_near``, ``oscillator_long``,
``gravity_bounce``, ``soft_wall``, ``inelastic_wall``, ``abcab``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaa.environment import resolve_reflection
from research.aaa_1k.streams import build_stream

from .runner import StreamBatch

SUITE_VERSION = "aaa.1k.stress.v1"
DT = 0.02
LOWER, UPPER = 0.0, 1.0


@dataclass(frozen=True)
class Realization:
    truth: np.ndarray
    observed: np.ndarray
    labels: dict[str, list[str]]
    metadata: dict[str, Any]


def _quarters(n: int) -> list[str]:
    return [f"q{min(4, 1 + (4 * i) // n)}" for i in range(n)]


def _hard(position: float, velocity: float) -> tuple[float, float, bool]:
    result = resolve_reflection(position, velocity, LOWER, UPPER)
    return result.position, result.velocity, result.bounced


def _integrate(
    rng: np.random.Generator,
    steps: int,
    *,
    speed: float | None = None,
    acceleration: Callable[[int, float, float], float] | None = None,
    wall: str = "hard",
    restitution: float = 1.0,
    start: tuple[float, float] | None = None,
) -> tuple[np.ndarray, list[bool]]:
    """Semi-implicit Euler with a declared wall law. Returns positions (steps+1) and bounce flags."""

    if start is None:
        s = float(rng.uniform(0.12, 0.32)) if speed is None else float(speed)
        position = float(rng.uniform(0.2, 0.8))
        velocity = s if rng.integers(0, 2) else -s
    else:
        position, velocity = start
    positions = [position]
    bounces = [False]
    for index in range(steps):
        if acceleration is not None:
            velocity += acceleration(index, position, velocity) * DT
        new = position + velocity * DT
        bounced = False
        if wall == "hard":
            new, velocity, bounced = _hard(new, velocity)
        elif wall == "inelastic":
            new, reflected, bounced = _hard(new, velocity)
            velocity = reflected * (restitution if bounced else 1.0)
        elif wall == "soft" and not LOWER <= new <= UPPER:
            raise RuntimeError("soft-wall trajectory left the interval; the spring constant is too weak")
        position = new
        positions.append(position)
        bounces.append(bounced)
    return np.asarray(positions, dtype=float), bounces


def _geometric_gaps(
    rng: np.random.Generator, steps: int, *, warmup: int, mean_interval: float, lengths: tuple[int, int]
) -> np.ndarray:
    observed = np.ones(steps + 1, dtype=bool)
    cursor = warmup + int(rng.geometric(1.0 / mean_interval))
    while cursor < steps:
        length = int(rng.integers(lengths[0], lengths[1] + 1))
        observed[cursor : min(steps + 1, cursor + length)] = False
        cursor += length + int(rng.geometric(1.0 / mean_interval))
    observed[: warmup + 1] = True
    return observed


# ----------------------------------------------------------------------
# historical regression families (unchanged generators)
# ----------------------------------------------------------------------
def _historical(family: str, options: dict[str, Any]) -> Callable[[int], Realization]:
    def generate(seed: int) -> Realization:
        stream = build_stream(family, seed, **dict(options))
        truth = np.asarray([step.true_position for step in stream.steps])
        observed = np.asarray([step.observed for step in stream.steps])
        regimes = [str(step.regime) for step in stream.steps]
        return Realization(
            truth,
            observed,
            {"condition": regimes, "quarter": _quarters(len(truth))},
            {"generator": f"research.aaa_1k.streams.build_stream({family!r}, seed, **{options})"},
        )

    return generate


# ----------------------------------------------------------------------
# stress families
# ----------------------------------------------------------------------
def long_bouncing(seed: int, steps: int = 2000) -> Realization:
    rng = np.random.default_rng(seed)
    truth, _ = _integrate(rng, steps)
    return Realization(truth, np.ones(steps + 1, bool), {"quarter": _quarters(steps + 1)}, {"steps": steps})


def _coarse(
    seed: int, steps: int, quantum: float, slow: float, fast: float, regime_length: int
) -> Realization:
    stream = build_stream(
        "coarse_speed_v1",
        seed,
        steps=steps,
        quantum=quantum,
        slow_speed=slow,
        fast_speed=fast,
        regime_length=regime_length,
    )
    truth = np.asarray([s.true_position for s in stream.steps])
    regimes = [s.regime for s in stream.steps]
    return Realization(
        truth,
        np.ones(len(truth), bool),
        {"condition": regimes, "quarter": _quarters(len(truth))},
        {"quantum": quantum, "slow": slow, "fast": fast, "regime_length": regime_length},
    )


def long_coarse(seed: int, steps: int = 2000) -> Realization:
    return _coarse(seed, steps, 0.005, 0.12, 0.30, 60)


def random_gaps(seed: int, steps: int = 800) -> Realization:
    rng = np.random.default_rng(seed)
    truth, _ = _integrate(rng, steps)
    observed = _geometric_gaps(rng, steps, warmup=12, mean_interval=15.0, lengths=(1, 12))
    return Realization(truth, observed, {"quarter": _quarters(steps + 1)}, {"gap_lengths": [1, 12]})


def wall_gaps(seed: int, steps: int = 1600) -> Realization:
    """Faster motion (speed 0.3-0.6) so walls are met often; gaps cover wall contacts."""

    rng = np.random.default_rng(seed)
    truth, bounces = _integrate(rng, steps, speed=float(rng.uniform(0.3, 0.6)))
    observed = np.ones(steps + 1, dtype=bool)
    near = np.zeros(steps + 1, dtype=bool)
    for index in np.nonzero(bounces)[0]:
        near[max(0, index - 3) : index + 4] = True
        if rng.uniform() < 0.8:
            before, after = int(rng.integers(0, 9)), int(rng.integers(0, 9))
            observed[max(13, index - before) : min(steps + 1, index + after + 1)] = False
    observed[:13] = True
    return Realization(
        truth,
        observed,
        {"condition": ["near_wall" if flag else "open" for flag in near], "quarter": _quarters(steps + 1)},
        {"hide_probability": 0.8, "speed": [0.3, 0.6], "window": [0, 8]},
    )


def _recall(seed: int, lengths: Sequence[int], steps: int = 1000, window: int = 20) -> Realization:
    rng = np.random.default_rng(seed)
    truth, _ = _integrate(rng, steps)
    observed = np.ones(steps + 1, dtype=bool)
    position = np.full(steps + 1, "visible", dtype=object)
    cursor = 50
    while cursor < steps:
        length = int(rng.choice(np.asarray(lengths)))
        end = min(steps + 1, cursor + length)
        observed[cursor:end] = False
        span = max(1, end - cursor)
        for offset in range(end - cursor):
            third = (3 * offset) // span
            position[cursor + offset] = ("gap_early", "gap_middle", "gap_late")[third]
        cursor = end + window
    return Realization(
        truth,
        observed,
        {"condition": [str(p) for p in position], "quarter": _quarters(steps + 1)},
        {"gap_lengths": list(lengths), "visible_window": window},
    )


def long_gap_recall(seed: int) -> Realization:
    return _recall(seed, (16, 32, 64))


def long_gap_recall_extended(seed: int) -> Realization:
    return _recall(seed, (96, 128), steps=1400)


def stationary_mixed(seed: int, steps: int = 800) -> Realization:
    rng = np.random.default_rng(seed)
    position = float(rng.uniform(0.2, 0.8))
    velocity = 0.0
    truth = [position]
    phase = []
    moving = False
    remaining = int(rng.integers(20, 61))
    for _ in range(steps):
        if remaining == 0:
            moving = not moving
            remaining = int(rng.integers(20, 61))
            speed = float(rng.uniform(0.12, 0.32))
            velocity = (speed if rng.integers(0, 2) else -speed) if moving else 0.0
        remaining -= 1
        new, velocity, _ = _hard(position + velocity * DT, velocity)
        position = new
        truth.append(position)
        phase.append("moving" if moving else "stationary")
    observed = _geometric_gaps(rng, steps, warmup=12, mean_interval=20.0, lengths=(1, 8))
    return Realization(
        np.asarray(truth),
        observed,
        {"condition": ["stationary", *phase], "quarter": _quarters(steps + 1)},
        {"note": "genuine zero displacement alternates with motion; gaps hide both"},
    )


def _noisy(truth: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noisy = truth + rng.normal(0.0, sigma, size=truth.shape)
    return np.clip(noisy, LOWER, UPPER)


def noise(seed: int, steps: int = 600, sigma: float = 0.002) -> Realization:
    """Observation noise; the cell sees noisy positions and is scored against latent truth."""

    rng = np.random.default_rng(seed)
    truth, _ = _integrate(rng, steps)
    return Realization(
        truth,
        np.ones(steps + 1, bool),
        {"quarter": _quarters(steps + 1)},
        {"sigma": sigma, "observation": _noisy(truth, rng, sigma)},
    )


def noise_gaps(seed: int, steps: int = 600, sigma: float = 0.002) -> Realization:
    rng = np.random.default_rng(seed)
    truth, _ = _integrate(rng, steps)
    observed = _geometric_gaps(rng, steps, warmup=12, mean_interval=15.0, lengths=(1, 10))
    return Realization(
        truth,
        observed,
        {"quarter": _quarters(steps + 1)},
        {"sigma": sigma, "observation": _noisy(truth, rng, sigma)},
    )


def _speed_changes(rng: np.random.Generator, steps: int) -> tuple[np.ndarray, list[str]]:
    speed = float(rng.uniform(0.12, 0.32))
    position = float(rng.uniform(0.2, 0.8))
    velocity = speed if rng.integers(0, 2) else -speed
    truth, labels = [position], ["pre"]
    next_change = int(rng.integers(100, 201))
    since = 0
    for index in range(steps):
        if index == next_change:
            factor = 0.5 if rng.integers(0, 2) == 0 else 1.8
            velocity = math.copysign(min(0.45, max(0.06, abs(velocity) * factor)), velocity)
            next_change = index + int(rng.integers(100, 201))
            since = 0
        position, velocity, _ = _hard(position + velocity * DT, velocity)
        truth.append(position)
        labels.append("post_change_30" if since < 30 and index >= 100 else "steady")
        since += 1
    return np.asarray(truth), labels


def change_gaps(seed: int, steps: int = 800) -> Realization:
    rng = np.random.default_rng(seed)
    truth, labels = _speed_changes(rng, steps)
    observed = _geometric_gaps(rng, steps, warmup=12, mean_interval=15.0, lengths=(1, 10))
    return Realization(truth, observed, {"condition": labels, "quarter": _quarters(steps + 1)}, {})


def change_noise(seed: int, steps: int = 800, sigma: float = 0.002) -> Realization:
    rng = np.random.default_rng(seed)
    truth, labels = _speed_changes(rng, steps)
    return Realization(
        truth,
        np.ones(steps + 1, bool),
        {"condition": labels, "quarter": _quarters(steps + 1)},
        {"sigma": sigma, "observation": _noisy(truth, rng, sigma)},
    )


def accel_switch(seed: int, steps: int = 800) -> Realization:
    rng = np.random.default_rng(seed)
    schedule: list[float] = []
    value = float(rng.choice([-0.5, -0.2, 0.2, 0.5]))
    while len(schedule) < steps:
        schedule.extend([value] * int(rng.integers(100, 201)))
        value = float(rng.choice([-0.5, -0.2, 0.2, 0.5]))
    truth, _ = _integrate(rng, steps, acceleration=lambda i, _p, v: schedule[i] - 0.5 * v)
    return Realization(
        truth,
        np.ones(steps + 1, bool),
        {"condition": ["accelerating"] * (steps + 1), "quarter": _quarters(steps + 1)},
        {"accelerations": [-0.5, -0.2, 0.2, 0.5], "linear_drag": 0.5},
    )


def quantized(seed: int, steps: int = 600) -> Realization:
    rng = np.random.default_rng(seed)
    quantum = float(rng.choice([0.0025, 0.004, 0.006, 0.008]))
    speed = float(rng.uniform(0.08, 0.30))
    truth, _ = _integrate(rng, steps, speed=speed)
    observation = np.round(truth / quantum) * quantum
    return Realization(
        observation,
        np.ones(steps + 1, bool),
        {"condition": [f"quantum_{quantum:g}"] * (steps + 1), "quarter": _quarters(steps + 1)},
        {"quantum": quantum, "speed": speed, "scoring": "quantized observable, as in coarse_speed_v1"},
    )


def coarse_near(seed: int, steps: int = 600) -> Realization:
    rng = np.random.default_rng(seed)
    slow = float(rng.choice([0.08, 0.10, 0.15]))
    fast = float(rng.choice([0.25, 0.35]))
    quantum = float(rng.choice([0.004, 0.006]))
    return _coarse(int(rng.integers(0, 2**62)), steps, quantum, slow, fast, 60)


def oscillator_long(seed: int, steps: int = 1200) -> Realization:
    rng = np.random.default_rng(seed)
    midpoint = 0.5
    omega, damping = float(rng.uniform(1.5, 8.0)), float(rng.uniform(0.02, 0.15))
    position = float(rng.uniform(0.25, 0.75))
    velocity = float(rng.uniform(-0.25, 0.25))
    truth, labels = [position], ["segment_1"]
    segment = 1
    for index in range(steps):
        if index > 0 and index % 300 == 0:
            omega, damping = float(rng.uniform(1.5, 8.0)), float(rng.uniform(0.02, 0.15))
            velocity += float(rng.uniform(-0.2, 0.2))
            segment += 1
        acceleration = -2.0 * damping * omega * velocity - omega * omega * (position - midpoint)
        velocity += acceleration * DT
        position, velocity, _ = _hard(position + velocity * DT, velocity)
        truth.append(position)
        labels.append(f"segment_{segment}")
    return Realization(
        np.asarray(truth), np.ones(steps + 1, bool), {"condition": labels}, {"change_every": 300}
    )


def gravity_bounce(seed: int, steps: int = 800) -> Realization:
    rng = np.random.default_rng(seed)
    g = float(rng.uniform(0.2, 1.0))
    truth, bounces = _integrate(rng, steps, acceleration=lambda _i, _p, _v: -g)
    near = np.zeros(steps + 1, dtype=bool)
    for index in np.nonzero(bounces)[0]:
        near[max(0, index - 3) : index + 4] = True
    return Realization(
        truth,
        np.ones(steps + 1, bool),
        {"condition": ["near_wall" if f else "open" for f in near], "quarter": _quarters(steps + 1)},
        {"gravity": g},
    )


def soft_wall(seed: int, steps: int = 800, stiffness: float = 400.0, margin: float = 0.1) -> Realization:
    """Spring walls inside the interval: acceleration near the boundary, no hard reflection."""

    rng = np.random.default_rng(seed)

    def spring(_i: int, p: float, _v: float) -> float:
        if p < LOWER + margin:
            return stiffness * (LOWER + margin - p)
        if p > UPPER - margin:
            return -stiffness * (p - (UPPER - margin))
        return 0.0

    speed = float(rng.uniform(0.12, 0.32))
    truth, _ = _integrate(
        rng,
        steps,
        acceleration=spring,
        wall="soft",
        start=(float(rng.uniform(0.3, 0.7)), speed if rng.integers(0, 2) else -speed),
    )
    in_spring = (truth < LOWER + margin) | (truth > UPPER - margin)
    return Realization(
        truth,
        np.ones(steps + 1, bool),
        {"condition": ["spring" if f else "free" for f in in_spring], "quarter": _quarters(steps + 1)},
        {"stiffness": stiffness, "margin": margin},
    )


def inelastic_wall(seed: int, steps: int = 800) -> Realization:
    rng = np.random.default_rng(seed)
    restitution = float(rng.uniform(0.6, 0.9))
    kicks = set(range(0, steps, 200))

    truth, _ = _integrate(
        rng,
        steps,
        acceleration=lambda i, _p, v: (
            (math.copysign(float(np.random.default_rng([seed, i]).uniform(0.12, 0.32)), v or 1.0) - v) / DT
            if i in kicks and i > 0
            else 0.0
        ),
        wall="inelastic",
        restitution=restitution,
    )
    return Realization(
        truth,
        np.ones(steps + 1, bool),
        {"quarter": _quarters(steps + 1)},
        {"restitution": restitution, "speed_reset_every": 200},
    )


def abcab(seed: int, segment: int = 250) -> Realization:
    rng = np.random.default_rng(seed)
    g = float(rng.uniform(0.3, 0.8))
    omega, damping = float(rng.uniform(4.0, 8.0)), float(rng.uniform(0.03, 0.1))
    order = ["A", "B", "C", "A", "B"]
    speed = float(rng.uniform(0.12, 0.32))
    position = float(rng.uniform(0.2, 0.8))
    velocity = speed if rng.integers(0, 2) else -speed
    truth, labels = [position], ["A"]
    for index in range(segment * len(order)):
        regime = order[index // segment]
        if regime == "B":
            velocity += (-2.0 * damping * omega * velocity - omega * omega * (position - 0.5)) * DT
        elif regime == "C":
            velocity += -g * DT
        position, velocity, _ = _hard(position + velocity * DT, velocity)
        truth.append(position)
        labels.append(regime)
    return Realization(
        np.asarray(truth),
        np.ones(len(truth), bool),
        {"condition": labels},
        {"order": order, "segment": segment, "gravity": g, "omega": omega, "damping": damping},
    )


@dataclass(frozen=True)
class Family:
    name: str
    role: str
    generate: Callable[[int], Realization]
    description: str


FAMILIES: dict[str, Family] = {
    family.name: family
    for family in (
        Family(
            "v1_motion_straight",
            "development",
            _historical("motion_compat", {"scenario": "straight", "steps": 160, "change_step": None}),
            "round-3 compatibility: straight",
        ),
        Family(
            "v1_motion_bouncing",
            "development",
            _historical("motion_compat", {"scenario": "bouncing", "steps": 160, "change_step": None}),
            "round-3 compatibility: bouncing",
        ),
        Family(
            "v1_motion_changed",
            "development",
            _historical("motion_compat", {"scenario": "changed", "steps": 160, "change_step": 80}),
            "round-3 compatibility: speed change",
        ),
        Family(
            "v1_motion_dynamics",
            "development",
            _historical("motion_compat", {"scenario": "dynamics_change", "steps": 160, "change_step": 80}),
            "round-3 compatibility: oscillator change",
        ),
        Family(
            "v1_occlusion", "development", _historical("occlusion_v1", {"steps": 200}), "round-3 occlusion_v1"
        ),
        Family(
            "v1_coarse_speed",
            "development",
            _historical("coarse_speed_v1", {"steps": 240}),
            "round-3 coarse_speed_v1",
        ),
        Family("v1_aba", "development", _historical("aba_v1", {"segment_steps": 120}), "round-3 aba_v1"),
        Family("long_bouncing", "development", long_bouncing, "2000-step reflecting motion"),
        Family(
            "long_coarse",
            "development",
            long_coarse,
            "2000-step coarse_speed construction (the M2 condition)",
        ),
        Family("random_gaps", "development", random_gaps, "random (geometric) gap schedule, lengths 1-12"),
        Family(
            "wall_gaps",
            "development",
            wall_gaps,
            "fast motion; gaps placed before, during and after wall contacts",
        ),
        Family(
            "long_gap_recall",
            "development",
            long_gap_recall,
            "occlusions of 16, 32 and 64 steps; memory of velocity",
        ),
        Family(
            "stationary_mixed",
            "development",
            stationary_mixed,
            "genuine stationary periods mixed with motion and gaps",
        ),
        Family(
            "noise", "development", noise, "Gaussian observation noise sigma 0.002, scored on latent truth"
        ),
        Family("change_gaps", "development", change_gaps, "unannounced speed changes plus random gaps"),
        Family("quantized", "development", quantized, "fixed speed through quanta 0.0025-0.008"),
        Family(
            "oscillator_long",
            "development",
            oscillator_long,
            "damped oscillator whose coefficients change every 300 steps",
        ),
        Family("noise_gaps", "attack", noise_gaps, "noise plus random gaps"),
        Family("change_noise", "attack", change_noise, "speed changes plus noise"),
        Family(
            "accel_switch", "attack", accel_switch, "switching constant acceleration with drag, hard walls"
        ),
        Family("coarse_near", "attack", coarse_near, "coarse construction at nearby speeds and quanta"),
        Family(
            "long_gap_recall_extended", "attack", long_gap_recall_extended, "occlusions of 96 and 128 steps"
        ),
        Family(
            "gravity_bounce",
            "held_out",
            gravity_bounce,
            "constant gravity toward the lower wall; acceleration at the wall",
        ),
        Family(
            "soft_wall",
            "held_out",
            soft_wall,
            "spring walls: no hard reflection, the public boundary map is wrong",
        ),
        Family(
            "inelastic_wall",
            "held_out",
            inelastic_wall,
            "restitution 0.6-0.9 at walls, speed reset every 200 steps",
        ),
        Family("abcab", "held_out", abcab, "A (constant velocity), B (oscillator), C (gravity), A, B"),
    )
}

ROLES = ("development", "attack", "held_out")


def families(role: str | None = None) -> list[str]:
    return [name for name, family in FAMILIES.items() if role is None or family.role == role]


def realize(family: str, seed: int) -> Realization:
    return FAMILIES[family].generate(seed)


def batch(family: str, seeds: Sequence[int]) -> StreamBatch:
    """Stream batch for one family (noisy families expose their observation, not their truth)."""

    parts: list[StreamBatch] = []
    for seed in seeds:
        r = realize(family, seed)
        observation = r.metadata.get("observation")
        values = np.asarray(observation if observation is not None else r.truth, dtype=float)
        n = len(values)
        vocab: dict[str, list[str]] = {}
        codes: dict[str, np.ndarray] = {}
        for key, words in r.labels.items():
            unique = sorted(set(words))
            vocab[key] = unique
            codes[key] = np.asarray([unique.index(w) for w in words], dtype=np.int16)[None, :]
        parts.append(
            StreamBatch(
                truth=values[None, :],
                observed=r.observed[None, :],
                length=np.asarray([n]),
                stream_ids=[f"{family}:{seed}"],
                labels=codes,
                vocab=vocab,
                score=None if observation is None else np.asarray(r.truth, dtype=float)[None, :],
            )
        )
    return StreamBatch.stack(parts)
