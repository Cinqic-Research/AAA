"""The three causally available inputs, and the public constants behind them.

Every value here is derived from public knowledge of the observation format --
the interval bounds, the step size, the declared displacement scale -- or from
observations the agent has already been given. No scenario name, event flag,
velocity, change schedule, hidden coefficient, latent truth or future
observation appears in this module, and none may.

Input 1, centered position
    ``(x_t - midpoint) / L``

Input 2, normalized recent displacement
    ``(x_t - x_{t-1}) / displacement_scale`` with ``displacement_scale =
    dt * displacement_scale_speed``. Both constants are taken from the frozen
    v2.1 specification (``dt = 0.02``, ``displacement_scale_speed = 0.2``), so
    the AAA-1K arms and the existing RLS candidate normalize displacement with
    the identical public number. When an observation is held for several steps
    the displacement is divided by the elapsed step count, so the feature is
    always a per-step velocity rather than a multi-step jump.

Input 3, previous signed prediction error
    ``(revealed x_t - the prediction that was made for x_t) / displacement_scale``.
    This exists only after the previous target has been revealed. It is zero at
    the start of an episode and zero whenever the previous target was not
    revealed to the agent. This closes AAA's predict / reveal / error / adapt
    loop through the network itself. It is inspired by prediction-error-driven
    learning; it is not a claim that this is a biological predictive-coding
    network.

    **Declared deviation.** The AAA-1K brief specified ``/ interval_width``.
    Normalizing a prediction error by ``L`` makes it numerically inert: a
    typical realized error on these streams is about 0.003, against a centered
    position of order 0.5 and a velocity feature of order 1. A development
    probe confirmed the consequence -- the ``zero_error_input`` ablation was
    indistinguishable from the full model to five decimal places, so the
    mechanism could not have been measured either way. The error is a
    displacement-like quantity, so it is normalized by the same public
    displacement constant as input 2, which is what "the same kind of public
    normalization discipline already established in AAA" actually implies. The
    probe that motivated this is retained in the decision log.

Holding an unobserved step
--------------------------
When an observation is withheld the adapter presents the **last known
position** unchanged. Input 2 is then exactly zero and input 3 is exactly zero.
The adapter therefore carries no velocity memory of its own: anything the model
knows about motion across a gap has to be in its hidden state. That is the
point of the occlusion benchmark, and it is why the baseline suite includes a
dead-reckoning predictor that *does* carry velocity explicitly.

The all-zero input pair is the missingness code. It is a weak code -- a genuine
zero displacement and a genuine zero error would look the same -- and that is
recorded as a declared limitation rather than fixed with a fourth input, which
would change the frozen 994-parameter architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PublicScales:
    """Public normalization constants shared by every arm of every experiment."""

    lower_bound: float = 0.0
    upper_bound: float = 1.0
    dt: float = 0.02
    displacement_scale_speed: float = 0.2

    def __post_init__(self) -> None:
        for name in ("lower_bound", "upper_bound", "dt", "displacement_scale_speed"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must exceed lower_bound")
        if self.dt <= 0 or self.displacement_scale_speed <= 0:
            raise ValueError("dt and displacement_scale_speed must be positive")

    @property
    def width(self) -> float:
        """Interval width ``L``."""

        return float(self.upper_bound - self.lower_bound)

    @property
    def midpoint(self) -> float:
        return float((self.lower_bound + self.upper_bound) / 2.0)

    @property
    def displacement_scale(self) -> float:
        """``dt * displacement_scale_speed``, exactly as frozen in v2.1."""

        return float(self.dt * self.displacement_scale_speed)

    def to_dict(self) -> dict[str, float]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "dt": self.dt,
            "displacement_scale_speed": self.displacement_scale_speed,
            "width": self.width,
            "midpoint": self.midpoint,
            "displacement_scale": self.displacement_scale,
        }


def build_inputs(
    scales: PublicScales,
    *,
    known_position: float,
    previous_known_position: float,
    previous_signed_error: float,
) -> np.ndarray:
    """Return the three-value input vector for one step."""

    values = (known_position, previous_known_position, previous_signed_error)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("feature inputs must be finite")
    return np.asarray(
        [
            (float(known_position) - scales.midpoint) / scales.width,
            (float(known_position) - float(previous_known_position)) / scales.displacement_scale,
            float(previous_signed_error),
        ],
        dtype=float,
    )
