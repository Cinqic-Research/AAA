"""The closed-loop gain of the previous-error channel (audit finding R-01, mechanism M2).

Input 3 is the agent's previous signed prediction error in normalized
displacement units, ``e_{t-1} = d*_{t-1} - o_{t-1}``, fed back as a feature.
Away from the walls (where reflection is the identity) the error the agent
makes next is ``e_t = d*_t - o_t``, and ``o_t`` depends on ``e_{t-1}``. The
agent and the world therefore form a closed loop, with the parameters frozen
for the duration of one step:

    state     s_t = (h_t, e_t)                       17 values
    h_t       = GRU(h_{t-1}, x_t),   x_t[2] = e_{t-1}
    e_t       = d*_t - W_o[0] h_t - b_o[0]

Two read-only measurements of that loop, at the activations the model
actually realized for its latest forward pass:

``direct_gain``
    ``a_t = d e_t / d e_{t-1}`` holding ``h_{t-1}`` fixed
    ``= -W_o[0] . dh_t/dx_t[:, 2]``. If ``|a_t| > 1`` persists, an error is
    fed back larger than it arrived.
``closed_loop_radius``
    spectral radius of the 17x17 Jacobian of ``s_{t-1} -> s_t``, which adds
    every path through the hidden state.

Why the one-step gradient can walk this gain out of the stable region
---------------------------------------------------------------------
The feature vector is a constant for differentiation (see the model
docstring), so SGD fits ``o_t`` to ``d*_t`` *given* ``e_{t-1}`` and never sees
that changing the weight on input 3 changes the next input. For a displacement
sequence with deviations ``delta_t`` about its mean that alternate
(``delta_t = -delta_{t-1}``, as a sub-quantum speed on a quantizer produces),
a model ``o_t = m + k e_{t-1}`` settles at ``e_t = delta_t / (1 - k)``; the
regression of ``delta_t`` on that ``e_{t-1}`` has slope ``-(1 - k) = k - 1``.
The regression target is always one unit beyond the current value, so there is
no stable fixed point: the gradient keeps pushing ``a = -k`` upward, at a rate
set by the learning rate, through ``a = 1``. That argument is what the round-2
hypotheses test; it is a hypothesis, not a result.

With ``zero_error_input`` the loop is open by construction and both
measurements are reported as exactly zero feedback.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from research.aaa_1k.agents import NeuralAgent
from research.aaa_1k.model import AAA1KGRU

from .dynamics import gru_jacobian

ERROR_INPUT = 2


def input_jacobian(model: AAA1KGRU) -> np.ndarray | None:
    """``dh_t / dx_t`` (16 x 3) at the latest cached transition."""

    if not model._caches:
        return None
    cache = model._caches[-1]
    p = model.parameters
    z, r, n, h_prev = cache.z, cache.r, cache.n, cache.h_prev
    dz = (z * (1.0 - z))[:, None] * p["W_z"]
    dr = (r * (1.0 - r))[:, None] * p["W_r"]
    dn = (1.0 - n * n)[:, None] * (p["W_n"] + p["U_n"] @ (h_prev[:, None] * dr))
    return (h_prev - n)[:, None] * dz + (1.0 - z)[:, None] * dn


def closed_loop_jacobian(model: AAA1KGRU) -> np.ndarray | None:
    """Jacobian of ``(h_{t-1}, e_{t-1}) -> (h_t, e_t)`` at the latest transition."""

    state = gru_jacobian(model)
    inputs = input_jacobian(model)
    if state is None or inputs is None:
        return None
    column = inputs[:, ERROR_INPUT] if not model.zero_error_input else np.zeros(state.shape[0])
    readout = model.parameters["W_o"][0]
    top = np.hstack([state, column[:, None]])
    bottom = np.concatenate([-(readout @ state), [-(readout @ column)]])
    return np.vstack([top, bottom[None, :]])


def loop_measurements(model: AAA1KGRU) -> dict[str, float]:
    jacobian = closed_loop_jacobian(model)
    if jacobian is None:
        return {"direct_gain": math.nan, "closed_loop_radius": math.nan, "state_radius": math.nan}
    state = jacobian[:-1, :-1]
    return {
        "direct_gain": float(jacobian[-1, -1]),
        "closed_loop_radius": float(np.max(np.abs(np.linalg.eigvals(jacobian)))),
        "state_radius": float(np.max(np.abs(np.linalg.eigvals(state)))),
    }


class GainAgent(NeuralAgent):
    """A :class:`NeuralAgent` that records the loop measurements after every prediction.

    Read-only: the measurement uses the cache the forward pass just wrote and
    changes nothing the learner or the scorer can see.
    """

    def __init__(self, model: Any, *, name: str, **kwargs: Any) -> None:
        super().__init__(model, name=name, **kwargs)
        self.gain_trace: list[dict[str, float]] = []

    def predict(self) -> float:
        prediction = super().predict()
        self.gain_trace.append(loop_measurements(self.model))
        return prediction
