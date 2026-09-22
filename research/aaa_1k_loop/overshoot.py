"""Does one online SGD step make the sample it learned from worse? (M2, after the gain hypothesis fell)

Round 2 of iteration 0004 falsified "learned loop gain above one": the
runaway starts while the direct error-feedback gain is about 0.2-0.5, and a
learning rate of 0.01 never runs away even with three times the updates. A
hard learning-rate threshold is the signature of a different, classical
instability -- the stability condition of online least squares (LMS):

For a squared-error step on one sample with residual ``r = o - d*`` and output
Jacobian ``J = do/dtheta``, the linearized residual after the step is
``r' = r (1 - kappa)`` with ``kappa = 2 * lr * scale * |J|^2`` (``scale`` the
clip factor). ``0 < kappa < 2`` shrinks the error on the sample just seen;
``kappa > 2`` overshoots it and amplifies it. ``|J|^2`` is not constant here: the
gate-weight entries of ``J`` scale with the inputs -- including input 3, the
previous error -- and with ``|W_o|``. So a larger error makes a larger input,
a larger curvature, a larger ``kappa`` and a larger error.

:class:`OvershootGRU` measures this directly and exactly, at every learning
step, on the live champion's own state. It learns with the live rule,
unchanged (tested bitwise), and records:

``amplification``
    ``|r'| / |r|``: the *exact* residual on the same inputs after the actual
    (clipped) update over the one before it, both recomputed from the
    realized hidden state at the start of the TBPTT window, so only the step
    differs. Above one means the step made the sample it learned from worse.
``kappa``
    the linearized correction ratio ``-(J . dtheta) / r`` for the actual step.
``curvature``
    ``|J|^2`` for the displacement output, and the share of it carried by the
    input-3 columns of the gate matrices and by the output head.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from research.aaa_1k.model import AAA1KGRU, PARAMETER_NAMES

from .tbptt import backward_through, flatten, replay_window

ERROR_COLUMN = 2
GATE_INPUT_MATRICES = ("W_z", "W_r", "W_n")
HEAD = ("W_o", "b_o")


def displacement_jacobian(model: AAA1KGRU) -> dict[str, np.ndarray]:
    """``d o_t[0] / d theta`` through the stored window, with the live rule's matrices."""

    window = list(model._caches)
    unit = np.zeros(model.parameters["b_o"].shape)
    unit[0] = 1.0
    return backward_through(window, [model.parameters] * len(window), model.parameters["W_o"], unit)


def curvature_shares(jacobian: dict[str, np.ndarray]) -> dict[str, float]:
    total = float(sum(float(np.sum(v * v)) for v in jacobian.values()))
    error_columns = float(
        sum(float(np.sum(jacobian[name][:, ERROR_COLUMN] ** 2)) for name in GATE_INPUT_MATRICES)
    )
    head = float(sum(float(np.sum(jacobian[name] ** 2)) for name in HEAD))
    return {
        "curvature": total,
        "error_input_share": error_columns / total if total > 0 else math.nan,
        "head_share": head / total if total > 0 else math.nan,
    }


class OvershootGRU(AAA1KGRU):
    """The champion, learning exactly as the champion does, measuring each step's overshoot."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.overshoot: list[dict[str, float]] = []

    def learn(self, target_displacement: float) -> dict[str, float]:
        window_start = self._caches[0].h_prev.copy()
        residual = float(self._caches[-1].output[0]) - float(target_displacement)
        replayed_before = float(replay_window(self, self.parameters)[-1].output[0]) - float(
            target_displacement
        )
        jacobian = displacement_jacobian(self)
        before = {name: self.parameters[name].copy() for name in PARAMETER_NAMES}
        update = super().learn(target_displacement)
        step = flatten({name: self.parameters[name] - before[name] for name in PARAMETER_NAMES})
        linear = float(np.dot(flatten(jacobian), step))
        # Exact residual on the same inputs, same window start, new parameters.
        if not np.array_equal(window_start, self._caches[0].h_prev):
            raise RuntimeError("the TBPTT window moved during learn()")
        after = float(replay_window(self, self.parameters)[-1].output[0]) - float(target_displacement)
        record = {
            "residual": residual,
            "amplification": abs(after) / abs(replayed_before) if replayed_before != 0.0 else math.nan,
            "kappa": -linear / residual if residual != 0.0 else math.nan,
            "clip_scale": float(update["clip_scale"]),
            "error_input": float(self._caches[-1].x[ERROR_COLUMN]),
            "head_norm": float(np.linalg.norm(self.parameters["W_o"])),
            **curvature_shares(jacobian),
        }
        self.overshoot.append(record)
        return update
