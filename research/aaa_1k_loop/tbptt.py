"""What gradient does AAA-1K's online TBPTT actually compute? (audit finding R-02)

The champion learns online: :class:`research.aaa_1k.agents.NeuralAgent` calls
``model.learn()`` after essentially every observed transition. Each call runs
``AAA1KGRU.backward()`` over the last ``tbptt_steps`` cached transitions and
then applies one SGD step. The cached *activations* of older transitions were
produced under older parameter values -- the parameters have moved once per
step since -- but ``backward()`` multiplies through the *current* recurrent
matrices ``U_z``, ``U_r``, ``U_n``. The model docstring says the gradient is
"exact for the realized trajectory up to the truncation horizon". The
existing gradient checker cannot test that sentence: it holds the parameters
fixed across the whole sequence, where every cache and every matrix agree.

This module makes the question decidable by defining the alternatives
explicitly and comparing them on the same model state.

``live``
    exactly what ``AAA1KGRU.backward()`` returns: cached activations, current
    matrices.
``snapshot``
    the gradient the docstring describes. Each cached transition is
    backpropagated through the matrices *it was computed with*. This is the
    exact derivative of the latest step's loss with respect to one common
    perturbation added to the parameter value in force at every step of the
    window, holding the realized hidden state at the start of the window and
    the recorded inputs fixed. :func:`realized_window_loss` is its
    finite-difference reference.
``replay``
    the exact gradient, at the *current* parameters, of the latest step's loss
    after re-running the window forward from the realized hidden state at its
    start. This is what chunked TBPTT with parameters held constant across the
    backpropagated chunk computes. :func:`current_window_loss` is its
    finite-difference reference.
``t1``
    the latest transition only. With one transition there are no stale
    caches, so all three definitions coincide; it is the rule that has no
    ambiguity to resolve.

Nothing here edits :mod:`research.aaa_1k`: the alternatives are subclasses
that override ``backward()`` and ``forward()`` only, and a test asserts that
the shared backward routine reproduces ``AAA1KGRU.backward()`` bit for bit
when it is handed the live matrices.

In every rule the recorded input vector (including input 3, built from an
earlier prediction) is a constant for differentiation, exactly as in the
champion; the auxiliary error target has stop-gradient semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.model import AAA1KGRU, RECURRENT_PARAMETER_NAMES, StepCache, sigmoid, softplus

RULES = ("live", "snapshot", "replay", "t1")

_SNAPSHOT_ATTRIBUTE = "_probe_parameters"
"""Each cache created by :class:`SnapshotGRU` carries the parameters it was computed with."""


# ----------------------------------------------------------------------
# the one backward routine every rule shares
# ----------------------------------------------------------------------
def backward_through(
    window: Sequence[StepCache],
    matrices: Sequence[Mapping[str, np.ndarray]],
    output_weight: np.ndarray,
    output_gradient: np.ndarray,
) -> dict[str, np.ndarray]:
    """``AAA1KGRU.backward()`` with the recurrent matrices chosen per transition.

    ``matrices[k]`` supplies ``U_z``, ``U_r`` and ``U_n`` for ``window[k]``.
    ``output_weight`` is the ``W_o`` that produced the last output. The
    arithmetic is a line-for-line copy of the champion's loop so that handing
    it the live matrices reproduces the champion exactly (tested bitwise).
    """

    if len(window) != len(matrices) or not window:
        raise ValueError("one set of recurrent matrices is required per cached transition")
    last = window[-1]
    gradients = {
        name: np.zeros(shape)
        for name, shape in (
            ("W_z", (last.z.size, last.x.size)),
            ("U_z", (last.z.size, last.z.size)),
            ("b_z", (last.z.size,)),
            ("W_r", (last.z.size, last.x.size)),
            ("U_r", (last.z.size, last.z.size)),
            ("b_r", (last.z.size,)),
            ("W_n", (last.z.size, last.x.size)),
            ("U_n", (last.z.size, last.z.size)),
            ("b_n", (last.z.size,)),
            ("W_o", output_weight.shape),
            ("b_o", (output_weight.shape[0],)),
        )
    }
    d_output = output_gradient
    gradients["W_o"] += np.outer(d_output, last.h)
    gradients["b_o"] += d_output
    d_h = output_weight.T @ d_output

    for cache, p in zip(reversed(window), reversed(matrices), strict=True):
        d_z = d_h * (cache.h_prev - cache.n)
        d_n = d_h * (1.0 - cache.z)
        d_h_prev = d_h * cache.z

        d_a_n = d_n * (1.0 - cache.n * cache.n)
        gradients["W_n"] += np.outer(d_a_n, cache.x)
        gradients["U_n"] += np.outer(d_a_n, cache.hr)
        gradients["b_n"] += d_a_n
        d_hr = p["U_n"].T @ d_a_n

        d_r = d_hr * cache.h_prev
        d_h_prev = d_h_prev + d_hr * cache.r

        d_a_r = d_r * cache.r * (1.0 - cache.r)
        gradients["W_r"] += np.outer(d_a_r, cache.x)
        gradients["U_r"] += np.outer(d_a_r, cache.h_prev)
        gradients["b_r"] += d_a_r
        d_h_prev = d_h_prev + p["U_r"].T @ d_a_r

        d_a_z = d_z * cache.z * (1.0 - cache.z)
        gradients["W_z"] += np.outer(d_a_z, cache.x)
        gradients["U_z"] += np.outer(d_a_z, cache.h_prev)
        gradients["b_z"] += d_a_z
        d_h_prev = d_h_prev + p["U_z"].T @ d_a_z

        d_h = d_h_prev
    return gradients


def step(parameters: Mapping[str, np.ndarray], x: np.ndarray, h_prev: np.ndarray) -> StepCache:
    """One GRU transition, identical arithmetic to ``AAA1KGRU.forward()``."""

    p = parameters
    z = sigmoid(p["W_z"] @ x + p["U_z"] @ h_prev + p["b_z"])
    r = sigmoid(p["W_r"] @ x + p["U_r"] @ h_prev + p["b_r"])
    hr = r * h_prev
    n = np.tanh(p["W_n"] @ x + p["U_n"] @ hr + p["b_n"])
    h = z * h_prev + (1.0 - z) * n
    output = p["W_o"] @ h + p["b_o"]
    return StepCache(x=x.copy(), h_prev=h_prev.copy(), z=z, r=r, n=n, hr=hr, h=h, output=output)


def _copy(parameters: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.array(value, dtype=float, copy=True) for name, value in parameters.items()}


# ----------------------------------------------------------------------
# the rules, as drop-in models
# ----------------------------------------------------------------------
class SnapshotGRU(AAA1KGRU):
    """AAA-1K whose caches remember the full parameter set they were computed with.

    Its forward pass is the champion's, unchanged. Only ``backward()`` differs,
    and only when the parameters moved inside the window.
    """

    rule = "snapshot"

    def forward(self, inputs: Any, *, record: bool = True) -> np.ndarray:
        before = _copy(self.parameters) if record else None
        output = super().forward(inputs, record=record)
        if record:
            setattr(self._caches[-1], _SNAPSHOT_ATTRIBUTE, before)
        return output

    def window_parameters(self) -> list[dict[str, np.ndarray]]:
        versions = []
        for cache in self._caches:
            version = getattr(cache, _SNAPSHOT_ATTRIBUTE, None)
            if version is None:
                raise RuntimeError(
                    "a cached transition has no parameter snapshot (was it restored from a checkpoint?)"
                )
            versions.append(version)
        return versions

    def backward(self, target_displacement: float) -> dict[str, np.ndarray]:
        return snapshot_gradient(self, target_displacement)


class ReplayGRU(SnapshotGRU):
    """AAA-1K that re-runs the window under current parameters before backpropagating.

    Inherits snapshots only so that one instrumented model can report every
    rule; its own update never reads them.
    """

    rule = "replay"

    def backward(self, target_displacement: float) -> dict[str, np.ndarray]:
        return replay_gradient(self, target_displacement)


def live_gradient(model: AAA1KGRU, target: float) -> dict[str, np.ndarray]:
    return AAA1KGRU.backward(model, target)


def snapshot_gradient(model: SnapshotGRU, target: float) -> dict[str, np.ndarray]:
    window = list(model._caches)
    if not window:
        raise ValueError("snapshot gradient requires at least one recorded transition")
    versions = model.window_parameters()
    last_version = versions[-1]
    d_output = model._output_gradient(window[-1].output, target)
    return backward_through(window, versions, last_version["W_o"], d_output)


def replay_window(model: AAA1KGRU, parameters: Mapping[str, np.ndarray]) -> list[StepCache]:
    """Re-run the cached window from its realized starting state under ``parameters``."""

    window = list(model._caches)
    h = window[0].h_prev.copy()
    replayed = []
    for cache in window:
        new = step(parameters, cache.x, h)
        replayed.append(new)
        h = new.h
    return replayed


def replay_gradient(model: AAA1KGRU, target: float) -> dict[str, np.ndarray]:
    if not model._caches:
        raise ValueError("replay gradient requires at least one recorded transition")
    p = model.parameters
    replayed = replay_window(model, p)
    # The auxiliary target is |o0 - d| at the *replayed* output, stop-gradient.
    d_output = model._output_gradient(replayed[-1].output, target)
    return backward_through(replayed, [p] * len(replayed), p["W_o"], d_output)


def t1_gradient(model: AAA1KGRU, target: float) -> dict[str, np.ndarray]:
    last = model._caches[-1]
    d_output = model._output_gradient(last.output, target)
    return backward_through([last], [model.parameters], model.parameters["W_o"], d_output)


def build_rule_model(rule: str, seed: int, configuration: Mapping[str, Any]) -> AAA1KGRU:
    """The champion's construction with the named update rule. Same initialization for every rule."""

    config = dict(configuration)
    if rule == "live":
        return AAA1KGRU(seed=seed, **config)
    if rule == "snapshot":
        return SnapshotGRU(seed=seed, **config)
    if rule == "replay":
        return ReplayGRU(seed=seed, **config)
    if rule == "t1":
        return AAA1KGRU(seed=seed, **{**config, "tbptt_steps": 1})
    raise ValueError(f"unknown rule {rule!r}")


# ----------------------------------------------------------------------
# finite-difference references
# ----------------------------------------------------------------------
def _window_loss(model: AAA1KGRU, outputs_error_target: float, target: float, output: np.ndarray) -> float:
    signed = float(output[0]) - float(target)
    estimate = float(softplus(output[1:2])[0])
    return signed * signed + model.error_loss_weight * (estimate - outputs_error_target) ** 2


def realized_window_loss(
    model: SnapshotGRU, target: float, delta: Mapping[str, np.ndarray], *, error_target: float
) -> float:
    """Latest-step loss with ``delta`` added to *every* step's own parameter version."""

    window = list(model._caches)
    versions = model.window_parameters()
    h = window[0].h_prev.copy()
    output = None
    for cache, version in zip(window, versions, strict=True):
        perturbed = {name: version[name] + delta[name] for name in version}
        new = step(perturbed, cache.x, h)
        h, output = new.h, new.output
    assert output is not None
    return _window_loss(model, error_target, target, output)


def current_window_loss(
    model: AAA1KGRU, target: float, delta: Mapping[str, np.ndarray], *, error_target: float
) -> float:
    """Latest-step loss after replaying the window under current parameters plus ``delta``."""

    perturbed = {name: value + delta[name] for name, value in model.parameters.items()}
    output = replay_window(model, perturbed)[-1].output
    return _window_loss(model, error_target, target, output)


def finite_difference(
    loss: Any, model: AAA1KGRU, target: float, *, error_target: float, epsilon: float = 1e-6
) -> dict[str, np.ndarray]:
    zero = {name: np.zeros_like(value) for name, value in model.parameters.items()}
    gradient = {name: np.zeros_like(value) for name, value in model.parameters.items()}
    for name, array in zero.items():
        for index in np.ndindex(array.shape):
            plus = {**zero, name: array.copy()}
            plus[name][index] = epsilon
            minus = {**zero, name: array.copy()}
            minus[name][index] = -epsilon
            gradient[name][index] = (
                loss(model, target, plus, error_target=error_target)
                - loss(model, target, minus, error_target=error_target)
            ) / (2.0 * epsilon)
    return gradient


# ----------------------------------------------------------------------
# comparing rules on one state
# ----------------------------------------------------------------------
def flatten(gradients: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.ravel(gradients[name]) for name in sorted(gradients)])


def clipped_update(model: AAA1KGRU, gradients: Mapping[str, np.ndarray]) -> np.ndarray:
    """The parameter change ``apply_gradients`` would make, as one flat vector."""

    vector = flatten(gradients)
    norm = float(np.linalg.norm(vector))
    scale = 1.0
    if model.gradient_clip is not None and norm > model.gradient_clip:
        scale = model.gradient_clip / norm
    if model.freeze_recurrent:
        mask = np.concatenate(
            [
                np.full(gradients[name].size, name not in RECURRENT_PARAMETER_NAMES, dtype=float)
                for name in sorted(gradients)
            ]
        )
        vector = vector * mask
    return -model.learning_rate * scale * vector


def compare(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """Relative difference and cosine of ``other`` against ``reference``."""

    ref_norm = float(np.linalg.norm(reference))
    other_norm = float(np.linalg.norm(other))
    difference = float(np.linalg.norm(other - reference))
    if ref_norm == 0.0:
        relative = 0.0 if difference == 0.0 else math.inf
    else:
        relative = difference / ref_norm
    if ref_norm == 0.0 or other_norm == 0.0:
        cosine = 1.0 if difference == 0.0 else math.nan
    else:
        cosine = float(np.dot(reference, other) / (ref_norm * other_norm))
    return {
        "relative_difference": relative,
        "cosine": cosine,
        "norm_ratio": other_norm / ref_norm if ref_norm else math.nan,
    }


class ComparingGRU(SnapshotGRU):
    """The champion, learning with the *live* rule, reporting every other rule on its exact state.

    ``learn()`` computes the snapshot, replay and T=1 update vectors on the
    same caches and parameters before applying the live update, so the
    trajectory is bitwise the champion's (tested) and the comparison is
    paired step by step. It also records how far the parameters moved inside
    the window, which is the only thing that makes the rules differ.
    """

    rule = "live"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.comparisons: list[dict[str, float]] = []

    def backward(self, target_displacement: float) -> dict[str, np.ndarray]:
        return live_gradient(self, target_displacement)

    def learn(self, target_displacement: float) -> dict[str, float]:
        live = live_gradient(self, target_displacement)
        others = {
            "snapshot": snapshot_gradient(self, target_displacement),
            "replay": replay_gradient(self, target_displacement),
            "t1": t1_gradient(self, target_displacement),
        }
        live_raw = flatten(live)
        live_update = clipped_update(self, live)
        versions = self.window_parameters()
        current = flatten(self.parameters)
        drift = max(float(np.linalg.norm(flatten(version) - current)) for version in versions)
        record: dict[str, float] = {
            "window": float(len(versions)),
            "window_parameter_drift": drift,
            "live_gradient_norm": float(np.linalg.norm(live_raw)),
            "live_update_norm": float(np.linalg.norm(live_update)),
        }
        for rule, gradients in others.items():
            raw = compare(flatten(gradients), live_raw)
            update = compare(clipped_update(self, gradients), live_update)
            record[f"{rule}_gradient_relative_difference"] = raw["relative_difference"]
            record[f"{rule}_gradient_cosine"] = raw["cosine"]
            record[f"{rule}_update_relative_difference"] = update["relative_difference"]
            record[f"{rule}_update_cosine"] = update["cosine"]
        self.comparisons.append(record)
        return self.apply_gradients(live)
