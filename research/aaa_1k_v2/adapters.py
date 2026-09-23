"""Causal observation adapters: what a batch of cells may see, and what it learns from.

The adapter owns everything about the interaction history that is not the
network: the observation tracker (zero-order hold plus per-step velocity), the
previous signed prediction error, the pending raw and scored predictions, and
the target rule. It is the batched counterpart of
``research.aaa_1k.agents.NeuralAgent`` and reproduces its arithmetic
expression by expression, so that on the CPU a batch of Champion 1 cells can be
compared with the historical scalar implementation cell by cell.

The only channel into an adapter is ``accept(values, observed)``: a value per
cell and a flag saying whether that cell was shown it. No regime label, event
flag, hidden coefficient, change schedule or future value exists in this
module's signatures.

Feature sets
------------
``v1`` (3 inputs, Champion 1's)
    ``[(x - mid) / L, (x - x_prev) / s, previous error / s]``; the all-zero
    pair is the (weak) missingness code.
``v2`` (4 inputs)
    ``v1`` plus an explicit ``observed`` flag: 1 when the input position is a
    fresh observation, 0 when it is a held value. A genuinely stationary dot
    and a hidden dot are then distinguishable (research brief, R-10).

Target rules
------------
``folded``      the revealed observation as is;
``own``         unfolded around the cell's own raw prediction (Champion 0);
``reach_gated`` Champion 1's c10 rule: the own-prediction unfolding, refused
                when the input position is farther from the crossed wall than
                one observed step can reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaa.compute import Backend

from .engine import BatchedLearner, softplus

FEATURE_SETS = {"v1": 3, "v2": 4}
TARGET_RULES = ("folded", "own", "reach_gated")
_VECTOR_REFLECTIONS = 4


@dataclass(frozen=True)
class Scales:
    """Public normalization constants (the dot's are AAA-1K's ``PublicScales``)."""

    lower_bound: float = 0.0
    upper_bound: float = 1.0
    dt: float = 0.02
    displacement_scale_speed: float = 0.2

    @property
    def width(self) -> float:
        return float(self.upper_bound - self.lower_bound)

    @property
    def midpoint(self) -> float:
        return float((self.lower_bound + self.upper_bound) / 2.0)

    @property
    def displacement_scale(self) -> float:
        return float(self.dt * self.displacement_scale_speed)


def reflect(backend: Backend, position: Any, lower: float, upper: float) -> Any:
    """``aaa.predictors.reflect_prediction`` elementwise, without host synchronization.

    The first :data:`_VECTOR_REFLECTIONS` folds use the historical routine's
    exact arithmetic, so every prediction within that many interval widths of
    the interval is reflected bit for bit as the historical code reflects it.
    A prediction still outside after those folds (only a wildly diverged one)
    is folded in closed form, which can differ from the iterative routine in
    the last bit; one beyond 10,000 widths becomes NaN, which fails the cell,
    as the historical routine's refusal does.
    """

    xp = backend.xp
    width = upper - lower
    out = position
    for _ in range(_VECTOR_REFLECTIONS):
        out = xp.where(out > upper, upper - (out - upper), xp.where(out < lower, lower + (lower - out), out))
    stray = (out > upper) | (out < lower)
    folded = xp.mod(out - lower, 2.0 * width)
    folded = lower + xp.where(folded > width, 2.0 * width - folded, folded)
    out = xp.where(stray, folded, out)
    too_far = xp.abs(position - (lower + upper) / 2.0) > 10_000 * width
    return xp.where(too_far, xp.nan, out)


def unfold(backend: Backend, observed: Any, reference: Any, lower: float, upper: float) -> Any:
    """``aaa.predictors.unfold_observation`` elementwise, same candidates, same tie order."""

    xp = backend.xp
    width = upper - lower
    candidates = xp.stack(
        [
            observed,
            upper + 0 * 2 * width + (upper - observed),
            lower - 0 * 2 * width - (observed - lower),
            upper + 1 * 2 * width + (upper - observed),
            lower - 1 * 2 * width - (observed - lower),
        ],
        axis=1,
    )
    distance = xp.abs(candidates - reference[:, None])
    choice = xp.argmin(distance, axis=1)
    return xp.take_along_axis(candidates, choice[:, None], axis=1)[:, 0]


class DotAdapter:
    """Batched ``NeuralAgent`` for the bounded moving-dot observation format."""

    def __init__(
        self,
        backend: Backend,
        size: int,
        *,
        scales: Scales | None = None,
        features: str = "v1",
        target_rule: str = "reach_gated",
        reflect_predictions: bool = True,
    ) -> None:
        if features not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {features!r}")
        if target_rule not in TARGET_RULES:
            raise ValueError(f"unknown target rule {target_rule!r}")
        self.backend = backend
        self.xp = backend.xp
        self.size = int(size)
        self.scales = scales or Scales()
        self.features = features
        self.inputs = FEATURE_SETS[features]
        self.target_rule = target_rule
        self.reflect_predictions = bool(reflect_predictions)
        self.begin()

    # -- tracker ----------------------------------------------------------
    def begin(self) -> None:
        xp, B = self.xp, self.size
        zeros = self.backend.zeros
        self.known = zeros(B)
        self.previous_known = zeros(B)
        self.has_known = xp.zeros(B, dtype=bool)
        self.gap = xp.ones(B, dtype=np.int64)
        self.observed_last = xp.zeros(B, dtype=bool)
        self.pending = xp.zeros(B, dtype=np.int64)
        self.steps_seen = xp.zeros(B, dtype=np.int64)
        self.previous_signed_error = zeros(B)
        self.error_estimate = zeros(B)
        self.raw = zeros(B)
        self.scored = zeros(B)
        self.has_prediction = xp.zeros(B, dtype=bool)
        self.input_observed = xp.zeros(B, dtype=bool)
        self.input_position = zeros(B)
        self.trained_steps = xp.zeros(B, dtype=np.int64)
        self.skipped_targets = xp.zeros(B, dtype=np.int64)
        self.mirror_steps = xp.zeros(B, dtype=np.int64)
        self.gated_steps = xp.zeros(B, dtype=np.int64)
        self.any_prediction = False

    def velocity_estimate(self) -> Any:
        xp = self.xp
        velocity = (self.known - self.previous_known) / self.gap.astype(np.float64)
        return xp.where(self.observed_last & self.has_known, velocity, 0.0)

    def _track(self, values: Any, observed: Any) -> None:
        xp = self.xp
        self.steps_seen = self.steps_seen + 1
        self.previous_known = xp.where(
            observed, xp.where(self.has_known, self.known, values), self.previous_known
        )
        self.gap = xp.where(observed, self.pending + 1, self.gap)
        self.known = xp.where(observed, values, self.known)
        self.has_known = self.has_known | observed
        self.pending = xp.where(observed, 0, self.pending + 1)
        self.observed_last = observed

    # -- interface ----------------------------------------------------------
    def inputs_for_prediction(self) -> Any:
        s = self.scales
        known = self.known
        previous = known - self.velocity_estimate()
        columns = [
            (known - s.midpoint) / s.width,
            (known - previous) / s.displacement_scale,
            self.previous_signed_error,
        ]
        if self.features == "v2":
            columns.append(self.observed_last.astype(np.float64))
        return self.xp.stack(columns, axis=1)

    def predict(self, learner: BatchedLearner) -> Any:
        """Forward every cell once; return the scored (reflected) predictions."""

        xp = self.xp
        out = learner.forward(self.inputs_for_prediction())
        displacement = out[:, 0]
        raw = self.known + self.scales.displacement_scale * displacement
        self.error_estimate = softplus(xp, out[:, 1]) if out.shape[1] > 1 else xp.zeros_like(raw)
        self.raw = raw
        self.input_observed = self.observed_last | (self.steps_seen == 1)
        self.input_position = self.known
        self.has_prediction = xp.ones(self.size, dtype=bool)
        self.any_prediction = True
        scored = (
            reflect(self.backend, raw, self.scales.lower_bound, self.scales.upper_bound)
            if self.reflect_predictions
            else raw
        )
        learner._fail(~xp.isfinite(scored))
        self.scored = scored
        return scored

    def training_target(self, revealed: Any) -> Any:
        xp = self.xp
        s = self.scales
        p = self.input_position
        effective = revealed
        if self.target_rule != "folded":
            effective = unfold(self.backend, revealed, self.raw, s.lower_bound, s.upper_bound)
            mirrored = effective != revealed
            if self.target_rule == "reach_gated":
                distance = xp.where(effective < s.lower_bound, p - s.lower_bound, s.upper_bound - p)
                reach = xp.abs(self.velocity_estimate()) + xp.abs(revealed - p)
                gated = mirrored & (distance > reach)
                effective = xp.where(gated, revealed, effective)
                self._last_gated = gated
            self._last_mirrored = effective != revealed
        return (effective - p) / s.displacement_scale

    def accept(self, learner: BatchedLearner | None, values: Any, observed: Any, update: Any) -> None:
        """Reveal ``values`` to the cells in ``observed``; update the cells in ``update``."""

        xp = self.xp
        observed = xp.asarray(observed, dtype=bool)
        values = xp.where(observed, values, 0.0)
        trainable = observed & self.has_prediction & self.input_observed
        if learner is not None and self.any_prediction:
            target = self.training_target(values)
            learning = trainable & xp.asarray(update, dtype=bool) & ~learner.failed
            learner.learn(target, learning)
            self.trained_steps = self.trained_steps + learning.astype(np.int64)
            if self.target_rule != "folded":
                self.mirror_steps = self.mirror_steps + (learning & self._last_mirrored).astype(np.int64)
            if self.target_rule == "reach_gated":
                self.gated_steps = self.gated_steps + (learning & self._last_gated).astype(np.int64)
        self.skipped_targets = self.skipped_targets + (
            (~observed) | (observed & self.has_prediction & ~self.input_observed)
        ).astype(np.int64)
        new_error = (values - self.scored) / self.scales.displacement_scale
        self.previous_signed_error = xp.where(
            observed, xp.where(self.has_prediction, new_error, self.previous_signed_error), 0.0
        )
        self._track(values, observed)


class SeriesAdapter:
    """Batched adapter for an external scalar series (no walls, no unfolding).

    ``s_t`` is the observed input channel and ``y_t`` the target channel. For
    ordinary forecasting ``s`` and ``y`` are the same series shifted by one
    step and the target is the change ``y_{t+1} - s_t`` (``target_mode =
    "delta"``); for an input/output system such as NARMA the target is the
    output value itself (``target_mode = "absolute"``). Every scale is derived
    from the *training* portion of the series only and is passed in; nothing
    here can see test data.

    Features, in the same slots as the dot's ``v1``/``v2`` sets:
    ``[(s - center) / level_scale, (s - s_prev) / delta_scale,
    previous error / error_scale (, observed flag)]``.
    """

    def __init__(
        self,
        backend: Backend,
        size: int,
        *,
        center: Any,
        level_scale: Any,
        delta_scale: Any,
        target_center: Any,
        target_scale: Any,
        target_mode: str = "delta",
        features: str = "v1",
    ) -> None:
        if target_mode not in ("delta", "absolute"):
            raise ValueError("target_mode must be 'delta' or 'absolute'")
        if features not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {features!r}")
        self.backend = backend
        self.xp = backend.xp
        self.size = int(size)
        asarray = backend.asarray
        self.center = asarray(np.broadcast_to(center, (size,)).copy())
        self.level_scale = asarray(np.broadcast_to(level_scale, (size,)).copy())
        self.delta_scale = asarray(np.broadcast_to(delta_scale, (size,)).copy())
        self.target_center = asarray(np.broadcast_to(target_center, (size,)).copy())
        self.target_scale = asarray(np.broadcast_to(target_scale, (size,)).copy())
        for name in ("level_scale", "delta_scale", "target_scale"):
            if not bool(self.xp.all(getattr(self, name) > 0)):
                raise ValueError(f"{name} must be positive")
        self.target_mode = target_mode
        self.features = features
        self.inputs = FEATURE_SETS[features]
        self.begin()

    def begin(self) -> None:
        xp, B, zeros = self.xp, self.size, self.backend.zeros
        self.known = zeros(B)
        self.previous_known = zeros(B)
        self.has_known = xp.zeros(B, dtype=bool)
        self.observed_last = xp.zeros(B, dtype=bool)
        self.previous_signed_error = zeros(B)
        self.prediction = zeros(B)
        self.error_estimate = zeros(B)
        self.has_prediction = xp.zeros(B, dtype=bool)
        self.input_observed = xp.zeros(B, dtype=bool)
        self.trained_steps = xp.zeros(B, dtype=np.int64)

    def inputs_for_prediction(self) -> Any:
        xp = self.xp
        delta = xp.where(self.observed_last, self.known - self.previous_known, 0.0)
        columns = [
            (self.known - self.center) / self.level_scale,
            delta / self.delta_scale,
            self.previous_signed_error,
        ]
        if self.features == "v2":
            columns.append(self.observed_last.astype(np.float64))
        return xp.stack(columns, axis=1)

    def _error_scale(self) -> Any:
        return self.delta_scale if self.target_mode == "delta" else self.target_scale

    def predict(self, learner: BatchedLearner) -> Any:
        xp = self.xp
        out = learner.forward(self.inputs_for_prediction())
        if self.target_mode == "delta":
            prediction = self.known + self.delta_scale * out[:, 0]
        else:
            prediction = self.target_center + self.target_scale * out[:, 0]
        self.error_estimate = (
            softplus(xp, out[:, 1]) * self._error_scale() if out.shape[1] > 1 else xp.zeros_like(prediction)
        )
        learner._fail(~xp.isfinite(prediction))
        self.prediction = prediction
        self.input_observed = self.observed_last
        self.has_prediction = xp.ones(self.size, dtype=bool)
        return prediction

    def normalized_target(self, target: Any) -> Any:
        if self.target_mode == "delta":
            return (target - self.known) / self.delta_scale
        return (target - self.target_center) / self.target_scale

    def reveal_target(self, learner: BatchedLearner | None, target: Any, revealed: Any, update: Any) -> None:
        """Reveal the target of the prediction just made (``revealed`` False = not shown)."""

        xp = self.xp
        revealed = xp.asarray(revealed, dtype=bool)
        trainable = revealed & self.has_prediction & self.input_observed
        if learner is not None:
            learning = trainable & xp.asarray(update, dtype=bool) & ~learner.failed
            learner.learn(self.normalized_target(xp.where(revealed, target, 0.0)), learning)
            self.trained_steps = self.trained_steps + learning.astype(np.int64)
        error = (xp.where(revealed, target, 0.0) - self.prediction) / self._error_scale()
        self.previous_signed_error = xp.where(revealed & self.has_prediction, error, 0.0)

    def observe_input(self, values: Any, observed: Any) -> None:
        """Advance the input channel: ``values`` where ``observed``, a hold elsewhere."""

        xp = self.xp
        observed = xp.asarray(observed, dtype=bool)
        self.previous_known = xp.where(
            observed, xp.where(self.has_known, self.known, values), self.previous_known
        )
        self.known = xp.where(observed, values, self.known)
        self.has_known = self.has_known | observed
        self.observed_last = observed


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _cell_arrays(adapter: Any) -> list[str]:
    """Names of the adapter's persistent per-cell state arrays (shape ``(size,)``).

    Underscore-prefixed arrays are per-step scratch (for example the last
    step's mirror flags) and never carry information into the next step.
    """

    return [
        key
        for key, value in vars(adapter).items()
        if not key.startswith("_")
        and isinstance(getattr(value, "shape", None), tuple)
        and len(value.shape) == 1
        and value.shape[0] == adapter.size
    ]


def take_adapter(adapter: Any, indices: Any) -> Any:
    """Copy an adapter's per-cell state for ``indices`` (duplicates allowed)."""

    xp = adapter.xp
    index = xp.asarray(np.asarray(indices, dtype=np.int64))
    clone = object.__new__(type(adapter))
    arrays = set(_cell_arrays(adapter))
    for key, value in vars(adapter).items():
        setattr(clone, key, value[index].copy() if key in arrays else value)
    clone.size = len(indices)
    return clone


def adapter_state(adapter: Any) -> dict[str, Any]:
    """Host-side per-cell interaction state plus the adapter's scalar flags (JSON-serializable)."""

    host = adapter.backend.to_host
    arrays = _cell_arrays(adapter)
    state: dict[str, Any] = {"arrays": {}, "dtypes": {}, "flags": {}}
    for key in arrays:
        value = host(getattr(adapter, key))
        state["arrays"][key] = value.tolist()
        state["dtypes"][key] = str(value.dtype)
    for key, value in vars(adapter).items():
        if isinstance(value, bool) and key not in arrays:
            state["flags"][key] = value
    return state


def load_adapter_state(adapter: Any, state: dict[str, Any]) -> None:
    """Restore :func:`adapter_state` output into an adapter of the same size and kind."""

    expected = set(_cell_arrays(adapter))
    if set(state["arrays"]) != expected:
        raise ValueError("adapter state fields do not match this adapter")
    for key, values in state["arrays"].items():
        array = adapter.backend.asarray(
            np.asarray(values, dtype=state["dtypes"][key]), dtype=state["dtypes"][key]
        )
        if array.shape != (adapter.size,):
            raise ValueError(f"adapter state {key} has the wrong shape")
        setattr(adapter, key, array)
    for key, value in state["flags"].items():
        setattr(adapter, key, bool(value))
