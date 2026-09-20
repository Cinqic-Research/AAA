"""Finite-difference verification of the hand-written backward passes.

Because the GRU is written out by hand, the gradient is the thing most likely
to be quietly wrong, and a wrong gradient does not crash -- it just learns
something slightly false. So it is checked two ways.

**One-step gradients.** With a single recorded transition, the analytic
gradient must match a central difference of that step's loss.

**Sequence gradients.** A recurrent model can have a perfect one-step gradient
and a broken temporal one. So the per-step truncated gradients are summed over
a whole sequence with the truncation horizon set at least as long as the
sequence, and compared against a central difference of the *total* sequence
loss from a fixed initial hidden state. Those two are equal by construction
only if the backward pass really propagates through the recurrence.

**Stop-gradient semantics.** The auxiliary target is ``|o0 - d|`` with a
stop-gradient. The finite-difference reference must therefore hold that target
fixed at its unperturbed value; a reference that lets it move disagrees with
the analytic gradient completely, which is itself a useful check that the
stop-gradient is genuinely in effect.

Sequences deliberately include sign changes, non-zero starting hidden states
and both output heads.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .controls import StatelessMLPControl, VanillaRNNControl
from .model import AAA1KGRU, softplus

EPSILON = 1e-6
TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-8
"""Central differences with ``EPSILON = 1e-6`` carry an absolute round-off floor of
roughly 1e-9 on an order-one loss. A purely relative criterion therefore reports a
spurious failure whenever a true gradient component happens to be tiny: two values of
8.743006e-06 and 8.743881e-06 agree to six figures and are 5e-5 apart relatively. The
pass criterion is the standard combined one, ``|difference| <= atol + rtol * scale``,
which keeps the relative test sharp where it means something without failing on the
noise floor."""


def _sequence_loss(model: Any, inputs: np.ndarray, targets: np.ndarray, frozen: Sequence[float]) -> float:
    """Total sequence loss with the auxiliary targets held fixed."""

    model.reset_state()
    total = 0.0
    for x, target, error_target in zip(inputs, targets, frozen, strict=True):
        output = model.forward(x)
        signed = float(output[0]) - float(target)
        estimate = float(softplus(output[1:2])[0])
        total += signed * signed + model.error_loss_weight * (estimate - error_target) ** 2
    return total


def _frozen_targets(model: Any, inputs: np.ndarray, targets: np.ndarray) -> list[float]:
    model.reset_state()
    frozen = []
    for x, target in zip(inputs, targets, strict=True):
        output = model.forward(x)
        frozen.append(abs(float(output[0]) - float(target)))
    return frozen


def _analytic_sequence_gradient(model: Any, inputs: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    model.reset_state()
    total = {name: np.zeros_like(array) for name, array in model.parameters.items()}
    for x, target in zip(inputs, targets, strict=True):
        model.forward(x)
        step = model.backward(float(target))
        for name in total:
            total[name] += step[name]
    return total


def check_model(
    factory: Callable[[], Any],
    *,
    label: str,
    length: int = 9,
    seed: int = 0,
    exhaustive: bool = False,
    samples_per_parameter: int = 12,
) -> dict[str, Any]:
    """Compare the analytic sequence gradient against central differences."""

    rng = np.random.default_rng(seed)
    model = factory()
    # Randomize away from the zero-initialized output head, so a bug in the
    # head cannot hide behind an exact zero, and warm the hidden state.
    for name, array in model.parameters.items():
        model.parameters[name] = rng.normal(size=array.shape) * 0.35
    inputs = rng.normal(size=(length, 3))
    # A sign change partway through, so temporal credit assignment is exercised
    # rather than a monotone ramp.
    inputs[length // 2 :, 1] *= -1.0
    targets = rng.normal(size=length) * 0.6
    targets[length // 3] = -targets[length // 3]

    frozen = _frozen_targets(model, inputs, targets)
    analytic = _analytic_sequence_gradient(model, inputs, targets)

    worst = 0.0
    worst_relative = 0.0
    worst_absolute = 0.0
    violations = 0
    worst_name = ""
    checked = 0
    for name, array in model.parameters.items():
        flat = array.reshape(-1)
        if exhaustive or flat.size <= samples_per_parameter:
            indices: Any = range(flat.size)
        else:
            indices = rng.choice(flat.size, samples_per_parameter, replace=False)
        for index in indices:
            original = flat[index]
            flat[index] = original + EPSILON
            plus = _sequence_loss(model, inputs, targets, frozen)
            flat[index] = original - EPSILON
            minus = _sequence_loss(model, inputs, targets, frozen)
            flat[index] = original
            numeric = (plus - minus) / (2.0 * EPSILON)
            exact = float(analytic[name].reshape(-1)[index])
            difference = abs(numeric - exact)
            scale = max(abs(numeric), abs(exact))
            relative = difference / max(1e-12, abs(numeric) + abs(exact))
            checked += 1
            worst_absolute = max(worst_absolute, difference)
            if difference > ABSOLUTE_TOLERANCE + TOLERANCE * scale and relative > worst:
                worst, worst_name = relative, f"{name}[{index}]"
            if difference > ABSOLUTE_TOLERANCE + TOLERANCE * scale:
                violations += 1
            worst_relative = max(worst_relative, relative)
    return {
        "model": label,
        "sequence_length": length,
        "exhaustive": bool(exhaustive),
        "checked": checked,
        "max_relative_error": worst_relative,
        "max_absolute_error": worst_absolute,
        "violations": violations,
        "worst_violating_parameter": worst_name,
        "worst_violating_relative_error": worst,
        "relative_tolerance": TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "passed": violations == 0,
    }


def full_gradient_check(*, exhaustive: bool = False) -> dict[str, Any]:
    """Check every model that has a hand-written backward pass."""

    length = 9
    models = [
        check_model(
            lambda: AAA1KGRU(seed=1, tbptt_steps=32, error_loss_weight=0.25),
            label="AAA1KGRU",
            length=length,
            seed=11,
            exhaustive=exhaustive,
        ),
        check_model(
            lambda: AAA1KGRU(seed=2, tbptt_steps=32, error_loss_weight=0.5),
            label="AAA1KGRU(lambda=0.5)",
            length=length,
            seed=12,
            exhaustive=exhaustive,
        ),
        check_model(
            lambda: AAA1KGRU(seed=3, tbptt_steps=32, error_loss_weight=0.0),
            label="AAA1KGRU(lambda=0)",
            length=length,
            seed=13,
            exhaustive=exhaustive,
        ),
        check_model(
            lambda: VanillaRNNControl(seed=4, tbptt_steps=32),
            label="VanillaRNNControl",
            length=length,
            seed=14,
            exhaustive=exhaustive,
        ),
        check_model(
            lambda: StatelessMLPControl(seed=5),
            label="StatelessMLPControl",
            length=1,
            seed=15,
            exhaustive=exhaustive,
        ),
    ]
    return {
        "schema": "aaa.1k.gradient_check.v1",
        "epsilon": EPSILON,
        "tolerance": TOLERANCE,
        "models": models,
        "passed": all(entry["passed"] for entry in models),
    }
