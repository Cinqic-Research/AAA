"""Externally defined synthetic benchmarks: NARMA-10, NARMA-20 and Mackey-Glass.

Each generator is a frozen definition taken from a cited source. Nothing here
was designed around AAA.

NARMA-10 (Atiya & Parlos 2000)
    ``y(t+1) = 0.3 y(t) + 0.05 y(t) sum_{i=0}^{9} y(t-i) + 1.5 u(t-9) u(t) + 0.1``
    with ``u(t)`` i.i.d. uniform on ``[0, 0.5]``.
NARMA-20 (Rodan & Tino 2011, with the tanh wrapper they introduced to stop divergence)
    ``y(t+1) = tanh(0.3 y(t) + 0.05 y(t) sum_{i=0}^{19} y(t-i) + 1.5 u(t-19) u(t) + 0.01)``.
    NARMA-30 is deliberately not used: the literature gives at least two
    incompatible parameter sets for it (Schrauwen et al. 2008 and later
    restatements), and one benchmark name must not cover two systems.
Mackey-Glass, tau = 17 (Mackey & Glass 1977; protocol of Jaeger 2010 as summarised by
the 2024 reservoir-benchmark review, arXiv:2405.06561)
    ``dx/dt = 0.2 x(t-17) / (1 + x(t-17)^10) - 0.1 x(t)``, forward Euler with
    ``dt = 0.1``, subsampled every 10 steps (unit sampling), transformed
    ``y = tanh(x - 1)``, washout 1000 samples. The constant initial history is
    ``x0`` drawn uniformly from ``[1.1, 1.3]`` per realization, so realizations
    differ only through their initial history.

NARMA streams are checked for divergence (a non-finite or ``|y| > 10`` value,
which NARMA-10 is known to produce occasionally). A diverged realization is
replaced by drawing again from the *next* sub-seed of the same identity; the
number of redraws is recorded. This is a property of the generator, decided
before any model output exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

NARMA_SPECS: dict[str, dict[str, Any]] = {
    "narma10": {
        "order": 10,
        "alpha": 0.3,
        "beta": 0.05,
        "gamma": 1.5,
        "delta": 0.1,
        "tanh": False,
        "citation": (
            "Atiya, A. F. & Parlos, A. G. (2000). New results on recurrent network training: unifying the "
            "algorithms and accelerating convergence. IEEE Transactions on Neural Networks 11(3), 697-709."
        ),
    },
    "narma20": {
        "order": 20,
        "alpha": 0.3,
        "beta": 0.05,
        "gamma": 1.5,
        "delta": 0.01,
        "tanh": True,
        "citation": (
            "Rodan, A. & Tino, P. (2011). Minimum complexity echo state network. IEEE Transactions on Neural "
            "Networks 22(1), 131-144 (NARMA-20 with tanh saturation)."
        ),
    },
}
NARMA_INPUT_RANGE = (0.0, 0.5)
NARMA_DIVERGENCE_BOUND = 10.0


@dataclass(frozen=True)
class SyntheticSeries:
    inputs: np.ndarray
    targets: np.ndarray
    metadata: dict[str, Any]


def narma(task: str, seed: int, length: int, *, washout: int = 200) -> SyntheticSeries:
    """``inputs[t] = u(t)``, ``targets[t] = y(t)`` (``y(t)`` depends on ``u`` up to ``t-1``)."""

    spec = NARMA_SPECS[task]
    order = int(spec["order"])
    redraws = 0
    while True:
        rng = np.random.default_rng([seed, redraws])
        total = length + washout + order
        u = rng.uniform(*NARMA_INPUT_RANGE, size=total)
        y = np.zeros(total)
        ok = True
        for t in range(order - 1, total - 1):
            window = y[t - order + 1 : t + 1]
            value = (
                spec["alpha"] * y[t]
                + spec["beta"] * y[t] * float(np.sum(window))
                + spec["gamma"] * u[t - order + 1] * u[t]
                + spec["delta"]
            )
            if spec["tanh"]:
                value = math.tanh(value)
            if not math.isfinite(value) or abs(value) > NARMA_DIVERGENCE_BOUND:
                ok = False
                break
            y[t + 1] = value
        if ok:
            break
        redraws += 1
        if redraws > 100:
            raise RuntimeError(f"{task} seed {seed} diverged 100 times")
    start = washout + order
    return SyntheticSeries(
        inputs=u[start:],
        targets=y[start:],
        metadata={
            "task": task,
            "seed": seed,
            "redraws": redraws,
            "washout": washout,
            **{k: spec[k] for k in ("order", "alpha", "beta", "gamma", "delta", "tanh")},
        },
    )


MACKEY_GLASS: dict[str, Any] = {
    "tau": 17.0,
    "beta": 0.2,
    "gamma": 0.1,
    "n": 10.0,
    "euler_dt": 0.1,
    "subsample": 10,
    "washout": 1000,
    "transform": "tanh(x - 1)",
    "initial_history": [1.1, 1.3],
    "citation": (
        "Mackey, M. C. & Glass, L. (1977). Oscillation and chaos in physiological control systems. Science "
        "197(4300), 287-289; protocol per Jaeger (2010) as summarised in Wringe et al. (2024), Reservoir "
        "computing benchmarks: a review, a taxonomy, some best practices, arXiv:2405.06561."
    ),
}


def mackey_glass(seed: int, length: int) -> SyntheticSeries:
    """``targets[t] = y(t+1)`` and ``inputs[t] = y(t)``: one-step-ahead forecasting."""

    p = MACKEY_GLASS
    rng = np.random.default_rng(seed)
    x0 = float(rng.uniform(*p["initial_history"]))
    delay = round(p["tau"] / p["euler_dt"])
    steps = (length + 1 + p["washout"]) * p["subsample"]
    x = np.empty(steps + delay + 1)
    x[: delay + 1] = x0
    dt = p["euler_dt"]
    for k in range(delay, steps + delay):
        lagged = x[k - delay]
        x[k + 1] = x[k] + dt * (p["beta"] * lagged / (1.0 + lagged ** p["n"]) - p["gamma"] * x[k])
    sampled = x[delay :: p["subsample"]][p["washout"] : p["washout"] + length + 1]
    y = np.tanh(sampled - 1.0)
    return SyntheticSeries(
        inputs=y[:-1],
        targets=y[1:],
        metadata={
            "task": "mackey_glass17",
            "seed": seed,
            "x0": x0,
            **{k: v for k, v in p.items() if k != "citation"},
        },
    )
