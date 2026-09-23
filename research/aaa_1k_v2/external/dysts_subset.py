"""A bounded, rule-selected subset of the ``dysts`` chaotic-systems benchmark.

Source: W. Gilpin, *Chaos as an interpretable benchmark for forecasting and
data-driven modelling*, NeurIPS Datasets & Benchmarks 2021;
https://github.com/williamgilpin/dysts, version 0.96 (Apache License 2.0).
``data/dysts_0.96_chaotic_attractors.json`` is dysts' own metadata file,
vendored unchanged (sha256 recorded in :data:`METADATA_SHA256`). The
right-hand sides below are transcribed from ``dysts/flows.py`` 0.96 with the
published parameter values; copyright in those equations belongs to their
original authors and to the dysts project, used here under Apache-2.0.

Selection rule (declared before any AAA model saw any dysts data)
----------------------------------------------------------------
1. Eligible: autonomous, not a delay equation, embedding dimension 3 or 4, no
   unbounded coordinates, and both a dominant period and a positive estimated
   maximum Lyapunov exponent in the metadata (106 of 135 systems).
2. Rank by ``lambda_max * period`` (Lyapunov exponents per dominant period,
   i.e. how chaotic one period of the 100-point-per-period sampling is), break
   ties by name, and split into quartiles.
3. From each quartile draw three systems without replacement with
   ``numpy.random.default_rng(derive_seed("development",
   "external.dysts-selection", 0))``.

The draw gave: Hadley, SprottD, SprottK | KawczynskiStrizhak, SprottE,
ZhouChen | Chen, Halvorsen, SprottJerk | Colpitts, HastingsPowell, SprottMore.
All twelve happen to be three-dimensional; that is a property of the draw and
is reported, not repaired.

Protocol and declared deviations from dysts
-------------------------------------------
* Sampling follows dysts' ``make_trajectory(resample=True,
  pts_per_period=100)``: 100 samples per dominant period.
* Integration uses fixed-step classical RK4 at ``h = period / 100 / k`` with the
  smallest integer ``k`` making ``h`` no larger than the metadata ``dt``,
  instead of SciPy's Radau at ``rtol = atol = 1e-12`` (SciPy is not an AAA
  dependency). A chaotic trajectory from different integrators diverges after
  a few Lyapunov times, so the *attractor* is what is reproduced;
  ``docs/evidence/aaa_1k_v2/dysts_validation.json`` compares short-horizon
  agreement and attractor statistics against dysts itself.
* Each realization starts from the metadata initial condition plus a seeded
  perturbation of 1% of each coordinate's attractor standard deviation, then
  discards 20 periods of transient.
* The observed scalar is coordinate 0 (dysts' univariate convention),
  standardized with dysts' published attractor mean and standard deviation for
  that coordinate -- public constants, never statistics of the scored data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..identities import derive_seed

METADATA_PATH = Path(__file__).with_name("data") / "dysts_0.96_chaotic_attractors.json"
METADATA_SHA256 = "f750c02d69c18d76b3a82f00673db0a410456ae3fec4b4a726a4fdcb46f73174"
PTS_PER_PERIOD = 100
TRANSIENT_PERIODS = 20
PERTURBATION = 0.01

Vector = tuple[float, float, float]


def load_metadata() -> dict[str, Any]:
    raw = METADATA_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != METADATA_SHA256:
        raise RuntimeError("vendored dysts metadata does not match its recorded checksum")
    return dict(json.loads(raw))


def eligible(entry: dict[str, Any]) -> bool:
    return bool(
        not entry.get("nonautonomous")
        and not entry.get("delay")
        and entry.get("embedding_dimension") in (3, 4)
        and not entry.get("unbounded_indices")
        and entry.get("period")
        and (entry.get("maximum_lyapunov_estimated") or 0) > 0
    )


def select_systems(metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Apply the declared selection rule. Deterministic."""

    data = metadata or load_metadata()
    names = sorted(name for name, entry in data.items() if eligible(entry))

    def chaos(name: str) -> float:
        return float(data[name]["period"]) * float(data[name]["maximum_lyapunov_estimated"])

    order = sorted(names, key=lambda name: (chaos(name), name))
    rng = np.random.default_rng(derive_seed("development", "external.dysts-selection", 0))
    picked = []
    for quartile, part in enumerate(np.array_split(np.asarray(order), 4)):
        for name in sorted(rng.choice(part, size=3, replace=False).tolist()):
            picked.append(
                {
                    "system": name,
                    "quartile": quartile,
                    "lyapunov_per_period": chaos(name),
                    "dimension": int(data[name]["embedding_dimension"]),
                    "period": float(data[name]["period"]),
                    "maximum_lyapunov_estimated": float(data[name]["maximum_lyapunov_estimated"]),
                }
            )
    return picked


# ----------------------------------------------------------------------
# right-hand sides (dysts 0.96 flows.py), f(state, parameters) -> derivative
# ----------------------------------------------------------------------
def _hadley(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    return (
        -(y**2) - z**2 - p["a"] * x + p["a"] * p["f"],
        x * y - p["b"] * x * z - y + p["g"],
        p["b"] * x * y + x * z - z,
    )


def _sprott_d(s: Vector, _p: dict[str, float]) -> Vector:
    x, y, z = s
    return (-y, x + z, x * z + 3 * y**2)


def _sprott_k(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    return (x * y - z, x - y, x + p["a"] * z)


def _kawczynski_strizhak(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    g, mu = p["gamma"], p["mu"]
    return (
        g * y - g * x**3 + 3 * mu * g * x,
        -2 * mu * x - y - z + p["beta"],
        p["kappa"] * x - p["kappa"] * z,
    )


def _sprott_e(s: Vector, _p: dict[str, float]) -> Vector:
    x, y, z = s
    return (y * z, x**2 - y, 1 - 4 * x)


def _zhou_chen(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    return (p["a"] * x + p["b"] * y + y * z, p["c"] * y - x * z + p["d"] * y * z, p["e"] * z - x * y)


def _chen(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    a, b, c = p["a"], p["b"], p["c"]
    return (a * y - a * x, (c - a) * x - x * z + c * y, x * y - b * z)


def _halvorsen(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    a, b = p["a"], p["b"]
    return (-a * x - b * y - b * z - y**2, -a * y - b * z - b * x - z**2, -a * z - b * x - b * y - x**2)


def _sprott_jerk(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    return (y, z, -x + y**2 - p["mu"] * z)


def _colpitts(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    u = z - (p["e"] - 1)
    fz = -u * (1 - (1.0 if u > 0 else 0.0))  # np.heaviside(u, 0)
    return (y - p["a"] * fz, p["c"] - x - p["b"] * y - z, y - p["d"] * z)


def _hastings_powell(s: Vector, p: dict[str, float]) -> Vector:
    x, y, z = s
    f1 = p["a1"] * x / (1 + p["b1"] * x)
    f2 = p["a2"] * y / (1 + p["b2"] * y)
    return (x * (1 - x) - y * f1, y * f1 - z * f2 - p["d1"] * y, z * f2 - p["d2"] * z)


def _sprott_more(s: Vector, _p: dict[str, float]) -> Vector:
    x, y, z = s
    sign = 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)
    return (y, -x - sign * y, y**2 - math.exp(-(x**2)))


RHS: dict[str, Callable[[Vector, dict[str, float]], Vector]] = {
    "Hadley": _hadley,
    "SprottD": _sprott_d,
    "SprottK": _sprott_k,
    "KawczynskiStrizhak": _kawczynski_strizhak,
    "SprottE": _sprott_e,
    "ZhouChen": _zhou_chen,
    "Chen": _chen,
    "Halvorsen": _halvorsen,
    "SprottJerk": _sprott_jerk,
    "Colpitts": _colpitts,
    "HastingsPowell": _hastings_powell,
    "SprottMore": _sprott_more,
}


def _rk4(
    f: Callable[[Vector, dict[str, float]], Vector], state: Vector, p: dict[str, float], h: float
) -> Vector:
    def add(a: Vector, b: Vector, scale: float) -> Vector:
        return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])

    k1 = f(state, p)
    k2 = f(add(state, k1, h / 2), p)
    k3 = f(add(state, k2, h / 2), p)
    k4 = f(add(state, k3, h), p)
    return tuple(state[i] + h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class Trajectory:
    system: str
    observed: np.ndarray
    raw: np.ndarray
    metadata: dict[str, Any]


def integrate(
    system: str,
    samples: int,
    *,
    seed: int,
    metadata: dict[str, Any] | None = None,
    transient_periods: int = TRANSIENT_PERIODS,
    initial_condition: np.ndarray | None = None,
) -> Trajectory:
    """``samples`` points at 100 per period after a transient; coordinate 0 standardized."""

    data = (metadata or load_metadata())[system]
    params = {k: float(v) for k, v in data["parameters"].items()}
    period, dt_max = float(data["period"]), float(data["dt"])
    interval = period / PTS_PER_PERIOD
    substeps = max(1, math.ceil(interval / dt_max))
    h = interval / substeps
    rng = np.random.default_rng(seed)
    std = np.asarray(data["std"], dtype=float)
    ic = (
        np.asarray(data["initial_conditions"], dtype=float) + PERTURBATION * std * rng.standard_normal(3)
        if initial_condition is None
        else np.asarray(initial_condition, dtype=float)
    )
    state: Vector = (float(ic[0]), float(ic[1]), float(ic[2]))
    f = RHS[system]
    skip = transient_periods * PTS_PER_PERIOD
    out = np.empty((samples, 3))
    if skip == 0:
        out[0] = state
    for index in range(1 if skip == 0 else 0, skip + samples):
        for _ in range(substeps):
            state = _rk4(f, state, params, h)
        if not all(math.isfinite(v) for v in state):
            raise FloatingPointError(f"{system} integration diverged")
        if index >= skip:
            out[index - skip] = state
    mean0, std0 = float(data["mean"][0]), float(data["std"][0])
    return Trajectory(
        system=system,
        observed=(out[:, 0] - mean0) / std0,
        raw=out,
        metadata={
            "system": system,
            "seed": seed,
            "parameters": params,
            "period": period,
            "sampling_interval": interval,
            "rk4_substeps": substeps,
            "rk4_step": h,
            "transient_periods": transient_periods,
            "standardization": {"mean": mean0, "std": std0, "source": "dysts metadata (public constant)"},
            "citation": data.get("citation"),
        },
    )
