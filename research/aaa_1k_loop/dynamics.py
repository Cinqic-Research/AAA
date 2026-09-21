"""Read-only recurrent-dynamics instrumentation: the one-step state Jacobian.

For the gated core (keep-gate convention ``h = z*h_prev + (1-z)*n``)::

    dz/dh = diag(z(1-z)) U_z
    dr/dh = diag(r(1-r)) U_r
    dn/dh = diag(1-n^2) U_n (diag(r) + diag(h_prev) dr/dh)
    J     = diag(z) + diag(h_prev - n) dz/dh + diag(1-z) dn/dh

For the ungated control ``h = tanh(W x + U h_prev + b)``::

    J = diag(1-h^2) U

Both are evaluated at the realized activations the model stored for its most
recent step, so the measurement describes the dynamics the model actually ran,
and nothing is written back. The eigenvalues say which temporal patterns the
state can carry: all eigenvalues near +0.5 means every perturbation decays
within a few steps without changing sign; an eigenvalue near -1 is a
sign-alternating (period-2) mode; complex pairs near the unit circle are
longer oscillations.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from research.aaa_1k.controls import VanillaRNNControl
from research.aaa_1k.model import AAA1KGRU


def gru_jacobian(model: AAA1KGRU) -> np.ndarray | None:
    if not model._caches:
        return None
    cache = model._caches[-1]
    p = model.parameters
    z, r, n, h_prev = cache.z, cache.r, cache.n, cache.h_prev
    dz = (z * (1.0 - z))[:, None] * p["U_z"]
    dr = (r * (1.0 - r))[:, None] * p["U_r"]
    inner = np.diag(r) + h_prev[:, None] * dr
    dn = (1.0 - n * n)[:, None] * (p["U_n"] @ inner)
    return np.diag(z) + (h_prev - n)[:, None] * dz + (1.0 - z)[:, None] * dn


def rnn_jacobian(model: VanillaRNNControl) -> np.ndarray | None:
    if not model._caches:
        return None
    cache = model._caches[-1]
    return (1.0 - cache.h * cache.h)[:, None] * model.parameters["U"]


def jacobian_statistics(model: Any) -> dict[str, float]:
    """Spectral summary of the most recent one-step Jacobian, or NaNs before any step."""

    if isinstance(model, AAA1KGRU):
        jacobian = gru_jacobian(model)
    elif isinstance(model, VanillaRNNControl):
        jacobian = rnn_jacobian(model)
    else:
        jacobian = None
    if jacobian is None:
        return {
            "jacobian_spectral_radius": math.nan,
            "jacobian_min_real": math.nan,
            "jacobian_negative_mode": math.nan,
        }
    eigenvalues = np.linalg.eigvals(jacobian)
    negative = eigenvalues[eigenvalues.real < 0]
    return {
        "jacobian_spectral_radius": float(np.max(np.abs(eigenvalues))),
        "jacobian_min_real": float(np.min(eigenvalues.real)),
        # magnitude of the strongest sign-alternating mode; 0 when there is none
        "jacobian_negative_mode": float(np.max(np.abs(negative))) if negative.size else 0.0,
    }
