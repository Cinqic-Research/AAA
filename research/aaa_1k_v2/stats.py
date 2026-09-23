"""Paired statistics over the actual sampling structure of ``aaa.1k.v2``.

Every comparison is paired: two arms run the same streams from the same
initializations. The design is *crossed* -- every initialization meets every
stream -- so both factors are sources of uncertainty, and steps inside a
stream are never treated as independent draws.

``relative_crossed``
    ``mean(second) / mean(first) - 1`` over an ``[initialization, stream]``
    grid, with a percentile bootstrap that resamples initializations and
    streams independently.
``geometric_relative``
    across several families that share one set of initializations: each
    family's ratio of means, combined by a geometric mean. A bootstrap draw
    resamples the initializations *once* (they are shared) and the streams of
    each family separately; the families themselves are the declared suite
    and are not resampled.

Status rules (all three-valued; an interval crossing a bound is never a pass):

``noninferior``  upper <= margin -> PASS; lower > margin -> FAIL; else INCONCLUSIVE
``superior``     upper < bound   -> PASS; lower >= bound -> FAIL; else INCONCLUSIVE
``guard``        lower > margin  -> FAIL (a resolved regression); else PASS

``holm`` adjusts a family of exploratory p-values; confirmatory promotion uses
an intersection-union rule instead (every criterion must pass).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _check(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.size == 0:
        raise ValueError("crossed inputs must be matching non-empty [initialization, stream] grids")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("crossed inputs must be finite")
    return a, b


def relative_crossed(
    first: np.ndarray, second: np.ndarray, *, seed: int, draws: int = 4000, confidence: float = 0.95
) -> dict[str, Any]:
    a, b = _check(first, second)
    inits, streams = a.shape
    point = float(b.mean() / a.mean() - 1.0)
    out: dict[str, Any] = {
        "first_mean": float(a.mean()),
        "second_mean": float(b.mean()),
        "relative": point,
        "initializations": inits,
        "streams": streams,
        "per_initialization_relative": [float(v) for v in b.mean(axis=1) / a.mean(axis=1) - 1.0],
        "streams_favouring_second": int(np.sum(b.mean(axis=0) < a.mean(axis=0))),
        "draws": draws,
        "confidence": confidence,
    }
    if inits < 2 or streams < 2:
        out.update(interval_status="INSUFFICIENT_EVIDENCE", lower=None, upper=None)
        return out
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, inits, size=(draws, inits))
    cols = rng.integers(0, streams, size=(draws, streams))
    num = b[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
    den = a[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
    ratios = num / den - 1.0
    alpha = (1.0 - confidence) / 2.0
    out.update(
        interval_status="MEASURED",
        lower=float(np.quantile(ratios, alpha)),
        upper=float(np.quantile(ratios, 1.0 - alpha)),
    )
    return out


def geometric_relative(
    families: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    draws: int = 4000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Geometric mean over families of ``mean(second)/mean(first)``, with a shared-initialization bootstrap."""

    names = sorted(families)
    grids = {name: _check(*families[name]) for name in names}
    inits = {grid[0].shape[0] for grid in grids.values()}
    if len(inits) != 1:
        raise ValueError("families must share their initializations")
    n_init = inits.pop()
    ratios = {name: float(b.mean() / a.mean()) for name, (a, b) in grids.items()}
    point = float(math.exp(np.mean([math.log(r) for r in ratios.values()])))
    out: dict[str, Any] = {
        "families": len(names),
        "geometric_ratio": point,
        "family_ratios": ratios,
        "draws": draws,
        "confidence": confidence,
    }
    if n_init < 2 or any(a.shape[1] < 2 for a, _ in grids.values()):
        out.update(interval_status="INSUFFICIENT_EVIDENCE", lower=None, upper=None)
        return out
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n_init, size=(draws, n_init))
    logs = np.zeros(draws)
    for name in names:
        a, b = grids[name]
        cols = rng.integers(0, a.shape[1], size=(draws, a.shape[1]))
        num = b[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
        den = a[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
        logs += np.log(num / den)
    geometric = np.exp(logs / len(names))
    alpha = (1.0 - confidence) / 2.0
    out.update(
        interval_status="MEASURED",
        lower=float(np.quantile(geometric, alpha)),
        upper=float(np.quantile(geometric, 1.0 - alpha)),
    )
    return out


def noninferior(
    result: Mapping[str, Any], margin: float, *, key_upper: str = "upper", key_lower: str = "lower"
) -> str:
    if result.get("interval_status") != "MEASURED":
        return "INCONCLUSIVE"
    if result[key_upper] <= margin:
        return "PASS"
    if result[key_lower] > margin:
        return "FAIL"
    return "INCONCLUSIVE"


def superior(result: Mapping[str, Any], bound: float) -> str:
    if result.get("interval_status") != "MEASURED":
        return "INCONCLUSIVE"
    if result["upper"] < bound:
        return "PASS"
    if result["lower"] >= bound:
        return "FAIL"
    return "INCONCLUSIVE"


def guard(result: Mapping[str, Any], margin: float) -> str:
    if result.get("interval_status") != "MEASURED":
        return "INCONCLUSIVE"
    return "FAIL" if result["lower"] > margin else "PASS"


def combine(statuses: Sequence[str]) -> str:
    """Intersection-union: any FAIL -> FAIL; all PASS -> PASS; otherwise INCONCLUSIVE."""

    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if statuses and all(status == "PASS" for status in statuses):
        return "PASS"
    return "INCONCLUSIVE"


def holm(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    order = sorted(p_values, key=lambda key: p_values[key])
    m = len(order)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    for rank, key in enumerate(order):
        adjusted = min(1.0, max(running, (m - rank) * p_values[key]))
        running = adjusted
        out[key] = {"p": p_values[key], "p_holm": adjusted, "rejected": adjusted <= alpha}
    return out


def bootstrap_p_two_sided(first: np.ndarray, second: np.ndarray, *, seed: int, draws: int = 4000) -> float:
    """Crossed-bootstrap two-sided p-value for ``mean(second - first) == 0`` (centred percentile method)."""

    a, b = _check(first, second)
    d = b - a
    inits, streams = d.shape
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, inits, size=(draws, inits))
    cols = rng.integers(0, streams, size=(draws, streams))
    means = d[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
    centred = means - d.mean()
    return float(min(1.0, (1 + np.sum(np.abs(centred) >= abs(d.mean()))) / (draws + 1)))
