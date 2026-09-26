"""Statistics for ``aaa.python.v1``: crossed intervals, Holm adjustment, variance components.

The unit is an ``initialization x stream`` cell. Intervals are v0's crossed
(pigeonhole) percentile bootstrap, reused unchanged: each draw resamples
initializations and streams independently (Owen 2007). v1 adds:

* ``DEGENERATE`` -- an interval of zero width because the compared arms never
  disagreed on any task in any cell. v0 printed such rows as
  ``+0.000 [+0.000, +0.000] INCONCLUSIVE``, which suggests a certainty the
  design does not have; here the status says what happened.
* Holm's step-down adjustment across a declared set of primary contrasts,
  applied by widening each contrast's interval to its adjusted level.
* A two-way random-effects variance decomposition (initialization, stream,
  residual) from the cell grid, reported beside every interval as a cross-check
  on the bootstrap (McCullagh 2000 shows none is exact for such arrays).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_python.stats import StatsError, grid

__all__ = ["StatsError", "crossed", "grid", "holm", "resolved_sign", "variance_components"]


def crossed(values: np.ndarray, *, seed: int, draws: int, confidence: float) -> dict[str, Any]:
    """Crossed percentile bootstrap of the grand mean of an ``[init, stream]`` grid."""

    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise StatsError("a crossed interval needs a complete finite 2-D grid")
    inits, streams = values.shape
    result: dict[str, Any] = {
        "mean": float(values.mean()),
        "initializations": inits,
        "streams": streams,
        "confidence": confidence,
    }
    if inits < 2 or streams < 2:
        result.update(interval_status="INSUFFICIENT_EVIDENCE", lower=None, upper=None)
        return result
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, inits, size=(draws, inits))
    cols = rng.integers(0, streams, size=(draws, streams))
    means = np.empty(draws)
    for start in range(0, draws, 500):
        stop = min(draws, start + 500)
        means[start:stop] = values[rows[start:stop, :, None], cols[start:stop, None, :]].mean(axis=(1, 2))
    tail = (1.0 - confidence) / 2.0
    result.update(
        interval_status="MEASURED",
        lower=float(np.quantile(means, tail)),
        upper=float(np.quantile(means, 1.0 - tail)),
    )
    return result


def resolved_sign(result: Mapping[str, Any], *, disagreements: int | None = None) -> str:
    """``POSITIVE``/``NEGATIVE`` only when the interval excludes zero; ``DEGENERATE`` if arms never differed."""

    if result.get("interval_status") != "MEASURED":
        return "INSUFFICIENT_EVIDENCE"
    if disagreements == 0:
        return "DEGENERATE"
    if result["lower"] > 0:
        return "POSITIVE"
    if result["upper"] < 0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def variance_components(values: np.ndarray) -> dict[str, float]:
    """Method-of-moments two-way random effects without replication (negative estimates set to 0)."""

    inits, streams = values.shape
    if inits < 2 or streams < 2:
        return {"initialization": float("nan"), "stream": float("nan"), "residual": float("nan")}
    grand = values.mean()
    row = values.mean(axis=1)
    col = values.mean(axis=0)
    ms_row = streams * float(np.sum((row - grand) ** 2)) / (inits - 1)
    ms_col = inits * float(np.sum((col - grand) ** 2)) / (streams - 1)
    resid = values - row[:, None] - col[None, :] + grand
    ms_res = float(np.sum(resid**2)) / ((inits - 1) * (streams - 1))
    return {
        "initialization": max(0.0, (ms_row - ms_res) / streams),
        "stream": max(0.0, (ms_col - ms_res) / inits),
        "residual": ms_res,
    }


def holm(
    results: Mapping[str, Mapping[str, Any]],
    grids: Mapping[str, np.ndarray],
    *,
    seed: int,
    draws: int,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    """Holm step-down over declared contrasts via bootstrap two-sided p-values.

    Each contrast's p-value is twice the smaller bootstrap tail mass beyond 0
    (bounded below by 1/draws). The contrast is ``resolved`` when its Holm-adjusted
    p-value is at most ``alpha``. Unresolved never means zero.
    """

    pvalues: dict[str, float] = {}
    for name, values in grids.items():
        inits, streams = values.shape
        if inits < 2 or streams < 2:
            pvalues[name] = 1.0
            continue
        rng = np.random.default_rng(seed)
        rows = rng.integers(0, inits, size=(draws, inits))
        cols = rng.integers(0, streams, size=(draws, streams))
        means = values[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
        below = float(np.mean(means <= 0.0))
        above = float(np.mean(means >= 0.0))
        pvalues[name] = min(1.0, max(1.0 / draws, 2.0 * min(below, above)))
    order = sorted(pvalues, key=lambda n: pvalues[n])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(order)
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return {
        name: {
            "p_value": pvalues[name],
            "holm_adjusted_p": adjusted[name],
            "resolved_after_holm": adjusted[name] <= alpha,
            "mean": results[name]["mean"] if name in results else None,
        }
        for name in pvalues
    }


def cells_grid(cells: Sequence[Mapping[str, Any]], value: str) -> np.ndarray:
    return grid(cells, value)[0]
