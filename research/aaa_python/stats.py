"""Crossed-design statistics for ``aaa.python.v0``.

The experimental unit is an ``initialization x stream`` cell. A stream is a
declared sequence of tasks; tasks inside it, and tokens inside a task, are
never resampled as if they were independent. A bootstrap draw resamples
initializations and streams independently and recomputes the mean over the
selected cells. Cells must form a complete grid; a ragged or non-finite grid
is refused rather than flattened.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class StatsError(ValueError):
    pass


def grid(cells: Sequence[Mapping[str, Any]], value: str) -> tuple[np.ndarray, list[Any], list[Any]]:
    """``[init, stream]`` grid of ``value``; refuses duplicates, gaps and non-finite values."""

    inits = sorted({c["init"] for c in cells})
    streams = sorted({c["stream"] for c in cells})
    out = np.full((len(inits), len(streams)), np.nan)
    seen = set()
    for cell in cells:
        key = (cell["init"], cell["stream"])
        if key in seen:
            raise StatsError(f"duplicate cell {key}")
        seen.add(key)
        number = cell[value]
        if isinstance(number, bool) or not isinstance(number, int | float) or not math.isfinite(number):
            raise StatsError(f"cell {key}: {value} is not a finite number")
        out[inits.index(cell["init"]), streams.index(cell["stream"])] = float(number)
    if np.isnan(out).any():
        raise StatsError("the initialization x stream grid is incomplete")
    return out, inits, streams


def crossed_mean(values: np.ndarray, *, seed: int, draws: int, confidence: float) -> dict[str, Any]:
    inits, streams = values.shape
    result: dict[str, Any] = {
        "mean": float(values.mean()),
        "initializations": inits,
        "streams": streams,
        "per_initialization": [float(v) for v in values.mean(axis=1)],
    }
    if inits < 2 or streams < 2:
        result.update(interval_status="INSUFFICIENT_EVIDENCE", lower=None, upper=None)
        return result
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, inits, size=(draws, inits))
    cols = rng.integers(0, streams, size=(draws, streams))
    means = values[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
    tail = (1.0 - confidence) / 2.0
    result.update(
        interval_status="MEASURED",
        lower=float(np.quantile(means, tail)),
        upper=float(np.quantile(means, 1.0 - tail)),
    )
    return result


def resolved_sign(result: Mapping[str, Any]) -> str:
    """``POSITIVE`` / ``NEGATIVE`` only when the interval excludes zero; never from a point alone."""

    if result.get("interval_status") != "MEASURED":
        return "INSUFFICIENT_EVIDENCE"
    if result["lower"] > 0:
        return "POSITIVE"
    if result["upper"] < 0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def calibration(confidences: Sequence[float], correct: Sequence[bool], bins: int) -> dict[str, float]:
    """Brier score of the reported confidence and expected calibration error over equal-width bins."""

    p = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if p.size == 0:
        return {"brier": math.nan, "ece": math.nan}
    brier = float(np.mean((p - y) ** 2))
    edges = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = edges == b
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"brier": brier, "ece": ece}


def balanced_accuracy(answers: Sequence[Any], truths: Sequence[Any]) -> float:
    classes = sorted({repr(t) for t in truths})
    recalls = []
    for label in classes:
        rows = [a == t for a, t in zip(answers, truths, strict=True) if repr(t) == label]
        recalls.append(sum(rows) / len(rows))
    return float(np.mean(recalls)) if recalls else math.nan
