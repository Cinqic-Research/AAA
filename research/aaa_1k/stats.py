"""Paired statistics and error-head calibration for AAA-1K.

Two rules decide everything here.

**The resampling unit is the stream.** Steps inside one stream are not
independent draws -- they are one trajectory -- so resampling steps would
manufacture precision out of autocorrelation. Every interval in this module
resamples whole streams.

**Pairing is preserved.** Arms are compared on the same stream realizations, so
a bootstrap draw selects a stream and takes every arm's value from it. Breaking
that would throw away the pairing the design exists to create.

Intervals are percentile bootstrap intervals on the mean paired difference,
with the number of draws and the confidence level declared before anything is
observed. There is no hypothesis-test theatre: for a handful of exploratory
streams a p-value would be decoration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .seeds import derive_seed

BOOTSTRAP_DRAWS = 10_000
CONFIDENCE_LEVEL = 0.95


def paired_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    confidence: float = CONFIDENCE_LEVEL,
    bootstrap_index: int = 0,
) -> dict[str, Any]:
    """Summarize ``second - first`` over paired per-stream values.

    A positive mean means ``first`` had the lower value, which for error
    metrics means ``first`` was better. Both the absolute and the relative
    effect are reported, along with the per-stream sign count, because a mean
    that hides a four-to-one split is not a summary.
    """

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired inputs must be one-dimensional and the same length")
    if a.size == 0:
        raise ValueError("paired inputs must be non-empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("paired inputs must be finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if draws < 1:
        raise ValueError("draws must be positive")

    differences = b - a
    mean = float(np.mean(differences))
    rng = np.random.default_rng(derive_seed("bootstrap", bootstrap_index))
    if a.size == 1:
        low = high = float("nan")
        interval_status = "INSUFFICIENT_EVIDENCE"
    else:
        indices = rng.integers(0, a.size, size=(draws, a.size))
        resampled = np.mean(differences[indices], axis=1)
        alpha = (1.0 - confidence) / 2.0
        low = float(np.quantile(resampled, alpha))
        high = float(np.quantile(resampled, 1.0 - alpha))
        interval_status = "MEASURED"
    baseline = float(np.mean(b))
    return {
        "streams": int(a.size),
        "first_mean": float(np.mean(a)),
        "second_mean": baseline,
        "per_stream_difference": [float(value) for value in differences],
        "mean_difference": mean,
        "median_difference": float(np.median(differences)),
        "relative_difference": mean / baseline if baseline != 0 else float("nan"),
        "confidence": confidence,
        "draws": int(draws),
        "interval_status": interval_status,
        "ci_low": low,
        "ci_high": high,
        "favours_first": int(np.sum(differences > 0)),
        "favours_second": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
    }


def crossed_paired_difference(
    first: ArrayLike,
    second: ArrayLike,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    confidence: float = CONFIDENCE_LEVEL,
    bootstrap_index: int = 1,
) -> dict[str, Any]:
    """Summarize ``second - first`` over a crossed initialization-by-stream design.

    Every arm runs on every stream from every initialization, so the design is
    crossed rather than nested, and *both* factors are sources of uncertainty.
    Resampling only streams -- which is what a single-initialization interval
    does -- treats the model's starting weights as if they were fixed by
    nature, and reports an interval narrower than the evidence supports.

    Each bootstrap draw therefore resamples initializations and streams
    independently with replacement and recomputes the mean over the selected
    cells. The per-initialization means are reported alongside, because a mean
    difference that hides one initialization disagreeing with four is not a
    summary.
    """

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("crossed inputs must be two-dimensional [initializations, streams]")
    if a.size == 0:
        raise ValueError("crossed inputs must be non-empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("crossed inputs must be finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")

    differences = b - a
    initializations, streams = differences.shape
    mean = float(np.mean(differences))
    per_initialization = [float(value) for value in np.mean(differences, axis=1)]
    per_stream = [float(value) for value in np.mean(differences, axis=0)]

    if initializations < 2 or streams < 2:
        low = high = float("nan")
        status = "INSUFFICIENT_EVIDENCE"
    else:
        rng = np.random.default_rng(derive_seed("bootstrap", bootstrap_index))
        rows = rng.integers(0, initializations, size=(draws, initializations))
        columns = rng.integers(0, streams, size=(draws, streams))
        resampled = np.empty(draws, dtype=float)
        for index in range(draws):
            resampled[index] = differences[np.ix_(rows[index], columns[index])].mean()
        alpha = (1.0 - confidence) / 2.0
        low = float(np.quantile(resampled, alpha))
        high = float(np.quantile(resampled, 1.0 - alpha))
        status = "MEASURED"

    baseline = float(np.mean(b))
    signs = {1 if value > 0 else (-1 if value < 0 else 0) for value in per_initialization}
    return {
        "design": "crossed initializations x streams",
        "initializations": int(initializations),
        "streams": int(streams),
        "first_mean": float(np.mean(a)),
        "second_mean": baseline,
        "mean_difference": mean,
        "median_difference": float(np.median(differences)),
        "relative_difference": mean / baseline if baseline != 0 else float("nan"),
        "per_initialization_difference": per_initialization,
        "per_stream_difference": per_stream,
        "confidence": confidence,
        "draws": int(draws),
        "interval_status": status,
        "ci_low": low,
        "ci_high": high,
        "favours_first": int(np.sum(np.asarray(per_stream) > 0)),
        "favours_second": int(np.sum(np.asarray(per_stream) < 0)),
        "ties": int(np.sum(np.asarray(per_stream) == 0)),
        "initializations_agreeing_on_sign": len(signs) == 1,
    }


def achieved_precision(record: Mapping[str, Any]) -> dict[str, Any]:
    """Interval half-width against the effect actually measured.

    A precision target sized from a pilot is a guess about an effect nobody had
    seen yet. What a reader needs is how sharply the effect that *was* measured
    is resolved, so that is reported for every headline comparison.
    """

    if record.get("interval_status") != "MEASURED":
        return {"status": record.get("interval_status", "NOT_VERIFIED")}
    half_width = (float(record["ci_high"]) - float(record["ci_low"])) / 2.0
    effect = abs(float(record["mean_difference"]))
    ratio = half_width / effect if effect > 0 else float("inf")
    return {
        "status": "MEASURED",
        "effect": float(record["mean_difference"]),
        "half_width": half_width,
        "half_width_over_effect": ratio,
        "resolved_to_fraction_of_effect": ratio,
        "meets_quarter_effect_target": bool(ratio <= 0.25),
    }


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # average ties, so a constant prediction does not produce a spurious ordering
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(unique.size, dtype=float)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return float("nan")
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def calibration(predicted: ArrayLike, realized: ArrayLike, *, bins: int = 5) -> dict[str, Any]:
    """How informative the error head is about the error it will actually make.

    ``predicted`` is the model's error-magnitude estimate for a step and
    ``realized`` is the magnitude it then made, both in the same normalized
    displacement units. Reported separately from prediction accuracy, because
    a model can predict badly and know it, or predict well and not know it, and
    those are different capabilities.

    ``slope`` and ``intercept`` come from regressing realized on predicted, so
    a perfectly calibrated head would give slope 1 and intercept 0. Rank
    correlation is reported alongside, because a head that orders its own
    errors correctly but is miscalibrated in scale is still useful.
    """

    p = np.asarray(predicted, dtype=float)
    r = np.asarray(realized, dtype=float)
    if p.shape != r.shape or p.ndim != 1:
        raise ValueError("calibration inputs must be one-dimensional and the same length")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    mask = np.isfinite(p) & np.isfinite(r)
    p, r = p[mask], r[mask]
    if p.size < bins:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "samples": int(p.size),
            "reason": f"fewer finite pairs ({int(p.size)}) than requested bins ({bins})",
        }

    pearson = _correlation(p, r)
    spearman = _correlation(_rank(p), _rank(r))
    if np.std(p) > 0:
        slope, intercept = np.polyfit(p, r, 1)
    else:
        slope, intercept = float("nan"), float("nan")

    edges = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
    table: list[dict[str, float]] = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        selected = (p >= low) & (p <= high) if index == bins - 1 else (p >= low) & (p < high)
        if not np.any(selected):
            continue
        table.append(
            {
                "bin": index,
                "predicted_low": float(low),
                "predicted_high": float(high),
                "count": int(np.sum(selected)),
                "mean_predicted": float(np.mean(p[selected])),
                "mean_realized": float(np.mean(r[selected])),
            }
        )
    monotone = all(
        table[index]["mean_realized"] <= table[index + 1]["mean_realized"] for index in range(len(table) - 1)
    )
    return {
        "status": "MEASURED",
        "samples": int(p.size),
        "mean_predicted": float(np.mean(p)),
        "mean_realized": float(np.mean(r)),
        "bias": float(np.mean(p) - np.mean(r)),
        "pearson": pearson,
        "spearman": spearman,
        "slope": float(slope),
        "intercept": float(intercept),
        "bin_table": table,
        "bins_monotone": bool(monotone),
    }


def capability_vector(entries: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap per-dimension results without collapsing them into one number.

    There is deliberately no weighted total. A single "autonomy score" is the
    most efficient way to hide a failure inside a success, and this phase has
    no business producing one.
    """

    return {
        "schema": "aaa.1k.capability_vector.v1",
        "note": "dimensions are reported separately and are not combined into a single score",
        "dimensions": dict(entries),
    }


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be positive")
    if array.size == 0:
        return []
    out = []
    for index in range(array.size):
        start = max(0, index - window + 1)
        out.append(float(np.mean(array[start : index + 1])))
    return out


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None
