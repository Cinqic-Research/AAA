"""Metrics of the external benchmarks, each implemented as its source defines it.

``nmse``
    ``mean((yhat - y)^2) / var(y)``, the normalized mean squared error used for
    NARMA (e.g. Rodan & Tino 2011); ``var`` is the population variance of the
    scored targets.
``nrmse``
    ``sqrt(nmse)``.
``smape_monash``
    ``mean(2|F - Y| / (|F| + |Y|))`` per series, exactly as
    ``calculate_smape`` in the Monash archive's ``utils/error_calculator.R``
    (a fraction, not a percentage).
``mase_monash``
    ``mean|F - Y| / mean|Y_t - Y_{t-S}|`` with the in-sample seasonal-naive
    scale at lag ``min(seasonality)``, falling back to lag 1 when that scale is
    undefined, and dropping infinite or undefined per-series values -- a line
    by line port of ``calculate_mase`` in the same file.
``smape_percent``
    ``100 * mean(2|F - Y| / (|F| + |Y|))``, the percentage form reported by
    dysts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def nmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    variance = float(np.var(target))
    if variance <= 0 or prediction.shape != target.shape or target.size == 0:
        return float("nan")
    return float(np.mean((prediction - target) ** 2) / variance)


def nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    value = nmse(prediction, target)
    return math.sqrt(value) if math.isfinite(value) else float("nan")


def smape_monash(forecast: np.ndarray, actual: np.ndarray) -> float:
    forecast, actual = np.asarray(forecast, float), np.asarray(actual, float)
    denominator = np.abs(forecast) + np.abs(actual)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 2.0 * np.abs(forecast - actual) / denominator
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def smape_percent(forecast: np.ndarray, actual: np.ndarray) -> float:
    return 100.0 * smape_monash(forecast, actual)


def _seasonal_scale(training: np.ndarray, lag: int) -> float:
    if training.size <= lag:
        return float("nan")
    return float(np.mean(np.abs(training[lag:] - training[:-lag])))


def mase_monash_series(
    forecast: np.ndarray, actual: np.ndarray, training: np.ndarray, seasonality: Sequence[float]
) -> float:
    """One series' MASE with R's semantics: ``x/0 = Inf`` (later dropped), ``0/0`` or NA retries at lag 1."""

    forecast, actual, training = (np.asarray(v, float) for v in (forecast, actual, training))
    error = float(np.mean(np.abs(forecast - actual)))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(np.float64(error) / np.float64(_seasonal_scale(training, int(min(seasonality)))))
        if math.isnan(value):
            value = float(np.float64(error) / np.float64(_seasonal_scale(training, 1)))
    return value


def mase_monash(
    forecasts: Sequence[np.ndarray],
    actuals: Sequence[np.ndarray],
    trainings: Sequence[np.ndarray],
    seasonality: Sequence[float],
) -> dict[str, float | int]:
    values = [
        mase_monash_series(f, a, t, seasonality)
        for f, a, t in zip(forecasts, actuals, trainings, strict=True)
    ]
    kept = np.asarray([v for v in values if math.isfinite(v)], float)
    return {
        "mean": float(np.mean(kept)) if kept.size else float("nan"),
        "median": float(np.median(kept)) if kept.size else float("nan"),
        "series_scored": int(kept.size),
        "series_dropped": int(len(values) - kept.size),
    }
