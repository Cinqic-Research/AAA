"""External benchmark tasks as online problems for the AAA interface, plus their baselines.

The benchmark owns truth; an adapter turns it into causally available inputs:

``external benchmark -> causal observation adapter -> generic AAA interface -> model``

A :class:`SeriesProblem` is one online pass: at step ``t`` a cell sees input
``s_t``, predicts target ``y_t``, and only then is ``y_t`` revealed. For
forecasting tasks ``y_t = s_{t+1}`` (``delta`` target: the model predicts the
change); for the NARMA input/output systems ``s_t = u(t)`` and ``y_t = y(t)``
(``absolute`` target). Optionally a recursive forecast of ``h`` further values
follows the online pass, with weights frozen and nothing revealed.

Normalization never sees scored data:

* NARMA, Mackey-Glass, dysts: constants from *calibration* realizations drawn
  from the development identity namespace ``external.calibration``, disjoint
  from every scored realization;
* Monash: each series' own warm-up window (the first ``max(50, 10%)`` training
  observations); prequential scoring starts after it, so every scaled value
  a scored step uses was already in the past.

Tasks
-----
``narma10``, ``narma20``           prequential NMSE over the last third of 6000 steps
``mackey_glass17``                 prequential NRMSE plus the 84-step recursive error (NRMSE84)
``dysts:<system>``                 prequential NRMSE plus a one-period (100-step) recursive forecast
``monash:<dataset>``               recursive forecast of the archive's test horizon, Monash MASE
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..identities import derive_seed
from . import dysts_subset, monash, synthetic
from .metrics import mase_monash_series, nmse, nrmse, smape_monash, smape_percent

NARMA_LENGTH = 6000
NARMA_SCORE_FROM = 4000
MG_LENGTH = 4000
MG_SCORE_FROM = 2000
MG_HORIZON = 84
DYSTS_LENGTH = 3000
DYSTS_SCORE_FROM = 2000
DYSTS_HORIZON = 100
CALIBRATION_REALIZATIONS = 4


@dataclass(frozen=True)
class SeriesProblem:
    task: str
    series_id: str
    inputs: np.ndarray
    targets: np.ndarray
    target_mode: str
    scales: dict[str, float]
    score_from: int
    forecast_truth: np.ndarray | None = None
    last_value: float | None = None
    training: np.ndarray | None = None
    seasonality: tuple[float, ...] | None = None
    raw_transform: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.inputs.size)


def calibration_namespace(task: str) -> str:
    return "calibration." + task.replace(":", "-")


def _calibration_seed(task: str, index: int) -> int:
    """Registered development identity ``calibration.<task>``: never a scored realization."""

    return derive_seed("development", calibration_namespace(task), index)


# ----------------------------------------------------------------------
# NARMA
# ----------------------------------------------------------------------
def narma_scales(task: str) -> dict[str, float]:
    low, high = synthetic.NARMA_INPUT_RANGE
    input_std = (high - low) / math.sqrt(12.0)
    targets = np.concatenate(
        [
            synthetic.narma(task, _calibration_seed(task, i), 3000).targets
            for i in range(CALIBRATION_REALIZATIONS)
        ]
    )
    return {
        "center": (low + high) / 2.0,
        "level_scale": input_std,
        "delta_scale": math.sqrt(2.0) * input_std,
        "target_center": float(np.mean(targets)),
        "target_scale": float(np.std(targets)),
    }


def narma_problems(task: str, seeds: Sequence[int]) -> list[SeriesProblem]:
    scales = narma_scales(task)
    out = []
    for seed in seeds:
        series = synthetic.narma(task, seed, NARMA_LENGTH)
        out.append(
            SeriesProblem(
                task=task,
                series_id=f"{task}:{seed}",
                inputs=series.inputs,
                targets=series.targets,
                target_mode="absolute",
                scales=scales,
                score_from=NARMA_SCORE_FROM,
                metadata=series.metadata,
            )
        )
    return out


# ----------------------------------------------------------------------
# Mackey-Glass
# ----------------------------------------------------------------------
def _delta_scales(
    values: np.ndarray, center: float | None = None, level: float | None = None
) -> dict[str, float]:
    diffs = np.diff(values)
    c = float(np.mean(values)) if center is None else center
    s = float(np.std(values)) if level is None else level
    d = float(np.std(diffs))
    return {"center": c, "level_scale": s, "delta_scale": d, "target_center": c, "target_scale": s}


def mackey_glass_problems(seeds: Sequence[int]) -> list[SeriesProblem]:
    calibration = np.concatenate(
        [
            synthetic.mackey_glass(_calibration_seed("mackey_glass17", i), 3000).inputs
            for i in range(CALIBRATION_REALIZATIONS)
        ]
    )
    scales = _delta_scales(calibration)
    out = []
    for seed in seeds:
        series = synthetic.mackey_glass(seed, MG_LENGTH + MG_HORIZON)
        out.append(
            SeriesProblem(
                task="mackey_glass17",
                series_id=f"mackey_glass17:{seed}",
                inputs=series.inputs[:MG_LENGTH],
                targets=series.targets[:MG_LENGTH],
                target_mode="delta",
                scales=scales,
                score_from=MG_SCORE_FROM,
                forecast_truth=series.targets[MG_LENGTH : MG_LENGTH + MG_HORIZON],
                last_value=float(series.targets[MG_LENGTH - 1]),
                metadata=series.metadata,
            )
        )
    return out


# ----------------------------------------------------------------------
# dysts
# ----------------------------------------------------------------------
def dysts_problems(system: str, seeds: Sequence[int]) -> list[SeriesProblem]:
    metadata = dysts_subset.load_metadata()
    calibration = np.concatenate(
        [
            dysts_subset.integrate(
                system, 1500, seed=_calibration_seed(f"dysts:{system}", i), metadata=metadata
            ).observed
            for i in range(2)
        ]
    )
    scales = _delta_scales(calibration, center=0.0, level=1.0)
    out = []
    for seed in seeds:
        trajectory = dysts_subset.integrate(
            system, DYSTS_LENGTH + DYSTS_HORIZON + 1, seed=seed, metadata=metadata
        )
        x = trajectory.observed
        info = trajectory.metadata["standardization"]
        out.append(
            SeriesProblem(
                task=f"dysts:{system}",
                series_id=f"dysts:{system}:{seed}",
                inputs=x[:DYSTS_LENGTH],
                targets=x[1 : DYSTS_LENGTH + 1],
                target_mode="delta",
                scales=scales,
                score_from=DYSTS_SCORE_FROM,
                forecast_truth=x[DYSTS_LENGTH + 1 : DYSTS_LENGTH + 1 + DYSTS_HORIZON],
                last_value=float(x[DYSTS_LENGTH]),
                raw_transform=(float(info["mean"]), float(info["std"])),
                metadata={k: v for k, v in trajectory.metadata.items() if k != "citation"},
            )
        )
    return out


# ----------------------------------------------------------------------
# Monash
# ----------------------------------------------------------------------
def monash_problems(dataset: str, role: str, *, root: Any = None) -> list[SeriesProblem]:
    loaded = monash.load(dataset, root=root, role=role)
    spec: monash.MonashDataset = loaded["dataset"]
    out = []
    for series in loaded["series"]:
        training = series.training
        warmup = max(50, len(training) // 10)
        head = training[:warmup]
        diffs = np.diff(head)
        level = float(np.std(head))
        delta = float(np.std(diffs))
        scales = {
            "center": float(np.mean(head)),
            "level_scale": level if level > 0 else 1.0,
            "delta_scale": delta if delta > 0 else (level if level > 0 else 1.0),
        }
        scales["target_center"], scales["target_scale"] = scales["center"], scales["level_scale"]
        out.append(
            SeriesProblem(
                task=f"monash:{dataset}",
                series_id=f"monash:{dataset}:{series.name}",
                inputs=training[:-1],
                targets=training[1:],
                target_mode="delta",
                scales=scales,
                score_from=warmup,
                forecast_truth=series.test,
                last_value=float(training[-1]),
                training=training,
                seasonality=spec.seasonality,
                metadata={"warmup": warmup, "horizon": spec.horizon, "role": role},
            )
        )
    return out


# ----------------------------------------------------------------------
# metrics for one problem
# ----------------------------------------------------------------------
def problem_metrics(
    problem: SeriesProblem, predictions: np.ndarray, forecast: np.ndarray | None
) -> dict[str, Any]:
    window = slice(problem.score_from, problem.length)
    pred, target = predictions[window], problem.targets[window]
    record: dict[str, Any] = {
        "series_id": problem.series_id,
        "prequential_nmse": nmse(pred, target),
        "prequential_nrmse": nrmse(pred, target),
        "prequential_mae_normalized": float(np.mean(np.abs(pred - target)) / problem.scales["target_scale"]),
    }
    if (
        problem.task.startswith("monash:")
        and problem.training is not None
        and problem.seasonality is not None
    ):
        record["prequential_mase"] = mase_monash_series(
            pred, target, problem.training[: problem.score_from + 1], problem.seasonality
        )
    if forecast is not None and problem.forecast_truth is not None:
        truth = problem.forecast_truth
        finite = bool(np.all(np.isfinite(forecast)))
        record["forecast_finite"] = finite
        if problem.task == "mackey_glass17":
            record["forecast_error_at_horizon"] = float(forecast[-1] - truth[-1]) if finite else None
            record["forecast_nrmse"] = (
                float(np.sqrt(np.mean((forecast - truth) ** 2)) / problem.scales["target_scale"])
                if finite
                else None
            )
        if problem.task.startswith("dysts:") and problem.raw_transform is not None:
            mean, std = problem.raw_transform
            record["forecast_nrmse"] = float(np.sqrt(np.mean((forecast - truth) ** 2))) if finite else None
            record["forecast_smape_percent"] = (
                smape_percent(forecast * std + mean, truth * std + mean) if finite else None
            )
            record["prequential_smape_percent"] = smape_percent(pred * std + mean, target * std + mean)
        if (
            problem.task.startswith("monash:")
            and problem.training is not None
            and problem.seasonality is not None
        ):
            record["forecast_mase"] = (
                mase_monash_series(forecast, truth, problem.training, problem.seasonality) if finite else None
            )
            record["forecast_smape"] = smape_monash(forecast, truth) if finite else None
    return record


# ----------------------------------------------------------------------
# baselines (analytic or online-linear; no initialization seeds)
# ----------------------------------------------------------------------
def persistence(problem: SeriesProblem) -> tuple[np.ndarray, np.ndarray | None]:
    if problem.target_mode == "absolute":
        predictions = np.concatenate([[problem.scales["target_center"]], problem.targets[:-1]])
        return predictions, None
    horizon = None if problem.forecast_truth is None else problem.forecast_truth.size
    forecast = None if horizon is None else np.full(horizon, problem.last_value, dtype=float)
    return problem.inputs.copy(), forecast


def seasonal_naive(problem: SeriesProblem) -> tuple[np.ndarray, np.ndarray | None]:
    if problem.seasonality is None or problem.training is None or problem.forecast_truth is None:
        raise ValueError("seasonal naive needs a seasonal forecasting problem")
    m = int(min(problem.seasonality))
    history = problem.training
    predictions = np.asarray(
        [history[t + 1 - m] if t + 1 - m >= 0 else history[t] for t in range(problem.length)]
    )
    horizon = problem.forecast_truth.size
    forecast = np.asarray([history[len(history) - m + (k % m)] for k in range(horizon)])
    return predictions, forecast


def ar_rls(
    problem: SeriesProblem, order: int = 10, ridge: float = 100.0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Online recursive least squares on normalized lags; recursive forecast for delta tasks.

    Features: ``[1, s_t, ..., s_{t-order+1}]`` plus, for absolute targets,
    ``[y_{t-1}, ..., y_{t-order}]`` (revealed targets). Plain RLS, forgetting 1,
    initial covariance ``ridge * I``. Its adaptive state is ``d + d^2`` scalars.
    """

    sc = problem.scales
    s = (problem.inputs - sc["center"]) / sc["level_scale"]
    y = (problem.targets - sc["target_center"]) / sc["target_scale"]
    absolute = problem.target_mode == "absolute"
    d = 1 + order + (order if absolute else 0)
    w = np.zeros(d)
    P = np.eye(d) * ridge
    predictions = np.zeros(problem.length)

    def features(t: int, s_hist: np.ndarray, y_hist: np.ndarray) -> np.ndarray:
        lags = [s_hist[t - k] if t - k >= 0 else 0.0 for k in range(order)]
        phi = [1.0, *lags]
        if absolute:
            phi += [y_hist[t - 1 - k] if t - 1 - k >= 0 else 0.0 for k in range(order)]
        return np.asarray(phi)

    for t in range(problem.length):
        phi = features(t, s, y)
        estimate = float(w @ phi)
        predictions[t] = estimate
        Pphi = P @ phi
        gain = Pphi / (1.0 + phi @ Pphi)
        w = w + gain * (y[t] - estimate)
        P = P - np.outer(gain, Pphi)
    predictions = predictions * sc["target_scale"] + sc["target_center"]
    forecast = None
    if problem.forecast_truth is not None and not absolute and problem.last_value is not None:
        history = [*s, (problem.last_value - sc["center"]) / sc["level_scale"]]
        outputs = []
        for _ in range(problem.forecast_truth.size):
            t = len(history) - 1
            phi = np.asarray([1.0, *[history[t - k] if t - k >= 0 else 0.0 for k in range(order)]])
            value = float(w @ phi)
            outputs.append(value * sc["target_scale"] + sc["target_center"])
            history.append((outputs[-1] - sc["center"]) / sc["level_scale"])
        forecast = np.asarray(outputs)
    return predictions, forecast


BASELINES = {
    "persistence": persistence,
    "ar_rls": ar_rls,
}
