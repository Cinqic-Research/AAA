"""Independent hierarchical statistics for observation-noise evidence.

The production runner records primitive step rows.  This module turns those
rows into episode-level estimands and paired uncertainty without importing the
benchmark metric collector or any production gate implementation.  The
resampling hierarchy is fixed by the noise protocol:

    training lineage -> latent episode -> sensor realization

Every bootstrap draw keeps the same paired trial selections for every
predictor, so comparisons do not silently change from a paired estimand into
an unpaired one.  A small deterministic fixture can use fewer draws in tests;
formal execution reads the frozen 4,000-draw value from the protocol.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_LEVELS = (0.90, 0.95)


class StatisticsError(ValueError):
    """Raised when primitive evidence cannot support a declared estimand."""


@dataclass(frozen=True)
class EpisodeValue:
    """One predictor's normalized episode mean and its trial identity."""

    condition: str
    family: str
    channel: str
    scale: float
    role: str
    branch: str
    predictor: str
    lineage: int
    episode: int
    realization: int
    value: float


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatisticsError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise StatisticsError(f"{label} must be finite")
    return result


def _cell_key(trial: Mapping[str, Any], predictor: str) -> tuple[str, str, str, float, str, str, str]:
    return (
        str(trial["condition"]),
        str(trial["family"]),
        str(trial["channel"]),
        _finite(trial["scale"], "trial.scale"),
        str(trial["role"]),
        str(trial["branch"]),
        predictor,
    )


def _cell_label(key: tuple[str, str, str, float, str, str, str]) -> str:
    condition, family, channel, scale, role, branch, predictor = key
    return f"{condition}|{family}|{channel}|{scale:.7g}|{role}|{branch}|{predictor}"


def _episode_values(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str, float, str, str, str], list[EpisodeValue]]:
    """Aggregate steps into one value per lineage/episode/realization."""

    grouped: dict[tuple[tuple[str, str, str, float, str, str, str], int, int, int], list[float]] = (
        defaultdict(list)
    )
    for record in records:
        trial = record["trial"]
        if not isinstance(trial, Mapping):
            raise StatisticsError("record trial must be an object")
        lineage = trial["lineage"]
        episode = trial["episode"]
        realization = trial["realization"]
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in (lineage, episode, realization)
        ):
            raise StatisticsError("trial hierarchy identities must be integers")
        predictions = record.get("predictions")
        if not isinstance(predictions, Mapping) or not predictions:
            raise StatisticsError("record predictions must be a nonempty object")
        for predictor, prediction in predictions.items():
            if not isinstance(prediction, Mapping):
                raise StatisticsError("prediction must be an object")
            value = _finite(
                prediction["latent_normalized_absolute_error"],
                f"{predictor}.latent_normalized_absolute_error",
            )
            key = _cell_key(trial, str(predictor))
            grouped[(key, lineage, episode, realization)].append(value)
    output: dict[tuple[str, str, str, float, str, str, str], list[EpisodeValue]] = defaultdict(list)
    for (key, lineage, episode, realization), values in sorted(grouped.items(), key=str):
        if not values:
            raise StatisticsError("empty episode value")
        output[key].append(EpisodeValue(*key, lineage, episode, realization, float(np.mean(values))))
    return dict(output)


def _nested(values: Sequence[EpisodeValue]) -> dict[int, dict[int, dict[int, float]]]:
    nested: dict[int, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for item in values:
        target = nested[item.lineage][item.episode]
        if item.realization in target:
            raise StatisticsError("duplicate lineage/episode/realization value")
        target[item.realization] = item.value
    if not nested or any(not episodes for episodes in nested.values()):
        raise StatisticsError("hierarchical cell is empty")
    return {lineage: dict(episodes) for lineage, episodes in nested.items()}


def _hierarchical_mean(nested: Mapping[int, Mapping[int, Mapping[int, float]]]) -> float:
    lineage_means = []
    for episodes in nested.values():
        episode_means = [float(np.mean(list(realizations.values()))) for realizations in episodes.values()]
        lineage_means.append(float(np.mean(episode_means)))
    return float(np.mean(lineage_means))


def _draw_hierarchical(
    nested: Mapping[int, Mapping[int, Mapping[int, float]]], rng: np.random.Generator
) -> float:
    """Draw lineages, then episodes, then sensor realizations with replacement."""

    lineages = sorted(nested)
    selected_lineages = rng.choice(np.asarray(lineages, dtype=int), size=len(lineages), replace=True)
    lineage_means: list[float] = []
    for lineage in selected_lineages:
        episodes = nested[int(lineage)]
        episode_ids = sorted(episodes)
        selected_episodes = rng.choice(
            np.asarray(episode_ids, dtype=int), size=len(episode_ids), replace=True
        )
        episode_means: list[float] = []
        for episode in selected_episodes:
            realizations = episodes[int(episode)]
            realization_ids = sorted(realizations)
            selected_realizations = rng.choice(
                np.asarray(realization_ids, dtype=int), size=len(realization_ids), replace=True
            )
            episode_means.append(float(np.mean([realizations[int(item)] for item in selected_realizations])))
        lineage_means.append(float(np.mean(episode_means)))
    return float(np.mean(lineage_means))


def _intervals(draws: np.ndarray, levels: Sequence[float]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for level in levels:
        if not 0.0 < level < 1.0:
            raise StatisticsError("interval level must be between zero and one")
        tail = (1.0 - level) / 2.0
        result[f"{level:g}"] = {
            "lower": float(np.quantile(draws, tail, method="linear")),
            "upper": float(np.quantile(draws, 1.0 - tail, method="linear")),
        }
    return result


def hierarchical_bootstrap(
    records: Iterable[Mapping[str, Any]],
    *,
    draws: int = 4000,
    seed: int = 0,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Return balanced cell estimates and lineage/episode/sensor intervals."""

    if draws < 1:
        raise StatisticsError("draws must be positive")
    cells = _episode_values(records)
    if not cells:
        raise StatisticsError("no primitive records available for statistics")
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for key, values in sorted(cells.items(), key=lambda item: _cell_label(item[0])):
        nested = _nested(values)
        sample = np.asarray([_draw_hierarchical(nested, rng) for _ in range(draws)], dtype=float)
        output[_cell_label(key)] = {
            "condition": key[0],
            "family": key[1],
            "channel": key[2],
            "scale": key[3],
            "role": key[4],
            "branch": key[5],
            "predictor": key[6],
            "estimand": "balanced_mean_of_realization_episode_lineage_means",
            "point": _hierarchical_mean(nested),
            "lineages": len(nested),
            "episodes": sum(len(episodes) for episodes in nested.values()),
            "sensor_realizations": len(values),
            "draws": draws,
            "intervals": _intervals(sample, levels),
            "evidence_status": (
                "PASS"
                if len(nested) >= 2 and all(len(episodes) >= 2 for episodes in nested.values())
                else "INSUFFICIENT_EVIDENCE"
            ),
        }
    return {"method": "hierarchical_bootstrap", "draws": draws, "cells": output}


def _paired_values(
    records: Iterable[Mapping[str, Any]], baseline: str, targets: Sequence[str]
) -> dict[tuple[str, str, str, float, str, str], dict[tuple[int, int, int], dict[str, float]]]:
    by_trial: dict[tuple[tuple[str, str, str, float, str, str], int, int, int], dict[str, list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        trial = record["trial"]
        predictions = record["predictions"]
        cell = (
            str(trial["condition"]),
            str(trial["family"]),
            str(trial["channel"]),
            _finite(trial["scale"], "trial.scale"),
            str(trial["role"]),
            str(trial["branch"]),
        )
        identity = (int(trial["lineage"]), int(trial["episode"]), int(trial["realization"]))
        for predictor in (baseline, *targets):
            if predictor in predictions:
                by_trial[(cell, *identity)][predictor].append(
                    _finite(
                        predictions[predictor]["latent_normalized_absolute_error"],
                        f"{predictor}.latent_normalized_absolute_error",
                    )
                )
    output: dict[tuple[str, str, str, float, str, str], dict[tuple[int, int, int], dict[str, float]]] = (
        defaultdict(dict)
    )
    for (cell, lineage, episode, realization), predictors in by_trial.items():
        if baseline not in predictors or any(target not in predictors for target in targets):
            continue
        output[cell][(lineage, episode, realization)] = {
            predictor: float(np.mean(values)) for predictor, values in predictors.items()
        }
    return dict(output)


def _draw_paired(
    values: Mapping[tuple[int, int, int], Mapping[str, float]],
    baseline: str,
    target: str,
    rng: np.random.Generator,
) -> float:
    nested: dict[int, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for (lineage, episode, realization), row in values.items():
        nested[lineage][episode][realization] = row[target] - row[baseline]
    return _draw_hierarchical(nested, rng)


def _lineage_sign_flip_pvalue(
    values: Mapping[tuple[int, int, int], Mapping[str, float]],
    baseline: str,
    target: str,
    rng: np.random.Generator,
    draws: int,
) -> float:
    """Test zero paired effect by flipping complete independent lineages."""

    by_lineage: dict[int, list[float]] = defaultdict(list)
    for (lineage, _episode, _realization), row in values.items():
        by_lineage[lineage].append(row[target] - row[baseline])
    lineage_means = np.asarray([np.mean(by_lineage[lineage]) for lineage in sorted(by_lineage)], dtype=float)
    observed = float(np.mean(lineage_means))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(lineage_means)))
    null = np.mean(signs * lineage_means, axis=1)
    exceedances = int(np.sum(np.abs(null) >= abs(observed)))
    return float((exceedances + 1) / (draws + 1))


def holm_bonferroni(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down adjustment to a fixed family of p-values."""

    ordered = sorted((max(0.0, min(1.0, float(value))), key) for key, value in pvalues.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (pvalue, key) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * pvalue))
        adjusted[key] = running
    return adjusted


def paired_hierarchical_comparisons(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline: str = "constant_motion_reflected",
    targets: Sequence[str] = ("incumbent_square_root_rls",),
    draws: int = 4000,
    seed: int = 0,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Compare predictors to one baseline using one paired hierarchy."""

    if draws < 1:
        raise StatisticsError("draws must be positive")
    paired = _paired_values(records, baseline, targets)
    rng = np.random.default_rng(seed)
    rows: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    bootstrap_samples: dict[str, np.ndarray] = {}
    for cell, values in sorted(paired.items(), key=str):
        for target in targets:
            sample = np.asarray([_draw_paired(values, baseline, target, rng) for _ in range(draws)])
            key = "|".join((*map(str, cell), target))
            pvalue = _lineage_sign_flip_pvalue(values, baseline, target, rng, draws)
            pvalues[key] = pvalue
            bootstrap_samples[key] = sample
            nested: dict[int, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
            for (lineage, episode, realization), row in values.items():
                nested[lineage][episode][realization] = row[target] - row[baseline]
            rows[key] = {
                "condition": cell[0],
                "family": cell[1],
                "channel": cell[2],
                "scale": cell[3],
                "role": cell[4],
                "branch": cell[5],
                "baseline": baseline,
                "target": target,
                "estimand": "target_minus_baseline_balanced_hierarchical_mean",
                "point": _hierarchical_mean(nested),
                "lineages": len({identity[0] for identity in values}),
                "episodes": len({identity[:2] for identity in values}),
                "sensor_realizations": len(values),
                "draws": draws,
                "p_value": pvalue,
                "p_value_method": "paired_lineage_sign_flip",
                "intervals": _intervals(sample, levels),
                "evidence_status": (
                    "PASS"
                    if len({identity[0] for identity in values}) >= 2
                    and len({identity[:2] for identity in values}) >= 4
                    else "INSUFFICIENT_EVIDENCE"
                ),
            }
    adjusted = holm_bonferroni(pvalues)
    family_size = len(pvalues)
    for key, value in adjusted.items():
        rows[key]["holm_adjusted_p_value"] = value
        rows[key]["family_size"] = family_size
        rows[key]["familywise_intervals"] = {
            f"{level:g}": {
                "lower": float(
                    np.quantile(bootstrap_samples[key], (1.0 - level) / (2.0 * family_size), method="linear")
                ),
                "upper": float(
                    np.quantile(
                        bootstrap_samples[key],
                        1.0 - (1.0 - level) / (2.0 * family_size),
                        method="linear",
                    )
                ),
            }
            for level in levels
        }
    return {
        "method": "paired_hierarchical_bootstrap",
        "baseline": baseline,
        "draws": draws,
        "multiplicity": "holm_bonferroni_with_familywise_bounds",
        "cells": rows,
    }


def interval_statistics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize only intervals that were available before each target reveal."""

    grouped: dict[tuple[str, str, str, float, str, str, str, str], list[tuple[bool, float, float]]] = (
        defaultdict(list)
    )
    for record in records:
        trial = record["trial"]
        intervals = record.get("prediction_intervals", {})
        if not isinstance(intervals, Mapping):
            raise StatisticsError("prediction_intervals must be an object")
        target = _finite(record["raw_observation"], "raw_observation")
        for predictor, levels in intervals.items():
            if not isinstance(levels, Mapping):
                raise StatisticsError("predictor interval map must be an object")
            for level, interval in levels.items():
                if interval is None:
                    continue
                if not isinstance(interval, Mapping) or set(interval) != {"lower", "upper"}:
                    raise StatisticsError("interval must contain lower and upper bounds")
                lower = _finite(interval["lower"], "interval.lower")
                upper = _finite(interval["upper"], "interval.upper")
                if lower > upper:
                    raise StatisticsError("interval lower bound exceeds upper bound")
                key = (*_cell_key(trial, str(predictor)), str(level))
                grouped[key].append((lower <= target <= upper, upper - lower, 0.0))
                alpha = 1.0 - float(level)
                penalty = 0.0
                if target < lower:
                    penalty += 2.0 * (lower - target) / alpha
                if target > upper:
                    penalty += 2.0 * (target - upper) / alpha
                grouped[key][-1] = (lower <= target <= upper, upper - lower, upper - lower + penalty)
    rows: dict[str, Any] = {}
    for key, values in sorted(grouped.items(), key=str):
        label = f"{_cell_label(key[:-1])}|{key[-1]}"
        coverage = float(np.mean([float(item[0]) for item in values]))
        rows[label] = {
            "condition": key[0],
            "family": key[1],
            "channel": key[2],
            "scale": key[3],
            "role": key[4],
            "branch": key[5],
            "predictor": key[6],
            "level": key[7],
            "count": len(values),
            "coverage": coverage,
            "width": float(np.mean([item[1] for item in values])),
            "interval_score": float(np.mean([item[2] for item in values])),
            "evidence_status": "PASS" if len(values) >= 1 else "INSUFFICIENT_EVIDENCE",
        }
    return {"method": "causal_residual_quantile", "cells": rows}


def validate_null_behavior(*, seed: int, simulations: int = 64, bootstrap_draws: int = 128) -> dict[str, Any]:
    """Exercise finite-sample zero-effect behavior on synthetic data only."""

    if simulations < 1 or bootstrap_draws < 1:
        raise StatisticsError("null-validation counts must be positive")
    rng = np.random.default_rng(seed)
    excludes_zero = 0
    contains_zero = 0
    for _ in range(simulations):
        nested: dict[int, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        for lineage in range(8):
            for episode in range(4):
                for realization in range(2):
                    nested[lineage][episode][realization] = float(rng.normal(0.0, 1.0))
        sample = np.asarray([_draw_hierarchical(nested, rng) for _ in range(bootstrap_draws)])
        lower = float(np.quantile(sample, 0.025, method="linear"))
        upper = float(np.quantile(sample, 0.975, method="linear"))
        if lower <= 0.0 <= upper:
            contains_zero += 1
        else:
            excludes_zero += 1
    return {
        "status": "DIAGNOSTIC_ONLY",
        "simulations": simulations,
        "bootstrap_draws": bootstrap_draws,
        "nominal_interval": 0.95,
        "zero_in_interval_rate": float(contains_zero / simulations),
        "false_positive_rate": float(excludes_zero / simulations),
        "note": "Synthetic zero-effect diagnostic; it is not benchmark evidence or an acceptance gate.",
    }


def resource_statistics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate measured per-step cost and state diagnostics by comparison cell."""

    grouped: dict[tuple[str, str, str, float, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        trial = record["trial"]
        diagnostics = record.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            raise StatisticsError("diagnostics must be an object")
        for predictor, values in diagnostics.items():
            if not isinstance(values, Mapping):
                raise StatisticsError("predictor diagnostics must be an object")
            key = _cell_key(trial, str(predictor))
            bucket = grouped[key]
            for name in ("update_norm", "predict_latency_ns", "update_latency_ns", "detected_surprises"):
                bucket[name].append(_finite(values.get(name, 0.0), f"diagnostics.{name}"))
            for name in ("dead_zone_skips", "reflection_skips", "forgetting_suspensions"):
                bucket[name].append(_finite(values.get(name, 0.0), f"diagnostics.{name}"))
    output: dict[str, Any] = {}
    for key, values in sorted(grouped.items(), key=str):
        output[_cell_label(key)] = {
            "condition": key[0],
            "family": key[1],
            "channel": key[2],
            "scale": key[3],
            "role": key[4],
            "branch": key[5],
            "predictor": key[6],
            "count": len(values["update_norm"]),
            "mean_update_norm": float(np.mean(values["update_norm"])),
            "mean_predict_latency_ns": float(np.mean(values["predict_latency_ns"])),
            "mean_update_latency_ns": float(np.mean(values["update_latency_ns"])),
            "max_detected_surprises": float(max(values["detected_surprises"])),
            "max_dead_zone_skips": float(max(values["dead_zone_skips"])),
            "max_reflection_skips": float(max(values["reflection_skips"])),
            "max_forgetting_suspensions": float(max(values["forgetting_suspensions"])),
        }
    return {"method": "primitive_diagnostic_aggregation", "cells": output}
