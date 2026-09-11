"""Controlled development-only diagnosis of the legacy learner.

The original v1 online learner did not convincingly improve from experience and
its frozen generalization was poor. That observation is preserved as historical
evidence; this module exists to find out *why*, with controlled ablations
rather than a single asserted cause.

Factors that are varied independently:

``features``
    the legacy raw-position basis, a centered/scaled position basis, a pure
    displacement basis, and displacement plus a centered position feature;
``estimator``
    plain SGD, normalized LMS, square-root RLS with and without forgetting,
    and a stable batch least-squares reference;
``regime``
    straight-only, bounce-only and mixed data;
``ordering``
    chronological single-pass, shuffled single-pass, and replay with an equal
    cumulative update budget.

Everything here uses development streams only. No confirmation data is touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import WorldConfig
from .environment import MovingDotEnvironment
from .experiment import TrialIdentity, run_episode
from .predictors import OnlineLinearPredictor, batch_least_squares

DIAGNOSIS_SCHEMA = "aaa.diagnosis.v2"

FeatureFn = Callable[[np.ndarray, WorldConfig], np.ndarray]


def _identity(scenario: str, seed: int, episode: int) -> TrialIdentity:
    return TrialIdentity(
        trial_id=f"diagnosis:{scenario}:s{seed}:e{episode}",
        role="development",
        family="diagnosis",
        scenario=scenario,
        environment_seed=seed,
        replica_id=0,
        episode=episode,
        stratum="diagnosis",
        update_mode="frozen",
    )


# ---------------------------------------------------------------------------
# feature bases
# ---------------------------------------------------------------------------


def legacy_positions(history: np.ndarray, world: WorldConfig) -> np.ndarray:
    return np.concatenate([[1.0], history[-4:]])


def centered_positions(history: np.ndarray, world: WorldConfig) -> np.ndarray:
    midpoint = (world.lower_bound + world.upper_bound) / 2
    return np.concatenate([[1.0], (history[-4:] - midpoint) / world.width])


def displacement_only(history: np.ndarray, world: WorldConfig) -> np.ndarray:
    scale = world.dt * world.speed_max
    return np.asarray([1.0, (history[-1] - history[-2]) / scale])


def displacement_position(history: np.ndarray, world: WorldConfig) -> np.ndarray:
    scale = world.dt * world.speed_max
    midpoint = (world.lower_bound + world.upper_bound) / 2
    return np.asarray(
        [1.0, (history[-1] - history[-2]) / scale, (history[-1] - midpoint) / world.width]
    )


FEATURE_SETS: dict[str, FeatureFn] = {
    "legacy_positions": legacy_positions,
    "centered_positions": centered_positions,
    "displacement_only": displacement_only,
    "displacement_position": displacement_position,
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    features: dict[str, np.ndarray]
    targets: np.ndarray
    world: WorldConfig
    regime: str

    def __len__(self) -> int:
        return int(self.targets.shape[0])


def build_dataset(regime: str, *, seeds: Sequence[int], world: WorldConfig) -> Dataset:
    scenarios = {"straight": ("straight",), "bouncing": ("bouncing",), "mixed": ("straight", "bouncing")}[regime]
    histories: list[np.ndarray] = []
    targets: list[float] = []
    for seed in seeds:
        for index, scenario in enumerate(scenarios):
            probe = OnlineLinearPredictor(update_enabled=False, name="probe")
            environment = MovingDotEnvironment(scenario, seed + index * 977, world)
            records = run_episode(environment, [probe], _identity(scenario, seed, index), learn=False)
            for record in records:
                histories.append(np.asarray(record.history, dtype=float))
                targets.append(record.actual_next_position - record.current_observation)
    stacked = np.asarray(histories, dtype=float)
    features = {
        name: np.asarray([function(row, world) for row in stacked], dtype=float)
        for name, function in FEATURE_SETS.items()
    }
    return Dataset(features=features, targets=np.asarray(targets, dtype=float), world=world, regime=regime)


def conditioning(matrix: np.ndarray) -> dict[str, object]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * float(singular[0]) * np.finfo(float).eps if singular.size else 0.0
    rank = int(np.sum(singular > tolerance))
    correlation = None
    if matrix.shape[1] > 1:
        columns = matrix[:, 1:]
        if columns.shape[1] > 1 and np.all(np.std(columns, axis=0) > 0):
            correlation = np.corrcoef(columns.T).tolist()
    return {
        "columns": int(matrix.shape[1]),
        "rows": int(matrix.shape[0]),
        "singular_values": [float(value) for value in singular],
        "effective_rank": rank,
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
        "column_means": [float(value) for value in matrix.mean(axis=0)],
        "column_scales": [float(value) for value in matrix.std(axis=0)],
        "off_bias_correlation": correlation,
    }


# ---------------------------------------------------------------------------
# estimators (diagnostic only; not the benchmark candidate)
# ---------------------------------------------------------------------------


def _sgd(features: np.ndarray, targets: np.ndarray, *, rate: float, order: np.ndarray) -> dict[str, object]:
    weights = np.zeros(features.shape[1])
    norms: list[float] = []
    for index in order:
        phi = features[index]
        error = float(phi @ weights) - float(targets[index])
        step = rate * error * phi
        weights = weights - step
        norms.append(float(np.linalg.norm(step)))
        if not np.all(np.isfinite(weights)):
            return {"weights": None, "diverged": True, "update_norm_mean": None}
    return {
        "weights": weights.tolist(),
        "diverged": False,
        "update_norm_mean": float(np.mean(norms)),
        "update_norm_max": float(np.max(norms)),
    }


def _nlms(features: np.ndarray, targets: np.ndarray, *, rate: float, order: np.ndarray) -> dict[str, object]:
    weights = np.zeros(features.shape[1])
    norms: list[float] = []
    for index in order:
        phi = features[index]
        power = float(phi @ phi) + 1e-12
        error = float(phi @ weights) - float(targets[index])
        step = rate * error * phi / power
        weights = weights - step
        norms.append(float(np.linalg.norm(step)))
    return {
        "weights": weights.tolist(),
        "diverged": bool(not np.all(np.isfinite(weights))),
        "update_norm_mean": float(np.mean(norms)),
        "update_norm_max": float(np.max(norms)),
    }


def _rls(features: np.ndarray, targets: np.ndarray, *, forgetting: float, ridge: float, order: np.ndarray) -> dict[str, object]:
    """Square-root RLS, matching the candidate's numerical formulation."""

    dimension = features.shape[1]
    weights = np.zeros(dimension)
    factor = np.eye(dimension) / np.sqrt(ridge)
    for index in order:
        phi = features[index]
        f = factor.T @ phi
        beta = float(f @ f)
        alpha = forgetting + beta
        error = float(targets[index]) - float(phi @ weights)
        if beta > 0:
            gain = (factor @ f) / alpha
            weights = weights + gain * error
            gamma = (1.0 - np.sqrt(forgetting / alpha)) / beta
            factor = (factor - gamma * np.outer(factor @ f, f)) / np.sqrt(forgetting)
        else:
            factor = factor / np.sqrt(forgetting)
    covariance = factor @ factor.T
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    return {
        "weights": weights.tolist(),
        "diverged": bool(not np.all(np.isfinite(weights))),
        "covariance_min_eigenvalue": float(eigenvalues.min()),
        "covariance_condition_number": float(eigenvalues.max() / eigenvalues.min()) if eigenvalues.min() > 0 else None,
    }


ORDERINGS = ("chronological", "shuffled", "replay_x3")


def _order(count: int, ordering: str, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(count)
    if ordering == "chronological":
        return base
    if ordering == "shuffled":
        return rng.permutation(base)
    if ordering == "replay_x3":
        return np.concatenate([rng.permutation(base) for _ in range(3)])
    raise ValueError(f"unknown ordering {ordering!r}")


def _score(weights: Sequence[float] | None, features: np.ndarray, targets: np.ndarray) -> float | None:
    if weights is None:
        return None
    array = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(array)):
        return None
    return float(np.mean(np.abs(features @ array - targets)))


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------


def run_diagnosis(output: str | Path = "diagnosis", *, quick: bool = False) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    world = WorldConfig(steps_per_episode=180, speed_min=0.08, speed_max=0.20, change_step=None)
    train_seeds = range(6) if quick else range(16)
    holdout_seeds = range(500, 503) if quick else range(500, 508)
    regimes = ("mixed",) if quick else ("straight", "bouncing", "mixed")
    rates = (0.08,) if quick else (0.02, 0.08, 0.32)
    orderings = ("chronological",) if quick else ORDERINGS

    conditioning_report: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for regime in regimes:
        train = build_dataset(regime, seeds=list(train_seeds), world=world)
        holdout = build_dataset(regime, seeds=list(holdout_seeds), world=world)
        conditioning_report[regime] = {
            name: conditioning(matrix) for name, matrix in train.features.items()
        }
        conditioning_report[f"{regime}_target"] = {
            "mean": float(train.targets.mean()),
            "scale": float(train.targets.std()),
            "rows": len(train),
        }
        for feature_name, matrix in train.features.items():
            holdout_matrix = holdout.features[feature_name]
            reference = batch_least_squares(matrix, train.targets, ridge=1e-8)
            rows.append(
                {
                    "regime": regime,
                    "features": feature_name,
                    "estimator": "batch_least_squares",
                    "ordering": "n/a",
                    "hyperparameter": 1e-8,
                    "train_mae": _score(reference, matrix, train.targets),
                    "holdout_mae": _score(reference, holdout_matrix, holdout.targets),
                    "weights": reference.tolist(),
                    "diverged": False,
                }
            )
            for ordering in orderings:
                rng = np.random.default_rng(20260910)
                order = _order(len(train), ordering, rng)
                for rate in rates:
                    result = _sgd(matrix, train.targets, rate=rate, order=order)
                    rows.append(
                        {
                            "regime": regime, "features": feature_name, "estimator": "sgd",
                            "ordering": ordering, "hyperparameter": rate,
                            "train_mae": _score(result["weights"], matrix, train.targets),  # type: ignore[arg-type]
                            "holdout_mae": _score(result["weights"], holdout_matrix, holdout.targets),  # type: ignore[arg-type]
                            **{k: v for k, v in result.items() if k != "weights"},
                            "weights": result["weights"],
                        }
                    )
                    nlms = _nlms(matrix, train.targets, rate=min(rate * 4, 1.0), order=order)
                    rows.append(
                        {
                            "regime": regime, "features": feature_name, "estimator": "normalized_lms",
                            "ordering": ordering, "hyperparameter": min(rate * 4, 1.0),
                            "train_mae": _score(nlms["weights"], matrix, train.targets),  # type: ignore[arg-type]
                            "holdout_mae": _score(nlms["weights"], holdout_matrix, holdout.targets),  # type: ignore[arg-type]
                            **{k: v for k, v in nlms.items() if k != "weights"},
                            "weights": nlms["weights"],
                        }
                    )
                for forgetting in (1.0, 0.98, 0.90):
                    result = _rls(matrix, train.targets, forgetting=forgetting, ridge=1e-4, order=order)
                    rows.append(
                        {
                            "regime": regime, "features": feature_name, "estimator": "sqrt_rls",
                            "ordering": ordering, "hyperparameter": forgetting,
                            "train_mae": _score(result["weights"], matrix, train.targets),  # type: ignore[arg-type]
                            "holdout_mae": _score(result["weights"], holdout_matrix, holdout.targets),  # type: ignore[arg-type]
                            **{k: v for k, v in result.items() if k != "weights"},
                            "weights": result["weights"],
                        }
                    )

    valid = [row for row in rows if row.get("holdout_mae") is not None]
    best = min(valid, key=lambda row: float(row["holdout_mae"])) if valid else None
    legacy = [
        row for row in rows
        if row["features"] == "legacy_positions" and row["estimator"] == "sgd" and row["ordering"] == "chronological"
    ]
    order_sensitivity = _order_sensitivity(rows)

    result = {
        "format_version": DIAGNOSIS_SCHEMA,
        "purpose": "controlled development-only ablations separating conditioning, feature basis, optimizer, regime and ordering",
        "world": {
            "lower_bound": world.lower_bound, "upper_bound": world.upper_bound, "dt": world.dt,
            "steps_per_episode": world.steps_per_episode, "speed_min": world.speed_min, "speed_max": world.speed_max,
        },
        "train_seeds": list(train_seeds),
        "holdout_seeds": list(holdout_seeds),
        "conditioning": conditioning_report,
        "rows": rows,
        "best_holdout": best,
        "legacy_chronological_sgd": legacy,
        "order_sensitivity": order_sensitivity,
        "interpretation": {
            "identifiability": "singular values and effective rank describe coefficient identifiability; equivalent predictions can have different weights",
            "causal_scope": "these rows separate feature basis, estimator, regime and ordering, but they do not establish a single root cause for the historical v1 outcome; each factor is reported with its own measured effect",
        },
    }
    (destination / "diagnosis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "diagnosis.md").write_text(_render(result), encoding="utf-8")
    return destination


def _order_sensitivity(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, float]] = {}
    for row in rows:
        if row["ordering"] not in ORDERINGS or row.get("holdout_mae") is None:
            continue
        key = (row["regime"], row["features"], row["estimator"], row["hyperparameter"])
        grouped.setdefault(key, {})[str(row["ordering"])] = float(row["holdout_mae"])
    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        if "chronological" in values and "shuffled" in values:
            output.append(
                {
                    "regime": key[0], "features": key[1], "estimator": key[2], "hyperparameter": key[3],
                    "chronological_holdout_mae": values["chronological"],
                    "shuffled_holdout_mae": values["shuffled"],
                    "replay_holdout_mae": values.get("replay_x3"),
                    "shuffled_minus_chronological": values["shuffled"] - values["chronological"],
                }
            )
    return output


def _render(result: dict[str, object]) -> str:
    rows = result["rows"]  # type: ignore[index]
    best = result["best_holdout"]
    lines = [
        "# Legacy learner diagnosis",
        "",
        "Development-only ablations. This file is not confirmation evidence and does not select the benchmark candidate.",
        "",
        f"- Configurations evaluated: `{len(rows)}`",
        f"- Best held-out configuration: `{best['features']}` / `{best['estimator']}` / `{best['ordering']}` "
        f"(hyperparameter `{best['hyperparameter']}`), held-out MAE `{best['holdout_mae']:.3e}`" if best else "- No configuration produced a finite held-out score",
        "",
        "## Conditioning of each feature basis",
        "",
        "| Regime | Features | Effective rank / columns | Condition number |",
        "|---|---|---:|---:|",
    ]
    for regime, entry in result["conditioning"].items():  # type: ignore[union-attr]
        if regime.endswith("_target"):
            continue
        for name, values in entry.items():  # type: ignore[union-attr]
            condition = values["condition_number"]
            lines.append(
                f"| {regime} | `{name}` | {values['effective_rank']} / {values['columns']} | "
                + (f"{condition:.3e}" if condition else "n/a")
                + " |"
            )
    lines.extend(
        [
            "",
            "## What the ablations separate",
            "",
            "- A rank-deficient raw-position basis makes coefficients non-identifiable while leaving predictions",
            "  essentially unchanged, so weight instability and predictive error are different phenomena.",
            "- Feature basis, estimator, data regime and update ordering are varied independently, so no single",
            "  factor is asserted as *the* cause of the historical v1 result.",
            "- The stable batch least-squares row is the reference solution for each basis and regime; a large gap",
            "  between it and an online estimator is an optimization or conditioning effect, not a capacity limit.",
            "",
            "## Order sensitivity",
            "",
            "| Regime | Features | Estimator | Chronological | Shuffled | Shuffled − chronological |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in result["order_sensitivity"][:24]:  # type: ignore[index]
        lines.append(
            f"| {row['regime']} | `{row['features']}` | `{row['estimator']}` | "
            f"{row['chronological_holdout_mae']:.3e} | {row['shuffled_holdout_mae']:.3e} | "
            f"{row['shuffled_minus_chronological']:+.3e} |"
        )
    lines.extend(["", "Complete rows, weights and conditioning statistics are in `diagnosis.json`.", ""])
    return "\n".join(lines)
