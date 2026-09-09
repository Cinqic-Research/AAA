"""Reproducible diagnostics for the legacy learner and small-model candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import WorldConfig
from .environment import MovingDotEnvironment
from .experiment import run_episode
from .predictors import OnlineLinearPredictor


def run_diagnosis(output: str | Path = "diagnosis") -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    world = WorldConfig(steps_per_episode=180, speed_min=0.02, speed_max=0.06)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for seed in range(20):
        model = OnlineLinearPredictor(update_enabled=False)
        records = run_episode(MovingDotEnvironment("straight", seed, world), [model], learn=False)
        for record in records:
            rows.append(OnlineLinearPredictor.features(record.history))
            targets.append(record.actual_next_position - record.current_observation)
    matrix = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    weights, residuals, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
    fitted = matrix @ weights
    result = {
        "format_version": "aaa.diagnosis.v1",
        "dataset": {"episodes": 20, "rows": int(len(matrix)), "scenario": "straight", "world": world.__dict__},
        "legacy_feature_order": ["bias", "x[t-3]", "x[t-2]", "x[t-1]", "x[t]"],
        "matrix_rank": int(rank),
        "matrix_columns": int(matrix.shape[1]),
        "singular_values": singular_values.tolist(),
        "condition_number": float(singular_values[0] / singular_values[-1]) if singular_values[-1] else None,
        "least_squares_weights": weights.tolist(),
        "least_squares_residual_sum": residuals.tolist(),
        "least_squares_training_mae": float(np.mean(np.abs(fitted - target))),
        "feature_scales": {name: float(np.std(matrix[:, index])) for index, name in enumerate(["bias", "x[t-3]", "x[t-2]", "x[t-1]", "x[t]"])},
        "feature_correlation": np.corrcoef(matrix[:, 1:].T).tolist(),
        "interpretation": {
            "coefficient_identifiability": "rank and singular values describe coefficient identifiability; equivalent predictions may have different weights",
            "causal_limit": "generated rows are development evidence and do not isolate optimization, regime mixing, or sequential-order causes",
        },
    }
    (destination / "diagnosis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Legacy learner diagnosis",
        "",
        "This report is generated from newly generated straight-motion training rows and is not a confirmation result.",
        "",
        f"- Rows: `{len(matrix)}`; rank: `{rank}/{matrix.shape[1]}`",
        f"- Singular values: `{[float(value) for value in singular_values]}`",
        f"- Condition number: `{result['condition_number']}`",
        f"- Stable SVD least-squares training MAE: `{result['least_squares_training_mae']}`",
        f"- Least-squares weights: `{[float(value) for value in weights]}`",
        "",
        "The legacy model uses highly correlated position columns. A rank-deficient straight-motion matrix can have non-unique coefficients while retaining essentially identical predictions. The diagnostic therefore separates coefficient identifiability from predictive accuracy. Mixed-regime online SGD, normalization, update order, and sequential order require controlled ablations; this file does not claim one complete root cause.",
        "",
        "The benchmark candidate is a separately named zero-initialized normalized RLS model. Its full covariance state is serialized and finite/symmetric checks are applied after each update.",
        "",
    ]
    (destination / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    return destination
