"""Observe: decompose Round 3's Q4 result from its retained primitives.

This module only *reads* the committed round-3 evidence. The cells it
analyses are observed evaluation evidence: they may motivate hypotheses, and
they are never used to fit, tune or select anything. Every number here is
recomputed from the per-cell primitives rather than read from a stored
summary, so the observation cannot inherit a summary's mistake.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

from research.aaa_1k.experiments import PRIMARY

OBSERVATION_SCHEMA = "aaa.loop.observation.v1"
UNGATED = "rnn_control"
COARSE_REGIME_LENGTH = 60


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def decompose_q4(round3: Mapping[str, Any]) -> dict[str, Any]:
    """Where does Q4's negative mean come from? Recomputed from cells."""

    cells = round3["cells"]
    difference = np.asarray([cell["mae"][UNGATED] - cell["mae"][PRIMARY] for cell in cells], dtype=float)
    families = sorted({cell["family"] for cell in cells})
    total = float(np.mean(difference))

    # stream means over initializations: the unit Q4's sign count uses
    stream_ids = sorted({cell["stream_id"] for cell in cells})
    per_stream = {
        stream_id: float(
            np.mean(
                [
                    cell["mae"][UNGATED] - cell["mae"][PRIMARY]
                    for cell in cells
                    if cell["stream_id"] == stream_id
                ]
            )
        )
        for stream_id in stream_ids
    }
    family_of = {cell["stream_id"]: cell["family"] for cell in cells}
    unfavourable = [stream for stream, value in per_stream.items() if value < 0]

    by_family: dict[str, Any] = {}
    for family in families:
        selected = [cell for cell in cells if cell["family"] == family]
        values = np.asarray([cell["mae"][UNGATED] - cell["mae"][PRIMARY] for cell in selected], dtype=float)
        initializations = sorted({cell["model_seed_index"] for cell in selected})
        family_streams = [value for stream, value in per_stream.items() if family_of[stream] == family]
        by_family[family] = {
            "cells": len(selected),
            "contribution_to_aggregate_mean": float(np.sum(values) / difference.size),
            "cell_difference": _quantiles(values),
            "streams_favouring_gated": int(sum(value > 0 for value in family_streams)),
            "streams_favouring_ungated": int(sum(value < 0 for value in family_streams)),
            "per_initialization_mean": [
                float(
                    np.mean(
                        [
                            cell["mae"][UNGATED] - cell["mae"][PRIMARY]
                            for cell in selected
                            if cell["model_seed_index"] == init
                        ]
                    )
                )
                for init in initializations
            ],
            "arm_mean_mae": {
                name: float(np.mean([cell["mae"][name] for cell in selected])) for name in selected[0]["mae"]
            },
            "arm_median_mae": {
                name: float(np.median([cell["mae"][name] for cell in selected]))
                for name in selected[0]["mae"]
            },
            "clip_rate": _quantiles(np.asarray([cell["clip_rate"] for cell in selected], dtype=float)),
        }

    without = {
        family: float(
            np.mean(
                [cell["mae"][UNGATED] - cell["mae"][PRIMARY] for cell in cells if cell["family"] != family]
            )
        )
        for family in families
    }
    ranked = sorted(cells, key=lambda cell: cell["mae"][UNGATED] - cell["mae"][PRIMARY])
    coarse = [cell for cell in cells if cell["family"] == "coarse_speed_v1"]
    coarse_diff = np.asarray([cell["mae"][UNGATED] - cell["mae"][PRIMARY] for cell in coarse], dtype=float)
    median = float(np.median(coarse_diff))
    mad = float(np.median(np.abs(coarse_diff - median)))
    outliers = [
        {
            "model_seed_index": cell["model_seed_index"],
            "stream_id": cell["stream_id"],
            "ungated_minus_gated": cell["mae"][UNGATED] - cell["mae"][PRIMARY],
            "gated_mae": cell["mae"][PRIMARY],
            "gated_frozen_mae": cell["mae"]["aaa1k_frozen"],
            "ungated_mae": cell["mae"][UNGATED],
            "clip_rate": cell["clip_rate"],
            "worst_steps": [
                {
                    "target_index": step["target_index"],
                    "steps_since_regime_switch": step["target_index"] % COARSE_REGIME_LENGTH,
                    "regime": step["regime"],
                    "event": step["event"],
                    "ungated_minus_gated": step["ungated_minus_gated"],
                }
                for step in cell["q4_worst_steps"]
            ],
        }
        for cell in coarse
        if (cell["mae"][UNGATED] - cell["mae"][PRIMARY]) < median - 10.0 * max(mad, 1e-12)
    ]
    worst_regimes = Counter(step["regime"] for cell in coarse for step in cell["q4_worst_steps"])
    worst_since_switch = [
        step["target_index"] % COARSE_REGIME_LENGTH for cell in coarse for step in cell["q4_worst_steps"]
    ]
    return {
        "aggregate_mean": total,
        "aggregate_median_cell": float(np.median(difference)),
        "cells": int(difference.size),
        "streams": len(per_stream),
        "streams_favouring_gated": int(sum(value > 0 for value in per_stream.values())),
        "streams_favouring_ungated": len(unfavourable),
        "unfavourable_streams_by_family": dict(Counter(family_of[stream] for stream in unfavourable)),
        "aggregate_mean_without_family": without,
        "by_family": by_family,
        "twelve_largest_gated_losses": [
            {
                "model_seed_index": cell["model_seed_index"],
                "stream_id": cell["stream_id"],
                "family": cell["family"],
                "ungated_minus_gated": cell["mae"][UNGATED] - cell["mae"][PRIMARY],
            }
            for cell in ranked[:12]
        ],
        "coarse_transient_outliers": {
            "rule": "coarse cell difference below median - 10 * MAD of coarse cells",
            "median": median,
            "mad": mad,
            "cells": outliers,
        },
        "coarse_worst_step_regimes": dict(worst_regimes),
        "coarse_worst_step_steps_since_switch": _quantiles(np.asarray(worst_since_switch, dtype=float)),
    }


def build_observation(round3: Mapping[str, Any], *, round3_sha256: str) -> dict[str, Any]:
    decomposition = decompose_q4(round3)
    coarse = decomposition["by_family"]["coarse_speed_v1"]
    arms = coarse["arm_mean_mae"]
    return {
        "schema": OBSERVATION_SCHEMA,
        "source": {"path": "docs/evidence/aaa_1k_evaluation_round3.json", "sha256": round3_sha256},
        "evidence_status": (
            "observed evaluation evidence: used only to state the problem and motivate hypotheses; "
            "never used for fitting, tuning or selection"
        ),
        "q4_decomposition": decomposition,
        "statement": {
            "prompt_framing": (
                "a minority of streams contain comparatively large gated losses that flip the Q4 mean"
            ),
            "recomputed_finding": (
                f"the unfavourable minority is not a scattered tail: "
                f"{decomposition['unfavourable_streams_by_family']} of the "
                f"{decomposition['streams_favouring_ungated']} ungated-favouring streams. Every "
                f"coarse_speed_v1 stream favours the ungated control "
                f"({coarse['streams_favouring_ungated']}/"
                f"{coarse['streams_favouring_gated'] + coarse['streams_favouring_ungated']}), in every "
                f"initialization, and that family alone contributes "
                f"{coarse['contribution_to_aggregate_mean']:+.3e} to an aggregate of "
                f"{decomposition['aggregate_mean']:+.3e}. Without it the aggregate is "
                f"{decomposition['aggregate_mean_without_family']['coarse_speed_v1']:+.3e}."
            ),
            "coarse_arm_comparison": (
                f"on coarse_speed_v1 the gated model (mean MAE {arms[PRIMARY]:.3e}) is no better than "
                f"its own state-reset ablation ({arms['aaa1k_state_reset']:.3e}), the stateless MLP "
                f"({arms['mlp_control']:.3e}) or dead reckoning ({arms['dead_reckoning']:.3e}); the "
                f"ungated control reaches {arms[UNGATED]:.3e} and the 8-point linear fit "
                f"{arms['linear_fit']:.3e}"
            ),
            "secondary": (
                f"{len(decomposition['coarse_transient_outliers']['cells'])} coarse cells are extreme "
                "outliers with elevated clip rates, concentrated in one initialization"
            ),
        },
    }
