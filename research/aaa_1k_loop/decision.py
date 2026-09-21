"""Decide: PROMOTE, REJECT or INCONCLUSIVE, computed from confirmation primitives.

Each criterion returns ``PASS``, ``FAIL`` or ``INCONCLUSIVE`` from the
per-cell records alone; nothing stored as a summary is read. The outcome rule
is fixed:

* any ``FAIL`` -> ``REJECT``;
* every criterion ``PASS`` -> ``PROMOTE``;
* otherwise -> ``INCONCLUSIVE``. A criterion that could not be measured is
  never counted as passing.

:data:`THRESHOLDS` is copied into the freeze manifest; the confirmation
command refuses to run if the live values differ from the frozen ones, and the
recomputation refuses a confirmation whose recorded thresholds differ.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.stats import crossed_paired_difference

from .harness import matrix

DECISION_SCHEMA = "aaa.loop.decision.v1"

THRESHOLDS: dict[str, Any] = {
    "confidence": 0.95,
    "bootstrap_draws": 10_000,
    "C1_min_stream_share": 0.75,
    "C1_fail_below_stream_share": 0.5,
    "C3_share_supported": 0.7,
    "C3_share_contradicted": 0.3,
    "C5_negative_mode_absent": 0.2,
    "C5_negative_mode_present": 0.5,
    "divergence_factor": 2.0,
    "bootstrap_indices": {
        "C1": 700,
        "C2_coarse": 701,
        "C2_without": 702,
        "C3_switch": 703,
        "C3_fixed": 704,
        "C4": 705,
    },
}

CRITERIA: dict[str, str] = {
    "C1_systematic_family_deficit": (
        "coarse_speed_v1 plan cells: champion minus ungated error, crossed over initializations and streams. "
        "PASS: interval above zero, >= 75% of stream means and every initialization mean positive. FAIL: "
        "interval below zero or < 50% of streams positive. Otherwise INCONCLUSIVE."
    ),
    "C2_q4_sign_carried_by_coarse": (
        "Q4 statistic (ungated minus champion error). PASS: the coarse family's crossed interval lies below "
        "zero and the all-family point estimate excluding coarse_speed_v1 is >= 0. FAIL: the coarse interval "
        "lies above zero, or the interval excluding coarse lies below zero. Otherwise INCONCLUSIVE."
    ),
    "C3_memory_rewarded_without_switches": (
        "stateless minus ungated error. PASS: on fixed-speed coarse streams the interval is above zero and "
        "the point estimate is >= 70% of the switching-stream point estimate. FAIL: fixed-speed interval "
        "below zero, or share < 30%. Otherwise INCONCLUSIVE."
    ),
    "C4_fixed_speed_observation_is_the_champions": (
        "stateless minus champion error on fixed-speed coarse streams. PASS: point estimate <= 0 (the "
        "original probe's observation reproduces). FAIL: interval above zero. Otherwise INCONCLUSIVE."
    ),
    "C5_mechanism_signature": (
        "coarse_speed_v1 plan cells. PASS: mean slow-regime deficit (champion minus ungated) exceeds the "
        "fast-regime deficit, champion median sign-alternating-mode magnitude < 0.2 and ungated > 0.5. "
        "FAIL: slow deficit <= fast deficit, or champion mode >= 0.5. Otherwise INCONCLUSIVE."
    ),
}


def _crossed(records: Sequence[Mapping[str, Any]], first: str, second: str, index: int) -> dict[str, Any]:
    return crossed_paired_difference(
        matrix(records, first),
        matrix(records, second),
        bootstrap_index=index,
        draws=THRESHOLDS["bootstrap_draws"],
        confidence=THRESHOLDS["confidence"],
    )


def _require_complete(records: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> None:
    for record in records:
        missing = [arm for arm in arms if arm not in record["mae"]]
        if missing or record.get("failures"):
            raise ValueError(
                f"confirmation cell {record['stream_id']} is incomplete: {missing or record['failures']}"
            )


def evaluate(primitives: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Every criterion, from the confirmation records."""

    plan = list(primitives["plan"])
    fixed = list(primitives["fixed_speed"])
    long = list(primitives["long_fixed_speed"])
    _require_complete(plan, ("gru", "rnn28", "mlp"))
    _require_complete(fixed, ("gru", "rnn28", "mlp"))
    indices = THRESHOLDS["bootstrap_indices"]
    coarse = [record for record in plan if record["family"] == "coarse_speed_v1"]
    other = [record for record in plan if record["family"] != "coarse_speed_v1"]
    results: dict[str, Any] = {}

    deficit = _crossed(coarse, "rnn28", "gru", indices["C1"])
    share = deficit["favours_first"] / deficit["streams"]
    if (
        deficit["ci_low"] > 0
        and share >= THRESHOLDS["C1_min_stream_share"]
        and all(value > 0 for value in deficit["per_initialization_difference"])
    ):
        status = "PASS"
    elif deficit["ci_high"] < 0 or share < THRESHOLDS["C1_fail_below_stream_share"]:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    results["C1_systematic_family_deficit"] = {"status": status, "deficit": deficit, "stream_share": share}

    coarse_q4 = _crossed(coarse, "gru", "rnn28", indices["C2_coarse"])
    without = _crossed(other, "gru", "rnn28", indices["C2_without"])
    aggregate = float(np.mean([record["mae"]["rnn28"] - record["mae"]["gru"] for record in plan]))
    contributions = {
        family: float(
            np.sum([r["mae"]["rnn28"] - r["mae"]["gru"] for r in plan if r["family"] == family]) / len(plan)
        )
        for family in sorted({r["family"] for r in plan})
    }
    if coarse_q4["ci_high"] < 0 and without["mean_difference"] >= 0:
        status = "PASS"
    elif coarse_q4["ci_low"] > 0 or without["ci_high"] < 0:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    results["C2_q4_sign_carried_by_coarse"] = {
        "status": status,
        "coarse_family_q4": coarse_q4,
        "q4_without_coarse": without,
        "q4_aggregate_mean": aggregate,
        "contribution_by_family": contributions,
    }

    switching = _crossed(coarse, "rnn28", "mlp", indices["C3_switch"])
    steady = _crossed(fixed, "rnn28", "mlp", indices["C3_fixed"])
    memory_share = (
        steady["mean_difference"] / switching["mean_difference"] if switching["mean_difference"] > 0 else None
    )
    if memory_share is not None and steady["ci_low"] > 0 and memory_share >= THRESHOLDS["C3_share_supported"]:
        status = "PASS"
    elif steady["ci_high"] < 0 or (
        memory_share is not None and memory_share < THRESHOLDS["C3_share_contradicted"]
    ):
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    results["C3_memory_rewarded_without_switches"] = {
        "status": status,
        "ungated_memory_advantage_switching": switching,
        "ungated_memory_advantage_fixed_speed": steady,
        "share": memory_share,
    }

    champion_fixed = _crossed(fixed, "gru", "mlp", indices["C4"])
    if champion_fixed["mean_difference"] <= 0:
        status = "PASS"
    elif champion_fixed["ci_low"] > 0:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    results["C4_fixed_speed_observation_is_the_champions"] = {
        "status": status,
        "champion_memory_advantage_fixed_speed": champion_fixed,
    }

    regime = {
        label: float(
            np.mean(
                [
                    r["regime_mae"][label]["gru"] - r["regime_mae"][label]["rnn28"]
                    for r in coarse
                    if label in r["regime_mae"]
                ]
            )
        )
        for label in ("fast", "slow")
    }
    modes = {
        name: float(np.median([r["jacobian_negative_mode_median"][name] for r in coarse]))
        for name in ("gru", "rnn28")
    }
    if (
        regime["slow"] > regime["fast"]
        and modes["gru"] < THRESHOLDS["C5_negative_mode_absent"]
        and modes["rnn28"] > THRESHOLDS["C5_negative_mode_present"]
    ):
        status = "PASS"
    elif regime["slow"] <= regime["fast"] or modes["gru"] >= THRESHOLDS["C5_negative_mode_present"]:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    results["C5_mechanism_signature"] = {
        "status": status,
        "regime_deficit": regime,
        "jacobian_negative_mode": modes,
    }

    divergent = sum(r["mae"]["gru"] > THRESHOLDS["divergence_factor"] * r["mae"]["persistence"] for r in long)
    secondary = {
        "long_fixed_speed_champion_divergent_cells": int(divergent),
        "long_fixed_speed_cells": len(long),
        "long_fixed_speed_champion_mean": float(np.mean([r["mae"]["gru"] for r in long])) if long else None,
        "long_fixed_speed_persistence_mean": float(np.mean([r["mae"]["persistence"] for r in long]))
        if long
        else None,
        "note": "descriptive replication of the M2 runaway; not a decision criterion",
    }
    return {"criteria": results, "secondary": secondary}


def outcome_of(statuses: Mapping[str, str]) -> str:
    values = list(statuses.values())
    if not values:
        return "INCONCLUSIVE"
    if any(value == "FAIL" for value in values):
        return "REJECT"
    if all(value == "PASS" for value in values):
        return "PROMOTE"
    return "INCONCLUSIVE"


def decide(confirmation: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the decision from a confirmation artifact's primitives."""

    if confirmation.get("thresholds") != THRESHOLDS:
        raise ValueError("confirmation thresholds differ from the decision thresholds")
    evaluation = evaluate(confirmation["primitives"])
    statuses = {name: entry["status"] for name, entry in evaluation["criteria"].items()}
    missing = sorted(set(CRITERIA) - set(statuses))
    if missing:
        raise ValueError(f"criteria not evaluated: {missing}")
    return {
        "schema": DECISION_SCHEMA,
        "statuses": statuses,
        "outcome": outcome_of(statuses),
        "evaluation": evaluation,
    }
