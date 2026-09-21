"""Classify, Diagnose, Hypothesize, Falsify: the Q4 coarse-family deficit.

Every stream here comes from a ``diagnostic`` block of the loop identity
ledger; none is a round-1/2/3 stream, a development-selection stream or a
characterization stream, and :mod:`research.aaa_1k_loop.identities` proves it.
Initializations are the champion's five declared ``model_init`` seeds, crossed
completely with every stream.

The hypotheses, their predictions, what would contradict each one, and the
numeric rule that turns a measurement into a verdict are declared in
:data:`HYPOTHESES` and :func:`adjudicate`, and were committed before the
diagnostic run that they judge. Verdicts are computed, never typed.

Verdict vocabulary
------------------
``SUPPORTED``     the predicted effect was measured, its interval excludes
                  zero, and it is at least the declared practical size.
``PARTIAL``       the effect was measured and excludes zero but is smaller than
                  the declared practical size.
``CONTRADICTED``  the observation the hypothesis forbids was measured.
``NOT_RESOLVED``  the evidence cannot distinguish.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed
from research.aaa_1k.stats import crossed_paired_difference
from research.aaa_1k.streams import coarse_speed_stream

from .arms import ArmSet
from .harness import Cell, arm_errors, diagnostic_series, matrix, run_cells, summary
from .identities import block_seeds, find_block

DIAGNOSIS_SCHEMA = "aaa.loop.diagnosis.v1"
DIAGNOSTIC_BLOCK = "aaa1k-loop-0001/diagnostic/coarse"
DIAGNOSTIC_STREAMS = 16
DIAGNOSTIC_INITIALIZATIONS = 5
COARSE_OPTIONS: dict[str, Any] = {"steps": 280}
"""Exactly the round-3 ``coarse_speed_v1`` construction, on fresh identities."""

CONDITIONS: dict[str, dict[str, Any]] = {
    "switching": {"regime_length": 60},
    "no_switch": {"regime_length": 10_000},
}
REGIME_LENGTH = 60
SWITCH_WINDOW = 5
"""Steps after a regime switch counted as the switch neighbourhood (H2, H9)."""

PRACTICAL_CLOSURE = 0.5
"""A probe explains the deficit only if it closes at least half of it."""

BURST_THRESHOLD = 0.025
"""A step error above five quanta (normalized) is a transient burst (P2)."""

SATURATION_THRESHOLD = 0.2
"""H3: gates are called saturated if more than 20% of gate values sit beyond 0.01/0.99."""

ARMS = ArmSet(
    (
        "gru",
        "gru_frozen",
        "gru_state_reset",
        "gru_no_error_input",
        "probe:gru_lr0.01",
        "probe:gru_T16",
        "probe:gru_keep_bias_2",
        "rnn28",
        "probe:rnn28_lr0.03",
        "probe:rnn16",
        "mlp",
        "linear_fit",
        "dead_reckoning",
    )
)
GATED_INSTRUMENTED = ("gru", "probe:gru_lr0.01", "probe:gru_T16", "probe:gru_keep_bias_2")

HYPOTHESES: tuple[dict[str, str], ...] = (
    {
        "id": "H1",
        "name": "gating mechanism failure",
        "statement": "gated recurrence is structurally unable to integrate coarse observations at this size",
        "prediction": "no parameter-neutral change to the gated model's training closes the deficit",
        "contradicted_by": "any parameter-neutral gated probe closing >= 50% of the deficit",
    },
    {
        "id": "H2",
        "name": "slow recovery after regime switches",
        "statement": "the gated model re-estimates speed slowly after an unannounced switch",
        "prediction": "the deficit is concentrated after switches and vanishes without switches",
        "contradicted_by": "a no-switch deficit >= 50% of the switching deficit",
    },
    {
        "id": "H3",
        "name": "pathological gate saturation",
        "statement": "keep/reset gates saturate at 0 or 1 and block learning",
        "prediction": "more than 20% of gate values beyond 0.01/0.99 on coarse streams",
        "contradicted_by": "gate saturation fraction at or below 20%",
    },
    {
        "id": "H4",
        "name": "gate / previous-error-input interaction",
        "statement": "the large, quantized previous-error input disrupts the gated state",
        "prediction": "removing input 3 closes >= 50% of the deficit",
        "contradicted_by": "the no-error-input ablation closing < 50% or worsening the deficit",
    },
    {
        "id": "H5",
        "name": "optimization / learning rate",
        "statement": "the gated model's higher selected learning rate (0.03 vs 0.01) inflates SGD noise on quantized targets",
        "prediction": "the gated model at lr 0.01 closes >= 50%; the ungated model at 0.03 loses >= 50% of its lead",
        "contradicted_by": "lr 0.01 closing < 50% of the deficit",
    },
    {
        "id": "H6",
        "name": "hidden width at equal budget",
        "statement": "28 ungated units simply hold more state than 16 gated units",
        "prediction": "a 16-unit ungated RNN loses >= 50% of the ungated lead",
        "contradicted_by": "the 16-unit ungated RNN keeping > 50% of the lead",
    },
    {
        "id": "H7",
        "name": "initialization sensitivity",
        "statement": "a few unlucky initializations carry the deficit",
        "prediction": "initializations disagree on the sign of the deficit",
        "contradicted_by": "every initialization showing the deficit",
    },
    {
        "id": "H8",
        "name": "TBPTT / credit assignment",
        "statement": "a 4-step truncation prevents learning a multi-step integrator",
        "prediction": "a 16-step horizon closes >= 50% of the deficit",
        "contradicted_by": "T=16 closing < 50%",
    },
    {
        "id": "H9",
        "name": "event-dominated mean",
        "statement": "a small set of event steps (switch neighbourhoods) dominates the deficit",
        "prediction": ">= 50% of the deficit falls within 5 steps after a switch",
        "contradicted_by": "< 50% of the deficit near switches",
    },
    {
        "id": "H10",
        "name": "legitimate width tradeoff; no model change warranted",
        "statement": "the ungated control's 28 units legitimately beat 16 gated units on this family",
        "prediction": "H6 supported and no parameter-neutral gated probe closes >= 50%",
        "contradicted_by": "a parameter-neutral gated probe closing >= 50% while H6 is not supported",
    },
    {
        "id": "H11",
        "name": "ordinary statistical tail",
        "statement": "the deficit is a few noisy streams, not a mechanism",
        "prediction": "fewer than 75% of fresh diagnostic streams show the deficit",
        "contradicted_by": ">= 75% of fresh streams showing the deficit",
    },
    {
        "id": "H12",
        "name": "short keep-gate memory horizon",
        "statement": (
            "with zero-initialized keep-gate biases the gated state halves every step, so it cannot "
            "average over the many coarse transitions needed to resolve speed; the model then behaves "
            "like its own stateless ablation"
        ),
        "prediction": (
            "mean keep gate near 0.5, the champion no better than its state-reset ablation, and a "
            "keep-bias initialization closing >= 50% of the deficit"
        ),
        "contradicted_by": "the keep-bias probe closing < 50%",
    },
)


def diagnostic_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    block = find_block(ledger, DIAGNOSTIC_BLOCK)
    if block["role"] != "diagnostic":
        raise ValueError("diagnosis must run on a diagnostic block")
    seeds = block_seeds(block)[:DIAGNOSTIC_STREAMS]
    cells: list[Cell] = []
    for condition, options in CONDITIONS.items():
        streams = [coarse_speed_stream(seed, **COARSE_OPTIONS, **options) for seed in seeds]
        for init in range(DIAGNOSTIC_INITIALIZATIONS):
            init_seed = derive_seed("model_init", init)
            cells.extend(Cell(condition, init, init_seed, stream) for stream in streams)
    return cells


def _profile(errors: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    since = targets % REGIME_LENGTH
    near = since < SWITCH_WINDOW
    return {
        "near_switch_sum": float(np.sum(errors[near])),
        "far_sum": float(np.sum(errors[~near])),
        "near_switch_steps": int(np.sum(near)),
    }


def reduce_cell(_cell: Cell, result: RunResult, agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    targets = np.asarray([step.target_index for step in result.steps], dtype=int)
    regimes = np.asarray([step.regime for step in result.steps])
    record: dict[str, Any] = {
        "mae": {name: float(np.mean(values)) for name, values in errors.items()},
        "max_step_error": {name: float(np.max(values)) for name, values in errors.items()},
        "regime_mae": {
            regime: {name: float(np.mean(values[regimes == regime])) for name, values in errors.items()}
            for regime in ("fast", "slow")
            if np.any(regimes == regime)
        },
        "switch_profile": {name: _profile(values, targets) for name, values in errors.items()},
        "instrumentation": {},
    }
    by_name = {agent.name: agent for agent in agents}
    for name in GATED_INSTRUMENTED:
        record["instrumentation"][name] = {
            "keep_gate_mean": summary(diagnostic_series(result, name, "update_gate_mean")),
            "reset_gate_mean": summary(diagnostic_series(result, name, "reset_gate_mean")),
            "gate_saturated_fraction": summary(diagnostic_series(result, name, "gate_saturated_fraction")),
            "hidden_norm": summary(diagnostic_series(result, name, "hidden_norm")),
            "hidden_saturated_fraction": summary(
                diagnostic_series(result, name, "hidden_saturated_fraction")
            ),
            "gradient_norm": summary(diagnostic_series(result, name, "last_gradient_norm")),
            "final_keep_bias_mean": float(np.mean(by_name[name].model.parameters["b_z"])),
            "clip_events": int(by_name[name].model.clip_events),
            "updates": int(by_name[name].model.update_count),
        }
    for name in ("rnn28", "probe:rnn16", "probe:rnn28_lr0.03"):
        record["instrumentation"][name] = {
            "hidden_norm": summary(diagnostic_series(result, name, "hidden_norm")),
            "hidden_saturated_fraction": summary(
                diagnostic_series(result, name, "hidden_saturated_fraction")
            ),
            "gradient_norm": summary(diagnostic_series(result, name, "last_gradient_norm")),
        }
    return record


def run_diagnosis(ledger: Mapping[str, Any], *, workers: int | None = None) -> list[dict[str, Any]]:
    return run_cells(diagnostic_cells(ledger), ARMS, reduce_cell, collect_diagnostics=True, workers=workers)


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------
def _compare(records: Sequence[Mapping[str, Any]], first: str, second: str, index: int) -> dict[str, Any]:
    """``second - first`` crossed over initializations and streams (positive: first better)."""

    return crossed_paired_difference(matrix(records, first), matrix(records, second), bootstrap_index=index)


def _closure(records: Sequence[Mapping[str, Any]], probe: str, deficit: float, index: int) -> dict[str, Any]:
    """How much of the champion-minus-ungated deficit does ``probe`` remove?"""

    record = _compare(records, probe, "gru", index)
    return {
        "comparison": "champion minus probe mean normalized error (positive: probe better)",
        "improvement": record,
        "closure_fraction": record["mean_difference"] / deficit if deficit > 0 else None,
    }


def _verdict_from_closure(entry: Mapping[str, Any]) -> str:
    fraction = entry["closure_fraction"]
    record = entry["improvement"]
    if fraction is None or record["interval_status"] != "MEASURED":
        return "NOT_RESOLVED"
    if record["ci_low"] > 0 and fraction >= PRACTICAL_CLOSURE:
        return "SUPPORTED"
    if record["ci_low"] > 0:
        return "PARTIAL"
    return "CONTRADICTED"


def adjudicate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute every comparison and every hypothesis verdict from the records."""

    switching = [record for record in records if record["condition"] == "switching"]
    steady = [record for record in records if record["condition"] == "no_switch"]

    deficit_record = _compare(switching, "rnn28", "gru", 100)
    deficit = float(deficit_record["mean_difference"])
    steady_deficit = _compare(steady, "rnn28", "gru", 101)
    closures = {
        probe: _closure(switching, probe, deficit, 110 + offset)
        for offset, probe in enumerate(
            ("gru_no_error_input", "probe:gru_lr0.01", "probe:gru_T16", "probe:gru_keep_bias_2")
        )
    }
    memory_use = _compare(switching, "gru", "gru_state_reset", 120)
    width = _compare(switching, "rnn28", "probe:rnn16", 121)
    ungated_at_high_lr = _compare(switching, "rnn28", "probe:rnn28_lr0.03", 122)
    width_lost = width["mean_difference"] / deficit if deficit > 0 else None
    high_lr_lost = ungated_at_high_lr["mean_difference"] / deficit if deficit > 0 else None

    # H3: champion gate saturation, pooled over switching cells
    saturation = float(
        np.mean([r["instrumentation"]["gru"]["gate_saturated_fraction"]["mean"] for r in switching])
    )
    keep_gate = float(np.mean([r["instrumentation"]["gru"]["keep_gate_mean"]["mean"] for r in switching]))
    probe_keep_gate = float(
        np.mean([r["instrumentation"]["probe:gru_keep_bias_2"]["keep_gate_mean"]["mean"] for r in switching])
    )

    # H9: share of the (summed) deficit that lies within the switch neighbourhood
    near = sum(
        r["switch_profile"]["gru"]["near_switch_sum"] - r["switch_profile"]["rnn28"]["near_switch_sum"]
        for r in switching
    )
    far = sum(
        r["switch_profile"]["gru"]["far_sum"] - r["switch_profile"]["rnn28"]["far_sum"] for r in switching
    )
    near_share = near / (near + far) if (near + far) > 0 else None

    per_stream_deficit = deficit_record["per_stream_difference"]
    stream_share = float(np.mean([value > 0 for value in per_stream_deficit]))

    bursts = {
        name: int(sum(r["max_step_error"][name] > BURST_THRESHOLD for r in records))
        for name in ("gru", "probe:gru_lr0.01", "probe:gru_T16", "probe:gru_keep_bias_2", "rnn28")
    }
    online_worse_than_frozen = int(sum(r["mae"]["gru"] > r["mae"]["gru_frozen"] for r in records))

    verdicts: dict[str, dict[str, Any]] = {}
    parameter_neutral_closures = [
        closures[p]["closure_fraction"] or 0.0
        for p in ("gru_no_error_input", "probe:gru_lr0.01", "probe:gru_T16", "probe:gru_keep_bias_2")
        if closures[p]["improvement"]["ci_low"] > 0
    ]
    any_neutral_closes = any(value >= PRACTICAL_CLOSURE for value in parameter_neutral_closures)

    verdicts["H1"] = {"verdict": "CONTRADICTED" if any_neutral_closes else "NOT_RESOLVED"}
    ratio = steady_deficit["mean_difference"] / deficit if deficit > 0 else None
    verdicts["H2"] = {
        "verdict": "NOT_RESOLVED"
        if ratio is None
        else ("CONTRADICTED" if ratio >= PRACTICAL_CLOSURE else "PARTIAL"),
        "no_switch_over_switching_deficit": ratio,
    }
    verdicts["H3"] = {
        "verdict": "SUPPORTED" if saturation > SATURATION_THRESHOLD else "CONTRADICTED",
        "gate_saturated_fraction": saturation,
    }
    verdicts["H4"] = {"verdict": _verdict_from_closure(closures["gru_no_error_input"])}
    verdicts["H5"] = {
        "verdict": _verdict_from_closure(closures["probe:gru_lr0.01"]),
        "ungated_lead_lost_at_gated_learning_rate": high_lr_lost,
    }
    verdicts["H6"] = {
        "verdict": "NOT_RESOLVED"
        if width_lost is None
        else ("SUPPORTED" if width_lost >= PRACTICAL_CLOSURE else "CONTRADICTED"),
        "ungated_lead_lost_at_16_units": width_lost,
    }
    per_init = deficit_record["per_initialization_difference"]
    verdicts["H7"] = {
        "verdict": "CONTRADICTED" if all(value > 0 for value in per_init) else "SUPPORTED",
        "per_initialization_deficit": per_init,
    }
    verdicts["H8"] = {"verdict": _verdict_from_closure(closures["probe:gru_T16"])}
    verdicts["H9"] = {
        "verdict": "NOT_RESOLVED"
        if near_share is None
        else ("SUPPORTED" if near_share >= PRACTICAL_CLOSURE else "CONTRADICTED"),
        "near_switch_share_of_deficit": near_share,
    }
    verdicts["H10"] = {
        "verdict": "SUPPORTED"
        if verdicts["H6"]["verdict"] == "SUPPORTED" and not any_neutral_closes
        else ("CONTRADICTED" if any_neutral_closes else "NOT_RESOLVED"),
    }
    verdicts["H11"] = {
        "verdict": "CONTRADICTED" if stream_share >= 0.75 else "SUPPORTED",
        "share_of_streams_with_deficit": stream_share,
    }
    h12 = _verdict_from_closure(closures["probe:gru_keep_bias_2"])
    verdicts["H12"] = {
        "verdict": h12,
        "champion_mean_keep_gate": keep_gate,
        "probe_mean_keep_gate": probe_keep_gate,
        "champion_minus_state_reset": memory_use,
    }
    return {
        "deficit": deficit_record,
        "no_switch_deficit": steady_deficit,
        "closures": closures,
        "champion_memory_use_vs_state_reset": memory_use,
        "ungated_width_16_vs_28": width,
        "ungated_at_gated_learning_rate": ungated_at_high_lr,
        "bursts_above_threshold": bursts,
        "champion_online_worse_than_frozen_cells": online_worse_than_frozen,
        "cells": len(records),
        "verdicts": verdicts,
    }
