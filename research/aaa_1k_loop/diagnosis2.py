"""Diagnosis round 2: hypotheses formed from round 1, tested on fresh streams.

Round 1 contradicted ten of twelve predeclared hypotheses and left one clue:
in the *slow* regime the online gated model is no better than its own frozen,
persistence-like copy, while the ungated control is far better. Slow motion is
0.48 quanta per step, so predicting the next *quantized* observation means
tracking a sub-quantum phase that alternates roughly every step. With zero
gate biases the gated core starts with ``z = r = 0.5``, so its one-step
Jacobian is roughly ``0.5 I + 0.25 U_n``: every mode decays within a few steps
and none alternates in sign. The ungated core starts near spectral radius one.

These hypotheses were declared after round 1 and are judged only on a fresh
diagnostic block (``diagnostic/coarse-2``); round-1 streams cannot confirm a
hypothesis they inspired.
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

DIAGNOSIS2_SCHEMA = "aaa.loop.diagnosis.v2"
DIAGNOSTIC_BLOCK_2 = "aaa1k-loop-0001/diagnostic/coarse-2"
STREAMS = 16
INITIALIZATIONS = 5
CONDITIONS: dict[str, dict[str, Any]] = {
    "switching": {"steps": 280, "regime_length": 60},
    "no_switch": {"steps": 280, "regime_length": 10_000},
    "long_no_switch": {"steps": 1120, "regime_length": 10_000},
}
PRACTICAL_CLOSURE = 0.5
NEGATIVE_MODE_ABSENT = 0.2
NEGATIVE_MODE_PRESENT = 0.5

GAIN_PROBES = ("probe:gru_keep_bias_-2", "probe:gru_reset_bias_2", "probe:gru_keep_-2_reset_2")
ARMS = ArmSet(
    (
        "gru",
        "gru_state_reset",
        "probe:gru_keep_bias_2",
        *GAIN_PROBES,
        "rnn28",
        "mlp",
        "linear_fit",
    )
)
DYNAMICS_ARMS = ("gru", "probe:gru_keep_bias_2", *GAIN_PROBES, "rnn28")

HYPOTHESES: tuple[dict[str, str], ...] = (
    {
        "id": "H13",
        "name": "contractive gated dynamics cannot carry the sub-quantum phase",
        "statement": (
            "zero gate biases confine the gated core's one-step Jacobian near 0.5 I + 0.25 U_n; it has no "
            "sign-alternating mode, so within one episode it cannot learn the roughly period-2 phase of "
            "slow quantized motion, and behaves like a stateless predictor there"
        ),
        "prediction": (
            "(a) champion negative-mode magnitude < 0.2 while the ungated control's > 0.5; (b) the slow-regime "
            "deficit exceeds the fast-regime deficit; (c) at least one gain-raising bias initialization closes "
            ">= 50% of the deficit with an interval above zero; (d) the keep-bias +2 probe, which moves "
            "eigenvalues toward +1 rather than -1, does not"
        ),
        "contradicted_by": "no gain-raising probe closing >= 50% of the deficit",
    },
    {
        "id": "H14",
        "name": "learning speed rather than representational inability",
        "statement": "the gated model can represent the phase but needs more than one short episode to learn it",
        "prediction": (
            "on 1120-step fixed-speed streams the champion's last-quarter deficit to the ungated control "
            "is < 50% of its first-quarter deficit"
        ),
        "contradicted_by": "a last-quarter deficit >= 50% of the first-quarter deficit",
    },
    {
        "id": "H15",
        "name": "coarse_speed_v1 rewards memory even without regime switches",
        "statement": (
            "the existing characterization concluded that without regime switches the quantizer alone "
            "hands the advantage to the stateless arm; that was measured with the gated model only"
        ),
        "prediction": (
            "with a recurrent model able to use memory (the ungated control), the no-switch advantage over "
            "the stateless control is >= 70% of its switching-condition advantage"
        ),
        "contradicted_by": "a no-switch ungated-over-stateless advantage < 30% of the switching one",
    },
)


def diagnostic_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    block = find_block(ledger, DIAGNOSTIC_BLOCK_2)
    if block["role"] != "diagnostic":
        raise ValueError("diagnosis round 2 must run on a diagnostic block")
    seeds = block_seeds(block)[:STREAMS]
    cells: list[Cell] = []
    for condition, options in CONDITIONS.items():
        streams = [coarse_speed_stream(seed, **options) for seed in seeds]
        for init in range(INITIALIZATIONS):
            cells.extend(Cell(condition, init, derive_seed("model_init", init), stream) for stream in streams)
    return cells


def reduce_cell(_cell: Cell, result: RunResult, _agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    regimes = np.asarray([step.regime for step in result.steps])
    quarter = max(1, len(result.steps) // 4)
    return {
        "mae": {name: float(np.mean(values)) for name, values in errors.items()},
        "first_quarter_mae": {name: float(np.mean(values[:quarter])) for name, values in errors.items()},
        "last_quarter_mae": {name: float(np.mean(values[-quarter:])) for name, values in errors.items()},
        "regime_mae": {
            regime: {name: float(np.mean(values[regimes == regime])) for name, values in errors.items()}
            for regime in ("fast", "slow")
            if np.any(regimes == regime)
        },
        "dynamics": {
            name: {
                field: summary(diagnostic_series(result, name, field))["median"]
                for field in (
                    "jacobian_spectral_radius",
                    "jacobian_min_real",
                    "jacobian_negative_mode",
                    "update_gate_mean",
                    "reset_gate_mean",
                )
            }
            for name in DYNAMICS_ARMS
        },
    }


def run_diagnosis2(ledger: Mapping[str, Any], *, workers: int | None = None) -> list[dict[str, Any]]:
    return run_cells(diagnostic_cells(ledger), ARMS, reduce_cell, collect_diagnostics=True, workers=workers)


def _compare(records: Sequence[Mapping[str, Any]], first: str, second: str, index: int, key: str = "mae"):
    return crossed_paired_difference(
        matrix(records, first, key=key), matrix(records, second, key=key), bootstrap_index=index
    )


def _mean(records: Sequence[Mapping[str, Any]], *path: str) -> float:
    values = []
    for record in records:
        node: Any = record
        for key in path:
            node = node[key]
        if node is not None:
            values.append(float(node))
    return float(np.mean(values)) if values else float("nan")


def adjudicate2(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    switching = [r for r in records if r["condition"] == "switching"]
    steady = [r for r in records if r["condition"] == "no_switch"]
    long = [r for r in records if r["condition"] == "long_no_switch"]

    deficit = _compare(switching, "rnn28", "gru", 200)
    size = float(deficit["mean_difference"])
    closures: dict[str, Any] = {}
    for offset, probe in enumerate(("probe:gru_keep_bias_2", *GAIN_PROBES)):
        improvement = _compare(switching, probe, "gru", 210 + offset)
        closures[probe] = {
            "improvement": improvement,
            "closure_fraction": improvement["mean_difference"] / size if size > 0 else None,
        }

    def closes(probe: str) -> bool:
        entry = closures[probe]
        return bool(
            entry["closure_fraction"] is not None
            and entry["improvement"]["ci_low"] > 0
            and entry["closure_fraction"] >= PRACTICAL_CLOSURE
        )

    regime_deficit = {
        regime: _mean(switching, "regime_mae", regime, "gru")
        - _mean(switching, "regime_mae", regime, "rnn28")
        for regime in ("fast", "slow")
    }
    modes = {name: _mean(switching, "dynamics", name, "jacobian_negative_mode") for name in DYNAMICS_ARMS}
    radii = {name: _mean(switching, "dynamics", name, "jacobian_spectral_radius") for name in DYNAMICS_ARMS}
    min_real = {name: _mean(switching, "dynamics", name, "jacobian_min_real") for name in DYNAMICS_ARMS}

    part_a = modes["gru"] < NEGATIVE_MODE_ABSENT and modes["rnn28"] > NEGATIVE_MODE_PRESENT
    part_b = regime_deficit["slow"] > regime_deficit["fast"]
    part_c = any(closes(probe) for probe in GAIN_PROBES)
    part_d = not closes("probe:gru_keep_bias_2")
    h13 = (
        "SUPPORTED"
        if (part_a and part_b and part_c and part_d)
        else ("CONTRADICTED" if not part_c else "PARTIAL")
    )

    first_quarter = _compare(long, "rnn28", "gru", 220, key="first_quarter_mae")
    last_quarter = _compare(long, "rnn28", "gru", 221, key="last_quarter_mae")
    ratio = (
        last_quarter["mean_difference"] / first_quarter["mean_difference"]
        if first_quarter["mean_difference"] > 0
        else None
    )
    h14 = "NOT_RESOLVED" if ratio is None else ("SUPPORTED" if ratio < PRACTICAL_CLOSURE else "CONTRADICTED")

    memory_switching = _compare(switching, "rnn28", "mlp", 230)
    memory_steady = _compare(steady, "rnn28", "mlp", 231)
    share = (
        memory_steady["mean_difference"] / memory_switching["mean_difference"]
        if memory_switching["mean_difference"] > 0
        else None
    )
    if share is None:
        h15 = "NOT_RESOLVED"
    elif share >= 0.7:
        h15 = "SUPPORTED"
    elif share < 0.3:
        h15 = "CONTRADICTED"
    else:
        h15 = "PARTIAL"
    gated_memory_steady = _compare(steady, "gru", "mlp", 232)

    return {
        "deficit": deficit,
        "closures": closures,
        "regime_deficit_mean": regime_deficit,
        "jacobian_median_negative_mode": modes,
        "jacobian_median_spectral_radius": radii,
        "jacobian_median_min_real": min_real,
        "long_stream_first_quarter_deficit": first_quarter,
        "long_stream_last_quarter_deficit": last_quarter,
        "ungated_over_stateless_switching": memory_switching,
        "ungated_over_stateless_no_switch": memory_steady,
        "gated_over_stateless_no_switch": gated_memory_steady,
        "cells": len(records),
        "verdicts": {
            "H13": {
                "verdict": h13,
                "a_champion_lacks_negative_mode_and_ungated_has_one": part_a,
                "b_slow_deficit_exceeds_fast": part_b,
                "c_a_gain_probe_closes_half": part_c,
                "d_keep_bias_plus_2_does_not_close": part_d,
            },
            "H14": {"verdict": h14, "last_over_first_quarter_deficit": ratio},
            "H15": {"verdict": h15, "no_switch_over_switching_memory_advantage": share},
        },
    }
