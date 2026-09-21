"""Iteration aaa1k-loop-0003: repairing the Q4 interpretation (a claim, not a model).

Iterations 0001 and 0002 rejected six model candidates before any freeze. What
their diagnoses *did* establish contradicts claims the repository currently
makes about Q4 and about ``coarse_speed_v1``:

* the report and handoff describe Q4's unfavourable mean as a *minority of
  streams* with large gated losses; recomputed from round-3 primitives it is
  one entire family, ``coarse_speed_v1`` (24/24 streams, every
  initialization), and without that family the Q4 aggregate is positive;
* ``docs/limitations.md`` and the characterization finding state that
  ``coarse_speed_v1`` "genuinely tests hidden-regime inference" because, with
  the speed held fixed, the recurrent model is worse than the stateless
  control. That probe used only the gated champion. With the ungated control,
  diagnosis round 2 measured the memory advantage without switches at 99% of
  its size with switches (H15), and round 2 attributed the champion's failure
  to its zero-bias gate dynamics (H13).

This iteration's *challenger* is therefore a corrected interpretation, and its
*promotion* means adopting it in the documentation (issue ``AAA-162``). The
champion model is not touched in any outcome. The outer loop is the same one a
model challenger would face: attack on attack identities, a committed freeze,
fresh confirmation identities spent at observation, an independent
recomputation, and a PROMOTE / REJECT / INCONCLUSIVE decision computed from
primitives by :mod:`research.aaa_1k_loop.decision`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.experiments import EVALUATION_PLAN
from research.aaa_1k.runner import RunResult
from research.aaa_1k.stats import crossed_paired_difference
from research.aaa_1k.streams import build_stream, coarse_speed_stream

from .arms import ArmSet
from .develop import plan_entry_name
from .harness import Cell, arm_errors, diagnostic_series, matrix, run_cells, summary
from .identities import block_seeds, find_block, require_usable

ITERATION_ID = "aaa1k-loop-0003"
CLAIM_ID_V2 = "aaa1k-claim-q4-coarse-v2"
CLAIM_ID = "aaa1k-claim-q4-coarse-v3"

CHAMPION_CLAIM = (
    "Q4's unfavourable mean comes from a minority of streams with large gated losses; coarse_speed_v1 "
    "genuinely tests hidden-regime inference, because with the speed held fixed the recurrent model is "
    "worse than the stateless control (the quantizer alone hands the advantage to the stateless arm)."
)
CHALLENGER_CLAIM_V2 = (
    "Q4's unfavourable mean is carried by one family, coarse_speed_v1, where the gated champion is "
    "systematically worse than the ungated control in every initialization; without that family the Q4 "
    "aggregate is not negative. coarse_speed_v1 rewards memory even with the speed held fixed: a recurrent "
    "model able to track the sub-quantum phase (the ungated control) keeps most of its memory advantage "
    "without regime switches. The fixed-speed probe's observation -- the champion no better than the "
    "stateless control -- is a limitation of the champion's zero-bias gate dynamics, concentrated in the "
    "slow regime, where its one-step Jacobian has no sign-alternating mode."
)
"""Rejected by the first attack (T1): true at the declared construction, not across nearby ones."""

CHALLENGER_CLAIM = (
    "At the declared coarse_speed_v1 construction (quantum 0.005, speeds 0.12 and 0.30, regime length 60): "
    "Q4's unfavourable mean is carried by that family, where the gated champion is systematically worse "
    "than the ungated control in every initialization, and without it the Q4 aggregate is not negative. "
    "With the speed held fixed, the ungated control keeps most (>= 70%) of its memory advantage over the "
    "stateless control while the champion gains nothing, so the fixed-speed probe's observation is a "
    "limitation of the champion's zero-bias gate dynamics -- concentrated in the slow regime, where its "
    "one-step Jacobian has no sign-alternating mode -- not a property of the benchmark. How much of the "
    "family's memory advantage is regime inference rather than sub-quantum phase integration depends on "
    "the speed-to-quantum ratio: at quantum 0.006 or speeds 0.10/0.25 most of it needs the switches. "
    "Neither description is construction-independent."
)
"""Claim v3: v2 scoped to the declared construction, carrying the boundary the first attack found."""

ATTACK_ENV_BLOCK = f"{ITERATION_ID}/attack/env"
ATTACK_INIT_BLOCK = f"{ITERATION_ID}/attack/init"
ATTACK2_ENV_BLOCK = f"{ITERATION_ID}/attack-2/env"
ATTACK2_INIT_BLOCK = f"{ITERATION_ID}/attack-2/init"
CONFIRMATION_ENV_BLOCK = f"{ITERATION_ID}/confirmation/env"
CONFIRMATION_INIT_BLOCK = f"{ITERATION_ID}/confirmation/init"

COARSE: dict[str, Any] = {"steps": 280}
NO_SWITCH: dict[str, Any] = {"steps": 280, "regime_length": 10_000}
LONG_NO_SWITCH: dict[str, Any] = {"steps": 1120, "regime_length": 10_000}
NEARBY: dict[str, dict[str, Any]] = {
    "quantum_0.004": {"quantum": 0.004},
    "quantum_0.006": {"quantum": 0.006},
    "speeds_0.10_0.25": {"slow_speed": 0.10, "fast_speed": 0.25},
    "speeds_0.15_0.35": {"slow_speed": 0.15, "fast_speed": 0.35},
}
SHARE_SUPPORTED = 0.7
SHARE_CONTRADICTED = 0.3

# ----------------------------------------------------------------------
# attack
# ----------------------------------------------------------------------
ATTACK_CRITERIA = {
    "T1_nearby_conditions": (
        "in at least 3 of 4 nearby coarse constructions, the ungated control's fixed-speed memory advantage "
        "over the stateless control is >= 70% of its switching advantage, and the champion is not better "
        "than the stateless control at fixed speed (point estimate <= 0)"
    ),
    "T2_other_memory_capable_model": (
        "a 16-unit ungated RNN (354 parameters) also keeps >= 70% of its memory advantage at fixed speed"
    ),
    "T3_fresh_initializations": (
        "on round-3-construction coarse streams the champion's deficit to the ungated control is positive "
        "in every fresh attack initialization"
    ),
    "T4_gated_model_with_fast_dynamics": (
        "the keep-bias -2 gated probe (iteration 0001's c2, used here only as a probe) has a positive "
        "fixed-speed memory advantage over the stateless control with an interval above zero: gating as "
        "such does not forbid it"
    ),
}
ATTACK_ARMS = ArmSet(("gru", "rnn28", "probe:rnn16", "probe:gru_keep_bias_-2", "mlp"))


def attack_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    require_usable(ledger, ATTACK_ENV_BLOCK, purpose="selection")
    require_usable(ledger, ATTACK_INIT_BLOCK, purpose="selection")
    env = block_seeds(find_block(ledger, ATTACK_ENV_BLOCK))
    inits = block_seeds(find_block(ledger, ATTACK_INIT_BLOCK))
    cells: list[Cell] = []
    for label, options in NEARBY.items():
        for variant, base in (("switching", COARSE), ("no_switch", NO_SWITCH)):
            streams = [coarse_speed_stream(seed, **{**base, **options}) for seed in env[0:8]]
            cells.extend(
                Cell(f"{label}/{variant}", i, init, stream)
                for stream in streams
                for i, init in enumerate(inits)
            )
    for variant, base in (("switching", COARSE), ("no_switch", NO_SWITCH)):
        streams = [coarse_speed_stream(seed, **base) for seed in env[8:24]]
        cells.extend(
            Cell(f"standard/{variant}", i, init, stream) for stream in streams for i, init in enumerate(inits)
        )
    return cells


def reduce_mae(_cell: Cell, result: RunResult, _agents: Sequence[Any]) -> dict[str, Any]:
    return {"mae": {name: float(np.mean(values)) for name, values in arm_errors(result).items()}}


def run_attack(ledger: Mapping[str, Any], *, workers: int | None = None) -> list[dict[str, Any]]:
    return run_cells(attack_cells(ledger), ATTACK_ARMS, reduce_mae, workers=workers, isolate_failures=True)


def _advantage(records: Sequence[Mapping[str, Any]], arm: str, index: int) -> dict[str, Any]:
    """Stateless minus ``arm`` error: positive when memory helps ``arm``."""

    return crossed_paired_difference(matrix(records, arm), matrix(records, "mlp"), bootstrap_index=index)


def adjudicate_attack(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rows(condition: str) -> list[Mapping[str, Any]]:
        return [record for record in records if record["condition"] == condition]

    nearby: dict[str, Any] = {}
    holding = 0
    for offset, label in enumerate(NEARBY):
        switching = _advantage(rows(f"{label}/switching"), "rnn28", 600 + 3 * offset)
        steady = _advantage(rows(f"{label}/no_switch"), "rnn28", 601 + 3 * offset)
        champion = _advantage(rows(f"{label}/no_switch"), "gru", 602 + 3 * offset)
        share = (
            steady["mean_difference"] / switching["mean_difference"]
            if switching["mean_difference"] > 0
            else None
        )
        ok = share is not None and share >= SHARE_SUPPORTED and champion["mean_difference"] <= 0
        holding += int(ok)
        nearby[label] = {
            "ungated_switching": switching,
            "ungated_no_switch": steady,
            "champion_no_switch": champion,
            "share": share,
            "holds": ok,
        }
    switching16 = _advantage(rows("standard/switching"), "probe:rnn16", 620)
    steady16 = _advantage(rows("standard/no_switch"), "probe:rnn16", 621)
    share16 = (
        steady16["mean_difference"] / switching16["mean_difference"]
        if switching16["mean_difference"] > 0
        else None
    )
    deficit = crossed_paired_difference(
        matrix(rows("standard/switching"), "rnn28"),
        matrix(rows("standard/switching"), "gru"),
        bootstrap_index=622,
    )
    gated_fast = _advantage(rows("standard/no_switch"), "probe:gru_keep_bias_-2", 623)
    outcome = {
        "T1_nearby_conditions": {"conditions": nearby, "holding": holding, "passed": holding >= 3},
        "T2_other_memory_capable_model": {
            "switching": switching16,
            "no_switch": steady16,
            "share": share16,
            "passed": share16 is not None and share16 >= SHARE_SUPPORTED,
        },
        "T3_fresh_initializations": {
            "deficit": deficit,
            "passed": all(value > 0 for value in deficit["per_initialization_difference"]),
        },
        "T4_gated_model_with_fast_dynamics": {"advantage": gated_fast, "passed": gated_fast["ci_low"] > 0},
    }
    return {
        "criteria": ATTACK_CRITERIA,
        "outcome": outcome,
        "advance_to_freeze": all(entry["passed"] for entry in outcome.values()),
    }


# ----------------------------------------------------------------------
# attack 2: the scoped claim v3, on fresh attack identities
# ----------------------------------------------------------------------
ATTACK2_CRITERIA = {
    "R1_fresh_initializations": (
        "declared construction: the champion's deficit to the ungated control is positive in every fresh "
        "initialization and its interval lies above zero"
    ),
    "R2_core_claim": (
        "declared construction at fixed speed: the ungated memory advantage over the stateless control has "
        "an interval above zero and is >= 70% of its switching advantage; the champion's fixed-speed memory "
        "advantage point estimate is <= 0"
    ),
    "R3_other_memory_capable_model": "the 16-unit ungated RNN keeps >= 70% of its memory advantage at fixed speed",
    "R4_gated_model_with_fast_dynamics": (
        "the keep-bias -2 gated probe's fixed-speed memory advantage over the stateless control has an "
        "interval above zero"
    ),
    "R5_declared_boundary_replicates": (
        "at quantum 0.006 and at speeds 0.10/0.25 the ungated fixed-speed share is below 70%, as the claim "
        "states; if either reaches 70% the claim's boundary statement is wrong"
    ),
}
BOUNDARY = ("quantum_0.006", "speeds_0.10_0.25")


def attack2_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    require_usable(ledger, ATTACK2_ENV_BLOCK, purpose="selection")
    require_usable(ledger, ATTACK2_INIT_BLOCK, purpose="selection")
    env = block_seeds(find_block(ledger, ATTACK2_ENV_BLOCK))
    inits = block_seeds(find_block(ledger, ATTACK2_INIT_BLOCK))
    cells: list[Cell] = []
    for variant, base in (("switching", COARSE), ("no_switch", NO_SWITCH)):
        streams = [coarse_speed_stream(seed, **base) for seed in env[0:16]]
        cells.extend(
            Cell(f"standard/{variant}", i, init, stream) for stream in streams for i, init in enumerate(inits)
        )
    for label in BOUNDARY:
        for variant, base in (("switching", COARSE), ("no_switch", NO_SWITCH)):
            streams = [coarse_speed_stream(seed, **{**base, **NEARBY[label]}) for seed in env[16:24]]
            cells.extend(
                Cell(f"{label}/{variant}", i, init, stream)
                for stream in streams
                for i, init in enumerate(inits)
            )
    return cells


def run_attack2(ledger: Mapping[str, Any], *, workers: int | None = None) -> list[dict[str, Any]]:
    return run_cells(attack2_cells(ledger), ATTACK_ARMS, reduce_mae, workers=workers, isolate_failures=True)


def _share(steady: Mapping[str, Any], switching: Mapping[str, Any]) -> float | None:
    return (
        steady["mean_difference"] / switching["mean_difference"] if switching["mean_difference"] > 0 else None
    )


def adjudicate_attack2(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rows(condition: str) -> list[Mapping[str, Any]]:
        return [record for record in records if record["condition"] == condition]

    deficit = crossed_paired_difference(
        matrix(rows("standard/switching"), "rnn28"),
        matrix(rows("standard/switching"), "gru"),
        bootstrap_index=650,
    )
    switching = _advantage(rows("standard/switching"), "rnn28", 651)
    steady = _advantage(rows("standard/no_switch"), "rnn28", 652)
    champion = _advantage(rows("standard/no_switch"), "gru", 653)
    share = _share(steady, switching)
    switching16 = _advantage(rows("standard/switching"), "probe:rnn16", 654)
    steady16 = _advantage(rows("standard/no_switch"), "probe:rnn16", 655)
    share16 = _share(steady16, switching16)
    gated_fast = _advantage(rows("standard/no_switch"), "probe:gru_keep_bias_-2", 656)
    boundary: dict[str, dict[str, Any]] = {}
    for offset, label in enumerate(BOUNDARY):
        b_switch = _advantage(rows(f"{label}/switching"), "rnn28", 660 + 2 * offset)
        b_steady = _advantage(rows(f"{label}/no_switch"), "rnn28", 661 + 2 * offset)
        boundary[label] = {"switching": b_switch, "no_switch": b_steady, "share": _share(b_steady, b_switch)}
    outcome = {
        "R1_fresh_initializations": {
            "deficit": deficit,
            "passed": deficit["ci_low"] > 0 and all(v > 0 for v in deficit["per_initialization_difference"]),
        },
        "R2_core_claim": {
            "switching": switching,
            "no_switch": steady,
            "share": share,
            "champion_no_switch": champion,
            "passed": share is not None
            and steady["ci_low"] > 0
            and share >= SHARE_SUPPORTED
            and champion["mean_difference"] <= 0,
        },
        "R3_other_memory_capable_model": {
            "switching": switching16,
            "no_switch": steady16,
            "share": share16,
            "passed": share16 is not None and share16 >= SHARE_SUPPORTED,
        },
        "R4_gated_model_with_fast_dynamics": {"advantage": gated_fast, "passed": gated_fast["ci_low"] > 0},
        "R5_declared_boundary_replicates": {
            "conditions": boundary,
            "passed": all(
                row["share"] is not None and row["share"] < SHARE_SUPPORTED for row in boundary.values()
            ),
        },
    }
    return {
        "criteria": ATTACK2_CRITERIA,
        "outcome": outcome,
        "advance_to_freeze": all(entry["passed"] for entry in outcome.values()),
    }


# ----------------------------------------------------------------------
# confirmation design (frozen by the freeze manifest)
# ----------------------------------------------------------------------
CONFIRMATION_DESIGN: dict[str, Any] = {
    "plan": "round-3 evaluation plan: 6 entries x 24 streams, fully crossed with 5 fresh initializations",
    "streams_per_plan_entry": 24,
    "fixed_speed_coarse_streams": 24,
    "long_fixed_speed_coarse_streams": 16,
    "initializations": 5,
    "env_seed_slices": {"plan": [0, 144], "fixed_speed": [144, 168], "long_fixed_speed": [168, 184]},
    "arms_plan": ["gru", "gru_state_reset", "rnn28", "mlp", "persistence"],
    "arms_fixed_speed": ["gru", "rnn28", "mlp", "persistence"],
    "arms_long": ["gru", "persistence"],
    "instrumented": "coarse_speed_v1 plan cells collect the read-only one-step Jacobian",
}
CONFIRMATION_ENV_COUNT = 184
CONFIRMATION_INIT_COUNT = 5
PLAN_ARMS = ArmSet(tuple(CONFIRMATION_DESIGN["arms_plan"]))
FIXED_ARMS = ArmSet(tuple(CONFIRMATION_DESIGN["arms_fixed_speed"]))
LONG_ARMS = ArmSet(tuple(CONFIRMATION_DESIGN["arms_long"]))


def confirmation_cells(ledger: Mapping[str, Any]) -> dict[str, list[Cell]]:
    env_block = require_usable(ledger, CONFIRMATION_ENV_BLOCK, purpose="confirmation")
    init_block = require_usable(ledger, CONFIRMATION_INIT_BLOCK, purpose="confirmation")
    env = block_seeds(env_block)
    inits = block_seeds(init_block)
    if len(env) != CONFIRMATION_ENV_COUNT or len(inits) != CONFIRMATION_INIT_COUNT:
        raise ValueError("confirmation blocks do not match the frozen design")
    plan: list[Cell] = []
    cursor = 0
    for family, options in EVALUATION_PLAN:
        entry = plan_entry_name(family, options)
        for _ in range(CONFIRMATION_DESIGN["streams_per_plan_entry"]):
            stream = build_stream(family, env[cursor], **dict(options))
            plan.extend(Cell(entry, i, init, stream) for i, init in enumerate(inits))
            cursor += 1
    fixed = [
        Cell("coarse_fixed_speed", i, init, coarse_speed_stream(seed, **NO_SWITCH))
        for seed in env[144:168]
        for i, init in enumerate(inits)
    ]
    long = [
        Cell("coarse_long_fixed_speed", i, init, coarse_speed_stream(seed, **LONG_NO_SWITCH))
        for seed in env[168:184]
        for i, init in enumerate(inits)
    ]
    return {"plan": plan, "fixed_speed": fixed, "long_fixed_speed": long}


def reduce_confirmation(cell: Cell, result: RunResult, _agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    record: dict[str, Any] = {
        "plan_entry": cell.condition,
        "mae": {name: float(np.mean(values)) for name, values in errors.items()},
    }
    if cell.stream.family == "coarse_speed_v1":
        regimes = np.asarray([step.regime for step in result.steps])
        record["regime_mae"] = {
            regime: {name: float(np.mean(values[regimes == regime])) for name, values in errors.items()}
            for regime in ("fast", "slow")
            if np.any(regimes == regime)
        }
        if result.steps and result.steps[0].diagnostics:
            record["jacobian_negative_mode_median"] = {
                name: summary(diagnostic_series(result, name, "jacobian_negative_mode"))["median"]
                for name in ("gru", "rnn28")
            }
    return record


def run_confirmation(
    ledger: Mapping[str, Any], *, workers: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    cells = confirmation_cells(ledger)
    coarse = [cell for cell in cells["plan"] if cell.stream.family == "coarse_speed_v1"]
    other = [cell for cell in cells["plan"] if cell.stream.family != "coarse_speed_v1"]
    plan = run_cells(other, PLAN_ARMS, reduce_confirmation, workers=workers, isolate_failures=True)
    plan += run_cells(
        coarse,
        PLAN_ARMS,
        reduce_confirmation,
        workers=workers,
        isolate_failures=True,
        collect_diagnostics=True,
    )
    return {
        "plan": sorted(plan, key=lambda r: (r["condition"], r["init_index"], r["stream_id"])),
        "fixed_speed": run_cells(
            cells["fixed_speed"], FIXED_ARMS, reduce_confirmation, workers=workers, isolate_failures=True
        ),
        "long_fixed_speed": run_cells(
            cells["long_fixed_speed"], LONG_ARMS, reduce_confirmation, workers=workers, isolate_failures=True
        ),
    }
