"""Iteration aaa1k-loop-0002: the error-feedback runaway (M2).

Iteration 0001 targeted the Q4 coarse-family deficit (M1). Its diagnosis also
classified a second mechanism, M2: on long quantized streams the champion's
unbounded previous-error input closes a runaway loop (60 of 160 long coarse
cells worse than twice persistence; H16, H17 supported). By signature -- the
clip active on ~10% of updates, the online model worse than its frozen copy --
M2 is the likely source of Round 3's two largest Q4 gated losses (the
initialization-3 bursts); those evaluation cells were not re-run to confirm it.
Iteration 0001's attack found challenger c2 failing with the same signature
(error above twice persistence) on a fresh initialization; that failure's
mechanism was not diagnosed separately. Iteration 0001 ended REJECTED; this iteration starts from that
state, on its own identity salt, so none of its streams can coincide with
0001's.

Candidates, declared before any development result, in increasing departure
from the champion (all parameter-neutral; all bitwise identical to the
champion whenever ``|input 3| <= K``):

* c4: clip input 3 to ``[-8, 8]`` (a bound fixed by public arithmetic);
* c5: clip to ``[-4, 4]``;
* c6: ``K = 0``, i.e. input 3 removed -- the ``aaa1k_no_error_input``
  ablation, which diagnosis round 3 used to demonstrate the mechanism.

Disclosed before this module was committed: one unregistered sanity run of the
champion and c4 on a single arbitrary 1120-step coarse stream (seed 4242, not
a ledger identity), made while checking that the bounded agent is bitwise
equal to the champion when unclipped. On that stream both degraded (champion
0.0276, c4 0.0312). It is recorded in the iteration record and was not used by
any rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed
from research.aaa_1k.stats import crossed_paired_difference
from research.aaa_1k.streams import coarse_speed_stream, motion_compat_stream, occlusion_stream

from .arms import ArmSet
from .develop import plan_streams
from .harness import Cell, arm_errors, matrix, run_cells
from .identities import block_seeds, find_block, require_usable

ITERATION_ID = "aaa1k-loop-0002"
DEVELOPMENT_BLOCK = f"{ITERATION_ID}/development/all"
LONG_STEPS = 1120
LONG_CONDITIONS = ("long_coarse_no_switch", "long_coarse_switching", "long_bouncing", "long_occlusion")
DIVERGENCE_FACTOR = 2.0
PRACTICAL_MARGIN = 0.02
TARGET_FRACTION = 0.2
MINIMUM_CHAMPION_DIVERGENCES = 5

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": f"{ITERATION_ID}-c4",
        "arm": "cand:error_bound_8",
        "error_input_bound": 8.0,
        "departure": 1,
    },
    {
        "candidate_id": f"{ITERATION_ID}-c5",
        "arm": "cand:error_bound_4",
        "error_input_bound": 4.0,
        "departure": 2,
    },
    {
        "candidate_id": f"{ITERATION_ID}-c6",
        "arm": "gru_no_error_input",
        "error_input_bound": 0.0,
        "departure": 3,
    },
)

SELECTION_RULE = (
    "S1 target: on the development long coarse cells (no switch and switching) the candidate diverges "
    "(mean error > 2x persistence) in at most 20% as many cells as the champion, and the champion diverges "
    "in at least 5 (otherwise the target is not resolved and nothing is selected); "
    "S2 no non-finite failure anywhere; "
    "S3 no more divergent cells than the champion on long bouncing and long occlusion; "
    "S4 non-regression: on every standard plan entry the candidate's mean error is at most 2% above the "
    "champion's; "
    "choice: the eligible candidate with the smallest departure (c4 < c5 < c6)"
)

ARMS = ArmSet(("gru", *(candidate["arm"] for candidate in CANDIDATES), "rnn28", "persistence"))


def _long_stream(condition: str, seed: int) -> Any:
    if condition == "long_coarse_no_switch":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=10_000)
    if condition == "long_coarse_switching":
        return coarse_speed_stream(seed, steps=LONG_STEPS, regime_length=60)
    if condition == "long_bouncing":
        return motion_compat_stream("bouncing", seed, steps=LONG_STEPS, change_step=None)
    if condition == "long_occlusion":
        return occlusion_stream(seed, steps=LONG_STEPS)
    raise ValueError(condition)


def development_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    block = require_usable(ledger, DEVELOPMENT_BLOCK, purpose="selection")
    if block["role"] != "development":
        raise ValueError("iteration 0002 development must run on a development block")
    seeds = block_seeds(find_block(ledger, DEVELOPMENT_BLOCK))
    cells = [
        Cell(entry, init, derive_seed("model_init", init), stream)
        for entry, stream in plan_streams(seeds[:48], 8)
        for init in range(5)
    ]
    for condition in LONG_CONDITIONS:
        for seed in seeds[48:56]:
            stream = _long_stream(condition, seed)
            cells.extend(Cell(condition, init, derive_seed("model_init", init), stream) for init in range(5))
    return cells


def reduce_cell(_cell: Cell, result: RunResult, _agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    return {
        "mae": {name: float(np.mean(values)) for name, values in errors.items()},
        "max_step_error": {name: float(np.max(values)) for name, values in errors.items()},
    }


def run_development(ledger: Mapping[str, Any], *, workers: int | None = None) -> list[dict[str, Any]]:
    return run_cells(development_cells(ledger), ARMS, reduce_cell, workers=workers, isolate_failures=True)


def _diverged(record: Mapping[str, Any], arm: str) -> bool:
    return bool(record["mae"][arm] > DIVERGENCE_FACTOR * record["mae"]["persistence"])


def evaluate_candidates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    long_coarse = [r for r in records if r["condition"] in LONG_CONDITIONS[:2]]
    long_other = [r for r in records if r["condition"] in LONG_CONDITIONS[2:]]
    standard = [r for r in records if r["condition"] not in LONG_CONDITIONS]
    champion_target = sum(_diverged(r, "gru") for r in long_coarse)
    champion_other = sum(_diverged(r, "gru") for r in long_other)
    results: dict[str, Any] = {}
    for offset, candidate in enumerate(CANDIDATES):
        arm = candidate["arm"]
        failures = sum(arm in r.get("failures", {}) for r in records)
        if failures:
            results[candidate["candidate_id"]] = {**candidate, "failures": failures, "eligible": False}
            continue
        target = sum(_diverged(r, arm) for r in long_coarse)
        other = sum(_diverged(r, arm) for r in long_other)
        long_improvement = {
            condition: crossed_paired_difference(
                matrix([r for r in long_coarse if r["condition"] == condition], arm),
                matrix([r for r in long_coarse if r["condition"] == condition], "gru"),
                bootstrap_index=500 + 10 * offset + index,
            )
            for index, condition in enumerate(LONG_CONDITIONS[:2])
        }
        per_entry = {}
        for entry in sorted({r["condition"] for r in standard}):
            rows = [r for r in standard if r["condition"] == entry]
            champion = float(np.mean([r["mae"]["gru"] for r in rows]))
            mean = float(np.mean([r["mae"][arm] for r in rows]))
            per_entry[entry] = {
                "champion_mean": champion,
                "candidate_mean": mean,
                "relative_change": (mean - champion) / champion,
            }
        regressions = [entry for entry, row in per_entry.items() if row["relative_change"] > PRACTICAL_MARGIN]
        s1 = champion_target >= MINIMUM_CHAMPION_DIVERGENCES and target <= TARGET_FRACTION * champion_target
        s3 = other <= champion_other
        s4 = not regressions
        results[candidate["candidate_id"]] = {
            **candidate,
            "failures": 0,
            "long_coarse_divergent": target,
            "champion_long_coarse_divergent": champion_target,
            "long_other_divergent": other,
            "champion_long_other_divergent": champion_other,
            "long_coarse_improvement": long_improvement,
            "per_plan_entry": per_entry,
            "regressing_entries": regressions,
            "S1_target": s1,
            "S2_finite": True,
            "S3_other_long": s3,
            "S4_non_regression": s4,
            "eligible": bool(s1 and s3 and s4),
        }
    eligible = [cid for cid, row in results.items() if row["eligible"]]
    selected = min(eligible, key=lambda cid: results[cid]["departure"]) if eligible else None
    return {"rule": SELECTION_RULE, "candidates": results, "eligible": eligible, "selected": selected}
