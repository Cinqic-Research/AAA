"""Intervene minimally: the challenger candidates and the rule that picks one.

The intervention sentence
-------------------------
*This change should reduce the gated model's coarse_speed_v1 deficit to the
ungated control because diagnosis round 2 showed that zero-initialized gate
biases (z = r = 0.5) confine the champion's one-step state Jacobian to
contractive, non-alternating modes, so it cannot carry the roughly period-2
sub-quantum phase of slow quantized motion within an episode; initializing
the keep-gate bias negative restores a sign-alternating mode (H13).*

What changes, and what does not
-------------------------------
Only the initial value of ``b_z``. Equations, the 994 parameters, the
optimizer (plain SGD, zero state), the learning rate (0.03), the TBPTT horizon
(4), the auxiliary weight (0.25), the clip (10), the inputs, the outputs, the
state footprint (1414 scalars) and the causal loop are the champion's. The
other 993 initial values are drawn from the same seed exactly as the
champion's are. Because the specification states "all gate biases: zero", a
challenger is nonetheless a *new specification variant* with its own lineage
identity; it does not redefine ``aaa.1k.v1``.

Candidates considered, in increasing departure from the champion
----------------------------------------------------------------
Only two values were declared, both before any development result existed:
``-1`` (initial keep ~0.27) and ``-2`` (initial keep ~0.12, the value round 2
probed). Rejected before development, with reasons recorded in the pilot
report: reset-gate bias (+2 alone and with keep -2 destabilized two of five
initializations in round 2), a lower learning rate (worse on the target and
on the champion's own selection streams), a bounded or removed error input
(addresses the long-horizon runaway, not the Q4 deficit, and is a separate
iteration), and any width or parameter increase (round 1 contradicted width
as the cause: a 354-parameter ungated RNN keeps 86% of the lead).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.stats import crossed_paired_difference

from .harness import matrix

CANDIDATES: tuple[dict[str, Any], ...] = (
    {"candidate_id": "aaa1k-loop-0001-c1", "arm": "cand:keep_bias_-1", "keep_bias": -1.0},
    {"candidate_id": "aaa1k-loop-0001-c2", "arm": "cand:keep_bias_-2", "keep_bias": -2.0},
)
CANDIDATES_ROUND_2: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "aaa1k-loop-0001-c3",
        "arm": "cand:keep_bias_split",
        "keep_bias": "units 0-7: -2; units 8-15: +2",
        "departure": 2.0,
    },
)
"""Declared after development round 1 rejected c1 and c2, before round 2 ran.

Development round 1 found c2 (every unit at -2) improving the target and four
other plan entries but regressing occlusion by 5.0%. The mechanism predicts
that tradeoff: a keep gate near 0.12 on every unit is exactly the fast,
sign-alternating dynamics the coarse phase needs and exactly the wrong
dynamics for holding a velocity through an observation gap. c3 keeps half the
units fast and makes the other half holding (+2, keep ~0.88): still one
change to one initial vector, still zero added parameters.
"""

TARGET_FAMILY = "coarse_speed_v1"
PRACTICAL_MARGIN = 0.02
"""Relative regression tolerated on any other family's development mean, and the tie window."""

DIVERGENCE_FACTOR = 2.0

SELECTION_RULE = (
    "E1 stability: no non-finite failure and no more divergent cells (mean error > 2x persistence) "
    "than the champion on the development block; "
    "E2 target: champion-minus-candidate coarse_speed_v1 error, crossed over initializations and "
    "streams, has a 95% interval above zero; "
    "E3 non-regression: on every other development plan entry the candidate's mean error is at most "
    "2% above the champion's; "
    "choice: the eligible candidate with the largest coarse improvement; candidates within 2% of the "
    "best improvement are tied and the smaller departure (smaller |keep_bias|) wins; "
    "if no candidate is eligible there is no challenger and the iteration stops at REJECT"
)


def plan_key(record: Mapping[str, Any]) -> str:
    """Development plan entry of a record: family plus scenario for motion_compat."""

    return str(record["plan_entry"])


def _departure(candidate: Mapping[str, Any]) -> float:
    return float(candidate.get("departure", abs(float(candidate["keep_bias"]))))


def evaluate_candidates(
    records: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]] = CANDIDATES
) -> dict[str, Any]:
    """Apply :data:`SELECTION_RULE` to development records. Pure function of the records."""

    entries = sorted({plan_key(record) for record in records})
    target = [record for record in records if record["family"] == TARGET_FAMILY]

    def diverged(arm: str) -> int:
        return int(
            sum(record["mae"][arm] > DIVERGENCE_FACTOR * record["mae"]["persistence"] for record in records)
        )

    champion_divergences = diverged("gru")
    results: dict[str, Any] = {}
    for offset, candidate in enumerate(candidates):
        arm = candidate["arm"]
        nonfinite = int(sum(arm in record.get("failures", {}) for record in records))
        if nonfinite:
            results[candidate["candidate_id"]] = {
                **candidate,
                "nonfinite_failures": nonfinite,
                "E1_stability": False,
                "eligible": False,
                "reason": "the candidate raised a non-finite failure on the development block",
            }
            continue
        divergences = diverged(arm)
        improvement = crossed_paired_difference(
            matrix(target, arm), matrix(target, "gru"), bootstrap_index=300 + offset
        )
        per_entry = {}
        for entry in entries:
            rows = [record for record in records if plan_key(record) == entry]
            champion = float(np.mean([row["mae"]["gru"] for row in rows]))
            challenger = float(np.mean([row["mae"][arm] for row in rows]))
            per_entry[entry] = {
                "champion_mean": champion,
                "candidate_mean": challenger,
                "relative_change": (challenger - champion) / champion,
            }
        regressions = [
            entry
            for entry, row in per_entry.items()
            if not entry.startswith(TARGET_FAMILY) and row["relative_change"] > PRACTICAL_MARGIN
        ]
        q4 = crossed_paired_difference(
            matrix(records, arm), matrix(records, "rnn28"), bootstrap_index=320 + offset
        )
        e1 = nonfinite == 0 and divergences <= champion_divergences
        e2 = improvement["interval_status"] == "MEASURED" and improvement["ci_low"] > 0
        e3 = not regressions
        results[candidate["candidate_id"]] = {
            **candidate,
            "nonfinite_failures": nonfinite,
            "divergent_cells": divergences,
            "champion_divergent_cells": champion_divergences,
            "coarse_improvement": improvement,
            "per_plan_entry": per_entry,
            "regressing_entries": regressions,
            "q4_ungated_minus_candidate_all_families": q4,
            "E1_stability": e1,
            "E2_target": e2,
            "E3_non_regression": e3,
            "eligible": e1 and e2 and e3,
        }
    eligible = [cid for cid, row in results.items() if row["eligible"]]
    selected: str | None = None
    if eligible:
        best = max(results[cid]["coarse_improvement"]["mean_difference"] for cid in eligible)
        tied = [
            cid
            for cid in eligible
            if results[cid]["coarse_improvement"]["mean_difference"] >= best * (1.0 - PRACTICAL_MARGIN)
        ]
        selected = min(tied, key=lambda cid: _departure(results[cid]))
    return {
        "rule": SELECTION_RULE,
        "candidates": results,
        "eligible": eligible,
        "selected": selected,
    }
