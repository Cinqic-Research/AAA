"""Iteration records for 0004-0006 (the post-audit iterations), built from committed evidence.

Same schema and validator as the pilot's records (:mod:`research.aaa_1k_loop.records`).
0004 and 0005 stop at the development screen; 0006 runs the whole outer path,
so its record is validated with a decision function that recomputes the
outcome from the confirmation primitives. Timestamps are the UTC times of the
commits that created each artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import HISTORICAL_PILOT_PROTOCOL_VERSION
from .evidence import read_strict_json
from .iteration import ITERATION_SCHEMA, artifact_entry
from .records import CHAMPION, _compute, _parent

E1 = "docs/evidence/aaa1k_loop_0001"
E4 = "docs/evidence/aaa1k_loop_0004"
E5 = "docs/evidence/aaa1k_loop_0005"
E6 = "docs/evidence/aaa1k_loop_0006"
AUDIT = "AAA_Research_Audit_2026-09-21 (external document; findings R-01, R-02, R-03, R-04)"

TIMING = {
    "task_start_utc": "2026-09-22T02:20:00Z",
    "note": "scientific compute time is per artifact; narrative timing is in docs/loop_report_0004_0006.md",
}

ACCOUNTING_UNCHANGED = {
    "parameters_before": 994,
    "parameters_after": 994,
    "parameters_counted_from_arrays": 994,
    "adaptive_state_before": 1414,
    "adaptive_state_after": 1414,
    "compute_change": "none: every candidate changes only how the training target is built",
}

DIAGNOSES = [
    (f"{E4}/diagnosis.json", "diagnosis"),
    (f"{E4}/diagnosis_gain.json", "diagnosis"),
    (f"{E4}/diagnosis_overshoot.json", "diagnosis"),
    (f"{E4}/diagnosis_unfold.json", "diagnosis"),
]
HYPOTHESES = {
    "H20": "SUPPORTED: live TBPTT is numerically the realized-trajectory gradient where stable",
    "H21": "CONTRADICTED: the stale-matrix mismatch does not drive M2",
    "H22": "SUPPORTED: M2 does not need multi-step credit (T=1 diverges as often)",
    "H23": "CONTRADICTED: chunk-consistent credit does not prevent M2",
    "H24": "CONTRADICTED: the loop gain does not cross one before the runaway",
    "H25": "SUPPORTED",
    "H26": "INCONCLUSIVE",
    "H27": "CONTRADICTED: lr 0.01 never runs away even with 3x the updates",
    "H28": "INSUFFICIENT_EVIDENCE",
    "H29": "CONTRADICTED: no single-step overshoot before onset",
    "H30": "SUPPORTED",
    "H31": "SUPPORTED as declared, mechanistically immaterial (input-3 curvature share ~1%)",
    "H32": "SUPPORTED: a mirror-branch frame lock precedes every runaway (54/54)",
    "H33": "SUPPORTED: stable learners never lock",
    "H34": "SUPPORTED: removing the self-reference removes M2 (0/160 divergences for both probes)",
}


def _history(states: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"state": state, "at": at} for state, at in states]


def record_0004(root: Path) -> dict[str, Any]:
    artifacts = [
        (CHAMPION, "champion"),
        (f"{E1}/diagnosis_3.json", "observation"),
        *DIAGNOSES,
        (f"{E4}/development.json", "development"),
    ]
    development = read_strict_json(root / f"{E4}/development.json")
    screens = development["screens"]
    return {
        "schema": ITERATION_SCHEMA,
        "iteration_id": "aaa1k-loop-0004",
        "loop_protocol_version": HISTORICAL_PILOT_PROTOCOL_VERSION,
        "parent_champion": _parent(root),
        "parent_iteration": "aaa1k-loop-0003 (REJECTED)",
        "problem": (
            "M2: Champion 0's online learning runs away on long quantized streams (diagnosis round 3 of 0001); "
            "the audit also asked whether the online TBPTT rule is the gradient it claims to be (R-02)"
        ),
        "evidence_that_exposed_the_problem": [f"{E1}/diagnosis_3.json", AUDIT],
        "classification": {
            "initial": "optimization instability (the pilot's hypothesis: learned closed-loop gain above one)",
            "final": "defect in target construction: a self-confirming unfolding frame lock",
        },
        "hypotheses": HYPOTHESES,
        "targeted_tests": [
            "four online TBPTT rules on the same state, each proven against finite differences",
            "direct and closed-loop error-feedback gain along real trajectories",
            "exact same-sample amplification of every SGD step",
            "unfolding-branch trace, with folded-target and prediction-independent-reference probes",
        ],
        "diagnostic_identity_blocks": [
            "aaa1k-loop-0004/diagnostic/tbptt",
            "aaa1k-loop-0004/diagnostic/gain",
            "aaa1k-loop-0004/diagnostic/overshoot",
            "aaa1k-loop-0004/diagnostic/unfold",
        ],
        "candidates": [
            {
                "candidate_id": "c7",
                "intervention": "unfold around a prediction-independent reference (input + velocity estimate)",
                "intervention_class": "target construction",
                "status": "REJECTED_IN_DEVELOPMENT",
                "reason": (
                    "screen not passed: long bouncing "
                    f"{screens['c7_unfold_dr']['S3_long_non_coarse']['long:bouncing']['status']} and "
                    "motion_compat:changed "
                    f"{screens['c7_unfold_dr']['S2_standard_families']['plan:motion_compat:changed']['status']}; "
                    "M2 itself removed (0 of 80)"
                ),
                "evidence": f"{E4}/development.json",
            },
            {
                "candidate_id": "c8",
                "intervention": "train on folded observations (unfold_target=False)",
                "intervention_class": "target construction",
                "status": "REJECTED_IN_DEVELOPMENT",
                "reason": "screen FAIL: motion_compat:changed +8.2% and long bouncing +17%; M2 removed",
                "evidence": f"{E4}/development.json",
            },
        ],
        "challenger": None,
        "confirmation_freeze": None,
        "confirmation_evidence": None,
        "independent_recomputation": None,
        "outcome": "REJECT",
        "reason": "both precommitted candidates removed M2 but neither passed the uncertainty-aware screen",
        "accounting": ACCOUNTING_UNCHANGED,
        "deviations": [],
        "unregistered_observations": [
            "a post-hoc look at the overshoot block's already-observed cells (onset vs first bounce, one "
            "traced cell) chose H32-H34; disclosed in unfolding.py and judged on a fresh block"
        ],
        "limitations": [
            "the 1120-step long-horizon conditions are the only horizon tested; M2 at longer horizons for "
            "smooth families is untested"
        ],
        "history": _history(
            [
                ("OBSERVED", "2026-09-22T02:45:32Z"),
                ("CLASSIFIED", "2026-09-22T02:45:32Z"),
                ("DIAGNOSED", "2026-09-22T02:59:14Z"),
                ("TEST_DEFINED", "2026-09-22T03:05:28Z"),
                ("CHALLENGER_CREATED", "2026-09-22T03:06:52Z"),
                ("REJECTED", "2026-09-22T03:06:52Z"),
                ("PRESERVED", "2026-09-22T03:06:52Z"),
            ]
        ),
        "artifacts": [artifact_entry(root, path, role) for path, role in artifacts],
        "compute_seconds": _compute(root, [path for path, _ in artifacts]),
        "timing": TIMING,
    }


def record_0005(root: Path) -> dict[str, Any]:
    artifacts = [
        (CHAMPION, "champion"),
        (f"{E4}/development.json", "observation"),
        (f"{E4}/diagnosis_unfold.json", "diagnosis"),
        (f"{E5}/development.json", "development"),
    ]
    return {
        "schema": ITERATION_SCHEMA,
        "iteration_id": "aaa1k-loop-0005",
        "loop_protocol_version": HISTORICAL_PILOT_PROTOCOL_VERSION,
        "parent_champion": _parent(root),
        "parent_iteration": "aaa1k-loop-0004 (REJECTED)",
        "problem": "M2 remains; 0004's candidates removed it at a cost on smooth bouncing",
        "evidence_that_exposed_the_problem": [f"{E4}/development.json"],
        "classification": {"initial": "target construction (0004)", "final": "target construction"},
        "hypotheses": {"H32-H34": "carried from 0004"},
        "targeted_tests": ["the 0004 screen, unchanged, on a fresh development block"],
        "diagnostic_identity_blocks": ["aaa1k-loop-0004/diagnostic/unfold"],
        "candidates": [
            {
                "candidate_id": "c9",
                "intervention": "refuse a mirror whose implied one-step crossing exceeds |v| + |y - p|",
                "intervention_class": "target construction",
                "status": "REJECTED_IN_DEVELOPMENT",
                "reason": (
                    "screen FAIL: long bouncing +4.5% [+2.2%, +7.1%]; refusing the champion's post-bounce "
                    "mirrored targets costs accuracy on smooth motion; M2 removed (0 of 80)"
                ),
                "evidence": f"{E5}/development.json",
            }
        ],
        "challenger": None,
        "confirmation_freeze": None,
        "confirmation_evidence": None,
        "independent_recomputation": None,
        "outcome": "REJECT",
        "reason": "the one precommitted candidate failed the screen",
        "accounting": ACCOUNTING_UNCHANGED,
        "deviations": [],
        "unregistered_observations": [],
        "limitations": [],
        "history": _history(
            [
                ("OBSERVED", "2026-09-22T03:06:52Z"),
                ("CLASSIFIED", "2026-09-22T03:06:52Z"),
                ("DIAGNOSED", "2026-09-22T03:06:52Z"),
                ("TEST_DEFINED", "2026-09-22T03:12:09Z"),
                ("CHALLENGER_CREATED", "2026-09-22T03:13:34Z"),
                ("REJECTED", "2026-09-22T03:13:34Z"),
                ("PRESERVED", "2026-09-22T03:13:34Z"),
            ]
        ),
        "artifacts": [artifact_entry(root, path, role) for path, role in artifacts],
        "compute_seconds": _compute(root, [path for path, _ in artifacts]),
        "timing": TIMING,
    }


def record_0006(root: Path) -> dict[str, Any]:
    artifacts = [
        (CHAMPION, "champion"),
        (f"{E5}/development.json", "observation"),
        (f"{E4}/diagnosis_unfold.json", "diagnosis"),
        (f"{E6}/development.json", "development"),
        (f"{E6}/attack.json", "attack"),
        (f"{E6}/freeze_2.json", "freeze"),
        (f"{E6}/freeze.json", "freeze"),
        (f"{E6}/confirmation_2.json", "confirmation"),
        (f"{E6}/confirmation_attempt_1.json", "confirmation"),
    ]
    confirmation = read_strict_json(root / f"{E6}/confirmation_2.json")
    return {
        "schema": ITERATION_SCHEMA,
        "iteration_id": "aaa1k-loop-0006",
        "loop_protocol_version": HISTORICAL_PILOT_PROTOCOL_VERSION,
        "parent_champion": _parent(root),
        "parent_iteration": "aaa1k-loop-0005 (REJECTED)",
        "problem": "M2 remains; 0005's c9 refused useful short post-bounce mirror runs",
        "evidence_that_exposed_the_problem": [f"{E5}/development.json"],
        "classification": {"initial": "target construction", "final": "target construction; repaired"},
        "hypotheses": {"H32-H34": "carried from 0004"},
        "targeted_tests": ["the 0004 screen, attack and K1-K3 decision, unchanged, on fresh blocks"],
        "diagnostic_identity_blocks": ["aaa1k-loop-0004/diagnostic/unfold"],
        "candidates": [
            {
                "candidate_id": "c10",
                "intervention": "refuse a mirror when the input is farther from the crossed wall than |v| + |y - p|",
                "intervention_class": "target construction",
                "status": "FROZEN",
                "reason": "screen PASS (bitwise identical on every standard family); attack A1-A5 PASS",
                "evidence": f"{E6}/attack.json",
            }
        ],
        "challenger": "c10",
        "confirmation_freeze": f"{E6}/freeze_2.json",
        "confirmation_evidence": f"{E6}/confirmation_2.json",
        "independent_recomputation": f"{E6}/recomputation_2.json",
        "outcome": confirmation["decision"]["outcome"],
        "reason": (
            "fresh confirmation: long coarse divergence 21.3% -> 0%, difference 0.21 [0.08, 0.36]; every "
            "standard family and long non-coarse condition bitwise identical; recomputed independently"
        ),
        "accounting": ACCOUNTING_UNCHANGED,
        "deviations": [
            "confirmation attempt 1 aborted before observation (admission defect in confirmation_cells); its "
            "identities were burned unobserved, the defect fixed with an end-to-end test, and the unchanged "
            "challenger refrozen with new confirmation blocks (confirmation_attempt_1.json, freeze_2.json)"
        ],
        "unregistered_observations": [
            "a post-hoc look at 0005's already-observed development cells (mirror-run lengths on long bouncing "
            "and long coarse, two initializations) chose c10; development identities may inform selection"
        ],
        "limitations": [
            "four M2 candidates (c7-c10) were screened across 0004-0006; the fresh attack and fresh "
            "confirmation, not the screen, carry the promotion",
        ],
        "history": _history(
            [
                ("OBSERVED", "2026-09-22T03:13:34Z"),
                ("CLASSIFIED", "2026-09-22T03:13:34Z"),
                ("DIAGNOSED", "2026-09-22T03:13:34Z"),
                ("TEST_DEFINED", "2026-09-22T03:15:23Z"),
                ("CHALLENGER_CREATED", "2026-09-22T03:16:05Z"),
                ("CHALLENGER_ATTACKED", "2026-09-22T03:17:17Z"),
                ("FROZEN", "2026-09-22T03:23:35Z"),
                ("CONFIRMED", "2026-09-22T03:25:22Z"),
                ("DECIDED", "2026-09-22T03:25:22Z"),
                ("PRESERVED", "2026-09-22T03:36:00Z"),
            ]
        ),
        "artifacts": [artifact_entry(root, path, role) for path, role in artifacts],
        "compute_seconds": _compute(root, [path for path, _ in artifacts]),
        "timing": TIMING,
    }


def decide_0006(confirmation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recompute 0006's outcome from its primitives with the frozen decision function."""

    from .iteration6 import decide

    return decide(confirmation["primitives"], confirmation["challenger"])


RECORDS_POST_AUDIT = {
    "aaa1k-loop-0004": (record_0004, f"{E4}/iteration.json", None),
    "aaa1k-loop-0005": (record_0005, f"{E5}/iteration.json", None),
    "aaa1k-loop-0006": (record_0006, f"{E6}/iteration.json", decide_0006),
}
