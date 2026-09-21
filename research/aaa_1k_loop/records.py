"""The three iteration records of the pilot, built from the committed evidence.

Narrative fields (problem statement, classification, reasons) are written
here; every number a reader might want to check is either read from an
evidence artifact at build time or referenced by artifact path and SHA-256,
so interpretation and primitive measurement stay separate. Timestamps are the
UTC times of the commits that created each artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import LOOP_PROTOCOL_VERSION
from .evidence import read_strict_json
from .iteration import ITERATION_SCHEMA, artifact_entry

E1 = "docs/evidence/aaa1k_loop_0001"
E2 = "docs/evidence/aaa1k_loop_0002"
E3 = "docs/evidence/aaa1k_loop_0003"
CHAMPION = f"{E1}/champion_0.json"

TASK_TIMING = {
    "task_start_utc": "2026-09-21T14:19:36Z",
    "note": "human/AI task timing lives in docs/loop_pilot_report.md; scientific compute time is per artifact",
}


def _compute(root: Path, paths: list[str]) -> dict[str, float]:
    return {
        path: float(read_strict_json(root / path)["compute_seconds"])
        for path in paths
        if path.endswith(".json") and "compute_seconds" in read_strict_json(root / path)
    }


def _parent(root: Path) -> dict[str, Any]:
    champion = read_strict_json(root / CHAMPION)
    return {
        "champion_id": champion["champion_id"],
        "phase_fingerprint": champion["phase_fingerprint"],
        "source_commit": champion["source_commit"],
        "record": artifact_entry(root, CHAMPION, "champion"),
    }


ACCOUNTING_UNCHANGED = {
    "parameters_before": 994,
    "parameters_after": 994,
    "parameters_counted_from_arrays": 994,
    "adaptive_state_before": 1414,
    "adaptive_state_after": 1414,
    "compute_change": "none: no challenger was frozen; every candidate had 994 parameters and 1414 state scalars",
}


def record_0001(root: Path) -> dict[str, Any]:
    artifacts = [
        (CHAMPION, "champion"),
        (f"{E1}/observation.json", "observation"),
        (f"{E1}/diagnosis.json", "diagnosis"),
        (f"{E1}/diagnosis_2.json", "diagnosis"),
        (f"{E1}/diagnosis_3.json", "diagnosis"),
        (f"{E1}/development.json", "development"),
        (f"{E1}/development_2.json", "development"),
        (f"{E1}/attack.json", "attack"),
    ]
    return {
        "schema": ITERATION_SCHEMA,
        "iteration_id": "aaa1k-loop-0001",
        "loop_protocol_version": LOOP_PROTOCOL_VERSION,
        "parent_champion": _parent(root),
        "problem": (
            "Round 3 Q4 (ungated minus gated error) is INCONCLUSIVE with its mean and median disagreeing in "
            "sign: the gated model is slightly better on 113 of 144 stream means and much worse on the rest."
        ),
        "evidence_that_exposed_the_problem": [
            "docs/evidence/aaa_1k_evaluation_round3.json (observed, never tuned on)"
        ],
        "classification": {
            "initial": "baseline competitiveness failure with a suspicious tail (prompt framing: minority-tail failure)",
            "after_observation": (
                "not a tail: one entire family (coarse_speed_v1, 24/24 streams, every initialization) carries the "
                "Q4 sign; plus two initialization-3 burst cells"
            ),
            "final": (
                "M1 representation/optimization interaction: zero gate biases give the gated core contractive, "
                "non-alternating one-step dynamics that cannot carry the slow regime's sub-quantum phase (H13); "
                "M2 instability: a runaway through the previous-error input on long quantized streams (H16-H19); "
                "the keep-gate initial operating point is an expected architectural tradeoff (coarse/smooth motion "
                "versus holding through gaps), not a free fix"
            ),
        },
        "hypotheses": {
            "diagnosis_1": "H1-H12, verdicts in diagnosis.json (H12 contradicted, H11 contradicted, width H6 contradicted)",
            "diagnosis_2": "H13 supported, H14 contradicted (exposed M2), H15 supported, in diagnosis_2.json",
            "diagnosis_3": "H16-H19 supported, in diagnosis_3.json",
            "no_change_warranted_hypotheses_tested": [
                "H10 legitimate width tradeoff",
                "H11 statistical tail",
            ],
        },
        "targeted_tests": [
            "switching vs fixed-speed coarse streams (H2, H9, H15)",
            "read-only one-step Jacobian spectra, finite-difference checked (H13)",
            "gain-raising and gain-lowering bias probes as a discriminating pair (H13 c/d)",
            "1120-step streams across four families (H14, H16-H19)",
        ],
        "diagnostic_identity_blocks": [
            "aaa1k-loop-0001/diagnostic/coarse",
            "aaa1k-loop-0001/diagnostic/coarse-2",
            "aaa1k-loop-0001/diagnostic/long-3",
        ],
        "candidates": [
            {
                "candidate_id": "aaa1k-loop-0001-c1",
                "intervention": "keep-gate bias initialized at -1 (initialization only; parameter-neutral)",
                "intervention_class": "optimization/initialization",
                "status": "REJECTED_IN_DEVELOPMENT",
                "reason": "E1/E2/E3 fail: coarse 140% worse, 2 divergent cells against the champion's 0, occlusion +3.3%",
                "evidence": f"{E1}/development.json",
            },
            {
                "candidate_id": "aaa1k-loop-0001-c2",
                "intervention": "keep-gate bias initialized at -2 (initialization only; parameter-neutral)",
                "intervention_class": "optimization/initialization",
                "status": "REJECTED_IN_ATTACK",
                "reason": (
                    "failed development E3 on a real occlusion regression (+5.0% [+3.4%, +6.8%]) and was advanced "
                    "to attack as a labelled tradeoff challenger (deviation L-4); the attack then failed A1, A2 "
                    "and A7: on one of five fresh initializations it degrades to twice persistence on the target "
                    "family, and bias -1.5 fails on the same initialization"
                ),
                "evidence": f"{E1}/attack.json",
            },
            {
                "candidate_id": "aaa1k-loop-0001-c3",
                "intervention": "keep-gate bias initialized -2 on units 0-7 and +2 on units 8-15",
                "intervention_class": "optimization/initialization",
                "status": "REJECTED_IN_DEVELOPMENT",
                "reason": "E3 fails: motion_compat +3.6% to +11.0%, occlusion +2.1%; coarse only -13.4%",
                "evidence": f"{E1}/development_2.json",
            },
        ],
        "challenger": None,
        "confirmation_freeze": None,
        "confirmation_evidence": None,
        "independent_recomputation": None,
        "outcome": "REJECT",
        "reason": (
            "no candidate survived development screening and attack; Champion 0 is retained unchanged. The "
            "diagnosed mechanism is real (H13) but its parameter-neutral remedies trade occlusion for coarse and "
            "smooth-motion accuracy, and the best of them fails on an unseen initialization with the signature "
            "of the M2 runaway (error above twice persistence; its mechanism was not diagnosed further). M2 "
            "became iteration 0002's target"
        ),
        "accounting": ACCOUNTING_UNCHANGED,
        "deviations": [
            "L-4: c2 failed development screening on a single non-regression criterion and was advanced to "
            "attack anyway, as a labelled tradeoff challenger, with the failed criterion carried into the would-be "
            "promotion rules; the attack rejected it, so no confirmation followed",
            "the candidate budget (three) and the handling of screen-failed tradeoff candidates were not "
            "predeclared; both were decided during the iteration and are recorded as protocol gaps",
        ],
        "unregistered_observations": [
            "one exploratory inspection of a single diagnosis-round-2 cell (a diagnostic identity) that pointed "
            "at the previous-error input before diagnosis round 3 was declared"
        ],
        "limitations": [
            "five initializations in development hid an initialization-dependent instability that five fresh "
            "attack initializations exposed",
            "coarse_speed_v1 and occlusion_v1 were designed by the implementer whose model they evaluate",
        ],
        "history": [
            {"state": "OBSERVED", "at": "2026-09-21T14:28:42Z"},
            {"state": "CLASSIFIED", "at": "2026-09-21T14:28:42Z"},
            {"state": "DIAGNOSED", "at": "2026-09-21T14:37:59Z"},
            {"state": "TEST_DEFINED", "at": "2026-09-21T14:37:59Z"},
            {"state": "CHALLENGER_CREATED", "at": "2026-09-21T14:43:23Z"},
            {"state": "CHALLENGER_ATTACKED", "at": "2026-09-21T14:48:03Z"},
            {"state": "REJECTED", "at": "2026-09-21T14:48:03Z"},
            {"state": "PRESERVED", "at": "2026-09-21T14:48:03Z"},
        ],
        "artifacts": [artifact_entry(root, path, role) for path, role in artifacts],
        "compute_seconds": _compute(root, [path for path, _ in artifacts]),
        "timing": TASK_TIMING,
    }


def record_0002(root: Path) -> dict[str, Any]:
    artifacts = [
        (CHAMPION, "champion"),
        (f"{E1}/observation.json", "observation"),
        (f"{E1}/diagnosis_3.json", "diagnosis"),
        (f"{E2}/development.json", "development"),
    ]
    return {
        "schema": ITERATION_SCHEMA,
        "iteration_id": "aaa1k-loop-0002",
        "loop_protocol_version": LOOP_PROTOCOL_VERSION,
        "parent_champion": _parent(root),
        "parent_iteration": "aaa1k-loop-0001 (REJECTED); this iteration starts from its resulting state",
        "problem": (
            "the champion's online learning runs away on long quantized streams: 60 of 160 long coarse cells "
            "end worse than twice persistence (diagnosis round 3). By signature it is the likely source of round "
            "3's two largest Q4 gated losses (those cells were not re-run to confirm), and iteration 0001's "
            "fresh-initialization failure of c2 has the same signature"
        ),
        "evidence_that_exposed_the_problem": [f"{E1}/diagnosis_2.json (H14 test)", f"{E1}/diagnosis_3.json"],
        "classification": {
            "initial": "instability",
            "final": (
                "instability through the presence of the previous-error feedback channel at lr 0.03, not through "
                "the input's magnitude: bounding the input does nothing, removing it fixes the runaway at a large "
                "accuracy cost"
            ),
        },
        "hypotheses": {
            "diagnosis_3": "H16 (error feedback), H17 (learning rate), H18 (recurrence not required), H19 (quantization-specific)",
            "falsified_in_development": "the magnitude form of H16: clipping input 3 at 8 or 4 leaves the runaway intact",
            "open": "a closed-loop gain above one learned on alternating targets (not tested in this pilot)",
        },
        "targeted_tests": [
            "long-horizon divergence counts across four families on fresh development identities"
        ],
        "diagnostic_identity_blocks": ["aaa1k-loop-0001/diagnostic/long-3"],
        "candidates": [
            {
                "candidate_id": "aaa1k-loop-0002-c4",
                "intervention": "clip input 3 to [-8, 8] (encoding repair; parameter-neutral)",
                "intervention_class": "representation/encoding",
                "status": "REJECTED_IN_DEVELOPMENT",
                "reason": "S1 fails: 27 divergent long coarse cells against the champion's 27",
                "evidence": f"{E2}/development.json",
            },
            {
                "candidate_id": "aaa1k-loop-0002-c5",
                "intervention": "clip input 3 to [-4, 4] (encoding repair; parameter-neutral)",
                "intervention_class": "representation/encoding",
                "status": "REJECTED_IN_DEVELOPMENT",
                "reason": "S1 fails: 28 divergent long coarse cells against the champion's 27",
                "evidence": f"{E2}/development.json",
            },
            {
                "candidate_id": "aaa1k-loop-0002-c6",
                "intervention": "remove input 3 (K = 0; the existing no-error-input ablation)",
                "intervention_class": "mechanism removal",
                "status": "REJECTED_IN_DEVELOPMENT",
                "reason": "S4 fails: every standard plan entry regresses by 4-41% (aba +41%, dynamics_change +39%), although divergences fall to 3",
                "evidence": f"{E2}/development.json",
            },
        ],
        "challenger": None,
        "confirmation_freeze": None,
        "confirmation_evidence": None,
        "independent_recomputation": None,
        "outcome": "REJECT",
        "reason": "no candidate passed development screening; Champion 0 retained",
        "accounting": ACCOUNTING_UNCHANGED,
        "deviations": [],
        "unregistered_observations": [
            "one sanity run of the champion and c4 on an arbitrary 1120-step coarse stream (seed 4242, not a "
            "ledger identity) while checking bitwise equality; disclosed in research/aaa_1k_loop/iteration2.py "
            "before the development run"
        ],
        "limitations": [
            "the learning-rate remedy (H17) was excluded on AAA-1K's own development-selection evidence rather than re-tested"
        ],
        "history": [
            {"state": "OBSERVED", "at": "2026-09-21T14:37:59Z"},
            {"state": "CLASSIFIED", "at": "2026-09-21T14:37:59Z"},
            {"state": "DIAGNOSED", "at": "2026-09-21T14:37:59Z"},
            {"state": "TEST_DEFINED", "at": "2026-09-21T14:50:34Z"},
            {"state": "CHALLENGER_CREATED", "at": "2026-09-21T14:52:37Z"},
            {"state": "REJECTED", "at": "2026-09-21T14:52:37Z"},
            {"state": "PRESERVED", "at": "2026-09-21T14:52:37Z"},
        ],
        "artifacts": [artifact_entry(root, path, role) for path, role in artifacts],
        "compute_seconds": _compute(root, [path for path, _ in artifacts]),
        "timing": TASK_TIMING,
    }


def record_0003(root: Path) -> dict[str, Any]:
    artifacts = [
        (CHAMPION, "champion"),
        (f"{E1}/observation.json", "observation"),
        (f"{E1}/diagnosis.json", "diagnosis"),
        (f"{E1}/diagnosis_2.json", "diagnosis"),
        ("research/aaa_1k_loop/iteration3.py", "development"),
        (f"{E3}/attack.json", "attack"),
        (f"{E3}/attack_2.json", "attack"),
    ]
    return {
        "schema": ITERATION_SCHEMA,
        "iteration_id": "aaa1k-loop-0003",
        "loop_protocol_version": LOOP_PROTOCOL_VERSION,
        "parent_champion": _parent(root),
        "parent_iteration": "aaa1k-loop-0002 (REJECTED)",
        "problem": (
            "the repository documents Q4's unfavourable mean as a minority tail and states that coarse_speed_v1 "
            "genuinely tests hidden-regime inference because, at fixed speed, the recurrent model is worse than "
            "the stateless control; iteration 0001's diagnoses contradict both statements"
        ),
        "evidence_that_exposed_the_problem": [f"{E1}/observation.json", f"{E1}/diagnosis_2.json (H15)"],
        "classification": {
            "initial": "unsupported claim",
            "final": "unsupported claim; corrected claim not confirmed",
        },
        "hypotheses": {
            "H15": "supported on diagnostic identities",
            "T1-T4": "attack on claim v2",
            "R1-R5": "attack on claim v3",
        },
        "targeted_tests": [
            "nearby coarse constructions",
            "a second memory-capable model",
            "fresh initializations",
            "the claim's own boundary",
        ],
        "diagnostic_identity_blocks": [
            "aaa1k-loop-0001/diagnostic/coarse",
            "aaa1k-loop-0001/diagnostic/coarse-2",
        ],
        "candidates": [
            {
                "candidate_id": "aaa1k-claim-q4-coarse-v2",
                "intervention": "corrected Q4 / coarse_speed_v1 interpretation (documentation; no model change)",
                "intervention_class": "claim repair",
                "status": "REJECTED_IN_ATTACK",
                "reason": "T1 fails: the fixed-speed memory advantage is construction-dependent (share ~0.3 at quantum 0.006 and at speeds 0.10/0.25)",
                "evidence": f"{E3}/attack.json",
            },
            {
                "candidate_id": "aaa1k-claim-q4-coarse-v3",
                "intervention": "claim v2 scoped to the declared construction, carrying the boundary T1 found",
                "intervention_class": "claim repair",
                "status": "REJECTED_IN_ATTACK",
                "reason": (
                    "R4 fails: the gated fast-dynamics probe's fixed-speed memory advantage is +5.1e-4 with "
                    "interval [-4.3e-5, +1.04e-3]; R1, R2, R3 and R5 pass"
                ),
                "evidence": f"{E3}/attack_2.json",
            },
        ],
        "challenger": None,
        "confirmation_freeze": None,
        "confirmation_evidence": None,
        "independent_recomputation": None,
        "outcome": "REJECT",
        "reason": (
            "the corrected interpretation did not survive attack in either form; the existing claim remains "
            "unsupported by the pilot's diagnostic and attack evidence and is flagged (AAA-162), but no "
            "replacement claim is promoted"
        ),
        "accounting": ACCOUNTING_UNCHANGED,
        "deviations": [
            "a claim, not a model, was the challenger: the loop's classes include 'unsupported claim' and its "
            "interventions include measurement repair; the same outer loop would have applied"
        ],
        "unregistered_observations": [],
        "limitations": [
            "R4 used c2 as its instrument although iteration 0001 had already shown c2 to be "
            "initialization-unstable; a better-designed criterion would not depend on a known-fragile probe"
        ],
        "history": [
            {"state": "OBSERVED", "at": "2026-09-21T14:56:51Z"},
            {"state": "CLASSIFIED", "at": "2026-09-21T14:56:51Z"},
            {"state": "DIAGNOSED", "at": "2026-09-21T14:56:51Z"},
            {"state": "TEST_DEFINED", "at": "2026-09-21T14:56:51Z"},
            {"state": "CHALLENGER_CREATED", "at": "2026-09-21T14:56:51Z"},
            {"state": "CHALLENGER_ATTACKED", "at": "2026-09-21T14:59:58Z"},
            {"state": "REJECTED", "at": "2026-09-21T14:59:58Z"},
            {"state": "PRESERVED", "at": "2026-09-21T14:59:58Z"},
        ],
        "artifacts": [artifact_entry(root, path, role) for path, role in artifacts],
        "compute_seconds": _compute(root, [path for path, _ in artifacts]),
        "timing": TASK_TIMING,
    }


RECORDS = {
    "aaa1k-loop-0001": (record_0001, f"{E1}/iteration.json"),
    "aaa1k-loop-0002": (record_0002, f"{E2}/iteration.json"),
    "aaa1k-loop-0003": (record_0003, f"{E3}/iteration.json"),
}
