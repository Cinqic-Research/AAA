"""Command line for the AAA-1K loop pilot: ``python -m research.aaa_1k_loop``.

Each stage writes one strict-JSON artifact under ``docs/evidence/aaa1k_loop_0001``
and exits non-zero when an invariant fails. The inner-loop commands
(``diagnose``, ``develop``, ``attack``) may only read diagnostic, development
and attack identity blocks. ``confirm`` is the only command that may touch a
confirmation block, and it refuses to run unless the freeze is committed and
every frozen fingerprint still matches.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from research.aaa_1k.identity import git_provenance

from . import LOOP_PROTOCOL_VERSION, PILOT_ITERATION_ID
from .evidence import read_strict_json, write_strict_json
from .identities import LEDGER_PATH, empty_ledger, load_ledger, prove_fresh, reserve, write_ledger

EVIDENCE_DIR = Path("docs/evidence/aaa1k_loop_0001")

INNER_LOOP_BLOCKS: tuple[dict[str, Any], ...] = (
    {
        "block_id": "aaa1k-loop-0004/diagnostic/gain",
        "iteration_id": "aaa1k-loop-0004",
        "role": "diagnostic",
        "namespace": "diagnostic_gain",
        "count": 16,
        "purpose": (
            "iteration 0004 round 2 (audit R-01, M2): closed-loop gain of the previous-error channel; "
            "hypotheses H24-H28 declared first"
        ),
    },
    {
        "block_id": "aaa1k-loop-0004/diagnostic/scratch",
        "iteration_id": "aaa1k-loop-0004",
        "role": "diagnostic",
        "namespace": "diagnostic_scratch",
        "count": 8,
        "purpose": (
            "iteration 0004 scratch: instrument shakedown only (does it run, how long); never cited as evidence"
        ),
    },
    {
        "block_id": "aaa1k-loop-0004/diagnostic/tbptt",
        "iteration_id": "aaa1k-loop-0004",
        "role": "diagnostic",
        "namespace": "diagnostic_tbptt",
        "count": 64,
        "purpose": (
            "iteration 0004 round 1 (audit R-02): online TBPTT update rules on 16 long-horizon seeds "
            "x 4 conditions and 8 streams per standard plan entry; hypotheses H20-H23 declared first"
        ),
    },
    {
        "block_id": "aaa1k-loop-0003/attack-2/env",
        "iteration_id": "aaa1k-loop-0003",
        "role": "attack",
        "namespace": "attack2_env",
        "count": 24,
        "purpose": "iteration 0003 second attack, on the scoped claim v3",
    },
    {
        "block_id": "aaa1k-loop-0003/attack-2/init",
        "iteration_id": "aaa1k-loop-0003",
        "role": "attack",
        "namespace": "attack2_init",
        "count": 5,
        "purpose": "iteration 0003 fresh initialization seeds for the second attack",
    },
    {
        "block_id": "aaa1k-loop-0003/attack/env",
        "iteration_id": "aaa1k-loop-0003",
        "role": "attack",
        "namespace": "attack_env",
        "count": 24,
        "purpose": "iteration 0003 attack on the corrected Q4 interpretation",
    },
    {
        "block_id": "aaa1k-loop-0003/attack/init",
        "iteration_id": "aaa1k-loop-0003",
        "role": "attack",
        "namespace": "attack_init",
        "count": 5,
        "purpose": "iteration 0003 fresh initialization seeds for the attack",
    },
    {
        "block_id": "aaa1k-loop-0002/development/all",
        "iteration_id": "aaa1k-loop-0002",
        "role": "development",
        "namespace": "development_all",
        "count": 56,
        "purpose": "iteration 0002 candidate development: 48 standard plan streams and 8 long-horizon seeds",
    },
    {
        "block_id": "aaa1k-loop-0002/attack/env",
        "iteration_id": "aaa1k-loop-0002",
        "role": "attack",
        "namespace": "attack_env",
        "count": 96,
        "purpose": "iteration 0002 adversarial attack streams",
    },
    {
        "block_id": "aaa1k-loop-0002/attack/init",
        "iteration_id": "aaa1k-loop-0002",
        "role": "attack",
        "namespace": "attack_init",
        "count": 5,
        "purpose": "iteration 0002 fresh initialization seeds for the attack",
    },
    {
        "block_id": "aaa1k-loop-0001/diagnostic/coarse",
        "role": "diagnostic",
        "namespace": "diagnostic_coarse",
        "count": 16,
        "purpose": "coarse_speed_v1 streams for the Q4 hypothesis tests",
    },
    {
        "block_id": "aaa1k-loop-0001/diagnostic/coarse-2",
        "role": "diagnostic",
        "namespace": "diagnostic_coarse_2",
        "count": 16,
        "purpose": "fresh coarse_speed_v1 streams for the round-2 hypotheses (H13-H15), declared after round 1",
    },
    {
        "block_id": "aaa1k-loop-0001/diagnostic/long-3",
        "role": "diagnostic",
        "namespace": "diagnostic_long_3",
        "count": 16,
        "purpose": "long-horizon streams for the round-3 stability hypotheses (H16-H19), declared after round 2",
    },
    {
        "block_id": "aaa1k-loop-0001/development/all",
        "role": "development",
        "namespace": "development_all",
        "count": 48,
        "purpose": "challenger development on every round-3 family (8 streams per plan entry)",
    },
    {
        "block_id": "aaa1k-loop-0001/attack/env",
        "role": "attack",
        "namespace": "attack_env",
        "count": 96,
        "purpose": "adversarial attack streams: nearby conditions, other families, tails",
    },
    {
        "block_id": "aaa1k-loop-0001/attack/init",
        "role": "attack",
        "namespace": "attack_init",
        "count": 5,
        "purpose": "fresh model initialization seeds for the attack",
    },
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ledger_path(root: Path) -> Path:
    return root / LEDGER_PATH


def _stamp(payload: dict[str, Any], root: Path, started: float) -> dict[str, Any]:
    return {
        **payload,
        "loop_protocol_version": LOOP_PROTOCOL_VERSION,
        "iteration_id": PILOT_ITERATION_ID,
        "git": git_provenance(root),
        "compute_seconds": time.perf_counter() - started,
    }


def command_ledger_sync(_args: argparse.Namespace) -> int:
    """Append declared blocks the ledger lacks; never edit or remove an existing one."""

    root = project_root()
    path = _ledger_path(root)
    ledger = load_ledger(path) if path.exists() else empty_ledger()
    existing = {block["block_id"]: block for block in ledger["blocks"]}
    added = 0
    for declared in INNER_LOOP_BLOCKS:
        current = existing.get(declared["block_id"])
        declared = dict(declared)
        iteration = declared.pop("iteration_id", PILOT_ITERATION_ID)
        if current is None:
            ledger = reserve(ledger, **declared, iteration_id=iteration)
            added += 1
            continue
        drift = [
            key for key, value in {**declared, "iteration": iteration}.items() if current.get(key) != value
        ]
        if drift:
            print(
                f"refusing: declared block {declared['block_id']} differs from the ledger in {drift}",
                file=sys.stderr,
            )
            return 2
    prove_fresh(ledger, root)
    write_ledger(path, ledger)
    print(f"ledger holds {len(ledger['blocks'])} blocks ({added} added)")
    return 0


def command_prove_fresh(args: argparse.Namespace) -> int:
    root = project_root()
    proof = prove_fresh(load_ledger(_ledger_path(root)), root)
    if args.output:
        write_strict_json(Path(args.output), proof)
    print(
        f"{proof['status']}: {proof['loop_seed_count']} loop seeds against "
        f"{proof['aaa1k_declared_seed_count']} declared and {proof['aaa1k_evidence_seed_count']} evidence seeds"
    )
    return 0


def command_champion(args: argparse.Namespace) -> int:
    from .champion import build_champion, verify_champion

    root = project_root()
    output = Path(args.output)
    if args.verify:
        problems = verify_champion(read_strict_json(output), root)
        for problem in problems:
            print(f"STALE: {problem}", file=sys.stderr)
        print("champion verified" if not problems else f"{len(problems)} disagreement(s)")
        return 0 if not problems else 1
    started = time.perf_counter()
    record = build_champion(root, source_commit=args.source_commit)
    write_strict_json(output, _stamp(record, root, started))
    print(
        f"{record['champion_id']}: {record['parameter_count']} parameters, fingerprint {record['phase_fingerprint']}"
    )
    return 0


def command_observe(args: argparse.Namespace) -> int:
    from .champion import ROUND3_EVIDENCE, sha256_file
    from .observation import build_observation

    root = project_root()
    started = time.perf_counter()
    source = root / ROUND3_EVIDENCE
    record = build_observation(read_strict_json(source), round3_sha256=sha256_file(source))
    write_strict_json(Path(args.output), _stamp(record, root, started))
    print(record["statement"]["recomputed_finding"])
    return 0


def command_diagnose(args: argparse.Namespace) -> int:
    from .diagnosis import DIAGNOSIS_SCHEMA, HYPOTHESES, adjudicate, run_diagnosis

    root = project_root()
    started = time.perf_counter()
    ledger = load_ledger(_ledger_path(root))
    records = run_diagnosis(ledger, workers=args.workers)
    analysis = adjudicate(records)
    payload = {
        "schema": DIAGNOSIS_SCHEMA,
        "evidence_role": "diagnostic",
        "identity_block": "aaa1k-loop-0001/diagnostic/coarse",
        "hypotheses": list(HYPOTHESES),
        "analysis": analysis,
        "records": records,
    }
    write_strict_json(Path(args.output), _stamp(payload, root, started))
    for key, value in analysis["verdicts"].items():
        print(f"{key:4s} {value['verdict']}")
    return 0


def command_diagnose2(args: argparse.Namespace) -> int:
    from .diagnosis2 import DIAGNOSIS2_SCHEMA, DIAGNOSTIC_BLOCK_2, HYPOTHESES, adjudicate2, run_diagnosis2

    root = project_root()
    started = time.perf_counter()
    records = run_diagnosis2(load_ledger(_ledger_path(root)), workers=args.workers)
    analysis = adjudicate2(records)
    payload = {
        "schema": DIAGNOSIS2_SCHEMA,
        "evidence_role": "diagnostic",
        "identity_block": DIAGNOSTIC_BLOCK_2,
        "hypotheses": list(HYPOTHESES),
        "analysis": analysis,
        "records": records,
    }
    write_strict_json(Path(args.output), _stamp(payload, root, started))
    for key, value in analysis["verdicts"].items():
        print(f"{key:4s} {value['verdict']}")
    return 0


def command_diagnose3(args: argparse.Namespace) -> int:
    from .diagnosis3 import DIAGNOSIS3_SCHEMA, DIAGNOSTIC_BLOCK_3, HYPOTHESES, adjudicate3, run_diagnosis3

    root = project_root()
    started = time.perf_counter()
    records = run_diagnosis3(load_ledger(_ledger_path(root)), workers=args.workers)
    analysis = adjudicate3(records)
    payload = {
        "schema": DIAGNOSIS3_SCHEMA,
        "evidence_role": "diagnostic",
        "identity_block": DIAGNOSTIC_BLOCK_3,
        "hypotheses": list(HYPOTHESES),
        "analysis": analysis,
        "records": records,
    }
    write_strict_json(Path(args.output), _stamp(payload, root, started))
    for key, value in analysis["verdicts"].items():
        print(f"{key:4s} {value['verdict']}")
    return 0


def command_develop(args: argparse.Namespace) -> int:
    from .challengers import CANDIDATES, CANDIDATES_ROUND_2, evaluate_candidates
    from .develop import ARMS, ARMS_ROUND_2, DEVELOPMENT_BLOCK, DEVELOPMENT_SCHEMA, run_development

    root = project_root()
    started = time.perf_counter()
    declared = CANDIDATES if args.round == 1 else CANDIDATES_ROUND_2
    records = run_development(
        load_ledger(_ledger_path(root)), workers=args.workers, arms=ARMS if args.round == 1 else ARMS_ROUND_2
    )
    selection = evaluate_candidates(records, declared)
    payload = {
        "schema": DEVELOPMENT_SCHEMA,
        "development_round": args.round,
        "evidence_role": "development",
        "identity_block": DEVELOPMENT_BLOCK,
        "candidates_declared": list(declared),
        "selection": selection,
        "records": records,
    }
    write_strict_json(Path(args.output), _stamp(payload, root, started))
    for cid, row in selection["candidates"].items():
        print(f"{cid}: eligible={row['eligible']}")
    print(f"selected: {selection['selected']}")
    return 0


def command_attack(args: argparse.Namespace) -> int:
    from .attack import ATTACK_SCHEMA, CHALLENGER, ENV_BLOCK, INIT_BLOCK, adjudicate_attack, run_attack

    root = project_root()
    started = time.perf_counter()
    result = run_attack(load_ledger(_ledger_path(root)), workers=args.workers)
    adjudication = adjudicate_attack(result)
    payload = {
        "schema": ATTACK_SCHEMA,
        "evidence_role": "attack",
        "identity_blocks": [ENV_BLOCK, INIT_BLOCK],
        "challenger_arm": CHALLENGER,
        "challenger_id": "aaa1k-loop-0001-c2",
        "adjudication": adjudication,
        **result,
    }
    write_strict_json(Path(args.output), _stamp(payload, root, started))
    for key, value in adjudication["outcome"].items():
        print(f"{key:34s} {'PASS' if value['passed'] else 'FAIL'}")
    print(f"advance to freeze: {adjudication['advance_to_freeze']}")
    return 0


def command_develop2(args: argparse.Namespace) -> int:
    from .iteration2 import CANDIDATES, DEVELOPMENT_BLOCK, evaluate_candidates, run_development

    root = project_root()
    started = time.perf_counter()
    records = run_development(load_ledger(_ledger_path(root)), workers=args.workers)
    selection = evaluate_candidates(records)
    payload = {
        "schema": "aaa.loop.development.v2",
        "iteration": "aaa1k-loop-0002",
        "evidence_role": "development",
        "identity_block": DEVELOPMENT_BLOCK,
        "candidates_declared": list(CANDIDATES),
        "selection": selection,
        "records": records,
    }
    write_strict_json(Path(args.output), _stamp(payload, root, started))
    for cid, row in selection["candidates"].items():
        print(f"{cid}: eligible={row['eligible']}")
    print(f"selected: {selection['selected']}")
    return 0


EVIDENCE_DIR_3 = Path("docs/evidence/aaa1k_loop_0003")
FREEZE_3 = EVIDENCE_DIR_3 / "freeze.json"


def _frozen_content_3() -> dict[str, Any]:
    from .decision import CRITERIA, THRESHOLDS
    from .iteration3 import CHALLENGER_CLAIM, CHAMPION_CLAIM, CLAIM_ID, CONFIRMATION_DESIGN, ITERATION_ID

    return {
        "iteration_id": ITERATION_ID,
        "claim_id": CLAIM_ID,
        "champion_claim": CHAMPION_CLAIM,
        "challenger_claim": CHALLENGER_CLAIM,
        "design": CONFIRMATION_DESIGN,
        "thresholds": THRESHOLDS,
        "criteria": CRITERIA,
        "outcome_rule": "any FAIL -> REJECT; every criterion PASS -> PROMOTE; otherwise INCONCLUSIVE",
        "promotion_effect": (
            "PROMOTE adopts the challenger claim in the documentation and repairs AAA-162; REJECT keeps the "
            "champion claim and records that the correction did not confirm; INCONCLUSIVE marks the claim "
            "contested. Champion 0, the model, is unchanged in every outcome."
        ),
    }


def command_attack3(args: argparse.Namespace) -> int:
    from .iteration3 import ATTACK_ENV_BLOCK, ATTACK_INIT_BLOCK, CLAIM_ID_V2, adjudicate_attack, run_attack

    root = project_root()
    started = time.perf_counter()
    records = run_attack(load_ledger(_ledger_path(root)), workers=args.workers)
    adjudication = adjudicate_attack(records)
    payload = {
        "schema": "aaa.loop.attack.v2",
        "iteration": "aaa1k-loop-0003",
        "evidence_role": "attack",
        "identity_blocks": [ATTACK_ENV_BLOCK, ATTACK_INIT_BLOCK],
        "claim_id": CLAIM_ID_V2,  # the first attack judged claim v2
        "adjudication": adjudication,
        "records": records,
    }
    write_strict_json(Path(args.output), _stamp(payload, root, started))
    for key, value in adjudication["outcome"].items():
        print(f"{key:36s} {'PASS' if value['passed'] else 'FAIL'}")
    print(f"advance to freeze: {adjudication['advance_to_freeze']}")
    return 0


def command_attack3b(args: argparse.Namespace) -> int:
    from .iteration3 import ATTACK2_ENV_BLOCK, ATTACK2_INIT_BLOCK, CLAIM_ID, adjudicate_attack2, run_attack2

    root = project_root()
    started = time.perf_counter()
    records = run_attack2(load_ledger(_ledger_path(root)), workers=args.workers)
    adjudication = adjudicate_attack2(records)
    payload = {
        "schema": "aaa.loop.attack.v3",
        "iteration": "aaa1k-loop-0003",
        "evidence_role": "attack",
        "identity_blocks": [ATTACK2_ENV_BLOCK, ATTACK2_INIT_BLOCK],
        "claim_id": CLAIM_ID,
        "adjudication": adjudication,
        "records": records,
    }
    write_strict_json(Path(args.output), _stamp(payload, root, started))
    for key, value in adjudication["outcome"].items():
        print(f"{key:36s} {'PASS' if value['passed'] else 'FAIL'}")
    print(f"advance to freeze: {adjudication['advance_to_freeze']}")
    return 0


def command_freeze3(_args: argparse.Namespace) -> int:
    from .freeze import build_freeze
    from .iteration3 import (
        CONFIRMATION_ENV_BLOCK,
        CONFIRMATION_ENV_COUNT,
        CONFIRMATION_INIT_BLOCK,
        CONFIRMATION_INIT_COUNT,
        adjudicate_attack2,
    )

    root = project_root()
    output = root / FREEZE_3
    if output.exists():
        print(f"refusing: {FREEZE_3} exists; a freeze is never rewritten", file=sys.stderr)
        return 2
    attack_path = EVIDENCE_DIR_3 / "attack_2.json"
    attack = read_strict_json(root / attack_path)
    if not adjudicate_attack2(attack["records"])["advance_to_freeze"]:
        print("refusing: the recomputed attack does not advance this challenger", file=sys.stderr)
        return 1
    ledger = load_ledger(_ledger_path(root))
    ledger = reserve(
        ledger,
        block_id=CONFIRMATION_ENV_BLOCK,
        role="confirmation",
        namespace="confirmation_env",
        count=CONFIRMATION_ENV_COUNT,
        purpose="iteration 0003 fresh confirmation streams",
        iteration_id="aaa1k-loop-0003",
    )
    ledger = reserve(
        ledger,
        block_id=CONFIRMATION_INIT_BLOCK,
        role="confirmation",
        namespace="confirmation_init",
        count=CONFIRMATION_INIT_COUNT,
        purpose="iteration 0003 fresh confirmation initializations",
        iteration_id="aaa1k-loop-0003",
    )
    freshness = prove_fresh(ledger, root)
    manifest = build_freeze(
        root,
        ledger=ledger,
        freshness=freshness,
        champion_path=str(EVIDENCE_DIR / "champion_0.json"),
        attack_path=str(attack_path),
        blocks=(CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK),
        frozen_content=_frozen_content_3(),
    )
    write_ledger(_ledger_path(root), ledger)
    write_strict_json(
        output, {**manifest, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, overwrite=False
    )
    print(f"freeze written: {FREEZE_3}; commit it before running confirm3")
    return 0


def command_confirm3(args: argparse.Namespace) -> int:
    from aaa.noise.reservation import ReservationError, reserve_confirmation_batch

    from .decision import THRESHOLDS, decide
    from .freeze import (
        load_freeze,
        require_committed,
        require_committed_confirmation_source,
        verify_freeze,
    )
    from .identities import mark, require_usable
    from .iteration3 import CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK, run_confirmation

    root = project_root()
    output = root / args.output
    if output.exists():
        print(f"refusing: {args.output} exists; a confirmation is observed once", file=sys.stderr)
        return 2
    manifest = load_freeze(root / FREEZE_3)
    head = require_committed(root, str(FREEZE_3))
    require_committed_confirmation_source(root, manifest)
    ledger = load_ledger(_ledger_path(root))
    problems = verify_freeze(manifest, root, ledger=ledger, frozen_content=_frozen_content_3())
    for block_id in (CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK):
        try:
            require_usable(ledger, block_id, purpose="confirmation")
        except Exception as error:
            problems.append(str(error))
    if problems:
        for problem in problems:
            print(f"REFUSED: {problem}", file=sys.stderr)
        return 1
    prove_fresh(ledger, root)
    observer = f"aaa1k-loop-0003 confirmation at {head}"
    # Local ledger mutation is recoverable by reset/reclone and therefore is
    # not a durable claim.  Atomically own each confirmation block on the
    # shared remote before execution can observe it.  Immutable Git refs make
    # crashes, worktrees, clones and concurrent actors converge on one owner.
    fingerprint = manifest["confirmation_source_fingerprint"]["sha256"]
    for block_id in (CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK):
        batch_id = block_id.replace("/", "-")
        try:
            reserve_confirmation_batch(
                root,
                batch_id=batch_id,
                role="loop-confirmation",
                attempt_id=observer,
                scientific_fingerprint_sha256=fingerprint,
                resume=False,
            )
        except ReservationError as error:
            print(f"REFUSED: durable confirmation reservation failed: {error}", file=sys.stderr)
            return 1
    for block_id in (CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK):
        ledger = mark(ledger, block_id, status="spent", observed_by=observer)
    # Spent before the first cell runs: a crash after this point still consumes the identities.
    write_ledger(_ledger_path(root), ledger)
    started = time.perf_counter()
    primitives = run_confirmation(ledger, workers=args.workers)
    import hashlib

    payload: dict[str, Any] = {
        "schema": "aaa.loop.confirmation.v1",
        "iteration": "aaa1k-loop-0003",
        "evidence_role": "confirmation",
        "freeze": {
            "path": str(FREEZE_3),
            "sha256": hashlib.sha256((root / FREEZE_3).read_bytes()).hexdigest(),
            "commit": head,
        },
        "confirmation_source_fingerprint": manifest["confirmation_source_fingerprint"]["sha256"],
        "champion_phase_fingerprint": manifest["champion_phase_fingerprint"],
        "thresholds": THRESHOLDS,
        "identity_blocks": [CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK],
        "primitives": primitives,
    }
    try:
        payload["decision"] = decide(payload)
    except Exception as error:
        payload["decision"] = {"outcome": None, "error": f"{type(error).__name__}: {error}"}
        write_strict_json(output, _stamp(payload, root, started), overwrite=False)
        print(f"decision failed after observation; primitives retained: {error}", file=sys.stderr)
        return 1
    write_strict_json(output, _stamp(payload, root, started), overwrite=False)
    for name, status in payload["decision"]["statuses"].items():
        print(f"{name:46s} {status}")
    print(f"OUTCOME: {payload['decision']['outcome']}")
    return 0


def command_recompute3(args: argparse.Namespace) -> int:
    from .recompute import verify_confirmation

    root = project_root()
    result = verify_confirmation(root / args.confirmation, root / FREEZE_3, load_ledger(_ledger_path(root)))
    if args.output:
        write_strict_json(Path(args.output), result)
    for problem in result["problems"]:
        print(f"DISAGREES: {problem}", file=sys.stderr)
    print(f"independent outcome: {result['independent']['outcome']}; agrees with stored: {result['agrees']}")
    return 0 if result["agrees"] else 1


def command_records(_args: argparse.Namespace) -> int:
    from .records import RECORDS

    root = project_root()
    for iteration_id, (builder, path) in RECORDS.items():
        write_strict_json(root / path, builder(root))
        print(f"{iteration_id}: {path}")
    return 0


def command_validate(_args: argparse.Namespace) -> int:
    """Validate every iteration record, the ledger and Champion 0; fail closed."""

    from .champion import verify_champion
    from .decision import decide
    from .iteration import IterationError, validate_iteration
    from .records import RECORDS

    root = project_root()
    failures = 0
    for iteration_id, (_builder, path) in RECORDS.items():
        try:
            result = validate_iteration(read_strict_json(root / path), root, decide=decide)
            print(
                f"{iteration_id}: {result['status']} {' -> '.join(result['states'])} outcome={result['outcome']}"
            )
        except (IterationError, OSError, ValueError, KeyError) as error:
            failures += 1
            print(f"{iteration_id}: INVALID: {error}", file=sys.stderr)
    problems = verify_champion(read_strict_json(root / EVIDENCE_DIR / "champion_0.json"), root)
    for problem in problems:
        print(f"champion: STALE: {problem}", file=sys.stderr)
    proof = prove_fresh(load_ledger(_ledger_path(root)), root)
    print(f"ledger: {proof['status']} ({proof['loop_blocks']} blocks)")
    return 1 if failures or problems else 0


VOLATILE = frozenset({"git", "compute_seconds", "seconds_per_transition", "ratio", "wall_seconds"})
"""Provenance and wall-clock fields; every other committed value must reproduce exactly."""

REPRODUCIBLE: dict[str, tuple[str, dict[str, Any]]] = {
    "observe": (str(EVIDENCE_DIR / "observation.json"), {}),
    "diagnose": (str(EVIDENCE_DIR / "diagnosis.json"), {"workers": None}),
    "diagnose2": (str(EVIDENCE_DIR / "diagnosis_2.json"), {"workers": None}),
    "diagnose3": (str(EVIDENCE_DIR / "diagnosis_3.json"), {"workers": None}),
    "develop": (str(EVIDENCE_DIR / "development.json"), {"workers": None, "round": 1}),
    "develop-round-2": (str(EVIDENCE_DIR / "development_2.json"), {"workers": None, "round": 2}),
    "attack": (str(EVIDENCE_DIR / "attack.json"), {"workers": None}),
    "develop2": ("docs/evidence/aaa1k_loop_0002/development.json", {"workers": None}),
    "attack3": (str(EVIDENCE_DIR_3 / "attack.json"), {"workers": None}),
    "attack3b": (str(EVIDENCE_DIR_3 / "attack_2.json"), {"workers": None}),
}


def compare_evidence(committed: Any, fresh: Any, path: str = "$") -> tuple[list[str], list[str]]:
    """``(mismatches, added)``: every committed value must reappear exactly."""

    mismatches: list[str] = []
    added: list[str] = []
    if isinstance(committed, dict):
        if not isinstance(fresh, dict):
            return [f"{path}: type changed"], added
        for key, value in committed.items():
            if key in VOLATILE:
                continue
            if key not in fresh:
                mismatches.append(f"{path}.{key}: missing from the rerun")
                continue
            m, a = compare_evidence(value, fresh[key], f"{path}.{key}")
            mismatches += m
            added += a
        added += [f"{path}.{key}" for key in fresh if key not in committed and key not in VOLATILE]
    elif isinstance(committed, list):
        if not isinstance(fresh, list) or len(fresh) != len(committed):
            return [f"{path}: list length changed"], added
        for index, (c, f) in enumerate(zip(committed, fresh, strict=True)):
            m, a = compare_evidence(c, f, f"{path}[{index}]")
            mismatches += m
            added += a
    elif committed != fresh:
        mismatches.append(f"{path}: committed {committed!r} rerun {fresh!r}")
    return mismatches, added


def command_reproduce(args: argparse.Namespace) -> int:
    import tempfile

    root = project_root()
    committed_path, options = REPRODUCIBLE[args.stage]
    handlers = build_handlers()
    with tempfile.TemporaryDirectory() as scratch:
        fresh_path = Path(scratch) / "rerun.json"
        stage = "develop" if args.stage == "develop-round-2" else args.stage
        status = handlers[stage](argparse.Namespace(output=str(fresh_path), **options))
        if status != 0:
            print(f"{args.stage}: the rerun itself failed", file=sys.stderr)
            return 1
        mismatches, added = compare_evidence(
            read_strict_json(root / committed_path), read_strict_json(fresh_path)
        )
    for mismatch in mismatches[:20]:
        print(f"MISMATCH {mismatch}", file=sys.stderr)
    print(
        f"{args.stage}: {'REPRODUCED' if not mismatches else 'NOT REPRODUCED'} "
        f"({len(mismatches)} mismatches; {len(added)} fields added by later code)"
    )
    return 0 if not mismatches else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_1k_loop", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ledger-sync", help="append declared inner-loop blocks; never edit existing ones")
    fresh = sub.add_parser("prove-fresh", help="prove every ledger block disjoint from AAA-1K identities")
    fresh.add_argument("--output")

    champion = sub.add_parser("champion", help="derive, or verify, the Champion 0 record")
    champion.add_argument("--output", default=str(EVIDENCE_DIR / "champion_0.json"))
    champion.add_argument("--source-commit", default="1ce716c42500003881e379f6d23dc0d807301001")
    champion.add_argument("--verify", action="store_true")

    observe = sub.add_parser("observe", help="decompose the round-3 Q4 result from its primitives")
    observe.add_argument("--output", default=str(EVIDENCE_DIR / "observation.json"))

    diagnose = sub.add_parser("diagnose", help="run the diagnostic hypothesis tests")
    diagnose.add_argument("--output", default=str(EVIDENCE_DIR / "diagnosis.json"))
    diagnose.add_argument("--workers", type=int)
    diagnose2 = sub.add_parser("diagnose2", help="run the round-2 hypotheses on a fresh diagnostic block")
    diagnose2.add_argument("--output", default=str(EVIDENCE_DIR / "diagnosis_2.json"))
    diagnose2.add_argument("--workers", type=int)
    diagnose3 = sub.add_parser("diagnose3", help="run the round-3 long-horizon stability hypotheses")
    diagnose3.add_argument("--output", default=str(EVIDENCE_DIR / "diagnosis_3.json"))
    diagnose3.add_argument("--workers", type=int)
    develop = sub.add_parser("develop", help="evaluate the declared candidates on the development block")
    develop.add_argument("--round", type=int, choices=(1, 2), default=1)
    develop.add_argument("--output", default=str(EVIDENCE_DIR / "development.json"))
    develop.add_argument("--workers", type=int)
    attack = sub.add_parser("attack", help="attack challenger c2 on attack identities")
    attack.add_argument("--output", default=str(EVIDENCE_DIR / "attack.json"))
    attack.add_argument("--workers", type=int)
    develop2 = sub.add_parser("develop2", help="iteration 0002: evaluate the bounded-error candidates")
    develop2.add_argument("--output", default="docs/evidence/aaa1k_loop_0002/development.json")
    develop2.add_argument("--workers", type=int)
    attack3 = sub.add_parser("attack3", help="iteration 0003: attack the corrected Q4 interpretation")
    attack3.add_argument("--output", default=str(EVIDENCE_DIR_3 / "attack.json"))
    attack3.add_argument("--workers", type=int)
    attack3b = sub.add_parser(
        "attack3b", help="iteration 0003: attack the scoped claim v3 on fresh identities"
    )
    attack3b.add_argument("--output", default=str(EVIDENCE_DIR_3 / "attack_2.json"))
    attack3b.add_argument("--workers", type=int)
    sub.add_parser("freeze3", help="iteration 0003: reserve confirmation identities and write the freeze")
    confirm3 = sub.add_parser("confirm3", help="iteration 0003: run the frozen confirmation once")
    confirm3.add_argument("--output", default=str(EVIDENCE_DIR_3 / "confirmation.json"))
    confirm3.add_argument("--workers", type=int)
    recompute3 = sub.add_parser("recompute3", help="iteration 0003: independently recompute the decision")
    recompute3.add_argument("--confirmation", default=str(EVIDENCE_DIR_3 / "confirmation.json"))
    recompute3.add_argument("--output")
    sub.add_parser("records", help="write the iteration records from the committed evidence")
    sub.add_parser("validate", help="validate iteration records, Champion 0 and the identity ledger")
    reproduce = sub.add_parser("reproduce", help="rerun a stage and require its committed primitives exactly")
    reproduce.add_argument("stage", choices=sorted(REPRODUCIBLE))
    return parser


def build_handlers() -> dict[str, Any]:
    return {
        "ledger-sync": command_ledger_sync,
        "prove-fresh": command_prove_fresh,
        "champion": command_champion,
        "observe": command_observe,
        "diagnose": command_diagnose,
        "diagnose2": command_diagnose2,
        "diagnose3": command_diagnose3,
        "develop": command_develop,
        "attack": command_attack,
        "develop2": command_develop2,
        "attack3": command_attack3,
        "attack3b": command_attack3b,
        "freeze3": command_freeze3,
        "confirm3": command_confirm3,
        "recompute3": command_recompute3,
        "records": command_records,
        "validate": command_validate,
        "reproduce": command_reproduce,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(build_handlers()[args.command](args))
