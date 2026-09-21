"""Command line for the AAA-1K loop pilot: ``python -m research.aaa_1k_loop``.

Each stage writes one strict-JSON artifact under ``docs/evidence/loop_pilot_1``
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

EVIDENCE_DIR = Path("docs/evidence/loop_pilot_1")

INNER_LOOP_BLOCKS: tuple[dict[str, Any], ...] = (
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
        if current is None:
            ledger = reserve(ledger, **declared)
            added += 1
            continue
        drift = [key for key, value in declared.items() if current.get(key) != value]
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "ledger-sync": command_ledger_sync,
        "prove-fresh": command_prove_fresh,
        "champion": command_champion,
        "observe": command_observe,
        "diagnose": command_diagnose,
        "diagnose2": command_diagnose2,
        "diagnose3": command_diagnose3,
        "develop": command_develop,
        "attack": command_attack,
    }
    return handlers[args.command](args)
