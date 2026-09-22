"""Command line for iteration 0004 (post-audit work, starting with R-02).

A separate entry point, not a sub-command of :mod:`research.aaa_1k_loop.cli`,
because that module is a frozen confirmation source and may import only other
frozen sources. Identity blocks for this iteration are still declared in
``cli.INNER_LOOP_BLOCKS`` so that ``ledger-sync`` remains the one way a block
enters the ledger.

    python -m research.aaa_1k_loop.stages4 diagnose [--scratch] [--workers N]
    python -m research.aaa_1k_loop.stages4 gain [--scratch] [--workers N]
    python -m research.aaa_1k_loop.stages4 overshoot [--scratch] [--workers N]
    python -m research.aaa_1k_loop.stages4 unfold [--scratch] [--workers N]
    python -m research.aaa_1k_loop.stages4 reproduce STAGE   # rerun; committed primitives must reappear
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from .cli import _ledger_path, _stamp, project_root
from .diagnosis4 import (
    DIAGNOSIS4_SCHEMA,
    DIAGNOSTIC_BLOCK,
    HYPOTHESES,
    ITERATION_ID,
    SCRATCH_BLOCK,
    adjudicate4,
    run_diagnosis4,
)
from .evidence import read_strict_json, write_strict_json
from .identities import load_ledger

EVIDENCE_DIR = Path("docs/evidence/aaa1k_loop_0004")


def command_diagnose(args: argparse.Namespace) -> int:
    root = project_root()
    started = time.perf_counter()
    ledger = load_ledger(_ledger_path(root))
    if args.scratch:
        design: dict[str, Any] = {
            "block_id": SCRATCH_BLOCK,
            "long_streams": 1,
            "per_entry": 1,
            "initializations": 1,
        }
        records = run_diagnosis4(ledger, workers=args.workers, **design)
        adjudicate4(records)
        print(
            f"scratch shakedown: {len(records)} cells in {time.perf_counter() - started:.1f}s (not evidence)"
        )
        return 0
    records = run_diagnosis4(ledger, workers=args.workers)
    analysis = adjudicate4(records)
    payload = {
        "schema": DIAGNOSIS4_SCHEMA,
        "evidence_role": "diagnostic",
        "identity_block": DIAGNOSTIC_BLOCK,
        "hypotheses": list(HYPOTHESES),
        "analysis": analysis,
        "records": records,
    }
    write_strict_json(Path(args.output), {**_stamp(payload, root, started), "iteration_id": ITERATION_ID})
    for key, value in analysis["verdicts"].items():
        print(f"{key:4s} {value['verdict']}")
    return 0


def command_gain(args: argparse.Namespace) -> int:
    from . import diagnosis4b as d

    root = project_root()
    started = time.perf_counter()
    ledger = load_ledger(_ledger_path(root))
    if args.scratch:
        records = d.run_diagnosis4b(
            ledger,
            workers=args.workers,
            block_id=d.SCRATCH_BLOCK,
            streams=2,
            initializations=1,
            step_scale=0.25,
        )
        d.adjudicate4b(records)
        print(
            f"scratch shakedown: {len(records)} cells in {time.perf_counter() - started:.1f}s (not evidence)"
        )
        return 0
    records = d.run_diagnosis4b(ledger, workers=args.workers)
    analysis = d.adjudicate4b(records)
    payload = {
        "schema": d.DIAGNOSIS4B_SCHEMA,
        "evidence_role": "diagnostic",
        "identity_block": d.DIAGNOSTIC_BLOCK,
        "hypotheses": list(d.HYPOTHESES),
        "analysis": analysis,
        "records": records,
    }
    write_strict_json(Path(args.output), {**_stamp(payload, root, started), "iteration_id": ITERATION_ID})
    for key, value in analysis["verdicts"].items():
        print(f"{key:4s} {value['verdict']}")
    return 0


def command_overshoot(args: argparse.Namespace) -> int:
    from . import diagnosis4c as d

    root = project_root()
    started = time.perf_counter()
    ledger = load_ledger(_ledger_path(root))
    if args.scratch:
        records = d.run_diagnosis4c(
            ledger, workers=args.workers, block_id=d.SCRATCH_BLOCK, streams=2, initializations=1, steps=280
        )
        d.adjudicate4c(records)
        print(
            f"scratch shakedown: {len(records)} cells in {time.perf_counter() - started:.1f}s (not evidence)"
        )
        return 0
    records = d.run_diagnosis4c(ledger, workers=args.workers)
    analysis = d.adjudicate4c(records)
    payload = {
        "schema": d.DIAGNOSIS4C_SCHEMA,
        "evidence_role": "diagnostic",
        "identity_block": d.DIAGNOSTIC_BLOCK,
        "hypotheses": list(d.HYPOTHESES),
        "analysis": analysis,
        "records": records,
    }
    write_strict_json(Path(args.output), {**_stamp(payload, root, started), "iteration_id": ITERATION_ID})
    for key, value in analysis["verdicts"].items():
        print(f"{key:4s} {value['verdict']}")
    return 0


def command_unfold(args: argparse.Namespace) -> int:
    from . import diagnosis4d as d

    root = project_root()
    started = time.perf_counter()
    ledger = load_ledger(_ledger_path(root))
    if args.scratch:
        records = d.run_diagnosis4d(
            ledger, workers=args.workers, block_id=d.SCRATCH_BLOCK, streams=2, initializations=1, steps=280
        )
        d.adjudicate4d(records)
        print(
            f"scratch shakedown: {len(records)} cells in {time.perf_counter() - started:.1f}s (not evidence)"
        )
        return 0
    records = d.run_diagnosis4d(ledger, workers=args.workers)
    analysis = d.adjudicate4d(records)
    payload = {
        "schema": d.DIAGNOSIS4D_SCHEMA,
        "evidence_role": "diagnostic",
        "identity_block": d.DIAGNOSTIC_BLOCK,
        "hypotheses": list(d.HYPOTHESES),
        "analysis": analysis,
        "records": records,
    }
    write_strict_json(Path(args.output), {**_stamp(payload, root, started), "iteration_id": ITERATION_ID})
    for key, value in analysis["verdicts"].items():
        print(f"{key:4s} {value['verdict']}")
    return 0


def command_confirmation_primitives(args: argparse.Namespace) -> int:
    """Re-run 0006's confirmation cells from the spent blocks (read-only reproduction, no decision)."""

    from . import iteration6

    root = project_root()
    ledger = load_ledger(_ledger_path(root))
    env = next(b for b in ledger["blocks"] if b["block_id"] == iteration6.CONFIRMATION_ENV_BLOCK)
    observer = env["observed_by"]
    cells = iteration6.confirmation_cells(ledger, observer=observer)
    committed = read_strict_json(root / "docs/evidence/aaa1k_loop_0006/confirmation_2.json")
    primitives = iteration6.run(cells, arms=("gru", committed["challenger"]), workers=args.workers)
    write_strict_json(Path(args.output), {"primitives": primitives})
    return 0


# Every post-audit stage whose primitives can be regenerated from committed code and identities.
REPRODUCIBLE_4: dict[str, tuple[str, str]] = {
    "diagnose": ("stages4:diagnose", "docs/evidence/aaa1k_loop_0004/diagnosis.json"),
    "gain": ("stages4:gain", "docs/evidence/aaa1k_loop_0004/diagnosis_gain.json"),
    "overshoot": ("stages4:overshoot", "docs/evidence/aaa1k_loop_0004/diagnosis_overshoot.json"),
    "unfold": ("stages4:unfold", "docs/evidence/aaa1k_loop_0004/diagnosis_unfold.json"),
    "develop4": ("outer4:develop", "docs/evidence/aaa1k_loop_0004/development.json"),
    "develop5": ("outer5:develop", "docs/evidence/aaa1k_loop_0005/development.json"),
    "develop6": ("outer6:develop", "docs/evidence/aaa1k_loop_0006/development.json"),
    "attack6": ("outer6:attack", "docs/evidence/aaa1k_loop_0006/attack.json"),
    "confirmation6": ("stages4:confirmation-primitives", "docs/evidence/aaa1k_loop_0006/confirmation_2.json"),
}


def command_reproduce(args: argparse.Namespace) -> int:
    import importlib
    import tempfile

    from .cli import compare_evidence

    root = project_root()
    target, committed_path = REPRODUCIBLE_4[args.stage]
    module_name, command = target.split(":")
    with tempfile.TemporaryDirectory() as scratch:
        fresh_path = Path(scratch) / "rerun.json"
        namespace = argparse.Namespace(output=str(fresh_path), workers=args.workers, scratch=False)
        if module_name == "stages4":
            status = HANDLERS[command](namespace)
        else:
            module = importlib.import_module(f"research.aaa_1k_loop.{module_name}")
            status = {"develop": module.command_develop, "attack": module.command_attack}[command](namespace)
        if status != 0:
            print(f"{args.stage}: the rerun itself failed", file=sys.stderr)
            return 1
        committed = read_strict_json(root / committed_path)
        fresh = read_strict_json(fresh_path)
        if command == "confirmation-primitives":
            committed = {"primitives": committed["primitives"]}
        mismatches, added = compare_evidence(committed, fresh)
    for mismatch in mismatches[:20]:
        print(f"MISMATCH {mismatch}", file=sys.stderr)
    print(
        f"{args.stage}: {'REPRODUCED' if not mismatches else 'NOT REPRODUCED'} "
        f"({len(mismatches)} mismatches; {len(added)} fields added by later code)"
    )
    return 0 if not mismatches else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_1k_loop.stages4")
    sub = parser.add_subparsers(dest="command", required=True)
    diagnose = sub.add_parser("diagnose", help="online TBPTT update rules (audit R-02), hypotheses H20-H23")
    diagnose.add_argument("--output", default=str(EVIDENCE_DIR / "diagnosis.json"))
    diagnose.add_argument("--workers", type=int)
    diagnose.add_argument(
        "--scratch", action="store_true", help="shake the instrument down on the scratch block"
    )
    gain = sub.add_parser("gain", help="closed-loop gain of the previous-error channel (R-01), H24-H28")
    gain.add_argument("--output", default=str(EVIDENCE_DIR / "diagnosis_gain.json"))
    gain.add_argument("--workers", type=int)
    gain.add_argument("--scratch", action="store_true", help="shake the instrument down on the scratch block")
    overshoot = sub.add_parser("overshoot", help="single-step SGD overshoot (R-01), H29-H31")
    overshoot.add_argument("--output", default=str(EVIDENCE_DIR / "diagnosis_overshoot.json"))
    overshoot.add_argument("--workers", type=int)
    overshoot.add_argument(
        "--scratch", action="store_true", help="shake the instrument down on the scratch block"
    )
    unfold = sub.add_parser("unfold", help="target-unfolding frame lock (R-01), H32-H34")
    unfold.add_argument("--output", default=str(EVIDENCE_DIR / "diagnosis_unfold.json"))
    unfold.add_argument("--workers", type=int)
    unfold.add_argument(
        "--scratch", action="store_true", help="shake the instrument down on the scratch block"
    )
    primitives = sub.add_parser(
        "confirmation-primitives", help="re-run 0006's confirmation cells from the spent blocks (no decision)"
    )
    primitives.add_argument("--output", required=True)
    primitives.add_argument("--workers", type=int)
    reproduce = sub.add_parser("reproduce", help="rerun a stage; committed primitives must reappear exactly")
    reproduce.add_argument("stage", choices=sorted(REPRODUCIBLE_4))
    reproduce.add_argument("--workers", type=int)
    return parser


HANDLERS: dict[str, Any] = {
    "diagnose": command_diagnose,
    "gain": command_gain,
    "overshoot": command_overshoot,
    "unfold": command_unfold,
    "confirmation-primitives": command_confirmation_primitives,
    "reproduce": command_reproduce,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(HANDLERS[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
