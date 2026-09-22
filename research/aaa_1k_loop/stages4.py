"""Command line for iteration 0004 (post-audit work, starting with R-02).

A separate entry point, not a sub-command of :mod:`research.aaa_1k_loop.cli`,
because that module is a frozen confirmation source and may import only other
frozen sources. Identity blocks for this iteration are still declared in
``cli.INNER_LOOP_BLOCKS`` so that ``ledger-sync`` remains the one way a block
enters the ledger.

    python -m research.aaa_1k_loop.stages4 diagnose [--scratch] [--workers N]
    python -m research.aaa_1k_loop.stages4 gain [--scratch] [--workers N]
"""

from __future__ import annotations

import argparse
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
from .evidence import write_strict_json
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int({"diagnose": command_diagnose, "gain": command_gain}[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
