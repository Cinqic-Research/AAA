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
from collections.abc import Mapping
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


FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-15
"""Cross-machine float tolerance for ``reproduce``.

Bitwise reproduction holds on one machine (every stage reproduced with zero
mismatches locally), but long float computations are not bit-stable across
CPUs, SIMD widths and library builds: on CI the last digit of MAEs differs
(relative ~1e-15). Floats therefore reproduce when they agree to a relative
1e-9; strings, integers, booleans (divergence flags, locks, verdicts) and
structure must agree exactly, so no adjudicated outcome can drift. The
largest deviation is always reported, never hidden.
"""


def compare_tolerant(committed: Any, fresh: Any, path: str = "$") -> tuple[list[str], list[float]]:
    """``(mismatches, float relative deviations)``; volatile provenance fields are skipped."""

    from .cli import VOLATILE

    mismatches: list[str] = []
    deviations: list[float] = []
    if isinstance(committed, dict):
        if not isinstance(fresh, dict):
            return [f"{path}: type changed"], deviations
        for key, value in committed.items():
            if key in VOLATILE:
                continue
            if key not in fresh:
                mismatches.append(f"{path}.{key}: missing from the rerun")
                continue
            m, d = compare_tolerant(value, fresh[key], f"{path}.{key}")
            mismatches += m
            deviations += d
    elif isinstance(committed, list):
        if not isinstance(fresh, list) or len(fresh) != len(committed):
            return [f"{path}: list length changed"], deviations
        for index, (c, f) in enumerate(zip(committed, fresh, strict=True)):
            m, d = compare_tolerant(c, f, f"{path}[{index}]")
            mismatches += m
            deviations += d
    elif isinstance(committed, float) and isinstance(fresh, float):
        if committed != fresh:
            relative = abs(committed - fresh) / max(abs(committed), abs(fresh))
            deviations.append(relative)
            if abs(committed - fresh) > FLOAT_ATOL + FLOAT_RTOL * max(abs(committed), abs(fresh)):
                mismatches.append(
                    f"{path}: committed {committed!r} rerun {fresh!r} (relative {relative:.2e})"
                )
    elif type(committed) is not type(fresh) or committed != fresh:
        mismatches.append(f"{path}: committed {committed!r} rerun {fresh!r}")
    return mismatches, deviations


CELL_IDENTITY = ("condition", "init_index", "init_seed", "stream_id", "stream_seed")
RESULT_FIELDS = ("analysis", "screens", "selected_for_attack", "adjudication", "decision")
"""Aggregates recomputed from the cells; judged at the verdict level, never number by number."""


def chaotic(record: Any) -> bool:
    """A cell in which any arm diverged or failed: its trajectory is not bit-stable across platforms.

    The champion's frame-locked cells amplify a last-bit float difference into a
    different trajectory (observed on CI runners: MAE differences of up to ~30%
    in diverged cells, none above 1e-9 elsewhere). Detected from the cell's own
    primitives, so it needs no stage-specific knowledge.
    """

    if not isinstance(record, dict):
        return False
    if record.get("failures"):
        return True
    mae = record.get("mae")
    if isinstance(mae, dict) and isinstance(mae.get("persistence"), float):
        limit = 2.0 * mae["persistence"]
        return any(isinstance(v, float) and v > limit for k, v in mae.items() if k != "persistence")

    def any_diverged(node: Any) -> bool:
        if isinstance(node, dict):
            return node.get("diverged") is True or any(any_diverged(v) for v in node.values())
        if isinstance(node, list):
            return any(any_diverged(v) for v in node)
        return False

    return any_diverged(record.get("arms", {}))


def verdicts(payload: Mapping[str, Any], challenger: str | None = None) -> Any:
    """The adjudicated conclusions of a stage's artifact (recomputed for confirmation primitives)."""

    if "analysis" in payload:
        return {key: value["verdict"] for key, value in payload["analysis"]["verdicts"].items()}
    if "screens" in payload:
        return {
            "selected_for_attack": payload["selected_for_attack"],
            "screens": {
                arm: {
                    "passed": screen["passed"],
                    "S1": screen["S1_long_horizon_stability"]["status"],
                    **{c: v["status"] for c, v in screen["S2_standard_families"].items()},
                    **{c: v["status"] for c, v in screen["S3_long_non_coarse"].items()},
                }
                for arm, screen in payload["screens"].items()
            },
        }
    if "adjudication" in payload:
        return {
            "outcome": payload["adjudication"]["outcome"],
            "advance": payload["adjudication"]["advance_to_freeze"],
        }
    from . import iteration6

    decision = iteration6.decide(payload["primitives"], str(challenger))
    return {"statuses": decision["statuses"], "outcome": decision["outcome"]}


def compare_reproduction(
    committed: Mapping[str, Any], fresh: Mapping[str, Any], challenger: str | None = None
) -> dict[str, Any]:
    """Two-tier cross-platform comparison: stable cells numerically, conclusions exactly."""

    key = "records" if "records" in committed else "primitives"
    mismatches: list[str] = []
    deviations: list[float] = []
    top_committed = {k: v for k, v in committed.items() if k not in (key, *RESULT_FIELDS)}
    top_fresh = {k: v for k, v in fresh.items() if k not in (key, *RESULT_FIELDS)}
    m, d = compare_tolerant(top_committed, top_fresh)
    mismatches += m
    deviations += d
    old, new = committed[key], fresh[key]
    chaotic_cells = flag_flips = 0
    chaotic_deviation = 0.0
    if len(old) != len(new):
        mismatches.append(f"$.{key}: {len(old)} cells committed, {len(new)} rerun")
    for index, (c, f) in enumerate(zip(old, new, strict=False)):
        identity = [field for field in CELL_IDENTITY if c.get(field) != f.get(field)]
        if identity:
            mismatches.append(f"$.{key}[{index}]: cell identity differs in {identity}")
            continue
        if chaotic(c) or chaotic(f):
            chaotic_cells += 1
            flag_flips += int(chaotic(c) != chaotic(f))
            _m, cell_deviations = compare_tolerant(c, f)
            chaotic_deviation = max([chaotic_deviation, *cell_deviations])
            continue
        m, d = compare_tolerant(c, f, f"$.{key}[{index}]")
        mismatches += m
        deviations += d
    committed_verdicts = verdicts(committed, challenger)
    fresh_verdicts = verdicts(fresh, challenger)
    if committed_verdicts != fresh_verdicts:
        mismatches.append(f"verdicts differ: committed {committed_verdicts} rerun {fresh_verdicts}")
    return {
        "mismatches": mismatches,
        "cells": len(old),
        "chaotic_cells": chaotic_cells,
        "chaotic_flag_flips": flag_flips,
        "largest_stable_deviation": max(deviations, default=0.0),
        "stable_floats_not_bit_identical": len(deviations),
        "largest_chaotic_deviation": chaotic_deviation,
        "verdicts": committed_verdicts,
    }


def command_reproduce(args: argparse.Namespace) -> int:
    import importlib
    import tempfile

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
    challenger = committed.get("challenger")
    if command == "confirmation-primitives":
        committed = {"primitives": committed["primitives"]}
    result = compare_reproduction(committed, fresh, challenger)
    for mismatch in result["mismatches"][:20]:
        print(f"MISMATCH {mismatch}", file=sys.stderr)
    print(
        f"{args.stage}: {'REPRODUCED' if not result['mismatches'] else 'NOT REPRODUCED'}: "
        f"{len(result['mismatches'])} mismatches; verdicts {'identical' if not result['mismatches'] else 'see above'}; "
        f"{result['cells']} cells, {result['chaotic_cells']} chaotic (diverged/failed; "
        f"{result['chaotic_flag_flips']} changed divergence status, largest deviation "
        f"{result['largest_chaotic_deviation']:.2e}); stable cells: "
        f"{result['stable_floats_not_bit_identical']} floats not bit-identical, largest relative deviation "
        f"{result['largest_stable_deviation']:.2e} (tolerance {FLOAT_RTOL:g})"
    )
    return 0 if not result["mismatches"] else 1


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
