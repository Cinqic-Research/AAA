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
"""Float tolerance used to *report* cross-platform drift in ``reproduce`` (see ``AAA-173``).

On the evidence platform (and any CPU with the same NumPy kernels) every
stage is bit-identical; ``reproduce --exact`` gates on that. Across
instruction sets it is not, and the default mode gates on identities,
structure and verdicts only, counting the cells that drift beyond this
tolerance.
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
    committed: Mapping[str, Any],
    fresh: Mapping[str, Any],
    challenger: str | None = None,
    *,
    exact: bool = False,
) -> dict[str, Any]:
    """Reproduction of a stage artifact (``AAA-173``).

    Gating in every mode: cell identities, artifact structure outside the
    recomputed aggregates, and every adjudicated verdict recomputed from the
    rerun's own primitives. With ``exact`` every cell value must also be
    bit-identical, which is the guarantee on the platform that produced the
    evidence (and on any CPU with the same NumPy kernels). Without it, per-cell
    numeric drift is reported but does not gate: on AVX-512 (Zen 4) runners long
    online-learning trajectories drift -- by up to ~30% in diverged, chaotic
    cells and ~1e-3 in long non-diverged ones -- so no fixed numeric tolerance
    separates platform noise from a defect, while the conclusions must not move.
    """

    key = "records" if "records" in committed else "primitives"
    gating: list[str] = []
    top_committed = {k: v for k, v in committed.items() if k not in (key, *RESULT_FIELDS)}
    top_fresh = {k: v for k, v in fresh.items() if k not in (key, *RESULT_FIELDS)}
    gating += compare_tolerant(top_committed, top_fresh)[0]
    old, new = committed[key], fresh[key]
    if len(old) != len(new):
        gating.append(f"$.{key}: {len(old)} cells committed, {len(new)} rerun")
    stable_deviation = chaotic_deviation = 0.0
    chaotic_cells = flag_flips = drifting_cells = bit_different_cells = 0
    drift_examples: list[str] = []
    for index, (c, f) in enumerate(zip(old, new, strict=False)):
        identity = [field for field in CELL_IDENTITY if c.get(field) != f.get(field)]
        if identity:
            gating.append(f"$.{key}[{index}]: cell identity differs in {identity}")
            continue
        is_chaotic = chaotic(c) or chaotic(f)
        chaotic_cells += int(is_chaotic)
        flag_flips += int(chaotic(c) != chaotic(f))
        beyond, deviations = compare_tolerant(c, f, f"$.{key}[{index}]")
        if c != f:
            bit_different_cells += 1
            if exact:
                gating.append(f"$.{key}[{index}]: not bit-identical ({beyond[:1] or 'float noise'})")
        if beyond:
            drifting_cells += 1
            drift_examples += beyond[:1]
        largest = max(deviations, default=0.0)
        if is_chaotic:
            chaotic_deviation = max(chaotic_deviation, largest)
        else:
            stable_deviation = max(stable_deviation, largest)
    committed_verdicts = verdicts(committed, challenger)
    fresh_verdicts = verdicts(fresh, challenger)
    verdicts_identical = committed_verdicts == fresh_verdicts
    if not verdicts_identical:
        gating.insert(0, f"verdicts differ: committed {committed_verdicts} rerun {fresh_verdicts}")
    return {
        "mismatches": gating,
        "verdicts_identical": verdicts_identical,
        "verdicts": committed_verdicts,
        "cells": len(old),
        "bit_different_cells": bit_different_cells,
        "cells_beyond_float_tolerance": drifting_cells,
        "chaotic_cells": chaotic_cells,
        "chaotic_flag_flips": flag_flips,
        "largest_stable_deviation": stable_deviation,
        "largest_chaotic_deviation": chaotic_deviation,
        "drift_examples": drift_examples[:5],
    }


def command_reproduce(args: argparse.Namespace) -> int:
    import importlib
    import platform
    import tempfile

    import numpy

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
    result = compare_reproduction(committed, fresh, challenger, exact=args.exact)
    print(f"platform: {platform.machine()} {platform.processor() or ''} numpy {numpy.__version__}".strip())
    print(f"verdicts: {'IDENTICAL' if result['verdicts_identical'] else 'DIFFERENT'} {result['verdicts']}")
    print(
        f"cells: {result['cells']}; bit-different {result['bit_different_cells']}; beyond 1e-9 "
        f"{result['cells_beyond_float_tolerance']}; chaotic {result['chaotic_cells']} "
        f"({result['chaotic_flag_flips']} changed divergence status); largest relative drift: stable "
        f"{result['largest_stable_deviation']:.2e}, chaotic {result['largest_chaotic_deviation']:.2e}"
    )
    for example in result["drift_examples"]:
        print(f"drift (reported, not gating{'; --exact gates it' if args.exact else ''}): {example}")
    for mismatch in result["mismatches"][:20]:
        print(f"MISMATCH {mismatch}", file=sys.stderr)
    mode = "bitwise" if args.exact else "verdict-level"
    print(f"{args.stage}: {'REPRODUCED' if not result['mismatches'] else 'NOT REPRODUCED'} ({mode})")
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
    reproduce.add_argument(
        "--exact", action="store_true", help="require bit-identical cells (same-platform guarantee)"
    )
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
