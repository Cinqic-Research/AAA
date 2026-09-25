"""``python -m research.aaa_python_v1``: the ``aaa.python.v1`` command line.

    spec-hash      the packaged protocol's canonical hash
    fingerprint    the phase's scientific source identity (checkout only)
    audit          parameter and state accounting of the declared arms
    develop        run one declared development stage and write its evidence
    summarize      (re)derive a stage's summary from its retained primitives

Exit codes: 0 success, 1 a check failed, 2 refused or invalid input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION, stages
from . import spec as spec_module
from .experiment import DESIGN, ArmSpec, run_adapt, run_stage, v0_instrument_job
from .summarize import summarize

STAGES = ("encoders", "heads", "capacity", "optimization", "tool", "adapt", "plasticity", "attack")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> Any:
    def refuse(token: str) -> Any:
        raise ValueError(f"non-standard JSON constant {token} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def _selected(previous: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in previous:
        raise SystemExit(f"stage requires the {name} evidence (--previous)")
    return previous[name]


def plan(
    stage: str, previous: dict[str, Any]
) -> tuple[list[ArmSpec], list[Any], list[Any], list[str], dict[str, Any]]:
    """Arms, primary and secondary contrasts, baselines and selections for a stage."""

    if stage == "encoders":
        return (
            stages.encoder_arms(),
            stages.ENCODER_PRIMARY,
            stages.ENCODER_SECONDARY,
            ["uniform", "majority", "lookup", "v0_heuristic", "rules", "medoid", "visible_tests"],
            {},
        )
    encoder = stages.select_encoder(_selected(previous, "encoders")["summary"])
    if stage == "heads":
        return stages.head_arms(encoder), stages.head_primary(encoder), [], [], {"encoder": encoder}
    output, localize = stages.select_heads(_selected(previous, "heads")["summary"], encoder)
    selections = {"encoder": encoder, "output_head": output, "localize_head": localize}
    if stage == "capacity":
        return (
            stages.capacity_arms(encoder, output, localize),
            stages.capacity_primary(encoder),
            stages.capacity_secondary(encoder),
            [],
            selections,
        )
    capacity = _selected(previous, "capacity")["stage"]
    if stage == "optimization":
        return stages.optimization_arms(capacity, encoder), stages.optimization_primary(), [], [], selections
    if stage == "tool":
        return stages.tool_arms(capacity, encoder), stages.tool_primary(), [], ["visible_tests"], selections
    raise SystemExit(f"stage {stage} is run by its own command")


def run_attack_stage(previous: dict[str, Any], workers: int) -> dict[str, Any]:
    """The selected arms, with their development budgets and no re-tuning, on the attack pool."""

    from .experiment import run_attack
    from .summarize import summarize

    encoder_stage = _selected(previous, "encoders")["stage"]
    encoder = stages.select_encoder(encoder_stage["summary"])
    capacity = _selected(previous, "capacity")["stage"]
    arms = [stages._fixed(encoder_stage["arms"][n], n) for n in ("h16@e1", "h16@e2")]
    arms += [
        stages._fixed(capacity["arms"][f"{t}@{encoder}"], f"{t}@{encoder}")
        for t in ("1k", "4k", "10k", "20k")
    ]
    evidence: dict[str, Any] = {
        "stage": "attack",
        "arms": {a.name: {**a.to_json(), "parameters": a.parameters()} for a in arms},
    }
    evidence.update(
        run_attack(
            arms, ["majority", "lookup", "v0_heuristic", "rules", "medoid", "visible_tests"], workers=workers
        )
    )
    primary = [
        ("h16@e2", "h16@e1", "frozen"),
        (f"10k@{encoder}", f"1k@{encoder}", "frozen"),
        (f"10k@{encoder}", f"4k@{encoder}", "frozen"),
    ]
    evidence["primary"] = [list(c) for c in primary]
    evidence["secondary"] = []
    evidence["summary"] = summarize(evidence, primary, split="attack")
    return evidence


def run_life_stage(stage: str, previous: dict[str, Any], workers: int) -> dict[str, Any]:
    """The adaptation and plasticity stages: the tuned 1K and 10K arms of the capacity stage."""

    from concurrent.futures import ProcessPoolExecutor

    from .plasticity import PLASTICITY, life_job, summarize_plasticity
    from .summarize import summarize_adapt

    encoder = stages.select_encoder(_selected(previous, "encoders")["summary"])
    capacity = _selected(previous, "capacity")["stage"]
    arms = stages.adapt_or_plasticity_arms(capacity, encoder)
    selected = {a.name: {"learning_rate": a.learning_rate, "epochs": a.epochs} for a in arms}
    evidence: dict[str, Any] = {
        "stage": stage,
        "arms": {a.name: {**a.to_json(), "parameters": a.parameters()} for a in arms},
    }
    if stage == "adapt":
        evidence["rows"] = run_adapt(arms, selected, workers=workers)
        evidence["summary"] = summarize_adapt(evidence["rows"])
    else:
        jobs = [(a, a.learning_rate, i) for a in arms for i in range(PLASTICITY["initializations"])]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evidence["rows"] = list(pool.map(life_job, *zip(*jobs, strict=True)))
        evidence["design"] = PLASTICITY
        evidence["summary"] = summarize_plasticity(evidence["rows"])
    return evidence


def cmd_develop(args: argparse.Namespace) -> int:
    from .identity import provenance

    record = provenance()
    if record["source"].get("dirty"):
        print(
            "refusing: development evidence is produced only from a clean, committed checkout",
            file=sys.stderr,
        )
        return 2
    previous = {p.stem.split("_", 1)[-1] if "_" in p.stem else p.stem: read_json(p) for p in args.previous}
    started = time.time()
    if args.stage == "attack":
        evidence = run_attack_stage(previous, args.workers)
        document = {
            "schema": "aaa.python.v1.development_stage.v1",
            "protocol": PROTOCOL_VERSION,
            "status": "attack evidence: used once, after development selections, before the freeze",
            "spec_sha256": spec_module.spec_hash(),
            "design": DESIGN,
            "provenance": record,
            "wall_seconds": time.time() - started,
            "stage": evidence,
        }
        write_json(args.output, document)
        print(f"wrote {args.output} ({document['wall_seconds']:.0f} s)")
        return 0
    if args.stage in ("adapt", "plasticity"):
        evidence = run_life_stage(args.stage, previous, args.workers)
        document = {
            "schema": "aaa.python.v1.development_stage.v1",
            "protocol": PROTOCOL_VERSION,
            "status": "development evidence: not confirmation, not a capability claim",
            "spec_sha256": spec_module.spec_hash(),
            "design": DESIGN,
            "provenance": record,
            "wall_seconds": time.time() - started,
            "stage": evidence,
        }
        write_json(args.output, document)
        print(f"wrote {args.output} ({document['wall_seconds']:.0f} s)")
        return 0
    arms, primary, secondary, baselines, selections = plan(args.stage, previous)
    evidence = run_stage(args.stage, arms, baselines=baselines, workers=args.workers)
    if args.stage == "encoders":
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            evidence["anchors"] = list(
                pool.map(v0_instrument_job, range(DESIGN["evaluate"]["initializations"]))
            )
    evidence["selections"] = selections
    evidence["primary"] = [list(c) for c in primary]
    evidence["secondary"] = [list(c) for c in secondary]
    evidence["summary"] = summarize(evidence, primary, secondary)
    if args.stage == "capacity":
        evidence["capacity_verdict"] = stages.capacity_verdict(evidence["summary"], selections["encoder"])
    document = {
        "schema": "aaa.python.v1.development_stage.v1",
        "protocol": PROTOCOL_VERSION,
        "status": "development evidence: not confirmation, not a capability claim",
        "spec_sha256": spec_module.spec_hash(),
        "design": DESIGN,
        "provenance": record,
        "wall_seconds": time.time() - started,
        "stage": evidence,
    }
    write_json(args.output, document)
    print(f"wrote {args.output} ({document['wall_seconds']:.0f} s)")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    document = read_json(args.evidence)
    stage = document["stage"]
    if stage["stage"] == "adapt":
        from .summarize import summarize_adapt

        fresh = summarize_adapt(stage["rows"])
    elif stage["stage"] == "plasticity":
        from .plasticity import summarize_plasticity

        fresh = summarize_plasticity(stage["rows"])
    else:
        primary = [tuple(c) for c in stage["primary"]]
        secondary = [tuple(c) for c in stage["secondary"]]
        split = "attack" if stage["stage"] == "attack" else "development"
        fresh = summarize(stage, primary, secondary, split=split)
    same = json.dumps(fresh, sort_keys=True) == json.dumps(stage["summary"], sort_keys=True)
    print("summary reproduced from primitives" if same else "SUMMARY DIFFERS from its primitives")
    return 0 if same else 1


def cmd_audit(_: argparse.Namespace) -> int:
    rows = {arm.name: arm.parameters() for arm in stages.encoder_arms()}
    for encoder in ("e1", "e2"):
        for output, localize in (("onehot", "onehot"), ("gauss", "pointer")):
            for arm in stages.capacity_arms(encoder, output, localize):
                rows[f"{arm.name}:{output}/{localize}"] = {"hidden": arm.hidden, **arm.parameters()}
    for name, account in rows.items():
        print(
            f"{name:40s} hidden={account.get('hidden', '-')!s:>4} trainable={account['trainable_parameters']:>7} "
            f"optimizer={account['optimizer_state']} adaptive={account['adaptive_state_total']}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_python_v1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("spec-hash")
    sub.add_parser("fingerprint")
    sub.add_parser("audit")
    develop = sub.add_parser("develop")
    develop.add_argument("--stage", choices=STAGES, required=True)
    develop.add_argument("--output", type=Path, required=True)
    develop.add_argument("--previous", type=Path, nargs="*", default=[])
    develop.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    summ = sub.add_parser("summarize")
    summ.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "spec-hash":
            print(f"protocol: {PROTOCOL_VERSION}\nhash: {spec_module.spec_hash()}")
            return 0
        if args.command == "fingerprint":
            from .identity import fingerprint

            fp = fingerprint()
            print(f"{fp['sha256']}\n{fp['file_count']} files, phase {fp['phase_version']}")
            return 0
        if args.command == "audit":
            return cmd_audit(args)
        if args.command == "develop":
            return cmd_develop(args)
        return cmd_summarize(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


__all__ = ["main", "run_adapt"]
