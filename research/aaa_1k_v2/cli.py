"""``python -m research.aaa_1k_v2``: the aaa.1k.v2 phase commands.

audit                       recount every arm's parameters and adaptive state
fingerprint                 the v2 scientific source fingerprint
registry [--sync]           verify (or write) the identity registry from the plan
prove-fresh                 disjointness proof against every historical identity
develop --output PATH       the development stage (hyperparameter selection + screen)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

from . import PARAMETER_CAP, PHASE_VERSION, identities, plan
from .evidence import write

ROOT = Path(__file__).resolve().parents[2]


def environment(device: str) -> dict[str, Any]:
    from aaa.compute import backend_provenance, resolve_backend
    from research.aaa_1k.identity import git_provenance, phase_fingerprint

    from .identity import fingerprint
    from .parallel import describe_workers

    return {
        "phase": PHASE_VERSION,
        "protocol": plan.PROTOCOL_VERSION,
        "git": git_provenance(ROOT),
        "v2_fingerprint": fingerprint(ROOT)["sha256"],
        "aaa_1k_v1_fingerprint": phase_fingerprint(ROOT)["sha256"],
        "registry_sha256": hashlib.sha256((ROOT / identities.REGISTRY_PATH).read_bytes()).hexdigest(),
        "backend": backend_provenance(resolve_backend(device)),
        "workers": describe_workers("auto"),
    }


def command_audit(_args: argparse.Namespace) -> int:
    from .arms import CANDIDATE_TEMPLATES, CHAMPION_1, CONTROL_TEMPLATES, audit

    ok = True
    for arm in (CHAMPION_1, *CANDIDATE_TEMPLATES.values(), *CONTROL_TEMPLATES.values()):
        record = audit(arm)
        good = record["agree"] and record["within_cap"]
        ok &= good
        print(
            f"{arm.name:18s} trainable={record['trainable_parameters']:5d} (cap {PARAMETER_CAP}) "
            f"hidden={record['hidden_state_scalars']:3d} tbptt={record['tbptt_buffer_scalars']:4d} "
            f"traces={record['eligibility_trace_scalars']:3d} optimizer=0 total={record['total_adaptive_state_scalars']:5d} "
            f"{'OK' if good else 'FAIL'}"
        )
    return 0 if ok else 1


def command_fingerprint(_args: argparse.Namespace) -> int:
    from .identity import fingerprint

    record = fingerprint(ROOT)
    print(record["sha256"])
    print(f"{record['file_count']} files, phase {record['phase_version']}")
    return 0


def command_registry(args: argparse.Namespace) -> int:
    path = ROOT / identities.REGISTRY_PATH
    registry = identities.load_registry(path)
    if args.sync:
        registry = identities.reserve(registry, plan.declared_blocks())
        identities.write_registry(path, registry)
        print(f"registry synced: {len(registry['blocks'])} blocks")
        return 0
    declared = {block["block_id"] for block in plan.declared_blocks()}
    present = {block["block_id"] for block in registry["blocks"]}
    missing, extra = sorted(declared - present), sorted(present - declared)
    print(f"registry: {len(present)} blocks; missing {len(missing)}; undeclared {len(extra)}")
    return 0 if not missing and not extra else 1


def command_prove_fresh(_args: argparse.Namespace) -> int:
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    proof = identities.prove_disjoint(registry, ROOT)
    print(
        f"{proof['status']}: {proof['v2_seed_count']} v2 seeds (64-bit) against {proof['compared_against']}"
    )
    return 0


def command_develop(args: argparse.Namespace) -> int:
    from . import tournament

    output = Path(args.output)
    if output.exists():
        print(f"error: {output} exists; evidence is never overwritten", file=sys.stderr)
        return 2
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    # provenance is captured before anything runs, so later edits to the working tree cannot be
    # recorded as the code that produced this evidence (AAA-179)
    captured = environment(plan.DEVELOPMENT_BACKEND)
    started = time.time()
    raw_dir = Path(args.raw_root) / "aaa_1k_v2" if args.raw_root else None
    result = tournament.run_development(
        registry, workers=args.workers, role=args.role, raw_dir=raw_dir, log=lambda m: print(m, flush=True)
    )
    result["environment"] = captured
    result["wall_seconds"] = time.time() - started
    digest = write(output, result)
    if args.role == "development":
        for block in registry["blocks"]:
            if block["role"] == "development" and block["status"] == "reserved":
                registry = identities.mark(
                    registry, block["block_id"], status="used", observed_by=f"development:{output.name}"
                )
        identities.write_registry(ROOT / identities.REGISTRY_PATH, registry)
    print(f"wrote {output} sha256 {digest}; challenger: {result['challenger']}")
    return 0


def _load(path: str) -> Any:
    from .evidence import read

    return read(ROOT / path)


def _challenger_arm(development: Any) -> Any:
    from .attack import arm_from_record
    from .tournament import candidate_template

    name = development["challenger"]
    if name is None:
        return None
    return arm_from_record(development["candidates"][name], candidate_template(name))


def command_diagnose(args: argparse.Namespace) -> int:
    from . import diagnostics
    from .attack import arm_from_record
    from .tournament import candidate_template

    output = Path(args.output)
    if output.exists():
        print(f"error: {output} exists", file=sys.stderr)
        return 2
    path = ROOT / identities.REGISTRY_PATH
    registry = diagnostics.register(identities.load_registry(path))
    identities.write_registry(path, registry)
    captured = environment("cpu")
    development = _load(args.development)
    arm = _challenger_arm(development)
    if arm is None:
        scored = {
            n: c["decision"].get("selected_score")
            for n, c in development["candidates"].items()
            if c["decision"].get("selected")
        }
        best = min(scored, key=lambda n: scored[n])
        arm = arm_from_record(development["candidates"][best], candidate_template(best))
    started = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    result: dict[str, Any] = {
        "schema": "aaa.1k.v2.diagnostics.v1",
        "stage": "diagnostic",
        "hypotheses": diagnostics.HYPOTHESES,
        "m1": diagnostics.m1(registry, workers=args.workers, log=log),
        "aaa162": diagnostics.aaa162(registry),
        "aaa172": diagnostics.aaa172(registry),
        "optimizer": diagnostics.optimizer_probe(registry, arm, workers=args.workers, log=log),
        "environment": captured,
        "wall_seconds": None,
    }
    result["wall_seconds"] = time.time() - started
    digest = write(output, result)
    for block in registry["blocks"]:
        if block["role"] == "diagnostic" and block["status"] == "reserved":
            registry = identities.mark(
                registry, block["block_id"], status="used", observed_by=f"diagnostic:{output.name}"
            )
    identities.write_registry(path, registry)
    print(f"wrote {output} sha256 {digest}")
    return 0


def command_attack(args: argparse.Namespace) -> int:
    from . import attack

    output = Path(args.output)
    if output.exists():
        print(f"error: {output} exists", file=sys.stderr)
        return 2
    development = _load(args.development)
    challenger = _challenger_arm(development)
    if challenger is None:
        print("no challenger was designated in development; there is nothing to attack")
        return 3
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    captured = environment("cpu")
    started = time.time()
    result = attack.run_attack(
        registry, challenger, workers=args.workers, cuda_device=args.cuda, log=lambda m: print(m, flush=True)
    )
    result["environment"] = captured
    result["wall_seconds"] = time.time() - started
    digest = write(output, result)
    path = ROOT / identities.REGISTRY_PATH
    for block in registry["blocks"]:
        if block["role"] == "attack" and block["status"] == "reserved":
            registry = identities.mark(
                registry, block["block_id"], status="used", observed_by=f"attack:{output.name}"
            )
    identities.write_registry(path, registry)
    print(f"wrote {output} sha256 {digest}; statuses {result['statuses']}; advance {result['advance']}")
    return 0


def confirmation_arms(development: Any, challenger: Any) -> dict[str, tuple[Any, bool]]:
    """Every arm confirmation observes, with whether it runs the synthetic external suite."""

    from dataclasses import replace

    from . import arms as arm_module
    from .attack import arm_from_record
    from .engine import CellConfig
    from .tournament import candidate_template

    out: dict[str, tuple[Any, bool]] = {"c1_champion1": (arm_module.CHAMPION_1, True)}
    for name, record in development["candidates"].items():
        if record["decision"].get("selected"):
            arm = arm_from_record(record, candidate_template(name))
            out[name] = (arm, True)
    out["c0_champion0"] = (arm_module.CHAMPION_0, False)
    mlp = replace(
        arm_module.CONTROL_TEMPLATES["mlp_v2"], config=CellConfig(learning_rate=0.01, gradient_clip=None)
    )
    out["mlp_v2"] = (mlp, True)
    parents = [arm_module.CHAMPION_1] + ([challenger] if challenger is not None else [])
    for parent in parents:
        for name, ablation in arm_module.ablations(parent).items():
            out[name] = (ablation, True)
    return out


def command_freeze(args: argparse.Namespace) -> int:
    from . import freeze

    output = ROOT / args.output
    if output.exists():
        print(f"error: {output} exists", file=sys.stderr)
        return 2
    development = _load(args.development)
    challenger = _challenger_arm(development)
    if challenger is not None:
        if not args.attack:
            print("error: a challenger needs its attack evidence", file=sys.stderr)
            return 2
        attack = _load(args.attack)
        if not attack["advance"]:
            print("the attack rejected the challenger; freezing a characterization-only confirmation")
            challenger = None
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    arms = confirmation_arms(development, challenger)
    capability_arms = ["c1_champion1"] + ([challenger.name] if challenger is not None else [])
    monash_arms = [name for name in arms if name in ("c1_champion1", *plan.CANDIDATES)]
    content = freeze.frozen_content(
        challenger=challenger, arms=arms, capability_arms=capability_arms, monash_arms=monash_arms
    )
    manifest = freeze.build_manifest(
        ROOT,
        registry,
        content,
        development=args.development,
        attack=args.attack if challenger is not None or args.attack else None,
    )
    digest = write(output, manifest)
    print(
        f"wrote {args.output} sha256 {digest}; challenger {None if challenger is None else challenger.name}; commit it, then run confirm"
    )
    return 0


def command_confirm(args: argparse.Namespace) -> int:
    from . import freeze
    from .confirmation import run_confirmation
    from .evidence import read, sha256_file

    output = ROOT / args.output
    if output.exists():
        print(f"error: {output} exists", file=sys.stderr)
        return 2
    head = freeze.require_committed(ROOT, args.freeze)
    manifest = read(ROOT / args.freeze)
    manifest["_sha256"] = sha256_file(ROOT / args.freeze)
    path = ROOT / identities.REGISTRY_PATH
    registry = identities.load_registry(path)
    problems = freeze.verify(manifest, ROOT, registry)
    if problems:
        print(
            "REFUSED: the live repository disagrees with the freeze:", *problems, sep="\n  ", file=sys.stderr
        )
        return 2
    observer = f"confirmation:{manifest['_sha256'][:16]}"
    captured = environment(manifest["frozen"]["backend"])
    if args.claim_remote:
        from aaa.noise.reservation import ReservationError, reserve_confirmation_batch

        try:
            claim = reserve_confirmation_batch(
                ROOT,
                batch_id="aaa1k-v2-confirmation",
                role="confirmation",
                attempt_id=observer,
                scientific_fingerprint_sha256=manifest["v2_fingerprint"],
                resume=False,
            )
        except ReservationError as error:
            print(f"REFUSED: durable remote claim failed: {error}", file=sys.stderr)
            return 2
        print(f"remote claim created: {claim['claim_ref']}")
    for block_id in manifest["confirmation_blocks"]:
        registry = identities.mark(registry, block_id, status="spent", observed_by=observer)
    identities.write_registry(path, registry)
    import subprocess

    subprocess.run(["git", "-C", str(ROOT), "add", identities.REGISTRY_PATH], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "commit",
            "-q",
            "-m",
            f"claim(aaa-1k-v2): spend every confirmation block before observation\n\nobserver {observer}; freeze {args.freeze} at {head}",
        ],
        check=True,
    )
    print(f"confirmation blocks spent and committed by {observer}")
    arms = {name: freeze.arm_from_dict(entry["spec"]) for name, entry in manifest["frozen"]["arms"].items()}
    manifest["arms"] = manifest["frozen"]["arms"]
    manifest["capability_arms"] = manifest["frozen"]["capability_arms"]
    manifest["monash_arms"] = manifest["frozen"]["monash_arms"]
    manifest["challenger"] = manifest["frozen"]["challenger"]
    started = time.time()
    result = run_confirmation(
        registry,
        manifest,
        arms,
        observer=observer,
        workers=args.workers,
        data_root=args.data_root,
        log=lambda m: print(m, flush=True),
    )
    result["environment"] = captured
    result["wall_seconds"] = time.time() - started
    digest = write(output, result)
    print(
        f"wrote {args.output} sha256 {digest}; decision {result['decision']['outcome']} {result['decision'].get('statuses')}"
    )
    return 0


def command_recompute(args: argparse.Namespace) -> int:
    from .recompute import recompute

    manifest = _load(args.freeze)
    confirmation = _load(args.confirmation)
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    result = recompute(confirmation, manifest, registry)
    if args.output:
        write(ROOT / args.output, result)
    print(f"independent outcome: {result['outcome']}; agrees with stored: {result['agrees']}")
    return 0 if result["agrees"] else 1


def command_qualify(args: argparse.Namespace) -> int:
    from .qualify import qualify

    output = ROOT / args.output
    if output.exists():
        print(f"error: {output} exists", file=sys.stderr)
        return 2
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    captured = environment("cuda:0")
    started = time.time()
    result = qualify(registry, log=lambda m: print(m, flush=True))
    result["environment"] = captured
    result["wall_seconds"] = time.time() - started
    print(f"wrote {args.output} sha256 {write(output, result)}")
    return 0


def command_capacity(args: argparse.Namespace) -> int:
    from .capacity import run_capacity
    from .freeze import arm_from_dict

    output = ROOT / args.output
    if output.exists():
        print(f"error: {output} exists", file=sys.stderr)
        return 2
    confirmation = _load(args.confirmation)
    manifest = _load(args.freeze)
    frozen = manifest["frozen"]
    promoted = confirmation["decision"]["outcome"] == "PROMOTE"
    name = frozen["challenger"]["name"] if promoted else "c1_champion1"
    arm = arm_from_dict(frozen["arms"][name]["spec"])
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    captured = environment("cpu")
    started = time.time()
    result = run_capacity(registry, arm, workers=args.workers, log=lambda m: print(m, flush=True))
    result["final_1k_system"] = name
    result["environment"] = captured
    result["wall_seconds"] = time.time() - started
    digest = write(output, result)
    path = ROOT / identities.REGISTRY_PATH
    for block in registry["blocks"]:
        if block["role"] == "capacity" and block["status"] == "reserved":
            registry = identities.mark(
                registry, block["block_id"], status="used", observed_by=f"capacity:{output.name}"
            )
    identities.write_registry(path, registry)
    print(f"wrote {args.output} sha256 {digest}; verdict {result['verdict']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m research.aaa_1k_v2",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit").set_defaults(func=command_audit)
    sub.add_parser("fingerprint").set_defaults(func=command_fingerprint)
    registry = sub.add_parser("registry")
    registry.add_argument("--sync", action="store_true")
    registry.set_defaults(func=command_registry)
    sub.add_parser("prove-fresh").set_defaults(func=command_prove_fresh)
    develop = sub.add_parser("develop")
    develop.add_argument("--output", required=True)
    develop.add_argument("--role", choices=("development", "scratch"), default="development")
    develop.add_argument("--workers", default="auto")
    develop.add_argument("--raw-root", help="directory for full raw archives (e.g. $AAA_DATA_ROOT)")
    develop.set_defaults(func=command_develop)
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--development", default="docs/evidence/aaa_1k_v2/development.json")
    diagnose.add_argument("--workers", default="auto")
    diagnose.set_defaults(func=command_diagnose)
    attack = sub.add_parser("attack")
    attack.add_argument("--output", required=True)
    attack.add_argument("--development", default="docs/evidence/aaa_1k_v2/development.json")
    attack.add_argument("--workers", default="auto")
    attack.add_argument("--cuda", default="cuda:0", help="device for criterion A6, or 'none'")
    attack.set_defaults(func=command_attack)
    freeze_cmd = sub.add_parser("freeze")
    freeze_cmd.add_argument("--output", required=True)
    freeze_cmd.add_argument("--development", default="docs/evidence/aaa_1k_v2/development.json")
    freeze_cmd.add_argument("--attack")
    freeze_cmd.set_defaults(func=command_freeze)
    confirm = sub.add_parser("confirm")
    confirm.add_argument("--freeze", required=True)
    confirm.add_argument("--output", required=True)
    confirm.add_argument("--workers", default="auto")
    confirm.add_argument("--data-root", help="Monash data root (default $AAA_DATA_ROOT)")
    confirm.add_argument(
        "--claim-remote", action="store_true", help="create an immutable remote claim ref first"
    )
    confirm.set_defaults(func=command_confirm)
    recompute_cmd = sub.add_parser("recompute")
    recompute_cmd.add_argument("--freeze", required=True)
    recompute_cmd.add_argument("--confirmation", required=True)
    recompute_cmd.add_argument("--output")
    recompute_cmd.set_defaults(func=command_recompute)
    qualify_cmd = sub.add_parser("qualify")
    qualify_cmd.add_argument("--output", required=True)
    qualify_cmd.set_defaults(func=command_qualify)
    capacity_cmd = sub.add_parser("capacity")
    capacity_cmd.add_argument("--output", required=True)
    capacity_cmd.add_argument("--freeze", required=True)
    capacity_cmd.add_argument("--confirmation", required=True)
    capacity_cmd.add_argument("--workers", default="auto")
    capacity_cmd.set_defaults(func=command_capacity)
    args = parser.parse_args(argv)
    if getattr(args, "cuda", None) == "none":
        args.cuda = None
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
