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
    started = time.time()
    raw_dir = Path(args.raw_root) / "aaa_1k_v2" if args.raw_root else None
    result = tournament.run_development(
        registry, workers=args.workers, role=args.role, raw_dir=raw_dir, log=lambda m: print(m, flush=True)
    )
    result["environment"] = environment(plan.DEVELOPMENT_BACKEND)
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
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
