"""Command-line entry points for AAA.

Exit-code contract
------------------
``benchmark --role development``
    Returns 0 even when gates fail. Development exploration is allowed to
    record a failure and keep going.
``benchmark --role confirmation_a|confirmation_b|high_replication``
    Returns non-zero when any required gate is not ``PASS`` — including
    ``FAIL``, ``NOT_VERIFIED`` and ``INSUFFICIENT_EVIDENCE``. Absence of
    evidence is never reported as success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark.gates import PASS
from .benchmark.manifest import build_manifest, save_manifest
from .benchmark.recompute import recompute_run
from .benchmark.runner import (
    ConfirmationError,
    checkpoint_digest,
    default_project_root,
    run_benchmark,
    train_replicas,
)
from .benchmark.seeds import (
    CONFIRMATION_ROLES,
    DEFAULT_GOLDEN_CASES,
    ROLES,
    ConfirmationBatchRegistry,
    golden_seed_fixture,
    lineage_for,
)
from .benchmark.spec import canonical_spec_path, load_spec, spec_hash
from .config import ExperimentConfig

REQUIRE_PASS_ROLES = tuple(role for role in ROLES if role != "development")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aaa", description="AAA — Accurate Autonomous Adaptation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text, default_label in (
        ("smoke", "Run a small CPU smoke experiment on the historical v1 track.", "smoke"),
        ("full", "Run the historical v1 development selection and evaluation.", "full"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--output-root", type=Path, default=Path("runs"))
        command_parser.add_argument("--label", default=default_label)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a benchmark v2.1 attempt.")
    benchmark_parser.add_argument("--role", choices=ROLES, default="development")
    benchmark_parser.add_argument(
        "--batch-id", help="Predeclared confirmation batch identity (confirmation roles)."
    )
    benchmark_parser.add_argument("--attempt-label", help="Label for a development attempt directory.")
    benchmark_parser.add_argument("--output-root", type=Path, default=Path("runs"))
    benchmark_parser.add_argument(
        "--replicas", type=int, help="Development override only; rejected below confirmation minimums."
    )
    benchmark_parser.add_argument(
        "--episodes", type=int, help="Development override only; rejected below confirmation minimums."
    )
    benchmark_parser.add_argument("--spec", type=Path, help="Development-only specification override.")
    benchmark_parser.add_argument(
        "--resume", action="store_true", help="Continue an interrupted attempt directory."
    )
    benchmark_parser.add_argument(
        "--reproduce", action="store_true", help="Re-run an already consumed confirmation batch."
    )

    recompute_parser = subparsers.add_parser(
        "recompute", help="Recompute metrics and gates from retained raw evidence."
    )
    recompute_parser.add_argument("run_dir", type=Path)
    recompute_parser.add_argument("--spec", type=Path)
    recompute_parser.add_argument("--no-verify-checksums", action="store_true")

    freeze_parser = subparsers.add_parser("freeze", help="Write the confirmation freeze manifest.")
    freeze_parser.add_argument(
        "--batch", action="append", default=[], help="Planned confirmation batch id (repeatable)."
    )
    freeze_parser.add_argument("--notes", default="")
    freeze_parser.add_argument("--project-root", type=Path)

    declare_parser = subparsers.add_parser("declare-batch", help="Predeclare a confirmation batch.")
    declare_parser.add_argument("batch_id")
    declare_parser.add_argument("--role", choices=CONFIRMATION_ROLES, required=True)
    declare_parser.add_argument("--notes", default="")
    declare_parser.add_argument("--project-root", type=Path)

    batches_parser = subparsers.add_parser("batches", help="List declared confirmation batches.")
    batches_parser.add_argument("--project-root", type=Path)

    subparsers.add_parser("spec-hash", help="Print the canonical specification path and hash.")

    golden_parser = subparsers.add_parser("write-golden-seeds", help="Regenerate the committed seed fixture.")
    golden_parser.add_argument("--project-root", type=Path)

    diagnosis_parser = subparsers.add_parser(
        "diagnose", help="Run the reproducible learner diagnosis and ablations."
    )
    diagnosis_parser.add_argument("--output", type=Path, default=Path("diagnosis"))
    diagnosis_parser.add_argument(
        "--quick", action="store_true", help="Reduced ablation grid for smoke checks."
    )

    selection_parser = subparsers.add_parser(
        "select-candidate", help="Run development-only candidate selection."
    )
    selection_parser.add_argument(
        "--output", type=Path, default=Path("docs/evidence/candidate_selection.json")
    )
    selection_parser.add_argument("--quick", action="store_true")

    animation_parser = subparsers.add_parser(
        "animate", help="Launch the optional interactive moving-dot animation."
    )
    animation_parser.add_argument("--checkpoint", type=Path, help="RLS or legacy linear JSON checkpoint.")
    animation_parser.add_argument(
        "--scenario", choices=("straight", "bouncing", "changed"), default="changed"
    )
    animation_parser.add_argument("--seed", type=int, default=201)
    animation_parser.add_argument(
        "--frozen", action="store_true", help="Do not update the model after scoring."
    )

    return parser


def _project(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "project_root", None) or default_project_root())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "spec-hash":
        spec = load_spec()
        print(f"path: {canonical_spec_path()}")
        print(f"version: {spec.spec_version}")
        print(f"status: {spec.status}")
        print(f"hash: {spec_hash(spec)}")
        return 0

    if args.command == "batches":
        spec = load_spec()
        registry = ConfirmationBatchRegistry.load(_project(args) / spec.confirmation.batch_registry)
        for batch in registry.batches():
            print(
                f"{batch.batch_id}\t{batch.role}\t{batch.status}\t{batch.outcome or '-'}\t{batch.spec_hash[:12]}"
            )
        return 0

    if args.command == "declare-batch":
        spec = load_spec()
        path = _project(args) / spec.confirmation.batch_registry
        registry = ConfirmationBatchRegistry.load(path)
        registry.declare(args.batch_id, args.role, spec_hash(spec), notes=args.notes)
        registry.save()
        print(f"declared {args.batch_id} for {args.role} against spec {spec_hash(spec)[:12]}")
        return 0

    if args.command == "write-golden-seeds":
        spec = load_spec()
        destination = _project(args) / "benchmarks" / "golden_seeds.json"
        payload = {
            "schema_version": "aaa.golden_seeds.v1",
            "root_seed": spec.randomness.root_seed,
            "seeds": golden_seed_fixture(spec.randomness.root_seed, DEFAULT_GOLDEN_CASES),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {destination}")
        return 0

    if args.command == "freeze":
        spec = load_spec()
        project = _project(args)
        if project.resolve() != default_project_root().resolve():
            raise SystemExit("freeze source root must contain the loaded AAA implementation")
        if not args.batch:
            raise SystemExit("freeze requires at least one --batch planned confirmation batch id")
        lineage = lineage_for("confirmation_a", spec.confirmation.ab_relationship)
        trained = train_replicas(
            spec, lineage=lineage, replicas=spec.confirmation.replicas, role="development"
        )
        manifest = build_manifest(
            spec,
            project_root=project,
            checkpoint_hashes=[checkpoint_digest(item.model.state_dict()) for item in trained],
            training_seeds={item.replica: list(item.seeds) for item in trained},
            planned_batches=args.batch,
            notes=args.notes,
        )
        destination = project / spec.confirmation.freeze_manifest
        save_manifest(manifest, destination)
        print(f"wrote {destination}")
        print(f"spec hash: {manifest.spec_hash}")
        print(f"checkpoint hashes: {[h[:12] for h in manifest.checkpoint_hashes]}")
        return 0

    if args.command == "recompute":
        result = recompute_run(args.run_dir, spec_path=args.spec, verify=not args.no_verify_checksums)
        stored = result.get("stored_gates") or {}
        recomputed = result["gates"]
        print(f"run: {result['run_id']}")
        print(f"spec hash: {result['spec_hash']}")
        print(f"checksums: {result['checksums']}")
        print(f"recomputed all_required_gates_pass: {recomputed['all_required_gates_pass']}")
        for label in ("result_comparison", "gate_comparison"):
            comparison = result[label]
            if not comparison["equivalent"]:
                print(f"{label.replace('_', ' ')} failed: {comparison}")
                return 3
        if stored:
            stored_statuses = {gate["name"]: gate["status"] for gate in stored.get("gates", [])}
            new_statuses = {gate["name"]: gate["status"] for gate in recomputed["gates"]}
            differing = {
                name: (stored_statuses.get(name), new_statuses.get(name))
                for name in sorted(set(stored_statuses) | set(new_statuses))
                if stored_statuses.get(name) != new_statuses.get(name)
            }
            if differing:
                print("gate status differences (stored -> recomputed):")
                for name, (before, after) in differing.items():
                    print(f"  {name}: {before} -> {after}")
                return 3
            print("every stored gate status was reproduced from raw evidence")
        return 0

    if args.command == "diagnose":
        from .diagnosis import run_diagnosis

        output = run_diagnosis(args.output, quick=args.quick)
        print(f"AAA diagnosis complete: {output}")
        return 0

    if args.command == "select-candidate":
        from .selection import run_candidate_selection

        output = run_candidate_selection(args.output, quick=args.quick)
        print(f"AAA candidate selection complete: {output}")
        return 0

    if args.command == "animate":
        from .animation import launch_animation

        launch_animation(
            checkpoint=args.checkpoint, scenario=args.scenario, seed=args.seed, online=not args.frozen
        )
        return 0

    if args.command == "benchmark":
        try:
            outcome = run_benchmark(
                role=args.role,
                batch_id=args.batch_id,
                output_root=args.output_root,
                attempt_label=args.attempt_label,
                replicas=args.replicas,
                episodes=args.episodes,
                spec_path=args.spec,
                resume=args.resume,
                reproduce=args.reproduce,
            )
        except ConfirmationError as error:
            print(f"confirmation refused: {error}", file=sys.stderr)
            return 2
        print(f"AAA benchmark complete: {outcome.directory}")
        print(f"Summary: {outcome.directory / 'summary.json'}")
        print(f"Report: {outcome.directory / 'report.md'}")
        for gate in outcome.summary["gates"]["gates"]:
            marker = " " if gate["status"] == PASS else "!"
            print(f" {marker} {gate['status']:<22} {gate['name']}")
        if outcome.passed:
            print("All required gates PASS.")
        else:
            print(f"Required gates not satisfied: {', '.join(outcome.unmet)}", file=sys.stderr)
        if args.role in REQUIRE_PASS_ROLES and not outcome.passed:
            return 1
        return 0

    from .evaluation import run_full_evaluation

    config = ExperimentConfig().quick() if args.command == "smoke" else ExperimentConfig()
    run_dir = run_full_evaluation(config, output_root=args.output_root, label=args.label)
    print(f"AAA run complete: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"Report: {run_dir / 'experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
