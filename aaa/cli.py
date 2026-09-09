"""Command-line entry points for AAA."""

from __future__ import annotations

import argparse
from pathlib import Path

from .animation import launch_animation
from .benchmark import run_benchmark_v2
from .diagnosis import run_diagnosis
from .config import ExperimentConfig
from .evaluation import run_full_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AAA — Accurate Autonomous Adaptation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text, default_label in (
        ("smoke", "Run a small CPU smoke experiment.", "smoke"),
        ("full", "Run development selection and the full reproducible evaluation.", "full"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument(
            "--output-root",
            type=Path,
            default=Path("runs"),
            help="Parent directory for unique run directories (default: runs).",
        )
        command_parser.add_argument("--label", default=default_label, help="Label included in the run directory name.")

    benchmark_parser = subparsers.add_parser("benchmark-v2", help="Run the predeclared benchmark v2 attempt.")
    benchmark_parser.add_argument("--output-root", type=Path, default=Path("runs"))
    benchmark_parser.add_argument("--role", choices=("development", "confirmation_a", "confirmation_b"), default="confirmation_a")
    benchmark_parser.add_argument("--attempt-id", help="Immutable attempt identifier; never reuse an existing ID.")
    benchmark_parser.add_argument("--replicas", type=int, help="Override only for development/smoke work.")
    benchmark_parser.add_argument("--episodes", type=int, help="Override only for development/smoke work.")
    benchmark_parser.add_argument("--spec", type=Path, help="Benchmark spec JSON; defaults to benchmarks/benchmark_v2.json.")

    diagnosis_parser = subparsers.add_parser("diagnose", help="Run the reproducible legacy-learner diagnosis.")
    diagnosis_parser.add_argument("--output", type=Path, default=Path("diagnosis"))

    animation_parser = subparsers.add_parser("animate", help="Launch the optional interactive moving-dot animation.")
    animation_parser.add_argument("--checkpoint", type=Path, help="Optional JSON checkpoint from a completed run.")
    animation_parser.add_argument(
        "--scenario", choices=("straight", "bouncing", "changed"), default="changed"
    )
    animation_parser.add_argument("--seed", type=int, default=201)
    animation_parser.add_argument("--frozen", action="store_true", help="Do not update the model after scoring.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "diagnose":
        output = run_diagnosis(args.output)
        print(f"AAA diagnosis complete: {output}")
        return 0
    if args.command == "benchmark-v2":
        if args.role.startswith("confirmation") and ((args.replicas is not None and args.replicas < 5) or (args.episodes is not None and args.episodes < 100)):
            raise SystemExit("confirmation attempts require at least 5 replicas and 100 episodes")
        run_dir = run_benchmark_v2(
            output_root=args.output_root,
            role=args.role,
            attempt_id=args.attempt_id,
            replicas=args.replicas,
            episodes=args.episodes,
            spec_path=args.spec,
        )
        print(f"AAA benchmark v2 complete: {run_dir}")
        print(f"Summary: {run_dir / 'summary.json'}")
        print(f"Report: {run_dir / 'report.md'}")
        return 0
    if args.command == "animate":
        launch_animation(checkpoint=args.checkpoint, scenario=args.scenario, seed=args.seed, online=not args.frozen)
        return 0

    config = ExperimentConfig().quick() if args.command == "smoke" else ExperimentConfig()
    run_dir = run_full_evaluation(config, output_root=args.output_root, label=args.label)
    print(f"AAA run complete: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"Report: {run_dir / 'experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
