"""Command-line entry points for AAA."""

from __future__ import annotations

import argparse
from pathlib import Path

from .animation import launch_animation
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

    animation_parser = subparsers.add_parser("animate", help="Launch the optional interactive moving-dot animation.")
    animation_parser.add_argument("--checkpoint", type=Path, help="Optional JSON checkpoint from a completed run.")
    animation_parser.add_argument(
        "--scenario", choices=("straight", "bouncing", "changed"), default="changed"
    )
    animation_parser.add_argument("--seed", type=int, default=201)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "animate":
        launch_animation(checkpoint=args.checkpoint, scenario=args.scenario, seed=args.seed)
        return 0

    config = ExperimentConfig().quick() if args.command == "smoke" else ExperimentConfig()
    run_dir = run_full_evaluation(config, output_root=args.output_root, label=args.label)
    print(f"AAA run complete: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"Report: {run_dir / 'experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
