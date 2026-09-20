"""Command line for the AAA-1K phase.

    python -m research.aaa_1k fingerprint
    python -m research.aaa_1k parameter-audit
    python -m research.aaa_1k select      --output docs/evidence/aaa_1k_development_selection.json
    python -m research.aaa_1k evaluate    --output docs/evidence/aaa_1k_evaluation.json
    python -m research.aaa_1k report      --evidence ... --selection ... --output docs/aaa_1k_report.md
    python -m research.aaa_1k recompute   --evidence docs/evidence/aaa_1k_evaluation.json
    python -m research.aaa_1k visualize   --family occlusion_v1 --output runs/aaa_1k/dashboard.png
    python -m research.aaa_1k gradient-check --full

`recompute` is the independent one: it rebuilds every headline statistic from
the per-stream primitives in the evidence file, without rerunning a model, and
exits non-zero on a disagreement. A summary nobody can recompute is a summary
nobody has to believe.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .controls import (
    StatelessMLPControl,
    VanillaRNNControl,
    expected_mlp_parameter_count,
    expected_rnn_parameter_count,
)
from .experiments import PRIMARY, plan_replication, run_experiments
from .identity import git_provenance, lineage_record, phase_fingerprint
from .model import AAA1KGRU, expected_parameter_count
from .report import render_report
from .seeds import derive_seed
from .selection import Configuration, run_development_selection
from .stats import paired_difference

RECOMPUTE_TOLERANCE = 1e-9


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(payload, str):
        temporary.write_text(payload, encoding="utf-8")
    else:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_fingerprint(args: argparse.Namespace) -> int:
    fingerprint = phase_fingerprint(project_root())
    payload = {
        "fingerprint": fingerprint,
        "git": git_provenance(project_root()),
    }
    if args.output:
        _write(Path(args.output), payload)
    print(fingerprint["sha256"])
    print(f"{fingerprint['file_count']} files, phase {fingerprint['phase_version']}")
    return 0


def command_parameter_audit(args: argparse.Namespace) -> int:
    """Recompute every parameter count two independent ways and compare."""

    model = AAA1KGRU(seed=0)
    mlp = StatelessMLPControl(seed=0)
    rnn = VanillaRNNControl(seed=0)
    rows = [
        ("AAA1KGRU", model.parameter_count(), expected_parameter_count(), 994),
        ("StatelessMLPControl", mlp.parameter_count(), expected_mlp_parameter_count(), 982),
        ("VanillaRNNControl", rnn.parameter_count(), expected_rnn_parameter_count(), 954),
    ]
    payload: dict[str, Any] = {
        "schema": "aaa.1k.parameter_audit.v1",
        "models": [
            {
                "name": name,
                "counted_from_arrays": counted,
                "counted_from_formula": formula,
                "declared": declared,
                "agrees": counted == formula == declared,
            }
            for name, counted, formula, declared in rows
        ],
        "aaa1k_inventory": model.parameter_inventory(),
        "aaa1k_state_footprint": model.state_footprint(),
        "lineage": lineage_record(
            model_state_hash=model.state_hash(),
            parameter_count=model.parameter_count(),
            initialization_seed=model.seed,
            project_root=project_root(),
        ),
    }
    for entry in payload["models"]:
        print(
            f"{entry['name']:22s} arrays={entry['counted_from_arrays']:5d} "
            f"formula={entry['counted_from_formula']:5d} declared={entry['declared']:5d} "
            f"{'OK' if entry['agrees'] else 'MISMATCH'}"
        )
    if args.output:
        _write(Path(args.output), payload)
    return 0 if all(entry["agrees"] for entry in payload["models"]) else 1


def command_select(args: argparse.Namespace) -> int:
    selection = run_development_selection()
    selection["git"] = git_provenance(project_root())
    selection["scientific_fingerprint"] = phase_fingerprint(project_root())["sha256"]
    if args.output:
        _write(Path(args.output), selection)
    print(f"selected: {selection['selected_key']}")
    print(
        f"divergence boundary: {selection['stage_zero_divergence_probe']['lowest_diverging_learning_rate']}"
    )
    print(f"eligible learning rates: {selection['eligible_learning_rates']}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    chosen = selection["selected"]
    configuration = Configuration(
        float(chosen["learning_rate"]), int(chosen["tbptt_steps"]), float(chosen["error_loss_weight"])
    )
    plan = plan_replication(configuration)
    replicas = int(args.replicas) if args.replicas else plan["selected_replicas"]
    evidence = run_experiments(configuration, replicas=replicas)
    evidence["replication_plan"] = plan
    evidence["precision_objective_met"] = bool(plan["required_replicas"] <= replicas and not plan["bounded"])
    evidence["selection_key"] = selection["selected_key"]
    evidence["git"] = git_provenance(project_root())
    evidence["scientific_fingerprint"] = phase_fingerprint(project_root())["sha256"]
    if args.output:
        _write(Path(args.output), evidence)
    dims = evidence["capability_vector"]["dimensions"]
    print(f"streams: {len(evidence['streams'])}, replicas per family: {replicas}")
    print(f"precision objective met: {evidence['precision_objective_met']}")
    for key in ("q1_online_learning", "q3_hidden_state", "q4_gating"):
        record = (
            dims[key].get("overall") or dims[key].get("versus_stateless_mlp") or dims[key].get("all_families")
        )
        print(
            f"{key:26s} mean={record['mean_difference']:+.3e} ci=[{record['ci_low']:+.3e},{record['ci_high']:+.3e}]"
        )
    return 0


def command_report(args: argparse.Namespace) -> int:
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    identity = phase_fingerprint(project_root())
    text = render_report(evidence, selection, identity)
    _write(Path(args.output), text)
    print(f"wrote {args.output} ({len(text)} bytes)")
    return 0


def command_recompute(args: argparse.Namespace) -> int:
    """Rebuild every headline statistic from the retained per-stream primitives.

    Nothing is retrained and nothing is re-simulated. If a stored number cannot
    be reproduced from the primitives beside it, that is a finding and this
    command exits non-zero.
    """

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    per_stream = evidence["per_stream"]
    dims = evidence["capability_vector"]["dimensions"]
    failures: list[str] = []
    checks = 0

    def compare(label: str, stored: float, recomputed: float) -> None:
        nonlocal checks
        checks += 1
        if not (math.isfinite(stored) and math.isfinite(recomputed)):
            if math.isfinite(stored) != math.isfinite(recomputed):
                failures.append(f"{label}: stored={stored} recomputed={recomputed}")
            return
        if abs(stored - recomputed) > RECOMPUTE_TOLERANCE * max(1.0, abs(stored)):
            failures.append(f"{label}: stored={stored!r} recomputed={recomputed!r}")

    # Q1: first quarter minus last quarter, paired over every stream.
    q1 = paired_difference(
        [entry["last_quarter_mae"][PRIMARY] for entry in per_stream],
        [entry["first_quarter_mae"][PRIMARY] for entry in per_stream],
    )
    compare(
        "q1.mean_difference", dims["q1_online_learning"]["overall"]["mean_difference"], q1["mean_difference"]
    )
    compare("q1.ci_low", dims["q1_online_learning"]["overall"]["ci_low"], q1["ci_low"])
    compare("q1.ci_high", dims["q1_online_learning"]["overall"]["ci_high"], q1["ci_high"])

    # Q3 and Q4 on the declared memory families.
    memory = [entry for entry in per_stream if entry["family"] in ("occlusion_v1", "coarse_speed_v1")]
    for label, control, stored in (
        ("q3.state_reset", "aaa1k_state_reset", dims["q3_hidden_state"]["versus_state_reset"]),
        ("q3.stateless_mlp", "mlp_control", dims["q3_hidden_state"]["versus_stateless_mlp"]),
        ("q4.memory", "rnn_control", dims["q4_gating"]["memory_families"]),
    ):
        recomputed = paired_difference(
            [entry["mae"][PRIMARY] for entry in memory], [entry["mae"][control] for entry in memory]
        )
        compare(f"{label}.mean_difference", stored["mean_difference"], recomputed["mean_difference"])
        compare(f"{label}.ci_low", stored["ci_low"], recomputed["ci_low"])

    # Per-family means in the capability vector.
    for family, row in dims["prediction_accuracy"].items():
        entries = [entry for entry in per_stream if entry["family"] == family]
        for name, summary in row.items():
            compare(
                f"prediction_accuracy.{family}.{name}.mean",
                summary["mean"],
                float(np.mean([entry["mae"][name] for entry in entries])),
            )

    # Stability counters.
    compare(
        "numerical_stability.total_clip_events",
        dims["numerical_stability"]["total_clip_events"],
        float(sum(entry["clip_events"] for entry in per_stream)),
    )
    compare(
        "numerical_stability.total_nonfinite_events",
        dims["numerical_stability"]["total_nonfinite_events"],
        float(sum(entry["nonfinite_events"] for entry in per_stream)),
    )

    print(f"recomputed {checks} stored values from retained primitives")
    if failures:
        print(f"MISMATCH on {len(failures)} value(s):")
        for failure in failures[:20]:
            print(f"  {failure}")
        return 1
    print("all recomputed values agree")
    return 0


def command_visualize(args: argparse.Namespace) -> int:
    from .streams import build_stream
    from .visualize import record_trace, render

    options: dict[str, Any] = {}
    if args.family == "motion_compat":
        options = {"scenario": "changed", "steps": 240, "change_step": 120}
    elif args.family == "aba_v1":
        options = {"segment_steps": 150}
    stream = build_stream(args.family, derive_seed("benchmark_generation", args.index), **options)
    model = AAA1KGRU(
        seed=derive_seed("model_init", 0),
        learning_rate=args.learning_rate,
        tbptt_steps=args.tbptt_steps,
    )
    before = model.state_hash()
    trace = record_trace(stream, model, online=not args.frozen)
    after_record = model.state_hash()
    path = render(trace, args.output, title=f"AAA-1K on {args.family}")
    if model.state_hash() != after_record:
        print("rendering mutated model state", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    print(f"model state changed by running: {before != after_record} (expected True when online)")
    return 0


def command_gradient_check(args: argparse.Namespace) -> int:
    """Finite-difference every parameter of every model, on demand.

    The routine CI check samples parameters to stay cheap. This is the full
    version, and it is the one to run when the backward pass changes.
    """

    from .gradcheck import full_gradient_check

    report = full_gradient_check(exhaustive=args.full)
    for entry in report["models"]:
        print(
            f"{entry['model']:22s} checked={entry['checked']:5d} "
            f"max_rel={entry['max_relative_error']:.2e} "
            f"max_abs={entry['max_absolute_error']:.2e} "
            f"violations={entry['violations']} "
            f"{'OK' if entry['passed'] else 'FAIL'}"
        )
    if args.output:
        _write(Path(args.output), report)
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research.aaa_1k", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fingerprint = sub.add_parser("fingerprint", help="phase scientific source identity")
    fingerprint.add_argument("--output")
    fingerprint.set_defaults(handler=command_fingerprint)

    audit = sub.add_parser("parameter-audit", help="recount every parameter two ways")
    audit.add_argument("--output")
    audit.set_defaults(handler=command_parameter_audit)

    select = sub.add_parser("select", help="run the frozen development selection plan")
    select.add_argument("--output")
    select.set_defaults(handler=command_select)

    evaluate = sub.add_parser("evaluate", help="run the held-out evaluation")
    evaluate.add_argument("--selection", required=True)
    evaluate.add_argument("--replicas", type=int, default=None)
    evaluate.add_argument("--output")
    evaluate.set_defaults(handler=command_evaluate)

    report = sub.add_parser("report", help="render the research report")
    report.add_argument("--evidence", required=True)
    report.add_argument("--selection", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(handler=command_report)

    recompute = sub.add_parser("recompute", help="rebuild headline statistics from primitives")
    recompute.add_argument("--evidence", required=True)
    recompute.set_defaults(handler=command_recompute)

    visualize = sub.add_parser("visualize", help="render the diagnostic dashboard")
    visualize.add_argument("--family", default="occlusion_v1")
    visualize.add_argument("--index", type=int, default=0)
    visualize.add_argument("--learning-rate", type=float, default=0.03)
    visualize.add_argument("--tbptt-steps", type=int, default=4)
    visualize.add_argument("--frozen", action="store_true")
    visualize.add_argument("--output", default="runs/aaa_1k/dashboard.png")
    visualize.set_defaults(handler=command_visualize)

    gradient = sub.add_parser("gradient-check", help="finite-difference the backward pass")
    gradient.add_argument("--full", action="store_true", help="check every parameter, not a sample")
    gradient.add_argument("--output")
    gradient.set_defaults(handler=command_gradient_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    raise SystemExit(main())
