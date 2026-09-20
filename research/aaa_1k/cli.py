"""Command line for the AAA-1K phase.

    python -m research.aaa_1k fingerprint
    python -m research.aaa_1k parameter-audit
    python -m research.aaa_1k select      --output docs/evidence/aaa_1k_development_selection.json
    python -m research.aaa_1k evaluate    --output docs/evidence/aaa_1k_evaluation_round1_superseded.json
    python -m research.aaa_1k report      --evidence ... --selection ... --output docs/aaa_1k_report.md
    python -m research.aaa_1k characterize --selection ... --output docs/evidence/aaa_1k_characterization.json
    python -m research.aaa_1k round3     --selection ... --output docs/evidence/aaa_1k_evaluation_round3.json
    python -m research.aaa_1k report2    --evidence ... --selection ... --output docs/aaa_1k_report.md
    python -m research.aaa_1k recompute   --evidence docs/evidence/aaa_1k_evaluation_round3.json
    python -m research.aaa_1k adversarial-probes --selection ... --output docs/evidence/aaa_1k_adversarial_probes.json
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
    configuration = _configuration(args.selection)
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


def _recompute_round2(evidence: dict[str, Any]) -> int:
    """Rebuild round 2's headline statistics from its retained cells."""

    from .round2 import MEMORY_FAMILIES, _crossed, _in_families, _matrix, _select
    from .stats import crossed_paired_difference

    cells = evidence["cells"]
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

    q1 = crossed_paired_difference(
        _matrix(cells, PRIMARY, "last_quarter_mae"), _matrix(cells, PRIMARY, "first_quarter_mae")
    )
    for field in ("mean_difference", "ci_low", "ci_high"):
        compare(f"q1.{field}", dims["q1_online_learning"]["overall"][field], q1[field])

    memory = _in_families(cells, MEMORY_FAMILIES)
    for label, control, stored in (
        ("q3.state_reset", "aaa1k_state_reset", dims["q3_hidden_state"]["versus_state_reset"]),
        ("q3.stateless", "mlp_control", dims["q3_hidden_state"]["versus_stateless_mlp"]),
        ("q4.all", "rnn_control", dims["q4_gating"]["all_families"]),
    ):
        source = cells if label == "q4.all" else memory
        recomputed = _crossed(source, PRIMARY, control)
        for field in ("mean_difference", "ci_low", "ci_high"):
            compare(f"{label}.{field}", stored[field], recomputed[field])

    # The two new measurements are recomputed from their own retained trials.
    adaptation = dims["q2_adaptation"]
    trials = adaptation["trials"]
    compare(
        "q2.adaptation_mean",
        adaptation["adaptation_effect"]["mean_difference"],
        float(np.mean([trial["adaptation_effect"] for trial in trials])),
    )
    for trial in trials:
        compare(
            f"q2.trial[{trial['seed']}].adaptation",
            trial["adaptation_effect"],
            trial["changed"]["absolute_advantage"] - trial["control"]["absolute_advantage"],
        )
    retention = dims["q5_retention"]
    compare(
        "q5.forgetting_mean",
        retention["forgetting"]["mean_difference"],
        float(np.mean([trial["forgetting"] for trial in retention["trials"]])),
    )
    for trial in retention["trials"]:
        compare(
            f"q5.trial[{trial['seed']}].forgetting",
            trial["forgetting"],
            trial["probe_error"]["after_B"] - trial["probe_error"]["after_A1"],
        )

    for family, row in dims["prediction_accuracy"].items():
        entries = _select(cells, family=family)
        for name, summary in row.items():
            compare(
                f"prediction_accuracy.{family}.{name}.mean",
                summary["mean"],
                float(np.mean([cell["mae"][name] for cell in entries])),
            )

    compare(
        "numerical_stability.total_clip_events",
        dims["numerical_stability"]["total_clip_events"],
        float(sum(cell["clip_events"] for cell in cells)),
    )

    print(f"recomputed {checks} stored values from retained primitives (round 2)")
    if failures:
        print(f"MISMATCH on {len(failures)} value(s):")
        for failure in failures[:20]:
            print(f"  {failure}")
        return 1
    print("all recomputed values agree")
    return 0


def _recompute_round3(evidence: dict[str, Any]) -> int:
    """Rebuild round 3 from crossed retained cells and trials."""

    from .round2 import MEMORY_FAMILIES, _crossed, _in_families
    from .round3 import FROZEN_PRIMARY, _matrix, _trial_matrix
    from .stats import crossed_paired_difference

    cells = evidence["cells"]
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

    q1 = crossed_paired_difference(
        _matrix(cells, PRIMARY), _matrix(cells, FROZEN_PRIMARY), bootstrap_index=11
    )
    for field in ("mean_difference", "ci_low", "ci_high"):
        compare(f"q1.{field}", dims["q1_online_learning"]["overall"][field], q1[field])
    memory = _in_families(cells, MEMORY_FAMILIES)
    for label, source, control, stored in (
        ("q3.state_reset", memory, "aaa1k_state_reset", dims["q3_hidden_state"]["versus_state_reset"]),
        ("q3.stateless", memory, "mlp_control", dims["q3_hidden_state"]["versus_stateless_mlp"]),
        ("q4.all", cells, "rnn_control", dims["q4_gating"]["all_families"]),
    ):
        recomputed = _crossed(source, PRIMARY, control)
        for field in ("mean_difference", "ci_low", "ci_high"):
            compare(f"{label}.{field}", stored[field], recomputed[field])
    for key, dimension, field, bootstrap_index in (
        ("adaptation_effect", "q2_adaptation", "adaptation_effect", 13),
        ("continued_learning_effect", "q2_adaptation", "continued_learning_effect", 14),
        ("forgetting", "q5_retention", "forgetting", 15),
    ):
        trials = dims[dimension]["trials"]
        matrix = _trial_matrix(trials, key)
        recomputed = crossed_paired_difference(np.zeros_like(matrix), matrix, bootstrap_index=bootstrap_index)
        stored = dims[dimension][field]
        for statistic in ("mean_difference", "ci_low", "ci_high"):
            compare(f"{dimension}.{statistic}", stored[statistic], recomputed[statistic])
    print(f"recomputed {checks} stored values from retained primitives (round 3)")
    if failures:
        print(f"MISMATCH on {len(failures)} value(s):")
        for failure in failures[:20]:
            print(f"  {failure}")
        return 1
    print("all recomputed values agree")
    return 0


def command_report2(args: argparse.Namespace) -> int:
    from .report import render_round2_report

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    characterization = (
        json.loads(Path(args.characterization).read_text(encoding="utf-8")) if args.characterization else None
    )
    text = render_round2_report(evidence, selection, phase_fingerprint(project_root()), characterization)
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
    if evidence.get("schema") == "aaa.1k.experiments.v3":
        return _recompute_round3(evidence)
    if evidence.get("schema") == "aaa.1k.experiments.v2":
        return _recompute_round2(evidence)
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


def _configuration(path: str) -> Configuration:
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    chosen = selection["selected"]
    return Configuration(
        float(chosen["learning_rate"]),
        int(chosen["tbptt_steps"]),
        float(chosen["error_loss_weight"]),
        chosen.get("gradient_clip", 1.0),
    )


def command_characterize(args: argparse.Namespace) -> int:
    """Development-only probes that settle the open interpretation questions."""

    from .characterization import run_characterization

    report = run_characterization(_configuration(args.selection))
    for probe in report["probes"]:
        print(f"{probe['probe']}: {probe['finding']}")
    if args.output:
        _write(Path(args.output), report)
    return 0


def command_round2(args: argparse.Namespace) -> int:
    """Run the corrected evaluation round on fresh stream identities."""

    from .round2 import run_round2

    configuration = _configuration(args.selection)
    fairness = None
    if args.characterization:
        report = json.loads(Path(args.characterization).read_text(encoding="utf-8"))
        fairness = next((p for p in report["probes"] if p["probe"] == "per_architecture_selection"), None)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    evidence = run_round2(
        configuration,
        architecture_configurations=selection.get("architecture_configurations"),
        replicas=args.replicas,
        initializations=args.initializations,
        adaptation_trials=args.adaptation_trials,
        retention_trials=args.retention_trials,
        fairness=fairness,
    )
    evidence["git"] = git_provenance(project_root())
    evidence["scientific_fingerprint"] = phase_fingerprint(project_root())["sha256"]
    if args.output:
        _write(Path(args.output), evidence)
    dims = evidence["capability_vector"]["dimensions"]
    print(
        f"cells: {len(evidence['cells'])} "
        f"({args.initializations} initializations x {len(evidence['streams'])} streams)"
    )
    for key, field in (
        ("q1_online_learning", "overall"),
        ("q2_adaptation", "adaptation_effect"),
        ("q3_hidden_state", "versus_stateless_mlp"),
        ("q4_gating", "all_families"),
        ("q5_retention", "forgetting"),
    ):
        record = dims[key][field]
        print(
            f"{key:26s} mean={record['mean_difference']:+.3e} "
            f"ci=[{record['ci_low']:+.3e},{record['ci_high']:+.3e}]"
        )
    return 0


def command_round3(args: argparse.Namespace) -> int:
    """Run the reviewer-corrected evaluation on fresh, fully crossed identities."""

    from .round3 import run_round3

    configuration = _configuration(args.selection)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    fairness = None
    if args.characterization:
        report = json.loads(Path(args.characterization).read_text(encoding="utf-8"))
        fairness = next((p for p in report["probes"] if p["probe"] == "per_architecture_selection"), None)
    evidence = run_round3(
        configuration,
        architecture_configurations=selection.get("architecture_configurations"),
        replicas=args.replicas,
        initializations=args.initializations,
        adaptation_environments=args.adaptation_environments,
        retention_environments=args.retention_environments,
        fairness=fairness,
    )
    evidence["git"] = git_provenance(project_root())
    evidence["scientific_fingerprint"] = phase_fingerprint(project_root())["sha256"]
    if args.output:
        _write(Path(args.output), evidence)
    dims = evidence["capability_vector"]["dimensions"]
    print(
        f"cells: {len(evidence['cells'])}; adaptation trials: {len(dims['q2_adaptation']['trials'])}; "
        f"retention trials: {len(dims['q5_retention']['trials'])}"
    )
    for key, field in (
        ("q1_online_learning", "overall"),
        ("q2_adaptation", "adaptation_effect"),
        ("q3_hidden_state", "versus_stateless_mlp"),
        ("q4_gating", "all_families"),
        ("q5_retention", "forgetting"),
    ):
        record = dims[key][field]
        print(
            f"{key:26s} mean={record['mean_difference']:+.3e} "
            f"ci=[{record['ci_low']:+.3e},{record['ci_high']:+.3e}]"
        )
    return 0


def command_probes(args: argparse.Namespace) -> int:
    """Run the adversarial probes that try to break this phase's conclusions."""

    from .adversarial import run_probes

    report = run_probes(_configuration(args.selection))
    for probe in report["probes"]:
        print(f"{probe['probe']}: {probe['finding']}")
    if args.output:
        _write(Path(args.output), report)
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

    report2 = sub.add_parser("report2", help="render the round-2 research report")
    report2.add_argument("--evidence", required=True)
    report2.add_argument("--selection", required=True)
    report2.add_argument("--characterization")
    report2.add_argument("--output", required=True)
    report2.set_defaults(handler=command_report2)

    report3 = sub.add_parser("report3", help="render the round-3 research report")
    report3.add_argument("--evidence", required=True)
    report3.add_argument("--selection", required=True)
    report3.add_argument("--characterization")
    report3.add_argument("--output", required=True)
    report3.set_defaults(handler=command_report2)

    recompute = sub.add_parser("recompute", help="rebuild headline statistics from primitives")
    recompute.add_argument("--evidence", required=True)
    recompute.set_defaults(handler=command_recompute)

    probes = sub.add_parser("adversarial-probes", help="try to break the phase's own conclusions")
    probes.add_argument("--selection", required=True)
    probes.add_argument("--output")
    probes.set_defaults(handler=command_probes)

    characterize = sub.add_parser(
        "characterize", help="development probes for the open interpretation questions"
    )
    characterize.add_argument("--selection", required=True)
    characterize.add_argument("--output")
    characterize.set_defaults(handler=command_characterize)

    round2 = sub.add_parser("round2", help="run the corrected evaluation round")
    round2.add_argument("--selection", required=True)
    round2.add_argument("--characterization")
    round2.add_argument("--replicas", type=int, default=24)
    round2.add_argument("--initializations", type=int, default=5)
    round2.add_argument("--adaptation-trials", type=int, default=24)
    round2.add_argument("--retention-trials", type=int, default=12)
    round2.add_argument("--output")
    round2.set_defaults(handler=command_round2)

    round3 = sub.add_parser("round3", help="run the reviewer-corrected evaluation round")
    round3.add_argument("--selection", required=True)
    round3.add_argument("--characterization")
    round3.add_argument("--replicas", type=int, default=24)
    round3.add_argument("--initializations", type=int, default=5)
    round3.add_argument("--adaptation-environments", type=int, default=24)
    round3.add_argument("--retention-environments", type=int, default=12)
    round3.add_argument("--output")
    round3.set_defaults(handler=command_round3)

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
