"""Human-readable benchmark reports generated from stored measurements only."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

SIGNIFICANT_DIGITS = 4


def fmt(value: Any, *, digits: int = SIGNIFICANT_DIGITS) -> str:
    """Human-scale precision. Machine artifacts keep full precision."""

    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e5:
            return f"{value:.{digits - 1}e}"
        return f"{value:.{digits}g}"
    return str(value)


def _interval(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return "n/a"
    if entry.get("status") != "OK":
        return f"**{entry.get('status')}** ({entry.get('reason')})"
    return f"{fmt(entry.get('estimate'))} [{fmt(entry.get('lower'))}, {fmt(entry.get('upper'))}]"


def write_report(path: str | Path, summary: Mapping[str, Any]) -> None:
    destination = Path(path)
    gates = summary["gates"]
    lines: list[str] = [
        f"# AAA benchmark {summary['spec_version']} — `{summary['run_id']}`",
        "",
        f"Role: `{summary['role']}`. Confirmation batch: `{summary.get('confirmation_batch') or 'n/a'}`.",
        f"Specification hash: `{summary['spec_hash'][:16]}…`.",
        "",
        "This is an engineering measurement report. It is not an approval decision, and it does not",
        "claim general intelligence, physical understanding, or independent goal formation.",
        "",
        "## Required-gate status",
        "",
        f"**All required gates pass: {fmt(gates['all_required_gates_pass'])}**",
        "",
    ]
    if gates["unmet_required_gates"]:
        lines.extend([f"Unmet required gates: `{', '.join(gates['unmet_required_gates'])}`.", ""])
    lines.extend(["| Gate | Required | Status | Observed | Description |", "|---|---|---|---:|---|"])
    for gate in gates["gates"]:
        description = str(gate["description"]).split(".")[0]
        lines.append(
            f"| `{gate['name']}` | {'yes' if gate['required'] else 'no'} | **{gate['status']}** | "
            f"{fmt(gate['observed'])} | {description}. |"
        )
    lines.append("")

    lines.extend(
        [
            "## Uncertainty intervals",
            "",
            "| Comparison | Estimate [95% interval] | p | replicas | episodes |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for key, entry in gates.get("intervals", {}).items():
        lines.append(
            f"| `{key}` | {_interval(entry)} | {fmt(entry.get('p_value'))} | "
            f"{fmt(entry.get('replicas'))} | {fmt(entry.get('episodes'))} |"
        )
    lines.append("")

    multiplicity = gates.get("multiplicity")
    if multiplicity:
        lines.extend(
            [
                "### Multiple-comparison treatment",
                "",
                f"Method `{multiplicity['method']}`, family size {multiplicity['family_size']} "
                f"(primary comparisons plus previously spent confirmation attempts), alpha {multiplicity['alpha']}.",
                "",
                "| Comparison | p | adjusted p | rejected |",
                "|---|---:|---:|---|",
            ]
        )
        for entry in multiplicity["entries"]:
            lines.append(
                f"| `{entry['name']}` | {fmt(entry['p_value'])} | {fmt(entry['adjusted_p_value'])} | "
                f"{fmt(entry['reject'])} |"
            )
        lines.append("")

    lines.extend(
        ["## Learning progress", "", "| Cumulative training episodes | Frozen probe MAE |", "|---:|---:|"]
    )
    curve = summary.get("learning_curve", {}).get("curve", {})
    for budget in summary.get("learning_curve", {}).get("budgets", []):
        lines.append(f"| {budget} | {fmt(curve[str(budget)]['mean'])} |")
    lines.extend(
        [
            "",
            "Every row is the same fixed development probe bank scored by a frozen checkpoint, so a changing",
            "episode difficulty cannot masquerade as learning.",
            "",
        ]
    )

    lines.extend(["## Family measurements", ""])
    for family, result in summary["results"].items():
        lines.extend(
            [
                f"### `{family}`",
                "",
                f"Episodes: {fmt(result['episodes'])}; replicas: {fmt(result['replicas'])}; "
                f"bounce events: {fmt(result['bounce_events'])}; "
                f"event episodes: {fmt(result['event_episodes'])}; "
                f"no-event episodes excluded from event statistics: {fmt(result['no_event_episodes'])}.",
                "",
                "| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, predictor in result["predictors"].items():
            lines.append(
                f"| `{name}` | {fmt(predictor['episode_balanced_mae'])} | {fmt(predictor['median'])} | "
                f"{fmt(predictor['p95'])} | {fmt(predictor['p99'])} | "
                f"{fmt(predictor['event_episode_balanced_mae'])} | {fmt(predictor['event_p95'])} | "
                f"{fmt(predictor['post_change_window_mae'])} | {fmt(predictor['worst_replica_mae'])} | "
                f"{fmt(predictor['signed_bias'])} |"
            )
        lines.append("")

    recovery_gate = next((gate for gate in gates["gates"] if gate["name"] == "recovery"), None)
    if recovery_gate:
        details = recovery_gate["details"]
        lines.extend(
            [
                "## Recovery",
                "",
                "| Status | Episodes |",
                "|---|---:|",
            ]
        )
        for status, count in (details.get("status_counts") or {}).items():
            lines.append(f"| `{status}` | {fmt(count)} |")
        times = details.get("recovery_time_steps") or {}
        lines.extend(
            [
                "",
                f"Eligible: {fmt(details.get('eligible'))}; recovered: {fmt(details.get('recovered'))}; "
                f"unrecovered within horizon: {fmt(details.get('unrecovered'))}.",
                f"Recovery time (transitions): median {fmt(times.get('median'))}, "
                f"p90 {fmt(times.get('p90'))}, p95 {fmt(times.get('p95'))}, max {fmt(times.get('max'))}.",
                "",
                "| Replica | Recovery rate |",
                "|---|---:|",
            ]
        )
        for replica, rate in (details.get("per_replica_recovery_rate") or {}).items():
            lines.append(f"| {replica} | {fmt(rate)} |")
        lines.append("")

    bounce_gate = next((gate for gate in gates["gates"] if gate["name"] == "bounce_event_accuracy"), None)
    if bounce_gate and bounce_gate["details"].get("advantage_decomposition"):
        lines.extend(
            [
                "## Where any bounce advantage comes from",
                "",
                "| Comparison | Relative improvement [95% interval] |",
                "|---|---|",
            ]
        )
        for label, entry in bounce_gate["details"]["advantage_decomposition"].items():
            lines.append(f"| `{label}` | {_interval(entry)} |")
        lines.extend(
            [
                "",
                "Reflection is programmed public knowledge of the observation format. The reflected",
                "constant-motion baseline uses the identical policy, so the difference between the two",
                "comparisons above is the part of any apparent advantage that is the boundary transform",
                "rather than a learned parameter.",
                "",
            ]
        )

    latency = summary.get("latency", {})
    lines.extend(
        [
            "## Latency (selected candidate, CPU)",
            "",
            "| Operation | p50 (ms) | p95 (ms) | p99 (ms) |",
            "|---|---:|---:|---:|",
            f"| predict | {fmt(latency.get('predict_p50_ms'))} | {fmt(latency.get('predict_p95_ms'))} | {fmt(latency.get('predict_p99_ms'))} |",
            f"| update | {fmt(latency.get('update_p50_ms'))} | {fmt(latency.get('update_p95_ms'))} | {fmt(latency.get('update_p99_ms'))} |",
            f"| predict+update | {fmt(latency.get('predict_plus_update_p50_ms'))} | {fmt(latency.get('predict_plus_update_p95_ms'))} | {fmt(latency.get('predict_plus_update_p99_ms'))} |",
            "",
            f"Samples: {fmt(latency.get('samples'))}; failures: {fmt(latency.get('failures'))}; "
            f"candidate forgetting factor: {fmt(latency.get('candidate_forgetting'))}.",
            "",
        ]
    )

    correctness = summary.get("correctness", {})
    reproducibility = summary.get("reproducibility", {})
    lines.extend(["## Verification", "", "| Check | Result |", "|---|---|"])
    for name, value in sorted((correctness.get("checks") or {}).items()):
        lines.append(f"| correctness: `{name}` | {fmt(value)} |")
    for name, value in sorted((reproducibility.get("checks") or {}).items()):
        lines.append(f"| reproducibility: `{name}` | {fmt(value)} |")
    lines.append("")

    metadata = summary.get("metadata", {})
    checkpoints = summary.get("checkpoints", {})
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Source commit: `{(metadata.get('git') or {}).get('commit')}`; dirty: "
            f"{fmt((metadata.get('git') or {}).get('dirty'))}",
            f"- Tree hash: `{(metadata.get('git') or {}).get('tree_hash')}`",
            f"- Dependency lock hash: `{(metadata.get('dependency_lock') or {}).get('hash')}`",
            f"- Checkpoint relationship: `{checkpoints.get('relationship')}`",
            f"- Checkpoint hashes: `{', '.join(h[:12] for h in checkpoints.get('hashes', []))}`",
            f"- Runtime: {fmt(summary.get('runtime_seconds'))} s",
            "",
            "## Reproduction",
            "",
            "```bash",
            f"python -m aaa.cli benchmark --role {summary['role']}"
            + (f" --batch-id {summary['confirmation_batch']}" if summary.get("confirmation_batch") else "")
            + " --output-root runs",
            "```",
            "",
            "Recompute every metric and gate from the retained raw evidence, without retraining or",
            "re-simulating:",
            "",
            "```bash",
            f"python -m aaa.cli recompute runs/benchmark-v2_1/{summary['run_id']}",
            "```",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
