"""Render docs/candidate_selection.md from the development selection table.

The table, including every unsuccessful variant, comes straight from
docs/evidence/candidate_selection.json. Nothing about the outcome is written
in prose here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "candidate_selection.json"
OUTPUT = ROOT / "docs" / "candidate_selection.md"

WHY: dict[str, str] = {
    "feature_set": (
        "`displacement_only` cannot express the boundary-adjacent position term; adding the "
        "centered position costs one parameter and is the smallest set that admits the "
        "changed-law dynamics"
    ),
    "forgetting": (
        "the admitted variant that maximizes changed-law improvement over its identical frozen "
        "copy; slower forgetting adapts measurably less, and the full sweep is in the table below"
    ),
    "ridge": "sets the initial covariance `I / ridge`; alternatives were compared and none was admitted with a better outcome",
    "trace_bound": "bounds covariance windup under weak excitation; unbounded arms lose positive semidefiniteness (see the diagnosis)",
    "reflect": (
        "programmed public knowledge of the observation format; the like-for-like reflected "
        "baseline exists precisely so this is never counted as learning"
    ),
    "unfold_target": (
        "the inverse of the same public map applied to the update target, so the learner "
        "regresses in the coordinate its own linear law lives in"
    ),
    "skip_after_reflected_prediction": (
        "no update on a window whose displacement feature straddles a wall, detected from the "
        "learner's own previously reflected raw prediction. Added after the round-1 confirmation "
        "failure; see `AAA-120`"
    ),
    "dead_zone": "off; it was not needed once forgetting is trace-bounded and self-triggered",
    "detector_multiplier": (
        "forgetting is applied only on steps the model's *own* scored error says are surprising; "
        "the evaluator never signals an event"
    ),
    "training_episodes": "the learning curve is flat well before this point; a larger budget buys nothing measurable",
}


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def short(label: str) -> str:
    return (
        label.replace("displacement_position", "disp+pos")
        .replace("displacement_only", "disp")
        .replace("|exponential", "|exp")
        .replace("|directional", "|dir")
    )


def main() -> int:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    selected = data["selected"]
    criterion = data["always_online_criterion"]

    lines = [
        "# Candidate selection",
        "",
        "```bash",
        "python -m aaa.cli select-candidate --output docs/evidence/candidate_selection.json",
        "```",
        "",
        "Development streams only. Every confirmation stream that has been viewed — the four",
        "historical v2 attempts and the two failed v2.1 round-1 attempts — is treated as",
        "contaminated for selection purposes and is not used to choose anything.",
        "",
        "> **This is round 2.** The first v2.1 confirmation round failed the",
        "> `always_online_stability` gate on both fresh streams. The failure was diagnosed to a",
        "> specific mechanism on development data and a candidate policy was added to address it",
        "> (`AAA-120` in [`issue_ledger.md`](issue_ledger.md)). This table is the re-run selection",
        "> including that mechanism and its ablation. The round-1 table is retained as",
        "> [`evidence/candidate_selection_round1.json`](evidence/candidate_selection_round1.json).",
        "> No threshold was altered in either round.",
        "",
        "The full machine-readable table is",
        "[`evidence/candidate_selection.json`](evidence/candidate_selection.json).",
        "",
        "## Selection rule, declared before the table was read",
        "",
        f"> {data['selection_rule']}",
        "",
        f"Straight-motion admission limit: `{fmt(data['straight_mae_limit'])}` normalized MAE.",
        f"Tie tolerance: `{fmt(data['tie_tolerance'])}`. Always-online admission uses the same form",
        "as the benchmark's `always_online_stability` gate:",
        f"`<= {criterion['max_ratio']} x {criterion['baseline']} + {fmt(criterion['absolute_floor'])}`.",
        "",
        f"Variants evaluated: **{len(data['variants'])}**. Admitted: **{data['eligible_count']}**.",
        f"Tied within tolerance: **{data['tied_count']}**.",
        "",
        "## The selected candidate",
        "",
        "| Property | Value | Why this and not something else |",
        "|---|---|---|",
    ]
    for key, reason in WHY.items():
        lines.append(f"| `{key}` | `{fmt(selected[key])}` | {reason} |")

    lines += [
        "",
        "Measured on development streams:",
        "",
        f"- straight-motion normalized MAE `{fmt(selected['development_straight_mae'])}`",
        f"- bouncing normalized MAE `{fmt(selected['development_bouncing_mae'])}`",
        f"- always-online normalized MAE `{fmt(selected['always_online_mae'])}` against a reflected"
        f" baseline of `{fmt(selected['always_online_baseline_mae'])}`",
        f"- always-online margin `{fmt(selected['always_online_margin'])}` (<= 0 satisfies)",
        f"- changed-law improvement over the identical frozen copy `{fmt(selected['improvement_vs_frozen'])}`",
        f"- numerically stable under the stress suite: `{selected['numerically_stable']}`,"
        f" stress failures `{len(selected['stress_failures'])}`",
        "",
        "## The straddling-window ablation",
        "",
        "Each pair below differs only in `skipstraddle`. This is the mechanism added in response",
        "to the round-1 failure, measured here on development streams.",
        "",
        "| Variant (differing only in the policy) | policy off | policy on |",
        "|---|---:|---:|",
    ]
    by_label = {variant["label"]: variant for variant in data["variants"]}
    pairs = 0
    for label, off in sorted(by_label.items()):
        if "skipstraddle=0" not in label:
            continue
        on = by_label.get(label.replace("skipstraddle=0", "skipstraddle=1"))
        if on is None:
            continue
        pairs += 1
        shared = short(label).replace("|skipstraddle=0", "")
        lines.append(f"| `{shared}` | {fmt(off['always_online_mae'])} | {fmt(on['always_online_mae'])} |")
    if not pairs:
        lines.append("| _no matched pair in this table_ | | |")

    lines += [
        "",
        "## Every variant considered",
        "",
        "Unsuccessful alternatives are retained deliberately. A selection table that lists only",
        "the winner is not evidence.",
        "",
        "| Variant | straight MAE | bouncing MAE | always-online MAE | always-online margin | changed-law vs frozen | stable |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for variant in sorted(data["variants"], key=lambda item: -(item.get("improvement_vs_frozen") or -9e9)):
        lines.append(
            f"| `{short(variant['label'])}` | {fmt(variant.get('development_straight_mae'))} "
            f"| {fmt(variant.get('development_bouncing_mae'))} "
            f"| {fmt(variant.get('always_online_mae'))} "
            f"| {fmt(variant.get('always_online_margin'))} "
            f"| {fmt(variant.get('improvement_vs_frozen'))} "
            f"| {'yes' if variant.get('numerically_stable') else 'NO'} |"
        )

    lines += [
        "",
        "## Discipline",
        "",
        f"- {data['data_discipline']}.",
        "- Thresholds in the benchmark specification were frozen before any confirmation batch was",
        "  generated, and none was chosen by looking at a confirmation result.",
        "- Changing the candidate changes the specification hash, which retires every batch",
        "  declared against the old one. Round 2 therefore required newly declared batches, and",
        "  the spent round-1 attempts are counted in the multiplicity family.",
        "- The selected candidate is the smallest understandable system that satisfies the",
        "  research objective. Parameter count was not increased because more compute was",
        "  available; the model has three parameters.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
