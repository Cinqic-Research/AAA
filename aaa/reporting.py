"""Generate the historical v1 experiment report from stored measurements.

Every statement is derived from the supplied summary. Nothing about the
outcome is written in prose ahead of time, and a missing measurement stays
missing instead of being rendered as a conclusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import stdev
from typing import Any

# Human reports round to four significant figures. Full precision stays in the
# machine-readable artifacts; a report that prints seventeen digits of a
# float invites unwarranted confidence in the last fourteen of them.
SIGNIFICANT_DIGITS = 4


def _fmt(value: Any, digits: int = SIGNIFICANT_DIGITS) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return f"{value:.{digits}g}"
    return str(value)


def _metric(summary: dict[str, Any], experiment: str, predictor: str, key: str) -> Any:
    return summary["experiments"][experiment]["predictors"][predictor][key]


def _replica_mae_sd(summary: dict[str, Any], experiment: str, predictor: str) -> float:
    replicas = summary["experiments"][experiment].get("replicas", {})
    values = [
        info["predictors"][predictor]["mae"]
        for info in replicas.values()
        if info["predictors"][predictor]["mae"] is not None
    ]
    return stdev(values) if len(values) > 1 else 0.0


def _recovery_line(summary: dict[str, Any], predictor: str) -> str:
    counts = _metric(summary, "online_adaptation", predictor, "recovery_status_counts")
    eligible = _metric(summary, "online_adaptation", predictor, "recovery_eligible_episodes")
    recovered = _metric(summary, "online_adaptation", predictor, "recovery_recovered_episodes")
    unrecovered = _metric(summary, "online_adaptation", predictor, "recovery_unrecovered_episodes")
    ineligible = {
        key: value for key, value in counts.items() if value and key not in ("recovered", "unrecovered")
    }
    return (
        f"{recovered} recovered / {unrecovered} unrecovered of {eligible} eligible; "
        f"ineligible reasons: {json.dumps(ineligible, sort_keys=True) if ineligible else 'none'}"
    )


def write_experiment_report(path: str | Path, summary: dict[str, Any]) -> None:
    """Render a report whose conclusions are derived from the supplied summary."""

    path = Path(path)
    learning = summary["experiments"]["learning_from_scratch"]
    linear_learning_delta = learning["early_late"]["linear_online"]["late_minus_early_mean"]
    online_post = _metric(summary, "online_adaptation", "linear_online", "post_change_window_mae_mean")
    frozen_post = _metric(summary, "online_adaptation", "linear_frozen", "post_change_window_mae_mean")
    lines = [
        "# AAA — historical v1 evaluation",
        "",
        "*Measured report generated from stored AAA artifacts. This is the historical",
        "v1 track, retained for regression and provenance. It is **not** benchmark",
        "acceptance evidence; see the benchmark protocol documentation for that.*",
        "",
        "## Hypothesis",
        "",
        "A short-history online predictor may improve with experience, generalize to",
        "unfamiliar deterministic episodes, and contribute to recovery after an",
        "unannounced change. These claims are evaluated separately and none implies",
        "another.",
        "",
        "## Method",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Learning rate selected on development evidence: `{_fmt(summary['dev_selection']['selected_learning_rate'])}`",
        f"- World configuration: `{json.dumps(summary['config']['world'], sort_keys=True)}`",
        f"- Development seeds: `{summary['config']['dev_seeds']}`",
        f"- Training seeds: `{summary['config']['training_seeds']}`",
        f"- Final evaluation seeds: `{summary['config']['final_seeds']}`",
        "- Temporal order: predict, record, advance, reveal and score, then update enabled models.",
        "- Every error in this report is normalized: absolute position error divided by",
        "  the interval width. Raw values remain in the step logs.",
        "",
        "## Results",
        "",
        "### Learning from scratch",
        "",
        "| Predictor | Normalized MAE | Episode MAE mean | Episode MAE SD |",
        "|---|---:|---:|---:|",
    ]
    for predictor in ("persistence", "constant_motion", "linear_online"):
        lines.append(
            f"| {predictor} "
            f"| {_fmt(_metric(summary, 'learning_from_scratch', predictor, 'mae'))} "
            f"| {_fmt(_metric(summary, 'learning_from_scratch', predictor, 'episode_mae_mean'))} "
            f"| {_fmt(_metric(summary, 'learning_from_scratch', predictor, 'episode_mae_sd'))} |"
        )
    lines.extend(
        [
            "",
            f"The measured late-minus-early difference for `linear_online` is "
            f"`{_fmt(linear_learning_delta)}`. A negative value indicates lower late error. "
            "This is a *within-run* early/late comparison across changing episodes, not a "
            "controlled learning curve; the benchmark's frozen-probe learning curve is the "
            "measurement designed to answer the learning question.",
            "",
            "### Frozen generalization",
            "",
            f"Measured checkpoint immutability during frozen evaluation: "
            f"`{summary['frozen_integrity']['checkpoint_weights_unchanged']}` across "
            f"`{summary['frozen_integrity']['evaluated_copies']}` evaluated copies "
            "(compared against the actual post-evaluation weights and update counts).",
            "",
            "| Predictor | Normalized MAE | Replica MAE SD | Bounce MAE | Non-bounce MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for predictor in ("persistence", "constant_motion", "linear_frozen"):
        lines.append(
            f"| {predictor} "
            f"| {_fmt(_metric(summary, 'frozen_generalization', predictor, 'mae'))} "
            f"| {_fmt(_replica_mae_sd(summary, 'frozen_generalization', predictor))} "
            f"| {_fmt(_metric(summary, 'frozen_generalization', predictor, 'bounce_mae'))} "
            f"| {_fmt(_metric(summary, 'frozen_generalization', predictor, 'non_bounce_mae'))} |"
        )
    lines.extend(
        [
            "",
            "### Online adaptation after an unannounced speed change",
            "",
            "| Predictor | Normalized MAE | Replica MAE SD | Post-change-window MAE | Recovery |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for predictor in ("persistence", "constant_motion", "linear_frozen", "linear_online"):
        lines.append(
            f"| {predictor} "
            f"| {_fmt(_metric(summary, 'online_adaptation', predictor, 'mae'))} "
            f"| {_fmt(_replica_mae_sd(summary, 'online_adaptation', predictor))} "
            f"| {_fmt(_metric(summary, 'online_adaptation', predictor, 'post_change_window_mae_mean'))} "
            f"| {_recovery_line(summary, predictor)} |"
        )
    lines.extend(
        [
            "",
            "## Evidence assessment",
            "",
            "- Correctness status is supported only by the recorded test command and the",
            "  artifacts supplied with this run.",
            f"- Training comparison: "
            f"`{'lower late error' if float(linear_learning_delta) < 0 else 'not lower late error'}` "
            "under the stored early/late summary.",
            f"- Frozen generalization: linear `{_fmt(_metric(summary, 'frozen_generalization', 'linear_frozen', 'mae'))}`; "
            f"persistence `{_fmt(_metric(summary, 'frozen_generalization', 'persistence', 'mae'))}`; "
            f"constant motion `{_fmt(_metric(summary, 'frozen_generalization', 'constant_motion', 'mae'))}`.",
            f"- Matched post-change comparison: online minus frozen MAE is "
            f"`{_fmt(None if online_post is None or frozen_post is None else float(online_post) - float(frozen_post))}`; "
            "this is evidence for this protocol only.",
            "- Missing, censored, or insufficient-coverage measurements remain unavailable",
            "  rather than being converted into PASS conclusions.",
            "",
            "## Reproducibility and limitations",
            "",
            "Configuration, package versions, git metadata, JSONL/CSV steps, checkpoint,",
            "plots, metrics, and this report are stored together under the run directory.",
            "Nothing is written outside the requested output root. The world is",
            "one-dimensional, deterministic, and noise-free; no general-intelligence or",
            "independent-goal claim follows from this experiment.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
