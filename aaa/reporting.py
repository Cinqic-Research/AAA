"""Generate reports from stored measurements, without hard-coded outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import stdev
from typing import Any


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 10 ** (-digits):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _metric(summary: dict[str, object], experiment: str, predictor: str, key: str) -> Any:
    return summary["experiments"][experiment]["predictors"][predictor][key]  # type: ignore[index]


def _seed_mae_sd(summary: dict[str, object], experiment: str, predictor: str) -> float:
    values = [seed_info["predictors"][predictor]["mae"] for seed_info in summary["experiments"][experiment]["seeds"].values()]  # type: ignore[index]
    return stdev(values) if len(values) > 1 else 0.0


def write_experiment_report(path: str | Path, summary: dict[str, object]) -> None:
    """Render a report whose conclusions are derived from the supplied summary."""

    path = Path(path)
    learning = summary["experiments"]["learning_from_scratch"]  # type: ignore[index]
    linear_learning_delta = learning["early_late"]["linear_online"]["late_minus_early_mean"]  # type: ignore[index]
    online_post = _metric(summary, "online_adaptation", "linear_online", "post_change_window_mae_mean")
    frozen_post = _metric(summary, "online_adaptation", "linear_frozen", "post_change_window_mae_mean")
    lines = [
        "# AAA — Accurate Autonomous Adaptation", "",
        "*Measured local experiment report generated from stored AAA artifacts.*", "",
        "## Hypothesis", "",
        "A short-history online predictor may improve with experience, generalize to unfamiliar deterministic episodes, and contribute to recovery after an unannounced change. These claims are evaluated separately.", "",
        "## Method", "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Learning rate selected on development evidence: `{_fmt(summary['dev_selection']['selected_learning_rate'])}`",
        f"- World configuration: `{json.dumps(summary['config']['world'], sort_keys=True)}`",
        f"- Development seeds: `{summary['config']['dev_seeds']}`",
        f"- Training seeds: `{summary['config']['training_seeds']}`",
        f"- Final evaluation seeds: `{summary['config']['final_seeds']}`",
        "- Temporal order: predict, record, advance, reveal and score, then update enabled models.",
        "- Normalized error is absolute position error divided by interval width; raw values remain in step logs.", "",
        "## Results", "", "### Learning from scratch", "",
        "| Predictor | MAE | Episode MAE mean | Episode MAE SD |", "|---|---:|---:|---:|",
    ]
    for predictor in ("persistence", "constant_motion", "linear_online"):
        lines.append(f"| {predictor} | {_fmt(_metric(summary, 'learning_from_scratch', predictor, 'mae'))} | {_fmt(_metric(summary, 'learning_from_scratch', predictor, 'episode_mae_mean'))} | {_fmt(_metric(summary, 'learning_from_scratch', predictor, 'episode_mae_sd'))} |")
    lines.extend([
        "", f"The measured late-minus-early training difference for `linear_online` is `{_fmt(linear_learning_delta)}`. A negative value indicates lower late error; the sign is reported as observed and is not relabeled as learning.",
        "", "### Frozen generalization", "",
        f"The checkpoint integrity check reports `{summary['frozen_integrity']['checkpoint_weights_unchanged']}` for weight immutability during frozen evaluation.", "",
        "| Predictor | Overall MAE | Seed MAE SD | Bounce MAE | Non-bounce MAE |", "|---|---:|---:|---:|---:|",
    ])
    for predictor in ("persistence", "constant_motion", "linear_frozen"):
        lines.append(f"| {predictor} | {_fmt(_metric(summary, 'frozen_generalization', predictor, 'mae'))} | {_fmt(_seed_mae_sd(summary, 'frozen_generalization', predictor))} | {_fmt(_metric(summary, 'frozen_generalization', predictor, 'bounce_mae'))} | {_fmt(_metric(summary, 'frozen_generalization', predictor, 'non_bounce_mae'))} |")
    lines.extend([
        "", "### Online adaptation", "",
        "| Predictor | Overall MAE | Seed MAE SD | Post-change-window MAE | Recovery episodes | Mean recovery steps |", "|---|---:|---:|---:|---:|---:|",
    ])
    for predictor in ("persistence", "constant_motion", "linear_frozen", "linear_online"):
        lines.append(f"| {predictor} | {_fmt(_metric(summary, 'online_adaptation', predictor, 'mae'))} | {_fmt(_seed_mae_sd(summary, 'online_adaptation', predictor))} | {_fmt(_metric(summary, 'online_adaptation', predictor, 'post_change_window_mae_mean'))} | {_fmt(_metric(summary, 'online_adaptation', predictor, 'recovery_recovered_episodes'))} / {_fmt(_metric(summary, 'online_adaptation', predictor, 'recovery_applicable_episodes'))} | {_fmt(_metric(summary, 'online_adaptation', predictor, 'recovery_time_steps_mean'))} |")
    lines.extend([
        "", "## Evidence assessment", "",
        "- Correctness status is supported only by the recorded test command and artifacts supplied with this run.",
        f"- Training comparison: `{('lower late error' if float(linear_learning_delta) < 0 else 'not lower late error')}` under the stored early/late summary.",
        f"- Frozen generalization comparison: linear MAE `{_fmt(_metric(summary, 'frozen_generalization', 'linear_frozen', 'mae'))}`; persistence `{_fmt(_metric(summary, 'frozen_generalization', 'persistence', 'mae'))}`; constant motion `{_fmt(_metric(summary, 'frozen_generalization', 'constant_motion', 'mae'))}`.",
        f"- Matched post-change comparison: online minus frozen MAE is `{_fmt(None if online_post is None or frozen_post is None else float(online_post) - float(frozen_post))}`; this is evidence for this protocol only.",
        "- Missing, censored, or insufficient-coverage measurements remain unavailable rather than being converted into PASS conclusions.",
        "", "## Reproducibility and limitations", "",
        "Configuration, package versions, git metadata, JSONL/CSV steps, checkpoint, plots, metrics, and this report are stored with the run. The historical `results/final` snapshot is preserved as a separate regression track. The world is one-dimensional, deterministic, and noise-free; no general-intelligence or independent-goal claim follows from this experiment.", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
