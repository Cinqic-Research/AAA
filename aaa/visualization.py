"""Static matplotlib plots for the reproducible AAA run."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ExperimentConfig
from .experiment import StepRecord
from .metrics import mean_absolute_error, rolling_mean


COLORS = {
    "persistence": "#4C78A8",
    "constant_motion": "#F58518",
    "linear_online": "#54A24B",
    "linear_frozen": "#B279A2",
}


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _episode(records: Sequence[StepRecord], seed: int, episode: int) -> list[StepRecord]:
    return sorted(
        [record for record in records if record.seed == seed and record.episode == episode],
        key=lambda record: record.step,
    )


def plot_representative_predictions(records: Sequence[StepRecord], path: Path) -> str:
    if not records:
        raise ValueError("cannot plot empty records")
    first = _episode(records, records[0].seed, records[0].episode)
    names = list(first[0].predictions)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    x = [record.target_step for record in first]
    axis.plot(x, [record.actual_next_position for record in first], color="black", linewidth=2, label="actual next position")
    for name in names:
        axis.plot(
            x,
            [record.predictions[name]["scored"] for record in first],
            linewidth=1.25,
            color=COLORS.get(name, None),
            label=name,
        )
    for record in first:
        if record.changed:
            axis.axvline(record.target_step, color="crimson", linestyle="--", alpha=0.7, label="unannounced speed change")
        if record.bounced:
            axis.axvline(record.target_step, color="gray", linestyle=":", alpha=0.35)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8)
    axis.set_title("AAA: predictions aligned to their target time step (online adaptation comparison)")
    axis.set_xlabel("target transition step")
    axis.set_ylabel("position in [0, 1]")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.2)
    return _save(fig, path)


def plot_learning_rolling_error(records: Sequence[StepRecord], config: ExperimentConfig, path: Path) -> str:
    if not records:
        raise ValueError("cannot plot empty records")
    groups: dict[tuple[int, int], list[StepRecord]] = defaultdict(list)
    for record in records:
        groups[(record.seed, record.episode)].append(record)
    for group in groups.values():
        group.sort(key=lambda record: record.step)
    names = list(records[0].predictions)
    values: dict[str, dict[int, list[float]]] = {name: defaultdict(list) for name in names}
    for group in groups.values():
        for name in names:
            rolling = rolling_mean(
                [record.predictions[name]["absolute_error"] for record in group], config.rolling_window
            )
            for index, value in enumerate(rolling):
                values[name][index].append(value)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for name in names:
        x = sorted(values[name])
        axis.plot(
            x,
            [mean(values[name][index]) for index in x],
            color=COLORS.get(name, None),
            linewidth=1.7,
            label=name,
        )
    axis.set_title(f"AAA: rolling MAE during learning from scratch (window={config.rolling_window})")
    axis.set_xlabel("scored step within episode")
    axis.set_ylabel("absolute position error")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    return _save(fig, path)


def plot_change_errors(records: Sequence[StepRecord], config: ExperimentConfig, path: Path) -> str:
    if not records:
        raise ValueError("cannot plot empty records")
    first = _episode(records, records[0].seed, records[0].episode)
    changed = [record for record in first if record.changed]
    if not changed:
        raise ValueError("adaptation plot requires a changed-motion event")
    event = changed[0].target_step
    selected = [record for record in first if event - 25 <= record.target_step <= event + config.post_change_window + 12]
    names = list(first[0].predictions)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for name in names:
        axis.plot(
            [record.target_step for record in selected],
            [record.predictions[name]["absolute_error"] for record in selected],
            color=COLORS.get(name, None),
            linewidth=1.7,
            marker=".",
            label=name,
        )
    axis.axvline(event, color="crimson", linestyle="--", linewidth=1.4, label="unannounced speed change")
    axis.axvspan(event, event + config.post_change_window - 1, color="crimson", alpha=0.07, label="post-change window")
    axis.set_title("AAA: error around the unannounced movement change (online adaptation)")
    axis.set_xlabel("target transition step")
    axis.set_ylabel("absolute position error")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    return _save(fig, path)


def plot_aggregate_performance(records: Sequence[StepRecord], path: Path) -> str:
    if not records:
        raise ValueError("cannot plot empty records")
    names = list(records[0].predictions)
    by_seed: dict[int, list[StepRecord]] = defaultdict(list)
    for record in records:
        by_seed[record.seed].append(record)
    means: list[float] = []
    deviations: list[float] = []
    for name in names:
        seed_errors = [
            mean_absolute_error(record.predictions[name]["absolute_error"] for record in seed_records)
            for seed_records in by_seed.values()
        ]
        clean = [value for value in seed_errors if value is not None]
        means.append(mean(clean))
        deviations.append(stdev(clean) if len(clean) > 1 else 0.0)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    positions = range(len(names))
    axis.bar(
        list(positions),
        means,
        yerr=deviations,
        capsize=4,
        color=[COLORS.get(name, "#777777") for name in names],
        alpha=0.9,
    )
    axis.set_xticks(list(positions), names, rotation=15)
    axis.set_title("AAA: frozen generalization MAE across final evaluation seeds")
    axis.set_ylabel("mean absolute position error (mean ± seed SD)")
    axis.grid(axis="y", alpha=0.2)
    return _save(fig, path)


def write_all_plots(
    directory: str | Path,
    *,
    learning_records: Sequence[StepRecord],
    frozen_records: Sequence[StepRecord],
    adaptation_records: Sequence[StepRecord],
    config: ExperimentConfig,
) -> dict[str, str]:
    directory = Path(directory)
    return {
        "representative_predictions": plot_representative_predictions(
            adaptation_records, directory / "representative_predictions.png"
        ),
        "learning_rolling_error": plot_learning_rolling_error(
            learning_records, config, directory / "learning_rolling_error.png"
        ),
        "movement_change_error": plot_change_errors(
            adaptation_records, config, directory / "movement_change_error.png"
        ),
        "aggregate_generalization": plot_aggregate_performance(
            frozen_records, directory / "aggregate_generalization.png"
        ),
    }
