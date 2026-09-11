"""Static plots rendered through Matplotlib's object-oriented API.

``matplotlib.pyplot`` is never imported here. Figures are created directly with
``matplotlib.figure.Figure`` and rendered by the Agg canvas, which is safe in a
headless process without any backend switching, and leaves the interactive
backend choice entirely to :mod:`aaa.animation`.

Axis limits are derived from the configured bounds and the data. Nothing
assumes the interval is ``[0, 1]``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .config import ExperimentConfig
from .experiment import StepRecord
from .metrics import mean_absolute_error, normalized_errors, rolling_mean

COLORS = {
    "persistence": "#4C78A8",
    "constant_motion": "#F58518",
    "constant_motion_reflected": "#72B7B2",
    "linear_online": "#54A24B",
    "linear_frozen": "#B279A2",
    "zero_control": "#9D755D",
    "candidate_frozen": "#E45756",
    "candidate_online": "#B82E2E",
    "candidate_no_reflect": "#EECA3B",
    "frozen": "#B279A2",
    "online": "#E45756",
}


def _save(figure: Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    FigureCanvasAgg(figure).print_figure(path, dpi=150)
    return str(path)


def _new(figsize: tuple[float, float]) -> tuple[Figure, Any]:
    figure = Figure(figsize=figsize)
    axis = figure.add_subplot(111)
    return figure, axis


def _episode(records: Sequence[StepRecord], replica: int, episode: int) -> list[StepRecord]:
    return sorted(
        [record for record in records if record.replica_id == replica and record.episode == episode],
        key=lambda record: record.step,
    )


def _bounds(config: ExperimentConfig) -> tuple[float, float]:
    return config.world.lower_bound, config.world.upper_bound


# ---------------------------------------------------------------------------
# historical v1 plots
# ---------------------------------------------------------------------------


def plot_representative_predictions(records: Sequence[StepRecord], config: ExperimentConfig, path: Path) -> str:
    if not records:
        raise ValueError("cannot plot empty records")
    first = _episode(records, records[0].replica_id, records[0].episode)
    names = list(first[0].predictions)
    figure, axis = _new((10, 4.8))
    x = [record.target_step for record in first]
    axis.plot(x, [record.actual_next_position for record in first], color="black", linewidth=2, label="actual next position")
    for name in names:
        axis.plot(x, [record.predictions[name]["scored"] for record in first], linewidth=1.25,
                  color=COLORS.get(name), label=name)
    for record in first:
        if record.changed:
            axis.axvline(record.target_step, color="crimson", linestyle="--", alpha=0.7, label="unannounced speed change")
        if record.bounced:
            axis.axvline(record.target_step, color="gray", linestyle=":", alpha=0.35)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8)
    axis.set_title("Predictions aligned to their target time step")
    axis.set_xlabel("target transition step")
    lower, upper = _bounds(config)
    axis.set_ylabel(f"position in [{lower:g}, {upper:g}]")
    margin = 0.05 * (upper - lower)
    axis.set_ylim(lower - margin, upper + margin)
    axis.grid(alpha=0.2)
    return _save(figure, path)


def plot_within_episode_rolling_error(records: Sequence[StepRecord], config: ExperimentConfig, path: Path) -> str:
    """Within-episode short-term behaviour. This is *not* a learning curve."""

    if not records:
        raise ValueError("cannot plot empty records")
    groups: dict[tuple[int, int], list[StepRecord]] = defaultdict(list)
    for record in records:
        groups[(record.replica_id, record.episode)].append(record)
    for group in groups.values():
        group.sort(key=lambda record: record.step)
    names = list(records[0].predictions)
    values: dict[str, dict[int, list[float]]] = {name: defaultdict(list) for name in names}
    for group in groups.values():
        for name in names:
            for index, value in enumerate(rolling_mean(normalized_errors(group, name), config.rolling_window)):
                values[name][index].append(value)
    figure, axis = _new((10, 4.8))
    for name in names:
        x = sorted(values[name])
        axis.plot(x, [mean(values[name][index]) for index in x], color=COLORS.get(name), linewidth=1.7, label=name)
    axis.set_title(f"Within-episode rolling normalized MAE (window={config.rolling_window}) — not a learning curve")
    axis.set_xlabel("scored step within episode")
    axis.set_ylabel("normalized absolute position error")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    return _save(figure, path)


def plot_change_errors(records: Sequence[StepRecord], config: ExperimentConfig, path: Path) -> str:
    if not records:
        raise ValueError("cannot plot empty records")
    first = _episode(records, records[0].replica_id, records[0].episode)
    changed = [record for record in first if record.changed]
    if not changed:
        raise ValueError("adaptation plot requires a changed-motion event")
    event = changed[0].target_step
    selected = [record for record in first if event - 25 <= record.target_step <= event + config.post_change_window + 12]
    names = list(first[0].predictions)
    figure, axis = _new((10, 4.8))
    for name in names:
        axis.plot(
            [record.target_step for record in selected],
            normalized_errors(selected, name),
            color=COLORS.get(name), linewidth=1.7, marker=".", label=name,
        )
    axis.axvline(event, color="crimson", linestyle="--", linewidth=1.4, label="unannounced speed change")
    axis.axvspan(event, event + config.post_change_window - 1, color="crimson", alpha=0.07, label="post-change window")
    axis.set_title("Error around the unannounced movement change")
    axis.set_xlabel("target transition step")
    axis.set_ylabel("normalized absolute position error")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    return _save(figure, path)


def plot_aggregate_performance(records: Sequence[StepRecord], path: Path) -> str:
    if not records:
        raise ValueError("cannot plot empty records")
    names = list(records[0].predictions)
    by_replica: dict[int, list[StepRecord]] = defaultdict(list)
    for record in records:
        by_replica[record.replica_id].append(record)
    means: list[float] = []
    deviations: list[float] = []
    for name in names:
        values = [mean_absolute_error(normalized_errors(group, name)) for group in by_replica.values()]
        clean = [value for value in values if value is not None]
        means.append(mean(clean))
        deviations.append(stdev(clean) if len(clean) > 1 else 0.0)
    figure, axis = _new((9, 4.8))
    positions = list(range(len(names)))
    axis.bar(positions, means, yerr=deviations, capsize=4,
             color=[COLORS.get(name, "#777777") for name in names], alpha=0.9)
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=15)
    axis.set_title("Frozen generalization normalized MAE across evaluation seeds")
    axis.set_ylabel("normalized MAE (mean ± seed SD)")
    axis.grid(axis="y", alpha=0.2)
    return _save(figure, path)


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
            adaptation_records, config, directory / "representative_predictions.png"
        ),
        "within_episode_rolling_error": plot_within_episode_rolling_error(
            learning_records, config, directory / "within_episode_rolling_error.png"
        ),
        "movement_change_error": plot_change_errors(
            adaptation_records, config, directory / "movement_change_error.png"
        ),
        "aggregate_generalization": plot_aggregate_performance(
            frozen_records, directory / "aggregate_generalization.png"
        ),
    }


# ---------------------------------------------------------------------------
# benchmark plots
# ---------------------------------------------------------------------------


def plot_learning_curve(summary: Mapping[str, Any], path: Path) -> str:
    """Cumulative training progression on a fixed frozen probe bank."""

    curve = summary["learning_curve"]["curve"]
    budgets = summary["learning_curve"]["budgets"]
    figure, axis = _new((8, 4.6))
    replicas = sorted({replica for budget in budgets for replica in curve[str(budget)]["replica_episode_values"]})
    for replica in replicas:
        values = [mean(curve[str(budget)]["replica_episode_values"][replica]) for budget in budgets]
        axis.plot(budgets, values, alpha=0.35, linewidth=1.0, color="#777777")
    axis.plot(budgets, [curve[str(budget)]["mean"] for budget in budgets],
              color="#E45756", linewidth=2.2, marker="o", label="mean across replicas")
    axis.set_yscale("log")
    axis.set_title("Learning progress: frozen checkpoints on one fixed probe bank")
    axis.set_xlabel("cumulative training episodes")
    axis.set_ylabel("normalized MAE on the fixed probe bank (log scale)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2, which="both")
    return _save(figure, path)


def plot_stratum_performance(summary: Mapping[str, Any], path: Path, *, family: str = "constant_velocity",
                             predictor: str = "candidate_frozen") -> str:
    strata = summary["results"][family]["predictors"][predictor]["strata"]
    names = [name for name in sorted(strata) if name != "training"]
    figure, axis = _new((11, 5.0))
    axis.bar(range(len(names)), [strata[name]["mae"] for name in names], color="#4C78A8", alpha=0.9)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=90, fontsize=6)
    axis.set_yscale("log")
    axis.set_title(f"{family}: normalized MAE by required stratum ({predictor})")
    axis.set_ylabel("normalized MAE (log scale)")
    axis.grid(axis="y", alpha=0.2, which="both")
    return _save(figure, path)


def plot_recovery_distribution(summary: Mapping[str, Any], path: Path) -> str:
    gate = next(item for item in summary["gates"]["gates"] if item["name"] == "recovery")
    counts = gate["details"].get("status_counts", {})
    names = list(counts)
    figure, axis = _new((8, 4.4))
    axis.bar(range(len(names)), [counts[name] for name in names], color="#54A24B", alpha=0.9)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=20, fontsize=8)
    axis.set_title("Recovery status distribution (every episode is in exactly one bucket)")
    axis.set_ylabel("episodes")
    axis.grid(axis="y", alpha=0.2)
    return _save(figure, path)


def plot_paired_scatter(summary: Mapping[str, Any], path: Path, *, family: str = "changed_law:changed",
                        candidate: str = "online", baseline: str = "frozen") -> str:
    result = summary["results"][family]["predictors"]
    left = result[candidate]["replica_mae"]
    right = result[baseline]["replica_mae"]
    replicas = sorted(set(left) & set(right))
    figure, axis = _new((5.6, 5.4))
    axis.scatter([right[key] for key in replicas], [left[key] for key in replicas], color="#E45756", s=42)
    limit = max([right[key] for key in replicas] + [left[key] for key in replicas] + [1e-12])
    axis.plot([0, limit], [0, limit], color="black", linewidth=1.0, linestyle="--", label="parity")
    axis.set_xlabel(f"{baseline} normalized MAE")
    axis.set_ylabel(f"{candidate} normalized MAE")
    axis.set_title(f"{family}: paired replica comparison")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    return _save(figure, path)


def plot_covariance_diagnostics(summary: Mapping[str, Any], path: Path) -> str:
    diagnostics = summary["checkpoints"]["diagnostics"]
    replicas = sorted(diagnostics)
    figure, axis = _new((7.4, 4.4))
    axis.plot(range(len(replicas)), [diagnostics[key]["condition_number"] for key in replicas],
              marker="o", color="#F58518", label="condition number")
    axis.plot(range(len(replicas)), [diagnostics[key]["min_eigenvalue"] for key in replicas],
              marker="s", color="#4C78A8", label="min eigenvalue")
    axis.set_yscale("log")
    axis.set_xticks(range(len(replicas)))
    axis.set_xticklabels([f"r{key}" for key in replicas])
    axis.set_title("Trained candidate covariance diagnostics per replica")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2, which="both")
    return _save(figure, path)


def write_benchmark_plots(directory: str | Path, summary: Mapping[str, Any]) -> dict[str, str]:
    """Plots that support scientific auditing of one benchmark attempt."""

    target = Path(directory)
    requested = (
        ("learning_curve", plot_learning_curve, "learning_curve.png"),
        ("stratum_performance", plot_stratum_performance, "stratum_performance.png"),
        ("recovery_distribution", plot_recovery_distribution, "recovery_distribution.png"),
        ("paired_scatter", plot_paired_scatter, "changed_law_paired_scatter.png"),
        ("covariance_diagnostics", plot_covariance_diagnostics, "covariance_diagnostics.png"),
    )
    plots: dict[str, str] = {}
    for name, render, filename in requested:
        try:
            plots[name] = render(summary, target / filename)
        except (KeyError, IndexError, StopIteration):
            # The summary does not carry the section this plot describes. A
            # missing optional figure must not cost the other four.
            continue
    return plots
