"""Raw-data-backed plots for an observation-noise attempt."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .verifier import iter_records


def _summary(values: list[float]) -> tuple[float, float, float]:
    """Return a mean and empirical 5th/95th percentiles for plotted rows."""

    import numpy as np

    if not values:
        raise ValueError("cannot summarize an empty plotted group")
    quantiles = np.quantile(values, (0.05, 0.95), method="linear")
    return float(np.mean(values)), float(quantiles[0]), float(quantiles[1])


def _bar_summary(values: list[float]) -> tuple[float, float, float]:
    """Return a median-centered interval so asymmetric outliers remain visible."""

    import numpy as np

    _mean, lower, upper = _summary(values)
    return float(np.median(values)), lower, upper


def _errorbar(axis: Any, x: float, values: list[float], *, label: str, color: str) -> None:
    center, lower, upper = _bar_summary(values)
    axis.errorbar(
        x,
        center,
        yerr=[[center - lower], [upper - center]],
        fmt="o",
        capsize=3,
        color=color,
        label=label,
    )


def _save(figure: Any, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def write_plots(run_dir: str | Path) -> list[Path]:
    """Write diagnostic figures whose provenance is the primitive archive.

    Error bars in the figures are empirical 5th/95th percentiles of retained
    primitive rows around their median, not the aggregate hierarchical
    intervals in ``summary``.
    The latter remain the authoritative statistical uncertainty artifact.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_dir)
    rows = list(iter_records(root))
    output: list[Path] = []
    by_scale: dict[tuple[str, float], list[float]] = defaultdict(list)
    by_step: dict[tuple[str, int], list[float]] = defaultdict(list)
    surprises: dict[int, list[float]] = defaultdict(list)
    update_norms: dict[tuple[str, int], list[float]] = defaultdict(list)
    latencies: dict[tuple[str, int], list[float]] = defaultdict(list)
    controls: dict[str, list[float]] = defaultdict(list)
    strata: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        trial = row["trial"]
        branch = str(trial["branch"])
        for name, prediction in row["predictions"].items():
            error = float(prediction["latent_normalized_absolute_error"])
            if name == "incumbent_square_root_rls" and branch == "stationary":
                by_scale[(str(trial["channel"]), float(trial["scale"]))].append(error)
            if name in {"incumbent_branch_frozen", "incumbent_branch_online"}:
                by_step[(name, int(row["step"]))].append(error)
            if name == "incumbent_square_root_rls":
                diagnostics = row.get("diagnostics", {}).get(name, {})
                surprises[int(row["step"])].append(float(diagnostics.get("detected_surprises", 0)))
                update_norms[(name, int(row["step"]))].append(float(diagnostics.get("update_norm", 0)))
                latencies[(name, int(row["step"]))].append(
                    float(diagnostics.get("predict_latency_ns", 0))
                    + float(diagnostics.get("update_latency_ns", 0))
                )
                if branch.startswith("sensor_shift_"):
                    controls[branch].append(error)
                stratum = str(trial.get("stratum", "unstratified"))
                if branch == "stationary" and stratum != "unstratified":
                    strata[(str(trial["family"]), stratum)].append(error)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    colors = {
        "gaussian": "tab:blue",
        "uniform": "tab:orange",
        "correlated": "tab:green",
        "impulsive": "tab:red",
    }
    labeled_channels: set[str] = set()
    for (channel, scale), values in sorted(by_scale.items()):
        color = colors.get(channel, "tab:gray")
        axis.scatter([scale] * len(values), values, s=4, alpha=0.10, color=color)
        label = f"{channel} median, 5-95%" if channel not in labeled_channels else "_nolegend_"
        _errorbar(axis, scale, values, label=label, color=color)
        labeled_channels.add(channel)
    axis.set_title(
        "Incumbent latent error by observed-noise scale\n(points: primitive rows; bars: median with empirical 5-95%)"
    )
    axis.set_xlabel("RMS noise scale")
    axis.set_ylabel("normalized absolute error")
    axis.set_yscale("symlog", linthresh=1e-6)
    if by_scale:
        axis.legend(loc="best", fontsize="small")
    output.append(_save(figure, root / "error_vs_noise_scale.png"))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, color in (
        ("incumbent_branch_frozen", "tab:gray"),
        ("incumbent_branch_online", "tab:blue"),
    ):
        points = sorted(
            (step, _summary(values)[0]) for (label, step), values in by_step.items() if label == name
        )
        if points:
            axis.plot([point[0] for point in points], [point[1] for point in points], label=name, color=color)
            first_step, first_error = points[0]
            axis.scatter([first_step], [first_error], color=color, marker="*", s=65, zorder=3)
            axis.annotate(
                "first post-change error",
                (first_step, first_error),
                xytext=(5, 8),
                textcoords="offset points",
            )
    axis.axvline(300, color="black", linestyle="--", linewidth=0.8, label="law change")
    axis.set_title(
        "Matched changed-law branch trajectory\n(first post-intervention error is included and marked)"
    )
    axis.set_xlabel("transition")
    axis.set_ylabel("normalized latent absolute error")
    if by_step:
        axis.legend(loc="best", fontsize="small")
    output.append(_save(figure, root / "matched_adaptation_trajectory.png"))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    labels = sorted(controls)
    if labels:
        positions = list(range(len(labels)))
        means = []
        lowers = []
        uppers = []
        for label in labels:
            center, lower, upper = _bar_summary(controls[label])
            means.append(center)
            lowers.append(lower)
            uppers.append(upper)
        axis.errorbar(
            positions,
            means,
            yerr=[
                [mean - lower for mean, lower in zip(means, lowers, strict=True)],
                [upper - mean for mean, upper in zip(means, uppers, strict=True)],
            ],
            fmt="o",
            capsize=3,
        )
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_title(
        "Sensor-shift controls: unchanged and changed latent dynamics\n(median with empirical primitive 5-95% bars)"
    )
    axis.set_xlabel("factorial branch")
    axis.set_ylabel("incumbent normalized latent absolute error")
    output.append(_save(figure, root / "noise_shift_controls.png"))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    worst = sorted(
        ((key, _summary(values)[0], values) for key, values in strata.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    if worst:
        labels = [f"{family}\n{stratum}" for (family, stratum), _mean, _values in worst]
        means = [_bar_summary(values)[0] for _key, _mean, values in worst]
        stratum_lowers = [_bar_summary(values)[1] for _key, _mean, values in worst]
        stratum_uppers = [_bar_summary(values)[2] for _key, _mean, values in worst]
        positions = list(range(len(worst)))
        axis.errorbar(
            positions,
            means,
            yerr=[
                [mean - lo for mean, lo in zip(means, stratum_lowers, strict=True)],
                [hi - mean for mean, hi in zip(means, stratum_uppers, strict=True)],
            ],
            fmt="o",
            capsize=3,
        )
        axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_title(
        "Worst realized incumbent strata\n(top strata by primitive-row mean; bars are median with empirical 5-95%)"
    )
    axis.set_xlabel("predeclared family and stratum")
    axis.set_ylabel("normalized latent absolute error")
    output.append(_save(figure, root / "worst_stratum.png"))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    points = sorted((step, max(values)) for step, values in surprises.items())
    if points:
        axis.plot([point[0] for point in points], [point[1] for point in points], color="tab:red")
    axis.set_title("Incumbent detector activity")
    axis.set_xlabel("transition")
    axis.set_ylabel("cumulative detected surprises")
    output.append(_save(figure, root / "detector_activity.png"))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    update_points = sorted(
        (step, _summary(values)[0])
        for (name, step), values in update_norms.items()
        if name == "incumbent_square_root_rls"
    )
    latency_points = sorted(
        (step, _summary(values)[0])
        for (name, step), values in latencies.items()
        if name == "incumbent_square_root_rls"
    )
    if update_points:
        axes[0].plot(
            [point[0] for point in update_points],
            [point[1] for point in update_points],
            color="tab:purple",
        )
    if latency_points:
        axes[1].plot(
            [point[0] for point in latency_points],
            [point[1] for point in latency_points],
            color="tab:green",
        )
    axes[0].set_title("Update-state movement")
    axes[0].set_xlabel("transition")
    axes[0].set_ylabel("mean update norm")
    axes[1].set_title("Measured predict + update cost")
    axes[1].set_xlabel("transition")
    axes[1].set_ylabel("mean nanoseconds / step")
    output.append(_save(figure, root / "update_resource_diagnostics.png"))

    records_path = root / "records.jsonl"
    provenance = {
        "schema_version": "aaa.observation_noise_plot_provenance.v1",
        "source": "records.jsonl",
        "source_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "source_rows": len(rows),
        "uncertainty_in_figures": "median-centered empirical primitive-row 5th/95th percentiles",
        "authoritative_aggregate_uncertainty": "summary.json statistics hierarchical and paired artifacts",
        "figures": [path.name for path in output],
    }
    provenance_path = root / "plot_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.append(provenance_path)
    return output
