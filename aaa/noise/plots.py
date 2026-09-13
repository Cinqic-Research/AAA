"""Raw-data-backed plots for an observation-noise attempt."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .verifier import iter_records


def write_plots(run_dir: str | Path) -> list[Path]:
    """Write small diagnostic figures whose provenance is the primitive archive."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_dir)
    rows = list(iter_records(root))
    output: list[Path] = []
    by_scale: dict[tuple[str, float], list[float]] = defaultdict(list)
    by_step: dict[tuple[str, int], list[float]] = defaultdict(list)
    surprises: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        trial = row["trial"]
        for name, prediction in row["predictions"].items():
            if name == "incumbent_square_root_rls" and trial["branch"] == "stationary":
                by_scale[(trial["channel"], float(trial["scale"]))].append(
                    float(prediction["latent_normalized_absolute_error"])
                )
            if name in {"incumbent_branch_frozen", "incumbent_branch_online"}:
                by_step[(name, int(row["step"]))].append(
                    float(prediction["latent_normalized_absolute_error"])
                )
            if name == "incumbent_square_root_rls":
                detected = float(row.get("diagnostics", {}).get(name, {}).get("detected_surprises", 0))
                surprises[int(row["step"])].append(detected)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for (channel, scale), values in sorted(by_scale.items()):
        axis.scatter([scale] * len(values), values, s=4, alpha=0.15, label=channel if scale == 0.0 else None)
    axis.set_title("Incumbent latent error by observed-noise scale (primitive records)")
    axis.set_xlabel("RMS noise scale")
    axis.set_ylabel("normalized absolute error")
    axis.set_yscale("symlog", linthresh=1e-6)
    axis.legend(loc="best")
    path = root / "error_vs_noise_scale.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    output.append(path)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name in ("incumbent_branch_frozen", "incumbent_branch_online"):
        points = sorted(
            (step, sum(values) / len(values)) for (label, step), values in by_step.items() if label == name
        )
        if points:
            axis.plot([point[0] for point in points], [point[1] for point in points], label=name)
    axis.axvline(300, color="black", linestyle="--", linewidth=0.8, label="law change")
    axis.set_title("Matched changed-law branch trajectory")
    axis.set_xlabel("transition")
    axis.set_ylabel("normalized latent absolute error")
    axis.legend(loc="best")
    path = root / "matched_adaptation_trajectory.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    output.append(path)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    points = sorted((step, max(values)) for step, values in surprises.items())
    if points:
        axis.plot([point[0] for point in points], [point[1] for point in points])
    axis.set_title("Incumbent detector activity")
    axis.set_xlabel("transition")
    axis.set_ylabel("cumulative detected surprises")
    path = root / "detector_activity.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    output.append(path)
    return output
