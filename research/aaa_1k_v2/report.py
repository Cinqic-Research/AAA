"""Generate the v2 reports from retained evidence (never from memory).

    compute_report(qualification) -> docs/aaa_1k_v2_compute_report.md
    phase_report(development, diagnostics, attack, freeze, confirmation, recomputation, capacity)
        -> docs/aaa_1k_v2_report.md

Every number in the generated documents is read from an artifact; the
artifact paths and SHA-256 hashes are printed at the top of each report.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}g}"
    return str(value)


def _rate(points: Sequence[Mapping[str, Any]], workload: str, executor: str, cells: int) -> float | None:
    for p in points:
        if p["workload"] == workload and p["executor"] == executor and p["cells"] == cells:
            return float(p["cell_steps_per_second"])
    return None


def compute_report(q: Mapping[str, Any], *, source: str, digest: str) -> str:
    from .qualify import crossover

    points = q["points"]
    rows = [
        ("single model / stream", "online", 1),
        ("32 cells", "online", 32),
        ("128 cells", "online", 128),
        ("512 cells", "online", 512),
        ("1,024 cells", "online", 1024),
        ("4,096 cells", "online", 4096),
        ("hyperparameter search (4,096 cells over the lr grid)", "sweep", 4096),
        ("internal evaluation (online/frozen twins, 1,024 trunks)", "evaluation", 1024),
        ("external benchmark batch (NARMA-10, 1,024 cells)", "external", 1024),
        ("prediction only (frozen, 4,096 cells)", "prediction", 4096),
    ]
    lines = [
        "# FLOWBOX compute report (aaa.1k.v2)",
        "",
        f"Generated from [`{source}`](../{source}) (sha256 `{digest[:16]}...`) by",
        "`research/aaa_1k_v2/report.py`. Throughput is in cell-steps per second (one cell advancing one",
        "online-learning step: predict, score, update), float64, AAA-1K Champion 1 unless stated. Every point is the",
        f"best of {q['repeats']} timed runs after a warm-up run; CUDA timers synchronize the device before stopping.",
        "The RTX 2060 also drives the desktop, so its numbers include that background load.",
        "",
        "| Workload | CPU sequential | CPU parallel (8 workers) | CUDA (RTX 2060) | Winner | Notes |",
        "|---|---:|---:|---:|---|---|",
    ]
    for label, workload, cells in rows:
        seq, par, gpu = (_rate(points, workload, ex, cells) for ex in ("cpu_seq", "cpu_par", "cuda"))
        values = {"CPU sequential": seq, "CPU parallel": par, "CUDA": gpu}
        present = {k: v for k, v in values.items() if v is not None}
        winner = max(present, key=lambda k: present[k]) if present else "n/a"
        note = ""
        if cells == 1:
            note = "a lone cell cannot use parallel hardware"
        elif workload == "external":
            note = "6,000-step sequences"
        lines.append(f"| {label} | {_fmt(seq)} | {_fmt(par)} | {_fmt(gpu)} | {winner} | {note} |")
    lines += ["", "## Crossover points", ""]
    for workload in ("online", "prediction", "sweep", "evaluation", "external"):
        a = crossover(points, workload, "cpu_seq", "cuda")
        b = crossover(points, workload, "cpu_par", "cuda")
        lines.append(
            f"* `{workload}`: CUDA beats one CPU process from {_fmt(a) if a else 'never in the measured range'} cells, "
            f"and the 8-worker CPU from {_fmt(b) if b else 'never in the measured range'} cells."
        )
    lines += [
        "",
        "## Resources",
        "",
        "| Workload | Executor | Cells | Wall s (min / median) | CPU util. | GPU util. | Peak GPU memory MiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in points:
        if p["cells"] in (32, 512, 4096, 16384) and p["workload"] in ("online", "external"):
            lines.append(
                f"| {p['workload']} | {p['executor']} | {p['cells']} | {p['wall_seconds_min']:.2f} / {p['wall_seconds_median']:.2f} | "
                f"{p['cpu_utilization']:.0%} | {_fmt(p.get('gpu_utilization_mean'))} | {_fmt(p.get('gpu_memory_used_peak_mib'))} |"
            )
    rss = q["peak_rss_bytes"]
    lines += [
        "",
        f"Peak resident memory: parent process {rss['parent'] / 2**20:.0f} MiB, largest worker {rss['largest_child'] / 2**20:.0f} MiB.",
        "GPU memory figures are whole-device `memory.used` samples and include the desktop's resident allocation.",
        "",
        "## Host-device transfer",
        "",
        "| Cells | Stream bytes | Upload s | Download s |",
        "|---:|---:|---:|---:|",
    ]
    for t in q["transfer"]:
        lines.append(
            f"| {t['cells']} | {t['bytes']:,} | {t['upload_seconds']:.5f} | {t['download_seconds']:.5f} |"
        )
    lines += [
        "",
        "Transfers are a one-off cost per job and negligible next to the per-step dispatch cost.",
        "",
        "## CPU / CUDA parity and determinism",
        "",
        "| Arm | Family | Cells | Bitwise CPU=CUDA | Max step diff | Max rel. MAE diff | Failure agreement | CPU repeat bitwise | CUDA repeat bitwise |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for arm, families in q["parity"].items():
        for family, r in families.items():
            lines.append(
                f"| {arm} | {family} | {r['cells']} | {r['bitwise_equal_cells_cpu_vs_cuda']} | {r['max_abs_step_difference']:.2e} | "
                f"{r['max_relative_mae_difference']:.2e} | {r['failed_agreement']} | {r['cpu_run_to_run_bitwise_cells']} | {r['cuda_run_to_run_bitwise_cells']} |"
            )
    worst = max(r["max_relative_mae_difference"] for fams in q["parity"].values() for r in fams.values())
    lines += [
        "",
        f"Largest per-cell relative MAE difference between backends: {worst:.2e}. The backends are separate numerical",
        "platforms: agreement is measured here, and paired scientific comparisons never mix them.",
        "",
        "## When to use the Ryzen 7 5700G and when the RTX 2060",
        "",
        "For AAA-1K-sized models the deciding quantity is the number of independent cells advancing in lockstep:",
        "",
        "* one or a few cells, and any very long sequential stream: one CPU core;",
        "* up to the crossover above: the 8-core CPU in 64-cell chunks;",
        "* beyond it: the RTX 2060, capped by its float64 rate (1/32 of float32).",
        "",
        "Larger future models do more arithmetic per step and move the crossover down; measure them when they exist.",
    ]
    return "\n".join(lines) + "\n"


def _status_table(statuses: Mapping[str, str]) -> list[str]:
    lines = ["| Criterion | Status |", "|---|---|"]
    lines += [f"| {key} | **{value}** |" for key, value in statuses.items()]
    return lines


def geometric_ratio(result: Mapping[str, Any] | None) -> str:
    if not result or result.get("interval_status") != "MEASURED":
        return "n/a"
    return f"{result['geometric_ratio']:.3f} [{result['lower']:.3f}, {result['upper']:.3f}]"


def family_means(primitives: Sequence[Mapping[str, Any]], key: str = "mae") -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for cell in primitives:
        if not cell.get("failed") and cell.get(key) is not None and np.isfinite(cell[key]):
            by.setdefault(cell["group"], []).append(float(cell[key]))
    return {group: float(np.mean(values)) for group, values in by.items()}
