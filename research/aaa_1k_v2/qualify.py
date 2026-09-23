"""FLOWBOX compute qualification: throughput, memory, parity and determinism.

Run on a quiet machine (no other AAA workload), on qualification identities.

Throughput
    Workloads: ``online`` (Champion 1 learning on 200-step occlusion streams),
    ``prediction`` (the same cells frozen), ``sweep`` (cells spread over the
    development learning-rate grid), ``evaluation`` (online/frozen twin
    branches) and ``external`` (NARMA-10 series cells). Executors:
    ``cpu_seq`` (one process, one batch), ``cpu_par`` (8 single-threaded
    workers, 64-cell chunks) and ``cuda`` (the RTX 2060). Each point is timed
    ``REPEATS`` times after a warm-up run; CUDA timers synchronize the device
    before stopping. Recorded: wall seconds, cell-steps per second, CPU
    utilization (process and children CPU time over wall time and logical
    threads), peak resident memory, GPU utilization and peak VRAM sampled from
    ``nvidia-smi``, and host-device transfer time.

Parity
    Every arm (Champion 1 and the six candidate templates at their template
    hyperparameters) on the same qualification cells on CPU and CUDA: per-step
    maximum absolute difference, per-cell relative MAE difference, bitwise-equal
    cells, and agreement of the failed/diverged classification.

Determinism
    Each backend runs the same cells twice; the number of bitwise-identical
    cells is recorded.
"""

from __future__ import annotations

import os
import resource
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from aaa.compute import backend_provenance, resolve_backend

from .arms import CANDIDATE_TEMPLATES, CHAMPION_1, ArmSpec
from .external import tasks as ext
from .jobs import DotJob, SeriesJob, run_jobs
from .runner import StreamBatch, run_dot
from .stress import batch as family_batch

SIZES = (1, 8, 32, 128, 512, 1024, 4096)
CUDA_EXTRA = (16384,)
REPEATS = 3
WORKLOADS = ("online", "prediction", "sweep", "evaluation", "external")
EXECUTORS = ("cpu_seq", "cpu_par", "cuda")


class GPUSampler:
    """Samples ``utilization.gpu`` and ``memory.used`` every 0.2 s while active."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                ).stdout.strip()
                util, mem = (float(v) for v in out.splitlines()[0].split(","))
                self.samples.append((util, mem))
            except Exception:
                pass
            self._stop.wait(0.2)

    def __enter__(self) -> GPUSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, float | None]:
        if not self.samples:
            return {"gpu_utilization_mean": None, "gpu_memory_used_peak_mib": None}
        util = [s[0] for s in self.samples]
        mem = [s[1] for s in self.samples]
        return {"gpu_utilization_mean": float(np.mean(util)), "gpu_memory_used_peak_mib": float(max(mem))}


def _stream_batch(seed_base: Sequence[int], size: int) -> StreamBatch:
    base = family_batch("v1_occlusion", list(seed_base))
    rows = [i % base.size for i in range(size)]
    return base.take(rows)


def _jobs(workload: str, size: int, device: str, seeds: Sequence[int], streams: Sequence[int]) -> list[Any]:
    init_seeds = tuple(seeds[i % len(seeds)] for i in range(size))
    if workload == "external":
        problems = ext.narma_problems("narma10", list(streams[:2]))
        probs = tuple(problems[i % len(problems)] for i in range(size))
        return [SeriesJob(arm=CHAMPION_1, seeds=init_seeds, problems=probs, device=device, tag="q:external")]
    batch = _stream_batch(streams, size)
    if workload == "prediction":
        return [
            DotJob(
                arm=replace(CHAMPION_1, update_enabled=False),
                seeds=init_seeds,
                batch=batch,
                device=device,
                tag="q:prediction",
            )
        ]
    if workload == "sweep":
        rates = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
        configs = tuple(replace(CHAMPION_1.config, learning_rate=rates[i % len(rates)]) for i in range(size))
        return [
            DotJob(
                arm=CHAMPION_1, seeds=init_seeds, batch=batch, configs=configs, device=device, tag="q:sweep"
            )
        ]
    if workload == "evaluation":
        return [
            DotJob(
                arm=CHAMPION_1, seeds=init_seeds, batch=batch, branch_at=50, device=device, tag="q:evaluation"
            )
        ]
    return [DotJob(arm=CHAMPION_1, seeds=init_seeds, batch=batch, device=device, tag="q:online")]


def _cpu_times() -> float:
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return self_usage.ru_utime + self_usage.ru_stime + children.ru_utime + children.ru_stime


def time_point(
    workload: str, executor: str, size: int, seeds: Sequence[int], streams: Sequence[int]
) -> dict[str, Any]:
    device = "cuda:0" if executor == "cuda" else "cpu"
    workers = 8 if executor == "cpu_par" else 1
    chunk = 64 if executor == "cpu_par" else max(size, 1)

    def once() -> tuple[float, int, float, dict[str, Any]]:
        jobs = _jobs(workload, size, device, seeds, streams)
        cpu_before = _cpu_times()
        with GPUSampler() as sampler:
            if executor == "cuda":
                resolve_backend("cuda:0").synchronize()
            started = time.perf_counter()
            results = run_jobs(jobs, workers=workers, max_cells_per_chunk=chunk)
            if executor == "cuda":
                resolve_backend("cuda:0").synchronize()
            wall = time.perf_counter() - started
        steps = sum(r["cell_steps"] for r in results)
        return wall, steps, _cpu_times() - cpu_before, sampler.summary()

    once()  # warm-up (CUDA kernel compilation, process start-up paths)
    runs = [once() for _ in range(REPEATS)]
    walls = [r[0] for r in runs]
    steps = runs[0][1]
    best = int(np.argmin(walls))
    return {
        "workload": workload,
        "executor": executor,
        "cells": size,
        "cell_steps": steps,
        "wall_seconds": walls,
        "wall_seconds_min": float(min(walls)),
        "wall_seconds_median": float(np.median(walls)),
        "cell_steps_per_second": float(steps / min(walls)),
        "cpu_utilization": float(runs[best][2] / min(walls) / (os.cpu_count() or 1)),
        **runs[best][3],
    }


def transfer_overhead(sizes: Sequence[int], streams: Sequence[int]) -> list[dict[str, Any]]:
    backend = resolve_backend("cuda:0")
    xp = backend.xp
    out = []
    for size in sizes:
        batch = _stream_batch(streams, size)
        backend.synchronize()
        started = time.perf_counter()
        truth = backend.asarray(batch.truth)
        backend.synchronize()
        upload = time.perf_counter() - started
        errors = xp.zeros_like(truth)
        backend.synchronize()
        started = time.perf_counter()
        backend.to_host(errors)
        download = time.perf_counter() - started
        out.append(
            {
                "cells": size,
                "bytes": int(batch.truth.nbytes),
                "upload_seconds": upload,
                "download_seconds": download,
            }
        )
    return out


def _run_cells(
    arm: ArmSpec, device: str, seeds: Sequence[int], batch: StreamBatch
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .jobs import build

    cell_seeds = [s for s in seeds for _ in range(batch.size)]
    rows = [r for _ in seeds for r in range(batch.size)]
    sub = batch.take(rows)
    backend, learner, adapter = build(arm, cell_seeds, [arm.config] * len(cell_seeds), device, _scales())
    run = run_dot(learner, adapter, sub)
    return run.errors, run.valid, backend.to_host(learner.failed)


def _scales() -> Any:
    from .adapters import Scales

    return Scales()


def parity(seeds: Sequence[int], families: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    arms = {"c1_champion1": CHAMPION_1, **CANDIDATE_TEMPLATES}
    out: dict[str, Any] = {}
    for name, arm in arms.items():
        per_family = {}
        for family, streams in families.items():
            batch = family_batch(family, list(streams))
            cpu = _run_cells(arm, "cpu", seeds, batch)
            cpu_again = _run_cells(arm, "cpu", seeds, batch)
            gpu = _run_cells(arm, "cuda:0", seeds, batch)
            gpu_again = _run_cells(arm, "cuda:0", seeds, batch)
            valid = cpu[1] & gpu[1]
            diff = np.where(valid, np.abs(cpu[0] - gpu[0]), 0.0)
            cpu_mae = np.where(cpu[1], cpu[0], 0).sum(1) / np.maximum(cpu[1].sum(1), 1)
            gpu_mae = np.where(gpu[1], gpu[0], 0).sum(1) / np.maximum(gpu[1].sum(1), 1)
            rel = np.abs(cpu_mae - gpu_mae) / np.maximum(cpu_mae, 1e-300)
            per_family[family] = {
                "cells": int(cpu[0].shape[0]),
                "bitwise_equal_cells_cpu_vs_cuda": int(
                    np.sum(np.all(np.where(valid, cpu[0] == gpu[0], True), axis=1))
                ),
                "max_abs_step_difference": float(diff.max()),
                "max_relative_mae_difference": float(rel.max()),
                "median_relative_mae_difference": float(np.median(rel)),
                "failed_agreement": bool(np.array_equal(cpu[2], gpu[2])),
                "cpu_run_to_run_bitwise_cells": int(
                    np.sum(np.all(np.where(cpu[1], cpu[0] == cpu_again[0], True), axis=1))
                ),
                "cuda_run_to_run_bitwise_cells": int(
                    np.sum(np.all(np.where(gpu[1], gpu[0] == gpu_again[0], True), axis=1))
                ),
            }
        out[name] = per_family
    return out


def qualify(registry: Any, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    from .identities import seeds_of

    seeds = seeds_of(dict(registry), "v2-qualification-init")
    streams = seeds_of(dict(registry), "v2-qualification-env.v1_occlusion")
    points = []
    for workload in WORKLOADS:
        for executor in EXECUTORS:
            sizes = SIZES + (
                CUDA_EXTRA if executor == "cuda" and workload in ("online", "prediction") else ()
            )
            for size in sizes:
                if executor == "cpu_par" and size < 32:
                    continue
                if workload == "external" and size > 1024:
                    continue
                point = time_point(workload, executor, size, seeds, streams)
                points.append(point)
                log(
                    f"{workload:10s} {executor:7s} {size:6d} cells: {point['cell_steps_per_second']:>12,.0f} cell-steps/s ({point['wall_seconds_min']:.2f}s)"
                )
    families = {
        f: seeds_of(dict(registry), f"v2-qualification-env.{f}")
        for f in ("v1_occlusion", "v1_coarse_speed", "long_coarse", "wall_gaps")
    }
    log("parity and determinism ...")
    parity_record = parity(seeds, families)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    children_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    return {
        "schema": "aaa.1k.v2.compute_qualification.v1",
        "cpu_backend": backend_provenance(resolve_backend("cpu")),
        "cuda_backend": backend_provenance(resolve_backend("cuda:0")),
        "sizes": list(SIZES),
        "repeats": REPEATS,
        "points": points,
        "transfer": transfer_overhead((32, 512, 4096, 16384), streams),
        "parity": parity_record,
        "peak_rss_bytes": {"parent": peak, "largest_child": children_peak},
        "config_note": "CUDA timings include the RTX 2060's concurrent desktop display load (Xorg/Cinnamon resident)",
    }


def crossover(points: Sequence[dict[str, Any]], workload: str, left: str, right: str) -> int | None:
    """Smallest measured cell count at which ``right`` beats ``left`` on ``workload``."""

    table: dict[str, dict[int, float]] = {}
    for p in points:
        if p["workload"] == workload:
            table.setdefault(p["executor"], {})[p["cells"]] = p["cell_steps_per_second"]
    common = sorted(set(table.get(left, {})) & set(table.get(right, {})))
    for size in common:
        if table[right][size] > table[left][size]:
            return size
    return None
