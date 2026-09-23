"""Jobs: a picklable description of "run these cells of this arm on these streams".

A job is executed by :func:`execute`, in the calling process, in a CPU worker
process (:func:`run_jobs` with ``workers > 1``), or on CUDA. Whatever the
executor, the returned primitives are per cell and host-side, and the result
records the backend it actually ran on.

Per-cell primitives for a dot job (all over *valid* scored steps):

``mae``                  mean normalized absolute error
``mae_hidden``           restricted to targets the cell was never shown
``mae_by``               per evaluator label code (e.g. regime or segment)
``failed`` / ``fail_step``  non-finite failure and when it happened
``clip_rate``            fraction of updates the gradient clip scaled
``trained_steps``        updates applied
``mirror_steps`` / ``gated_steps``  target-unfolding diagnostics
``error_head``           rank correlation, calibration slope and bias of the
                         error estimate against realized error (observed steps)

A *branch* job runs a trunk to ``branch_at`` and then continues two
bitwise-identical copies of every cell, one online and one frozen; it returns
the post-branch MAE of each.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aaa.compute import backend_provenance, resolve_backend

from .adapters import DotAdapter, Scales, SeriesAdapter
from .arms import ArmSpec
from .engine import BatchedLearner, CellConfig
from .external.tasks import SeriesProblem, problem_metrics
from .parallel import chunks
from .runner import StreamBatch, branch, forecast_series, run_dot, run_series


@dataclass(frozen=True)
class DotJob:
    arm: ArmSpec
    seeds: tuple[int, ...]
    batch: StreamBatch
    configs: tuple[CellConfig, ...] | None = None
    device: str = "cpu"
    branch_at: int | None = None
    label_keys: tuple[str, ...] = ()
    scales: Scales = field(default_factory=Scales)
    tag: str = ""

    def cell_configs(self) -> tuple[CellConfig, ...]:
        return self.configs if self.configs is not None else tuple([self.arm.config] * len(self.seeds))


@dataclass(frozen=True)
class SeriesJob:
    """Cells of one arm on external series problems (one problem per cell)."""

    arm: ArmSpec
    seeds: tuple[int, ...]
    problems: tuple[SeriesProblem, ...]
    configs: tuple[CellConfig, ...] | None = None
    device: str = "cpu"
    tag: str = ""

    def cell_configs(self) -> tuple[CellConfig, ...]:
        return self.configs if self.configs is not None else tuple([self.arm.config] * len(self.seeds))


def build_learner(arm: ArmSpec, seeds: Sequence[int], configs: Sequence[CellConfig], device: str) -> Any:
    """The arm's learner for these cells; ``keep_bias="chrono"`` expands per seed."""

    backend = resolve_backend(device)
    core = arm.build_core()
    init = arm.init_options()
    options = []
    for seed in seeds:
        option = dict(init)
        if option.get("keep_bias") == "chrono":
            from .arms import chrono_keep_bias

            option["keep_bias"] = chrono_keep_bias(core.hidden, seed)
        options.append(option)
    learner = BatchedLearner(
        core,
        backend,
        seeds=list(seeds),
        configs=list(configs),
        tbptt_steps=max(arm.tbptt_steps, 1),
        rule=arm.rule,
        init_options=options,
    )
    return backend, learner


def build(
    arm: ArmSpec, seeds: Sequence[int], configs: Sequence[CellConfig], device: str, scales: Scales
) -> Any:
    backend, learner = build_learner(arm, seeds, configs, device)
    adapter = DotAdapter(
        backend, len(seeds), scales=scales, features=arm.features, target_rule=arm.target_rule
    )
    return backend, learner, adapter


def execute_series(job: SeriesJob) -> dict[str, Any]:
    """Online pass (and recursive forecast where defined) for every cell's problem."""

    started = time.perf_counter()
    configs = job.cell_configs()
    backend, learner = build_learner(job.arm, job.seeds, configs, job.device)
    problems = job.problems
    B = len(problems)
    T = max(p.length for p in problems)
    inputs = np.zeros((B, T))
    targets = np.zeros((B, T))
    valid = np.zeros((B, T), dtype=bool)
    for b, problem in enumerate(problems):
        inputs[b, : problem.length] = problem.inputs
        targets[b, : problem.length] = problem.targets
        valid[b, : problem.length] = True
    modes = {p.target_mode for p in problems}
    if len(modes) != 1:
        raise ValueError("a series job must share one target mode")

    def column(key: str) -> np.ndarray:
        return np.asarray([p.scales[key] for p in problems], dtype=float)

    adapter = SeriesAdapter(
        backend,
        B,
        center=column("center"),
        level_scale=column("level_scale"),
        delta_scale=column("delta_scale"),
        target_center=column("target_center"),
        target_scale=column("target_scale"),
        target_mode=modes.pop(),
        features=job.arm.features,
    )
    run = run_series(learner, adapter, inputs, targets, valid, update=job.arm.update_enabled)
    horizons = {None if p.forecast_truth is None else p.forecast_truth.size for p in problems}
    forecast = None
    if len(horizons) == 1 and None not in horizons and all(p.length == T for p in problems):
        horizon = horizons.pop()
        assert horizon is not None
        last = np.asarray([p.last_value for p in problems], dtype=float)
        forecast = forecast_series(learner, adapter, horizon, last)
    elif None not in horizons:
        raise ValueError("forecast problems in one job must share their length and horizon")
    failed = backend.to_host(learner.failed)
    fail_step = backend.to_host(learner.fail_step)
    cells = []
    for b, problem in enumerate(problems):
        if failed[b]:
            record: dict[str, Any] = {
                "series_id": problem.series_id,
                "failed": True,
                "fail_step": int(fail_step[b]),
            }
        else:
            record = problem_metrics(
                problem, run.predictions[b, : problem.length], None if forecast is None else forecast[b]
            )
            record["failed"] = False
        record["seed"] = int(job.seeds[b])
        cells.append(record)
    backend.synchronize()
    return {
        "arm": job.arm.name,
        "tag": job.tag,
        "cells": cells,
        "seconds": time.perf_counter() - started,
        "cell_steps": int(valid.sum()),
        "backend": backend.resolved,
    }


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 3:
        return float("nan")

    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=float)
        ranks[order] = np.arange(values.size, dtype=float)
        # average ties
        sorted_values = values[order]
        start = 0
        for index in range(1, values.size + 1):
            if index == values.size or sorted_values[index] != sorted_values[start]:
                ranks[order[start:index]] = (start + index - 1) / 2.0
                start = index
        return ranks

    a, b = rank(first), rank(second)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def error_head_summary(estimates: np.ndarray, realized: np.ndarray) -> dict[str, float | None]:
    if estimates.size < 3 or not np.all(np.isfinite(estimates)):
        return {"spearman": None, "slope": None, "bias": None}
    rho = spearman(estimates, realized)
    variance = float(np.var(estimates))
    slope = float(np.cov(estimates, realized, bias=True)[0, 1] / variance) if variance > 0 else float("nan")
    bias = float(np.mean(estimates - realized))
    return {
        "spearman": rho if np.isfinite(rho) else None,
        "slope": slope if np.isfinite(slope) else None,
        "bias": bias,
    }


def reduce_dot(
    run: Any, learner: BatchedLearner, adapter: DotAdapter, batch: StreamBatch, job: DotJob
) -> list[dict[str, Any]]:
    host = learner.backend.to_host
    failed = host(learner.failed)
    fail_step = host(learner.fail_step)
    clips = host(learner.clip_events)
    updates = host(learner.update_count)
    trained = host(adapter.trained_steps)
    mirror = host(adapter.mirror_steps)
    gated = host(adapter.gated_steps)
    mae = run.mae()
    hidden = run.mae(~run.target_observed)
    scale = adapter.scales.displacement_scale / adapter.scales.width
    records = []
    for b in range(learner.size):
        use = run.valid[b] & run.target_observed[b]
        est = run.error_estimates[b][use] * scale
        realized = run.errors[b][use]
        record: dict[str, Any] = {
            "stream_id": batch.stream_ids[b],
            "seed": int(learner.seeds[b]),
            "mae": float(mae[b]) if np.isfinite(mae[b]) else None,
            "mae_hidden": float(hidden[b]) if np.isfinite(hidden[b]) else None,
            "failed": bool(failed[b]),
            "fail_step": int(fail_step[b]),
            "valid_steps": int(run.valid[b].sum()),
            "clip_rate": float(clips[b] / updates[b]) if updates[b] else 0.0,
            "trained_steps": int(trained[b]),
            "mirror_steps": int(mirror[b]),
            "gated_steps": int(gated[b]),
            "error_head": error_head_summary(est, realized) if learner.core.outputs > 1 else None,
        }
        by: dict[str, dict[str, float | None]] = {}
        for key in job.label_keys:
            codes = batch.labels[key][b, run.start + 1 : run.start + 1 + run.errors.shape[1]]
            entry = {}
            for code, word in enumerate(batch.vocab[key]):
                mask = run.valid[b] & (codes == code)
                entry[word] = float(run.errors[b][mask].mean()) if mask.any() else None
            by[key] = entry
        record["mae_by"] = by
        records.append(record)
    return records


def execute(job: DotJob) -> dict[str, Any]:
    """Run one dot job and return per-cell primitives plus timing and platform."""

    configs = job.cell_configs()
    started = time.perf_counter()
    backend, learner, adapter = build(job.arm, job.seeds, configs, job.device, job.scales)
    if job.branch_at is None:
        run = run_dot(learner, adapter, job.batch, update=job.arm.update_enabled)
        cells = reduce_dot(run, learner, adapter, job.batch, job)
    else:
        run_dot(learner, adapter, job.batch, update=True, stop=job.branch_at)
        twin, twin_adapter, mask = branch(learner, adapter)
        doubled = job.batch.take(list(range(job.batch.size)) * 2)
        post = run_dot(twin, twin_adapter, doubled, update=mask, start=job.branch_at, begin=False)
        post_cells = reduce_dot(post, twin, twin_adapter, doubled, job)
        n = job.batch.size
        cells = []
        for b in range(n):
            cells.append({"online": post_cells[b], "frozen": post_cells[n + b]})
    backend.synchronize()
    return {
        "arm": job.arm.name,
        "tag": job.tag,
        "cells": cells,
        "seconds": time.perf_counter() - started,
        "cell_steps": int(job.batch.length.sum() - job.batch.size) * (2 if job.branch_at is not None else 1),
        "backend": backend.resolved,
    }


def execute_any(job: Any) -> dict[str, Any]:
    return execute_series(job) if isinstance(job, SeriesJob) else execute(job)


def split(job: Any, parts: int) -> list[Any]:
    if isinstance(job, SeriesJob):
        configs = job.cell_configs()
        return [
            SeriesJob(
                arm=job.arm,
                seeds=tuple(job.seeds[i] for i in piece),
                problems=tuple(job.problems[i] for i in piece),
                configs=tuple(configs[i] for i in piece),
                device=job.device,
                tag=job.tag,
            )
            for piece in chunks(len(job.seeds), parts)
        ]
    return split_dot(job, parts)


def split_dot(job: DotJob, parts: int) -> list[DotJob]:
    """Split a job into contiguous per-cell chunks (a cell's arithmetic is unchanged by chunking)."""

    configs = job.cell_configs()
    out = []
    for piece in chunks(len(job.seeds), parts):
        index = list(piece)
        out.append(
            DotJob(
                arm=job.arm,
                seeds=tuple(job.seeds[i] for i in index),
                batch=job.batch.take(index),
                configs=tuple(configs[i] for i in index),
                device=job.device,
                branch_at=job.branch_at,
                label_keys=job.label_keys,
                scales=job.scales,
                tag=job.tag,
            )
        )
    return out


DEFAULT_CHUNK = 64
"""Cells per CPU work unit: measured optimum on FLOWBOX (a 64-cell chunk's arrays stay cache-resident)."""


def run_jobs(
    jobs: Sequence[Any], *, workers: int | str | None = 1, max_cells_per_chunk: int = DEFAULT_CHUNK
) -> list[dict[str, Any]]:
    """Execute jobs, returning results in job order.

    CPU jobs are split into chunks of at most ``max_cells_per_chunk`` cells and
    run by single-threaded worker processes; CUDA jobs run in this process
    *concurrently* with the CPU chunks. A study that compares arms must give
    all of them the same device: this function runs whatever it is given and
    never moves a job between backends.
    """

    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    from .parallel import THREAD_VARIABLES, resolve_workers

    count = resolve_workers(workers)
    pieces: list[DotJob] = []
    owners: list[int] = []
    for index, job in enumerate(jobs):
        if job.device == "cpu":
            parts = max(1, -(-len(job.seeds) // max_cells_per_chunk))
            for piece in split(job, parts):
                pieces.append(piece)
                owners.append(index)
    outputs: list[dict[str, Any]] = []
    gpu_results: dict[int, dict[str, Any]] = {}
    if count == 1 or len(pieces) <= 1:
        outputs = [execute_any(piece) for piece in pieces]
        for index, job in enumerate(jobs):
            if job.device != "cpu":
                gpu_results[index] = execute_any(job)
    else:
        saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
        for name in THREAD_VARIABLES:
            os.environ[name] = "1"
        try:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(count, len(pieces)), mp_context=context) as pool:
                futures = [pool.submit(execute_any, piece) for piece in pieces]
                for index, job in enumerate(jobs):
                    if job.device != "cpu":
                        gpu_results[index] = execute_any(job)
                outputs = [future.result() for future in futures]
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    merged: dict[int, dict[str, Any]] = dict(gpu_results)
    for owner, output in zip(owners, outputs, strict=True):
        if owner not in merged:
            merged[owner] = {**output, "cells": list(output["cells"]), "pieces": 1}
        else:
            merged[owner]["cells"].extend(output["cells"])
            merged[owner]["seconds"] += output["seconds"]
            merged[owner]["cell_steps"] += output["cell_steps"]
            merged[owner]["pieces"] += 1
    return [merged[index] for index in range(len(jobs))]


def platform_record(device: str) -> dict[str, Any]:
    return backend_provenance(resolve_backend(device))
