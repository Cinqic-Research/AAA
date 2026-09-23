"""Process-level parallelism for CPU cells, with BLAS oversubscription prevented.

Cells are independent, so a large CPU batch can be split into contiguous
chunks, each run by a single-threaded worker process. Worker processes are
started with the ``spawn`` method *after* the BLAS/OpenMP thread variables are
set to 1, so ``W`` workers use ``W`` threads rather than ``W x cores``.

Chunking changes nothing about a cell's arithmetic: every cell's operations
are elementwise across the batch or per-cell matrix products, so a cell
computed in a chunk of 64 is bitwise identical to the same cell computed in a
batch of 4096 (tested). ``workers="auto"`` uses the physical core count.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

T = TypeVar("T")
R = TypeVar("R")


def physical_cores() -> int:
    """Physical cores from ``/proc/cpuinfo`` (falls back to ``os.cpu_count() // 2``)."""

    try:
        pairs = set()
        physical = core = None
        with Path("/proc/cpuinfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("physical id"):
                    physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip() and physical is not None and core is not None:
                    pairs.add((physical, core))
                    physical = core = None
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def resolve_workers(workers: int | str | None) -> int:
    if workers is None or workers == "auto":
        return physical_cores()
    if isinstance(workers, str):
        if not workers.isdigit():
            raise ValueError(f"workers must be a positive integer or 'auto', got {workers!r}")
        workers = int(workers)
    if isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer or 'auto'")
    return int(workers)


def _pin_threads() -> None:
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"


def chunks(count: int, parts: int) -> list[range]:
    """Split ``range(count)`` into ``parts`` contiguous, near-equal ranges (empty ones dropped)."""

    parts = max(1, min(parts, count))
    base, extra = divmod(count, parts)
    out, start = [], 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        out.append(range(start, start + size))
        start += size
    return [r for r in out if len(r)]


def parallel_map(function: Callable[[T], R], jobs: Sequence[T], *, workers: int | str | None = 1) -> list[R]:
    """Map ``function`` over ``jobs`` in single-threaded worker processes, preserving order."""

    count = resolve_workers(workers)
    if count == 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    _pin_threads()
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(count, len(jobs)), mp_context=context) as pool:
            return list(pool.map(function, jobs))
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def describe_workers(workers: int | str | None) -> dict[str, Any]:
    return {
        "requested": workers,
        "resolved": resolve_workers(workers),
        "physical_cores": physical_cores(),
        "logical_threads": os.cpu_count(),
        "blas_threads_per_worker": 1,
        "start_method": "spawn",
    }
