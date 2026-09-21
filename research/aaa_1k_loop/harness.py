"""A deterministic, parallel cell runner on top of AAA-1K's causal loop.

Every scored transition still goes through :func:`research.aaa_1k.runner.run_stream`,
which is the single place the predict -> record -> advance -> reveal -> score
-> update order is enforced. This module only decides *which* (initialization,
stream, arm set) cells to run, runs independent cells in separate processes,
and reduces each cell to primitives before returning. Results are sorted by
cell key, so the output does not depend on the number of workers or on
scheduling order.

Instrumentation is read-only: :class:`InstrumentedAgent` extends the model's
``diagnostics()`` with the last update's gradient norm and clip scale, and the
runner reads it *after* the prediction and *before* the reveal, exactly where
AAA-1K's own ``collect_diagnostics`` reads it. It cannot feed anything back to
the learner.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.aaa_1k.agents import NeuralAgent
from research.aaa_1k.runner import RunResult, ScoredStep, run_stream
from research.aaa_1k.streams import Stream

from .dynamics import jacobian_statistics
from .evidence import finite_or_none


class InstrumentedAgent(NeuralAgent):
    """A :class:`NeuralAgent` whose diagnostics also report the last update."""

    def diagnostics(self) -> dict[str, float]:
        values = dict(self.model.diagnostics())
        values["last_gradient_norm"] = float(self.last_update.get("gradient_norm", float("nan")))
        values["last_clip_scale"] = float(self.last_update.get("clip_scale", float("nan")))
        values["hidden_norm"] = float(np.linalg.norm(getattr(self.model, "hidden", np.zeros(1))))
        values["previous_signed_error_input"] = float(self.previous_signed_error)
        values.update(jacobian_statistics(self.model))
        return values


@dataclass(frozen=True)
class Cell:
    """One (condition, initialization, stream) unit of a crossed design."""

    condition: str
    init_index: int
    init_seed: int
    stream: Stream

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.condition, self.init_index, self.stream.stream_id)


ArmFactory = Callable[[int], Sequence[Any]]
"""Build every agent for a cell from the cell's initialization seed."""

Reducer = Callable[[Cell, RunResult, Sequence[Any]], dict[str, Any]]


FAILURE_TYPES = (FloatingPointError, OverflowError, ValueError)


def _merge(results: Sequence[RunResult]) -> RunResult:
    """Combine single-agent runs over the same stream into one paired result.

    Agents never interact inside :func:`run_stream`, so running them one at a
    time over the same stream produces exactly the steps a joint run would.
    The merge checks that claim step by step instead of assuming it.
    """

    first = results[0]
    steps: list[ScoredStep] = []
    for index, step in enumerate(first.steps):
        merged = {
            field: {k: v for result in results for k, v in getattr(result.steps[index], field).items()}
            for field in (
                "predictions",
                "absolute_errors",
                "normalized_absolute_errors",
                "signed_errors",
                "error_estimates",
                "diagnostics",
            )
        }
        for result in results[1:]:
            other = result.steps[index]
            if (other.index, other.target_index, other.true_next_position) != (
                step.index,
                step.target_index,
                step.true_next_position,
            ):
                raise RuntimeError("isolated runs did not see the same stream")
        steps.append(
            ScoredStep(
                index=step.index,
                target_index=step.target_index,
                true_next_position=step.true_next_position,
                target_observed=step.target_observed,
                input_observed=step.input_observed,
                regime=step.regime,
                event=step.event,
                **merged,
            )
        )
    return RunResult(
        schema=first.schema,
        stream_summary=first.stream_summary,
        agent_names=[name for result in results for name in result.agent_names],
        steps=steps,
    )


def _run_one(payload: tuple[Cell, ArmFactory, Reducer, bool, bool]) -> dict[str, Any]:
    cell, factory, reducer, collect, isolate = payload
    agents = list(factory(cell.init_seed))
    failures: dict[str, str] = {}
    if isolate:
        results: list[RunResult] = []
        survivors: list[Any] = []
        for agent in agents:
            try:
                results.append(run_stream(cell.stream, [agent], collect_diagnostics=collect))
                survivors.append(agent)
            except FAILURE_TYPES as error:
                # A failure is evidence: it is recorded against the arm that
                # raised it and the arm is excluded from this cell's primitives.
                failures[agent.name] = f"{type(error).__name__}: {error}"
        if not results:
            raise RuntimeError(f"every arm failed on cell {cell.key}: {failures}")
        result = _merge(results)
        agents = survivors
    else:
        result = run_stream(cell.stream, agents, collect_diagnostics=collect)
    record = reducer(cell, result, agents)
    record["failures"] = failures
    record.setdefault("condition", cell.condition)
    record.setdefault("init_index", cell.init_index)
    record.setdefault("init_seed", cell.init_seed)
    record.setdefault("stream_id", cell.stream.stream_id)
    record.setdefault("family", cell.stream.family)
    record.setdefault("stream_seed", cell.stream.seed)
    return record


def default_workers() -> int:
    requested = os.environ.get("AAA_LOOP_WORKERS")
    if requested:
        return max(1, int(requested))
    return max(1, min(16, (os.cpu_count() or 1)))


def run_cells(
    cells: Sequence[Cell],
    factory: ArmFactory,
    reducer: Reducer,
    *,
    collect_diagnostics: bool = False,
    workers: int | None = None,
    isolate_failures: bool = False,
) -> list[dict[str, Any]]:
    """Run every cell and return reduced records in canonical key order.

    With ``isolate_failures`` each arm runs alone on the cell's stream, so a
    non-finite failure in one arm is recorded in ``record["failures"]`` instead
    of aborting the other arms. Without it, any failure aborts the run.
    """

    keys = [cell.key for cell in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("cells must have unique (condition, initialization, stream) keys")
    payloads = [(cell, factory, reducer, collect_diagnostics, isolate_failures) for cell in cells]
    count = workers or default_workers()
    if count == 1 or len(payloads) == 1:
        records = [_run_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=count) as pool:
            records = list(pool.map(_run_one, payloads, chunksize=1))
    return sorted(
        records, key=lambda record: (record["condition"], record["init_index"], record["stream_id"])
    )


def parallel_map(
    function: Callable[[Any], Any], payloads: Sequence[Any], *, workers: int | None = None
) -> list[Any]:
    """``[function(p) for p in payloads]`` in input order, across processes."""

    count = workers or default_workers()
    if count == 1 or len(payloads) <= 1:
        return [function(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=count) as pool:
        return list(pool.map(function, payloads, chunksize=1))


# ----------------------------------------------------------------------
# standard reductions
# ----------------------------------------------------------------------
def arm_errors(result: RunResult) -> dict[str, np.ndarray]:
    return {name: result.errors(name) for name in result.agent_names}


def mean_errors(result: RunResult) -> dict[str, float]:
    return {name: float(np.mean(values)) for name, values in arm_errors(result).items()}


def diagnostic_series(result: RunResult, arm: str, field: str) -> np.ndarray:
    return np.asarray(
        [step.diagnostics.get(arm, {}).get(field, math.nan) for step in result.steps], dtype=float
    )


def summary(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": None, "median": None, "q95": None, "max": None, "min": None}
    return {
        "mean": finite_or_none(float(np.mean(finite))),
        "median": finite_or_none(float(np.median(finite))),
        "q95": finite_or_none(float(np.quantile(finite, 0.95))),
        "max": finite_or_none(float(np.max(finite))),
        "min": finite_or_none(float(np.min(finite))),
    }


def matrix(records: Sequence[Mapping[str, Any]], arm: str, *, key: str = "mae") -> np.ndarray:
    """Reshape records into ``[initialization, stream]``; refuse a ragged crossing."""

    inits = sorted({record["init_index"] for record in records})
    streams = sorted({record["stream_id"] for record in records})
    lookup = {(record["init_index"], record["stream_id"]): record for record in records}
    if len(lookup) != len(records):
        raise ValueError("duplicate (initialization, stream) records cannot be flattened")
    if len(lookup) != len(inits) * len(streams):
        raise ValueError("records are not a complete initialization-by-stream crossing")
    return np.asarray([[lookup[(i, s)][key][arm] for s in streams] for i in inits], dtype=float)
