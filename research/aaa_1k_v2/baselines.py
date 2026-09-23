"""Analytic and incumbent baselines on the v2 streams, using the historical agents unchanged.

The agents are ``research.aaa_1k.agents``' own classes -- persistence,
reflected constant motion, dead reckoning, the eight-point windowed linear fit
and the unmodified v2.1 RLS candidate -- driven through the same causal order
as ``research.aaa_1k.runner.run_stream``: predict, score against latent truth,
then reveal the permitted observation. Baselines do not depend on a model
initialization, so each stream is run once.

Persistence's per-stream MAE also defines divergence for the neural arms
(``plan.DIVERGENCE_FACTOR``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from research.aaa_1k.agents import (
    ConstantMotionAgent,
    DeadReckoningAgent,
    PersistenceAgent,
    RLSAgent,
    WindowedLinearFitAgent,
)

from .parallel import chunks, parallel_map
from .runner import StreamBatch

BASELINE_NAMES = ("persistence", "constant_motion_reflected", "dead_reckoning", "linear_fit", "rls_online")


def _agents() -> list[Any]:
    return [
        PersistenceAgent(),
        ConstantMotionAgent(name="constant_motion_reflected", reflect=True),
        DeadReckoningAgent(),
        WindowedLinearFitAgent(),
        RLSAgent(name="rls_online"),
    ]


def run_baselines(batch: StreamBatch) -> list[dict[str, Any]]:
    """Per-stream MAE (all valid steps and hidden-target steps) for every baseline."""

    out = []
    scoring = batch.scoring
    for b in range(batch.size):
        n = int(batch.length[b])
        agents = _agents()
        for agent in agents:
            agent.begin_episode()
            agent.accept_observation(float(batch.truth[b, 0]))
        errors = {agent.name: np.zeros(n - 1) for agent in agents}
        for t in range(n - 1):
            predictions = {agent.name: float(agent.predict()) for agent in agents}
            truth = float(scoring[b, t + 1])
            for name, value in predictions.items():
                errors[name][t] = abs(value - truth)
            revealed = float(batch.truth[b, t + 1]) if batch.observed[b, t + 1] else None
            for agent in agents:
                agent.accept_observation(revealed)
        hidden = ~batch.observed[b, 1:n]
        record: dict[str, Any] = {"stream_id": batch.stream_ids[b]}
        for name, values in errors.items():
            record[name] = float(values.mean())
            record[f"{name}:hidden"] = float(values[hidden].mean()) if hidden.any() else None
        out.append(record)
    return out


def baselines_parallel(
    batch: StreamBatch, *, workers: int | str | None = 1, chunk: int = 16
) -> list[dict[str, Any]]:
    pieces = [batch.take(list(piece)) for piece in chunks(batch.size, max(1, -(-batch.size // chunk)))]
    results = parallel_map(run_baselines, pieces, workers=workers)
    return [record for piece in results for record in piece]


def persistence_mae(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {record["stream_id"]: float(record["persistence"]) for record in records}
