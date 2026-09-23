"""The causal loop for a batch of cells, and the online/frozen branch construction.

Order for every scored transition, identical to ``research.aaa_1k.runner``:

1. every cell holds only causally available observations;
2. every cell predicts;
3. predictions are recorded;
4. the stream advances;
5. only the permitted observation is revealed (``observed`` flag per cell);
6. the pre-reveal prediction is scored against latent truth;
7. only then may an enabled cell update;
8. the revealed value, or a hold, enters the cell's own state.

A :class:`StreamBatch` is host-side evaluator data: latent truth, the
``observed`` mask and evaluator-only labels. Only ``truth[:, t+1]`` masked by
``observed[:, t+1]`` ever reaches an adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aaa.compute import Backend

from .adapters import DotAdapter, SeriesAdapter, take_adapter
from .engine import BatchedLearner, CellConfig


@dataclass
class StreamBatch:
    """Padded streams, one per cell. ``length[b]`` counts steps including the seed.

    ``truth`` holds the values a cell is *shown* when a step is observed (the
    observation, which for a noisy family is corrupted). ``score`` holds what
    predictions are scored against (latent truth); it defaults to ``truth``.
    """

    truth: np.ndarray
    observed: np.ndarray
    length: np.ndarray
    stream_ids: list[str]
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)
    score: np.ndarray | None = None

    @property
    def scoring(self) -> np.ndarray:
        return self.truth if self.score is None else self.score

    def __post_init__(self) -> None:
        B, T = self.truth.shape
        if self.observed.shape != (B, T) or self.length.shape != (B,) or len(self.stream_ids) != B:
            raise ValueError("stream batch arrays disagree in shape")
        if not np.all(self.observed[np.arange(B), 0]):
            raise ValueError("the seed observation must always be shown")
        if np.any(self.length < 3) or np.any(self.length > T):
            raise ValueError("stream lengths must be in [3, padded length]")
        valid = np.arange(T)[None, :] < self.length[:, None]
        if not np.all(np.isfinite(self.truth[valid])):
            raise ValueError("stream truth must be finite")
        if self.score is not None and (
            self.score.shape != (B, T) or not np.all(np.isfinite(self.score[valid]))
        ):
            raise ValueError("stream scoring truth must match the batch and be finite")

    @property
    def size(self) -> int:
        return int(self.truth.shape[0])

    def take(self, indices: Sequence[int]) -> StreamBatch:
        idx = np.asarray(indices, dtype=np.int64)
        return StreamBatch(
            truth=self.truth[idx],
            observed=self.observed[idx],
            length=self.length[idx],
            stream_ids=[self.stream_ids[i] for i in idx],
            labels={key: value[idx] for key, value in self.labels.items()},
            vocab=dict(self.vocab),
            score=None if self.score is None else self.score[idx],
        )

    @staticmethod
    def stack(parts: Sequence[StreamBatch]) -> StreamBatch:
        T = max(part.truth.shape[1] for part in parts)

        def pad(array: np.ndarray, fill: Any) -> np.ndarray:
            out = np.full((array.shape[0], T), fill, dtype=array.dtype)
            out[:, : array.shape[1]] = array
            return out

        labels: dict[str, np.ndarray] = {}
        vocab: dict[str, list[str]] = {}
        keys = set().union(*(part.labels for part in parts))
        for key in sorted(keys):
            merged: list[str] = []
            for part in parts:
                for word in part.vocab.get(key, []):
                    if word not in merged:
                        merged.append(word)
            vocab[key] = merged
            rows = []
            for part in parts:
                codes = part.labels.get(key)
                local = part.vocab.get(key, [])
                if codes is None:
                    rows.append(np.full((part.size, T), -1, dtype=np.int16))
                    continue
                remap = np.asarray([merged.index(word) for word in local] + [-1], dtype=np.int16)
                rows.append(pad(remap[codes], -1))
            labels[key] = np.concatenate(rows)
        return StreamBatch(
            truth=np.concatenate([pad(p.truth, 0.0) for p in parts]),
            observed=np.concatenate([pad(p.observed, False) for p in parts]),
            length=np.concatenate([p.length for p in parts]),
            stream_ids=[sid for p in parts for sid in p.stream_ids],
            labels=labels,
            vocab=vocab,
            score=None
            if all(p.score is None for p in parts)
            else np.concatenate([pad(p.scoring, 0.0) for p in parts]),
        )


@dataclass
class DotRun:
    """Per-step primitives of one pass (host arrays, ``(B, steps)``)."""

    errors: np.ndarray
    error_estimates: np.ndarray
    predictions: np.ndarray
    valid: np.ndarray
    target_observed: np.ndarray
    start: int

    def mae(self, mask: np.ndarray | None = None) -> np.ndarray:
        use = self.valid if mask is None else (self.valid & mask)
        counts = use.sum(axis=1)
        sums = np.where(use, self.errors, 0.0).sum(axis=1)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def run_dot(
    learner: BatchedLearner,
    adapter: DotAdapter,
    batch: StreamBatch,
    *,
    update: Any = True,
    start: int = 0,
    stop: int | None = None,
    begin: bool = True,
) -> DotRun:
    """Run every cell over its own stream from ``start`` to ``stop`` (exclusive target index)."""

    backend: Backend = learner.backend
    xp = backend.xp
    B, T = batch.truth.shape
    if learner.size != B or adapter.size != B:
        raise ValueError("learner, adapter and batch sizes differ")
    last = T - 1 if stop is None else min(int(stop), T - 1)
    if not 0 <= start < last:
        raise ValueError("start must leave at least one scored transition")
    update_mask = xp.asarray(np.broadcast_to(np.asarray(update, dtype=bool), (B,)).copy())
    truth = backend.asarray(batch.truth)
    scoring = truth if batch.score is None else backend.asarray(batch.score)
    observed = backend.asarray(batch.observed, dtype=bool)
    length = batch.length
    width = adapter.scales.width
    steps = last - start
    errors = xp.zeros((B, steps))
    estimates = xp.zeros((B, steps))
    predictions = xp.zeros((B, steps))
    if begin:
        learner.reset_state()
        adapter.begin()
        adapter.accept(None, truth[:, start], xp.ones(B, dtype=bool), update_mask)
    first_step = learner.step_index
    with np.errstate(all="ignore"):  # failures are tracked per cell, not by warnings
        for offset, index in enumerate(range(start, last)):
            scored = adapter.predict(learner)
            target = truth[:, index + 1]
            errors[:, offset] = xp.abs(scored - scoring[:, index + 1]) / width
            estimates[:, offset] = adapter.error_estimate
            predictions[:, offset] = scored
            live = observed[:, index + 1]
            adapter.accept(learner, target, live, update_mask)
            learner.advance_clock()
    host_errors = backend.to_host(errors)
    step_index = np.arange(start + 1, last + 1)[None, :]
    valid = step_index < length[:, None]
    failed = backend.to_host(learner.failed)
    fail_step = backend.to_host(learner.fail_step)
    # a failed cell has no primitive from its failure step on
    valid &= ~(failed[:, None] & ((first_step + np.arange(steps))[None, :] >= fail_step[:, None]))
    return DotRun(
        errors=host_errors,
        error_estimates=backend.to_host(estimates),
        predictions=backend.to_host(predictions),
        valid=valid,
        target_observed=batch.observed[:, start + 1 : last + 1],
        start=start,
    )


def branch(
    learner: BatchedLearner, adapter: Any, *, frozen_configs: Sequence[CellConfig] | None = None
) -> tuple[BatchedLearner, Any, np.ndarray]:
    """Duplicate every cell: first half online, second half frozen. Returns the update mask.

    Both halves start from bitwise-identical complete state (checked by hash).
    """

    n = learner.size
    indices = list(range(n)) * 2
    configs = list(learner.configs) + list(frozen_configs if frozen_configs is not None else learner.configs)
    twin = learner.take(indices, configs)
    twin_adapter = take_adapter(adapter, indices)
    hashes = twin.cell_state_hashes()
    if hashes[:n] != hashes[n:]:
        raise RuntimeError("online and frozen twins do not share an identical starting state")
    mask = np.array([True] * n + [False] * n)
    return twin, twin_adapter, mask


@dataclass
class SeriesRun:
    predictions: np.ndarray
    targets: np.ndarray
    valid: np.ndarray
    error_estimates: np.ndarray


def run_series(
    learner: BatchedLearner,
    adapter: SeriesAdapter,
    inputs: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    *,
    update: Any = True,
    begin: bool = True,
) -> SeriesRun:
    """Online prequential pass: at step ``t`` see ``inputs[:, t]``, predict ``targets[:, t]``, then learn.

    For one-step forecasting ``targets[:, t] = series[:, t + 1]`` and
    ``inputs[:, t] = series[:, t]``. ``valid`` marks real (non-padding) steps.
    """

    backend = learner.backend
    xp = backend.xp
    B, T = inputs.shape
    if targets.shape != (B, T) or valid.shape != (B, T):
        raise ValueError("inputs, targets and valid must share a shape")
    update_mask = xp.asarray(np.broadcast_to(np.asarray(update, dtype=bool), (B,)).copy())
    x = backend.asarray(inputs)
    y = backend.asarray(targets)
    v = backend.asarray(valid, dtype=bool)
    predictions = xp.zeros((B, T))
    estimates = xp.zeros((B, T))
    if begin:
        learner.reset_state()
        adapter.begin()
    with np.errstate(all="ignore"):  # failures are tracked per cell, not by warnings
        for t in range(T):
            adapter.observe_input(x[:, t], v[:, t])
            predictions[:, t] = adapter.predict(learner)
            estimates[:, t] = adapter.error_estimate
            adapter.reveal_target(learner, y[:, t], v[:, t], update_mask)
            learner.advance_clock()
    return SeriesRun(
        predictions=backend.to_host(predictions),
        targets=targets,
        valid=valid,
        error_estimates=backend.to_host(estimates),
    )


def forecast_series(
    learner: BatchedLearner, adapter: SeriesAdapter, horizon: int, last_value: np.ndarray
) -> np.ndarray:
    """Recursive multi-step forecast after an online pass; weights frozen, own predictions fed back.

    The last known value is observed first, so the first forecast uses the
    genuinely revealed previous error. After that each prediction is fed back
    as an observed input, the previous-error input is zero because nothing is
    revealed, and no cell updates.
    """

    backend = learner.backend
    xp = backend.xp
    B = learner.size
    out = xp.zeros((B, horizon))
    yes = xp.ones(B, dtype=bool)
    with np.errstate(all="ignore"):
        adapter.observe_input(backend.asarray(last_value), yes)
        for step in range(horizon):
            prediction = adapter.predict(learner)
            out[:, step] = prediction
            adapter.previous_signed_error = xp.zeros(B)
            adapter.has_prediction = xp.zeros(B, dtype=bool)
            adapter.observe_input(prediction, yes)
            learner.advance_clock()
    return backend.to_host(out)
