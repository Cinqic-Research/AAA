"""Capability designs for confirmation: online learning, adaptation and retention.

These are the round-3 designs (``research.aaa_1k.measurements``) in batched
form, with the same stream generators and parameters, so the v2 capability
vector is comparable in kind with Champion 1's round-3 vector.

Online learning (Q1)
    Every cell runs a trunk to a quarter of its stream, then continues as two
    bitwise-identical twins, one updating and one frozen. The effect is
    ``frozen MAE - online MAE`` after the branch.
Adaptation (Q2, difference of differences)
    ``paired_change_streams(seed, steps=240, change_step=120)``: the changed
    and control streams are bit-identical until step 120. One trunk runs the
    shared prefix and then continues four ways: changed/online,
    changed/frozen, control/online, control/frozen. The effect is
    ``(frozen - online)_changed - (frozen - online)_control``.
Retention (Q5)
    ``aba_stream(seed, segment_steps=200)``; after each segment a frozen copy
    of every cell, its interaction state reset, is scored on a fixed bank of
    eight held-out regime-A episodes (``motion_compat`` bouncing, 80 steps).
    Forgetting is ``probe after B - probe after A1`` (positive is forgetting).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from research.aaa_1k.streams import aba_stream, motion_compat_stream, paired_change_streams

from .adapters import DotAdapter, Scales, take_adapter
from .arms import ArmSpec
from .jobs import build
from .runner import StreamBatch, run_dot

PAIRED_STEPS = 240
CHANGE_STEP = 120
SEGMENT_STEPS = 200
PROBE_STEPS = 80


def _batch(streams: Sequence[Any]) -> StreamBatch:
    T = max(len(s.steps) for s in streams)
    truth = np.zeros((len(streams), T))
    observed = np.zeros((len(streams), T), dtype=bool)
    length = np.zeros(len(streams), dtype=int)
    for i, stream in enumerate(streams):
        n = len(stream.steps)
        truth[i, :n] = [s.true_position for s in stream.steps]
        observed[i, :n] = [s.observed for s in stream.steps]
        length[i] = n
    return StreamBatch(truth, observed, length, [s.stream_id for s in streams])


def adaptation(arm: ArmSpec, inits: Sequence[int], seeds: Sequence[int]) -> list[dict[str, Any]]:
    """Q2 difference-of-differences for every (init, paired stream) trial."""

    pairs = [paired_change_streams(seed, steps=PAIRED_STEPS, change_step=CHANGE_STEP) for seed in seeds]
    changed = _batch([p[0] for p in pairs])
    control = _batch([p[1] for p in pairs])
    trial_seeds = [init for init in inits for _ in seeds]
    rows = [row for _ in inits for row in range(len(seeds))]
    trunk_batch = changed.take(rows)
    backend, learner, adapter = build(arm, trial_seeds, [arm.config] * len(trial_seeds), "cpu", Scales())
    run_dot(learner, adapter, trunk_batch, stop=CHANGE_STEP)
    n = len(trial_seeds)
    index = list(range(n)) * 4
    configs = list(learner.configs) * 4
    quad = learner.take(index, configs)
    quad_adapter = take_adapter(adapter, index)
    hashes = quad.cell_state_hashes()
    if any(hashes[k * n + i] != hashes[i] for k in range(4) for i in range(n)):
        raise RuntimeError("paired-change continuations did not start from one identical state")
    streams = StreamBatch.stack(
        [changed.take(rows), changed.take(rows), control.take(rows), control.take(rows)]
    )
    update = np.asarray([True] * n + [False] * n + [True] * n + [False] * n)
    post = run_dot(quad, quad_adapter, streams, update=update, start=CHANGE_STEP, begin=False)
    mae = post.mae()
    failed = backend.to_host(quad.failed)
    out = []
    for i in range(n):
        values = [mae[k * n + i] for k in range(4)]
        ok = not any(failed[k * n + i] for k in range(4)) and all(np.isfinite(values))
        record: dict[str, Any] = {
            "init": trial_seeds[i],
            "stream_id": changed.stream_ids[rows[i]],
            "failed": not ok,
        }
        if ok:
            changed_advantage = values[1] - values[0]
            control_advantage = values[3] - values[2]
            record.update(
                changed_online=values[0],
                changed_frozen=values[1],
                control_online=values[2],
                control_frozen=values[3],
                changed_advantage=changed_advantage,
                control_advantage=control_advantage,
                difference_of_differences=changed_advantage - control_advantage,
            )
        out.append(record)
    return out


def retention(
    arm: ArmSpec, inits: Sequence[int], seeds: Sequence[int], probe_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    """Q5 probe-bank retention for every (init, A/B/A stream) trial."""

    streams = _batch([aba_stream(seed, segment_steps=SEGMENT_STEPS) for seed in seeds])
    bank = _batch(
        [motion_compat_stream("bouncing", seed, steps=PROBE_STEPS, change_step=None) for seed in probe_seeds]
    )
    trial_seeds = [init for init in inits for _ in seeds]
    rows = [row for _ in inits for row in range(len(seeds))]
    batch = streams.take(rows)
    backend, learner, adapter = build(arm, trial_seeds, [arm.config] * len(trial_seeds), "cpu", Scales())
    n = len(trial_seeds)
    P = bank.size
    probes: dict[str, np.ndarray] = {}
    start = 0
    for label, stop in (
        ("after_A1", SEGMENT_STEPS),
        ("after_B", 2 * SEGMENT_STEPS),
        ("after_A2", 3 * SEGMENT_STEPS),
    ):
        run_dot(learner, adapter, batch, start=start, stop=stop, begin=(start == 0))
        index = [i for i in range(n) for _ in range(P)]
        frozen_configs = [replace(learner.configs[i]) for i in index]
        probe_learner = learner.take(index, frozen_configs)
        probe_adapter = DotAdapter(backend, len(index), features=arm.features, target_rule=arm.target_rule)
        updates_before = backend.to_host(probe_learner.update_count).copy()
        probe_run = run_dot(probe_learner, probe_adapter, bank.take(list(range(P)) * n), update=False)
        probes[label] = probe_run.mae().reshape(n, P).mean(axis=1)
        if not np.array_equal(backend.to_host(probe_learner.update_count), updates_before):
            raise RuntimeError("a probe copy updated its weights; the probe is not read-only")
        start = stop
    failed = backend.to_host(learner.failed)
    out = []
    for i in range(n):
        record: dict[str, Any] = {
            "init": trial_seeds[i],
            "stream_id": streams.stream_ids[rows[i]],
            "failed": bool(failed[i]),
        }
        if not failed[i]:
            record.update({label: float(values[i]) for label, values in probes.items()})
            record["forgetting"] = float(probes["after_B"][i] - probes["after_A1"][i])
            record["recovery"] = float(probes["after_A2"][i] - probes["after_B"][i])
        out.append(record)
    return out
