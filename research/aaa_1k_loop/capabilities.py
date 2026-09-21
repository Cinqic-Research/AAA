"""Q2 adaptation and Q5 retention for any registered gated arm.

These reproduce :mod:`research.aaa_1k.measurements` exactly -- same paired
streams, same branch construction and complete-state hash check, same fixed
probe bank, same frozen read-only probes -- with one difference: the agent is
built from the loop's arm registry, so the champion and a challenger are
measured by one code path from one seed. The AAA-1K functions construct the
champion's initialization internally and cannot build a challenger.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.aaa_1k.measurements import probe_bank
from research.aaa_1k.runner import run_online_frozen_branch, run_stream
from research.aaa_1k.streams import Stream, aba_stream, paired_change_streams

from .arms import ARMS

ADAPTATION_STEPS = 240
ADAPTATION_CHANGE_STEP = 120
RETENTION_SEGMENT_STEPS = 200


def adaptation_trial(arm: str, *, env_seed: int, init_seed: int) -> dict[str, Any]:
    """One difference-of-differences trial, as in round 3."""

    changed, control = paired_change_streams(
        env_seed, steps=ADAPTATION_STEPS, change_step=ADAPTATION_CHANGE_STEP
    )
    advantages: dict[str, dict[str, float]] = {}
    hashes: dict[str, str] = {}
    for label, stream in (("changed", changed), ("control", control)):
        trunk = ARMS[arm].build(init_seed)
        outcome = run_online_frozen_branch(stream, trunk, branch_index=ADAPTATION_CHANGE_STEP)
        advantages[label] = outcome.post_branch_advantage()
        hashes[label] = outcome.clone_state_hash
    if hashes["changed"] != hashes["control"]:
        raise RuntimeError("paired trunks diverged before the branch")
    return {
        "arm": arm,
        "env_seed": env_seed,
        "init_seed": init_seed,
        "trunk_state_hash": hashes["changed"],
        "changed": advantages["changed"],
        "control": advantages["control"],
        "adaptation_effect": advantages["changed"]["absolute_advantage"]
        - advantages["control"]["absolute_advantage"],
        "continued_learning_effect": advantages["control"]["absolute_advantage"],
    }


def _probe(agent: Any, bank: list[Stream], label: str) -> float:
    errors = []
    for index, stream in enumerate(bank):
        clone = agent.branch(name=f"probe_{label}_{index}", update_enabled=False)
        result = run_stream(stream, [clone], begin_episode=True)
        if clone.model.update_count != agent.model.update_count:
            raise RuntimeError("a probe clone updated its weights; the probe is not read-only")
        errors.append(result.mean_absolute_error(clone.name))
    return float(np.mean(errors))


def retention_trial(
    arm: str, *, env_seed: int, init_seed: int, bank: list[Stream] | None = None
) -> dict[str, Any]:
    """One A/B/A stream probed against the fixed AAA-1K probe bank, as in round 3."""

    probes_bank = bank if bank is not None else probe_bank()
    stream = aba_stream(env_seed, segment_steps=RETENTION_SEGMENT_STEPS)
    agent = ARMS[arm].build(init_seed)
    probes: dict[str, float] = {}
    start = 0
    for label, stop in zip(
        ("after_A1", "after_B", "after_A2"),
        (RETENTION_SEGMENT_STEPS, 2 * RETENTION_SEGMENT_STEPS, 3 * RETENTION_SEGMENT_STEPS),
        strict=True,
    ):
        run_stream(stream, [agent], start=start, stop=stop, begin_episode=(start == 0))
        probes[label] = _probe(agent, probes_bank, label)
        start = stop
    return {
        "arm": arm,
        "env_seed": env_seed,
        "init_seed": init_seed,
        "probe_error": probes,
        "forgetting": probes["after_B"] - probes["after_A1"],
        "reacquisition_gap": probes["after_A2"] - probes["after_A1"],
    }


def adaptation_task(payload: tuple[str, int, int]) -> dict[str, Any]:
    arm, env_seed, init_seed = payload
    return adaptation_trial(arm, env_seed=env_seed, init_seed=init_seed)


def retention_task(payload: tuple[str, int, int]) -> dict[str, Any]:
    arm, env_seed, init_seed = payload
    return retention_trial(arm, env_seed=env_seed, init_seed=init_seed)


def trial_matrix(trials: list[dict[str, Any]], arm: str, key: str) -> np.ndarray:
    """``[initialization, environment]``; refuses an incomplete crossing."""

    rows = [trial for trial in trials if trial["arm"] == arm]
    inits = sorted({trial["init_seed"] for trial in rows})
    envs = sorted({trial["env_seed"] for trial in rows})
    lookup = {(trial["init_seed"], trial["env_seed"]): trial for trial in rows}
    if len(lookup) != len(rows) or len(lookup) != len(inits) * len(envs):
        raise ValueError("trials are not a complete initialization-by-environment crossing")
    return np.asarray([[lookup[(i, e)][key] for e in envs] for i in inits], dtype=float)
