"""Running agents through the ``aaa.python.v1`` causal boundary.

Every task, for every agent, goes through :class:`~.episode.ToolEnvironment`:

    begin_task -> present -> [tool, if the agent uses it and the task is repair]
               -> act -> commit -> reveal -> learn (only if permitted)

A record per action is returned. Nothing here sees an answer key except the
environment's own scoring, which happens after the commit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_python.generator import Task
from research.aaa_python.learners import Agent

from .episode import ToolEnvironment


def tool_vector(results: Sequence[Sequence[bool]]) -> np.ndarray:
    """Per candidate: fraction of visible tests passed, and whether all passed (2 dense inputs)."""

    return np.array([[sum(r) / len(r), float(all(r))] for r in results], dtype=float)


def run_task(agent: Agent, env: ToolEnvironment, task: Task, *, learn: bool = True) -> dict[str, Any]:
    agent.begin_task()
    view = env.present(task)
    if getattr(agent, "uses_tool", False) and task.family == "repair":
        results = env.run_visible_tests(view)
        store = agent.tool_results  # type: ignore[attr-defined]
        store.clear()
        store[view.task_ref] = tool_vector(results) if hasattr(agent, "model") else results
    action = agent.act(view)
    env.commit(view, action)
    correct, feedback = env.reveal(view)
    updated = agent.learn(view, action, feedback) if learn else False
    return {
        "task_id": task.task_id,
        "family": task.family,
        "slice": task.slice,
        "answer": action.answer,
        "truth": task.answer,
        "confidence": float(action.confidence),
        "abstain": bool(action.abstain),
        "correct": bool(correct),
        "updated": bool(updated),
    }


def run_stream(
    agent: Agent, tasks: Sequence[Task], spec: Mapping[str, Any], *, feedback: bool = True, learn: bool = True
) -> list[dict[str, Any]]:
    env = ToolEnvironment(spec, feedback_enabled=feedback)
    records = []
    for position, task in enumerate(tasks):
        record = run_task(agent, env, task, learn=learn)
        record["position"] = position
        records.append(record)
    return records
