"""The ``aaa.python.v1`` causal boundary: v0's, plus one declared pre-action tool.

    present -> [tool, repair only] -> act -> commit -> reveal -> score -> learn

:class:`ToolEnvironment` extends v0's :class:`~research.aaa_python.episode.Environment`
unchanged in every other respect. Its only addition is :meth:`ToolEnvironment.run_visible_tests`:
for a presented, not-yet-committed ``repair`` task it executes each candidate,
substituted into the visible buggy function, on the task's **visible** tests
and returns pass/fail per candidate and test. It reads only fields of the
issued :class:`TaskView`, so it cannot reach hidden tests, the answer or the
oracle's record; the call is logged in the event sequence, so the order can
be audited.

Running the task's own program *is* the oracle for ``syntax``, ``outcome``,
``output`` and ``localize``; no tool exists for them, and the call refuses.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from research.aaa_python.episode import Action, BoundaryError, Environment, Feedback, TaskView, view_of
from research.aaa_python.oracle import run_many

__all__ = [
    "Action",
    "BoundaryError",
    "Feedback",
    "TaskView",
    "ToolEnvironment",
    "view_of",
    "visible_test_results",
]


def _function_name(source: str) -> str:
    first = source.split("\n", 1)[0]
    if not first.startswith("def ") or "(" not in first:
        raise BoundaryError("a repair task's source must begin with its function definition")
    return first[4 : first.index("(")]


@lru_cache(maxsize=100000)
def _run(
    source: str, repair_line: int, candidates: tuple[str, ...], tests: tuple[tuple[int, int], ...]
) -> tuple[tuple[bool, ...], ...]:
    from . import spec as spec_module

    spec = spec_module.load()
    lines = source.rstrip("\n").split("\n")
    name = _function_name(source)
    jobs = []
    for candidate in candidates:
        body = [*lines[: repair_line - 1], candidate, *lines[repair_line:]]
        jobs.append(("exec", "\n".join(body + [f"print({name}({i}))" for i, _ in tests]) + "\n"))
    results = []
    for outcome in run_many(jobs, spec):
        printed = outcome.stdout.split() if outcome.status == "ok" else []
        results.append(
            tuple(
                len(printed) == len(tests) and printed[k] == str(expected)
                for k, (_, expected) in enumerate(tests)
            )
        )
    return tuple(results)


def visible_test_results(view: TaskView) -> tuple[tuple[bool, ...], ...]:
    """Pass/fail of every candidate on the view's visible tests (a pure function of the view)."""

    if view.family != "repair" or view.repair_line is None:
        raise BoundaryError("the visible-test tool exists only for repair tasks")
    return _run(view.source, view.repair_line, view.candidates, view.visible_tests)


class ToolEnvironment(Environment):
    """v0's environment with the declared, logged, pre-action repair tool."""

    def __init__(self, spec: Mapping[str, Any], *, feedback_enabled: bool = True) -> None:
        super().__init__(spec, feedback_enabled=feedback_enabled)

    def run_visible_tests(self, view: TaskView) -> tuple[tuple[bool, ...], ...]:
        if self._current is None or view is not self._view:
            raise BoundaryError("the tool may be used only on the presented task")
        if self._action is not None:
            raise BoundaryError("the tool may be used only before the action is committed")
        results = visible_test_results(view)
        self._log("tool")
        return results
