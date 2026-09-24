"""The causal boundary of ``aaa.python.v0``.

    present -> act -> commit -> reveal -> score -> learn (only where permitted)

A learner receives a :class:`TaskView`, built field by field from the
evaluator's :class:`~.generator.Task`. The view holds *copies* of the allowed
fields and no reference to the task, so the answer key, the oracle
observation, hidden test cases and per-candidate results are not reachable
from it. :func:`view_of` is the only place a view is made; its allowed fields
are the declared ones and nothing else.

Feedback is what a real interpreter or test run would reveal *after* an
action, restricted to the family's declared ``feedback`` fields. For repair,
that is the hidden-test results of the *chosen* candidate only, never which
candidate was right. :class:`Environment` refuses to reveal before a commit,
refuses a second commit, refuses feedback for a task other than the committed
one, validates the action against the label space, and logs every event with
a sequence number so the order can be audited afterwards.

This boundary protects against accidental leakage by trusted research code.
It is not a sandbox against a learner deliberately written to subvert the
Python runtime; learners run in-process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .generator import Task

VIEW_FIELDS = ("task_ref", "family", "source", "labels", "repair_line", "candidates", "visible_tests")


class BoundaryError(RuntimeError):
    """An operation that would break the predict-before-reveal order."""


@dataclass(frozen=True)
class TaskView:
    """Everything a learner may see before acting. ``task_ref`` is an opaque per-episode position."""

    task_ref: int
    family: str
    source: str
    labels: tuple[Any, ...]
    repair_line: int | None = None
    candidates: tuple[str, ...] = ()
    visible_tests: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Action:
    answer: Any
    confidence: float
    abstain: bool = False


@dataclass(frozen=True)
class Feedback:
    """Post-action information a real run would reveal, per the family's declared feedback fields."""

    family: str
    fields: Mapping[str, Any] = field(default_factory=dict)


def label_space(task: Task, spec: Mapping[str, Any]) -> tuple[Any, ...]:
    family = spec["families"][task.family]
    if task.family == "output":
        return tuple(range(family["min"], family["max"] + 1))
    if task.family == "localize":
        return tuple(range(1, family["max_line"] + 1))
    if task.family == "repair":
        return tuple(range(family["candidates"]))
    return tuple(family["labels"])


def view_of(task: Task, position: int, spec: Mapping[str, Any]) -> TaskView:
    return TaskView(
        task_ref=position,
        family=task.family,
        source=str(task.source),
        labels=label_space(task, spec),
        repair_line=task.repair_line,
        candidates=tuple(str(c) for c in task.candidates),
        visible_tests=tuple((int(i), int(o)) for i, o in task.visible_tests),
    )


def feedback_of(task: Task, action: Action, spec: Mapping[str, Any]) -> Feedback:
    declared = spec["families"][task.family]["feedback"]
    oracle = task.oracle
    available: dict[str, Any] = {
        "compile_status": oracle.get("status"),
        "syntax_error_line": oracle.get("line") if oracle.get("status") == "syntax_error" else None,
        "status": oracle.get("status"),
        "exception": oracle.get("exception"),
        "error_line": oracle.get("line"),
        "stdout": oracle.get("stdout"),
    }
    if task.family == "repair":
        chosen = action.answer if isinstance(action.answer, int) and not action.abstain else None
        available["chosen_candidate_hidden_results"] = (
            None if chosen is None else list(task.candidate_hidden_results[chosen])
        )
    return Feedback(task.family, {name: available[name] for name in declared})


def correct(task: Task, action: Action) -> bool:
    return not action.abstain and action.answer == task.answer


class Environment:
    """One task at a time, in the causal order, with an auditable event log."""

    def __init__(self, spec: Mapping[str, Any], *, feedback_enabled: bool = True) -> None:
        self.spec = spec
        self.feedback_enabled = feedback_enabled
        self.events: list[tuple[int, str, int]] = []
        self._sequence = 0
        self._current: Task | None = None
        self._view: TaskView | None = None
        self._position = -1
        self._action: Action | None = None
        self._revealed = True

    def _log(self, event: str) -> None:
        self.events.append((self._sequence, event, self._position))
        self._sequence += 1

    def present(self, task: Task) -> TaskView:
        if not self._revealed:
            raise BoundaryError("the previous task was never revealed")
        self._current, self._action, self._revealed = task, None, False
        self._position += 1
        self._log("present")
        self._view = view_of(task, self._position, self.spec)
        return self._view

    def commit(self, view: TaskView, action: Action) -> None:
        if self._current is None or view is not self._view:
            raise BoundaryError("commit does not refer to the presented task")
        if self._action is not None:
            raise BoundaryError("an action was already committed for this task")
        if not isinstance(action, Action) or isinstance(action.confidence, bool):
            raise BoundaryError("malformed action")
        try:
            confidence = float(action.confidence)
        except (TypeError, ValueError):
            raise BoundaryError("malformed action: confidence is not a number") from None
        if not 0.0 <= confidence <= 1.0:  # also refuses NaN
            raise BoundaryError("malformed action: confidence outside [0, 1]")
        # Match type as well as value: True == 1 and 1.0 == 1 must not pass as label 1.
        if not action.abstain and not any(
            type(action.answer) is type(label) and action.answer == label for label in view.labels
        ):
            raise BoundaryError(f"answer {action.answer!r} is outside the label space")
        self._action = action
        self._log("commit")

    def reveal(self, view: TaskView) -> tuple[bool, Feedback | None]:
        """Score the committed action; return feedback only if the protocol permits learning from it."""

        if self._current is None or self._action is None:
            raise BoundaryError("nothing may be revealed before an action is committed")
        if view is not self._view or self._revealed:
            raise BoundaryError("reveal does not refer to the committed task")
        self._revealed = True
        score = correct(self._current, self._action)
        self._log("reveal")
        if not self.feedback_enabled:
            return score, None
        return score, feedback_of(self._current, self._action, self.spec)
