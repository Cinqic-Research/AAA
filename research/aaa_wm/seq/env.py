"""The causal boundary of ``aaa.python.seq.v0``.

    reset -> view -> (EDIT | RUN)* -> SUBMIT -> score -> post-episode feedback

The agent-facing :class:`SeqView` holds copies of the visible fields only: the
*current* program, the visible tests, and the history of the agent's own actions and
RUN observations. The reference, fault lines and hidden tests live only on the
:class:`~.generator.SeqTask`, which the environment never hands out. RUN executes the
*current* program on the visible tests only. Hidden tests run exactly once, at
SUBMIT, and the post-episode feedback (success bit and the submitted program's hidden
pass vector) is returned only after that. Every event is logged with a sequence number.
Observations produced by a model are never accepted here; only the environment
creates :class:`RunObservation` objects, and each is tagged ``REAL_OBSERVATION``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.aaa_python_v1.generator import mutations

from .execute import run_function
from .generator import SeqTask, load_spec

REAL = "REAL_OBSERVATION"


class BoundaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class TestResult:
    status: str  # "pass" | "wrong" | "error"
    got: Any  # int for wrong, exception class for error, expected value for pass


@dataclass(frozen=True)
class RunObservation:
    step: int
    results: tuple[TestResult, ...]
    provenance: str = REAL

    @property
    def passed(self) -> int:
        return sum(r.status == "pass" for r in self.results)


@dataclass(frozen=True)
class Edit:
    line: int  # 1-based
    text: str


RUN = "RUN"
SUBMIT = "SUBMIT"


@dataclass(frozen=True)
class SeqView:
    """Everything an agent may see at a decision point (immutable copy)."""

    task_ref: int
    lines: tuple[str, ...]
    visible_tests: tuple[tuple[int, int], ...]
    history: tuple[Any, ...]  # Edit | RunObservation, in order
    steps_left: int
    runs_left: int
    original: tuple[str, ...]

    def edits(self) -> list[Edit]:
        """Every legal edit of the current program: one symmetric mutation of one line."""

        return [Edit(n, m) for n, line in enumerate(self.lines, start=1) for m in mutations(line)]

    def last_run(self) -> RunObservation | None:
        for item in reversed(self.history):
            if isinstance(item, RunObservation):
                return item
        return None


def run_visible(lines: tuple[str, ...], tests: tuple[tuple[int, int], ...]) -> tuple[TestResult, ...]:
    source = "\n".join(lines) + "\n"
    out = []
    for x, expected in tests:
        status, value = run_function(source, x)
        if status == "ok" and value == expected:
            out.append(TestResult("pass", expected))
        elif status == "ok":
            out.append(TestResult("wrong", value))
        else:
            out.append(TestResult("error", value))
    return tuple(out)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    runs: int
    edits: int
    submitted: tuple[str, ...]
    hidden_pass: tuple[bool, ...]
    harmful_edits: int
    events: list[tuple[int, str]] = field(default_factory=list)


class SeqEnv:
    def __init__(
        self, task: SeqTask, position: int = 0, *, steps: int | None = None, runs: int | None = None
    ) -> None:
        budgets = load_spec()["budgets"]
        self._task = task
        self._lines = tuple(task.buggy)
        self._history: list[Any] = []
        self._steps_left = budgets["steps"] if steps is None else steps
        self._runs_left = budgets["runs"] if runs is None else runs
        self._position = position
        self._done = False
        self._events: list[tuple[int, str]] = []
        self._runs = self._edits = 0
        self._log("reset")

    def _log(self, event: str) -> None:
        self._events.append((len(self._events), event))

    def view(self) -> SeqView:
        if self._done:
            raise BoundaryError("episode is over")
        return SeqView(
            self._position,
            self._lines,
            tuple(self._task.visible_tests),
            tuple(self._history),
            self._steps_left,
            self._runs_left,
            tuple(self._task.buggy),
        )

    @property
    def done(self) -> bool:
        return self._done

    def step(self, action: Any) -> RunObservation | None:
        if self._done:
            raise BoundaryError("episode is over")
        if action == SUBMIT:
            raise BoundaryError("use submit()")
        if self._steps_left <= 0:
            raise BoundaryError("no steps left")
        if action == RUN:
            if self._runs_left <= 0:
                raise BoundaryError("no runs left")
            self._runs_left -= 1
            self._steps_left -= 1
            self._runs += 1
            obs = RunObservation(
                len(self._history), run_visible(self._lines, tuple(self._task.visible_tests))
            )
            self._history.append(obs)
            self._log("run")
            return obs
        if not isinstance(action, Edit) or not 1 <= action.line <= len(self._lines):
            raise BoundaryError(f"malformed action {action!r}")
        if action.text not in mutations(self._lines[action.line - 1]):
            raise BoundaryError("an edit must be one symmetric mutation of the current line")
        lines = list(self._lines)
        lines[action.line - 1] = action.text
        self._lines = tuple(lines)
        self._history.append(action)
        self._steps_left -= 1
        self._edits += 1
        self._log("edit")
        return None

    def submit(self) -> EpisodeResult:
        if self._done:
            raise BoundaryError("already submitted")
        self._done = True
        self._log("submit")
        hidden = run_visible(self._lines, tuple(self._task.hidden_tests))
        hidden_pass = tuple(r.status == "pass" for r in hidden)
        harmful = sum(
            1
            for n in range(1, len(self._lines) + 1)
            if n not in self._task.fault_lines and self._lines[n - 1] != self._task.reference[n - 1]
        )
        self._log("reveal")
        return EpisodeResult(
            all(hidden_pass),
            len(self._events),
            self._runs,
            self._edits,
            self._lines,
            hidden_pass,
            harmful,
            list(self._events),
        )


def play(
    agent: Any, task: SeqTask, position: int = 0, *, learn: bool = False, **budgets: int
) -> EpisodeResult:
    """Run one causal episode. ``agent.act(view)`` returns an Edit, RUN or SUBMIT."""

    env = SeqEnv(task, position, **budgets)
    if hasattr(agent, "begin"):
        agent.begin(env.view())
    while True:
        view = env.view()
        if view.steps_left <= 0:
            break
        action = agent.act(view)
        if action == SUBMIT:
            break
        obs = env.step(action)
        if hasattr(agent, "observe"):
            agent.observe(view, action, obs)
    result = env.submit()
    if learn and hasattr(agent, "learn_episode"):
        # Declared post-episode feedback only: success and the submitted program's hidden pass
        # vector. Evaluator-only metrics (fault lines, harmful edits) never reach the agent.
        agent.learn_episode(result.success, result.hidden_pass)
    return result
