"""The causal boundary of ``aaa.python.opaque.v0``.

    reset -> view -> (EDIT | RUN)* -> SUBMIT -> domain-equivalence score -> post-episode feedback

A :class:`View` carries only visible fields: the current program, the visible tests (input,
expected result), the agent's own action history with the real ``RUN`` observations, the
remaining budgets, and the library *names*. It never carries the library implementation,
the reference, the faults or the domain results. ``RUN`` executes the current program with
the environment's library on the visible inputs. Only this environment constructs
:class:`RunObservation`, and each carries ``provenance = REAL_OBSERVATION``. The success
check runs once, at ``SUBMIT``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .generator import OpaqueTask, domain, library, load_spec
from .program import Edit, apply, edits, run, signature

REAL = "REAL_OBSERVATION"
RUN = "RUN"
SUBMIT = "SUBMIT"


class BoundaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunObservation:
    source: str  # the program that was run (the agent's own current program)
    results: tuple[tuple[str, Any], ...]  # per visible test: ("ok", v) | ("error", cls)
    passed: tuple[bool, ...]
    provenance: str = REAL


@dataclass(frozen=True)
class View:
    task_ref: int
    source: str
    visible_tests: tuple[tuple[int, Any], ...]
    history: tuple[Any, ...]
    steps_left: int
    runs_left: int
    api_names: tuple[str, ...]

    def edits(self) -> tuple[Edit, ...]:
        return edits(self.source, self.api_names)

    def runs(self) -> list[RunObservation]:
        return [h for h in self.history if isinstance(h, RunObservation)]


@dataclass
class Outcome:
    success: bool
    runs: int
    edits: int
    submitted: str
    domain_match: tuple[bool, ...]
    events: list[str] = field(default_factory=list)


class OpaqueEnv:
    def __init__(self, task: OpaqueTask, position: int = 0, *, steps: int | None = None, runs: int | None = None) -> None:
        b = load_spec()["budgets"]
        self._task = task
        self._lib = library(task.library)
        self._source = task.buggy
        self._history: list[Any] = []
        self._steps = b["steps"] if steps is None else steps
        self._runs_left = b["runs"] if runs is None else runs
        self._pos = position
        self._done = False
        self._events = ["reset"]
        self._n_runs = self._n_edits = 0

    def view(self) -> View:
        if self._done:
            raise BoundaryError("episode is over")
        return View(
            self._pos,
            self._source,
            self._task.visible_tests,
            tuple(self._history),
            self._steps,
            self._runs_left,
            tuple(a.name for a in self._lib),
        )

    def step(self, action: Any) -> RunObservation | None:
        if self._done:
            raise BoundaryError("episode is over")
        if self._steps <= 0:
            raise BoundaryError("no steps left")
        if action == RUN:
            if self._runs_left <= 0:
                raise BoundaryError("no runs left")
            self._runs_left -= 1
            self._steps -= 1
            self._n_runs += 1
            results = tuple(run(self._source, self._lib, x) for x, _ in self._task.visible_tests)
            obs = RunObservation(
                self._source,
                results,
                tuple(r == e for r, (_, e) in zip(results, self._task.visible_tests, strict=True)),
            )
            self._history.append(obs)
            self._events.append("run")
            return obs
        if not isinstance(action, Edit) or action not in edits(self._source, tuple(a.name for a in self._lib)):
            raise BoundaryError(f"not a legal edit of the current program: {action!r}")
        self._source = apply(self._source, action)
        self._history.append(action)
        self._steps -= 1
        self._n_edits += 1
        self._events.append("edit")
        return None

    def submit(self) -> Outcome:
        if self._done:
            raise BoundaryError("already submitted")
        self._done = True
        self._events.append("submit")
        ref = signature(self._task.reference, self._lib, domain())
        got = signature(self._source, self._lib, domain())
        match = tuple(a == b for a, b in zip(got, ref, strict=True))
        self._events.append("reveal")
        return Outcome(all(match), self._n_runs, self._n_edits, self._source, match, list(self._events))


def play(agent: Any, task: OpaqueTask, position: int = 0, *, learn: bool = False, **budgets: int) -> Outcome:
    env = OpaqueEnv(task, position, **budgets)
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
    out = env.submit()
    if learn and hasattr(agent, "learn_episode"):
        agent.learn_episode(out.success, out.domain_match)
    return out
