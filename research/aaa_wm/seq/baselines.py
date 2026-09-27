# ruff: noqa: ARG002, ARG005 - agent interface methods keep the shared signature
"""Non-learned agents and planning ceilings for ``aaa.python.seq.v0``.

Every agent acts only on a :class:`~.env.SeqView`. Ceilings are labelled as such:
``OraclePlanner`` imagines edits with the *real* executor on the *visible* tests,
the perfect-visible-consequence world model. It never touches hidden tests, but it is not a
baseline, because it spends unlimited free executions. It bounds what any learned
consequence model could add at the same run budget.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from .env import RUN, SUBMIT, Edit, RunObservation, SeqView, run_visible


def diff_type(old: str, new: str) -> str:
    a, b = old.split(" "), new.split(" ")
    for x, y in zip(a, b, strict=False):
        if x != y:
            if x.isdigit() and y.isdigit():
                return f"lit{int(y) - int(x):+d}"
            return f"op:{x}>{y}"
    return "same"


def _apply(lines: tuple[str, ...], edits: Sequence[Edit]) -> tuple[str, ...]:
    out = list(lines)
    for e in edits:
        out[e.line - 1] = e.text
    return tuple(out)


def _second_edits(lines: tuple[str, ...], first: Edit) -> list[Edit]:
    from research.aaa_python_v1.generator import mutations

    after = _apply(lines, [first])
    return [Edit(n, m) for n, line in enumerate(after, start=1) if n != first.line for m in mutations(line)]


class SubmitAsIs:
    name = "submit_asis"

    def act(self, view: SeqView) -> Any:
        return SUBMIT


class Prior:
    """Edit-type prior (fitted on training fixes); no tool. Applies the top edit, twice at most."""

    name = "prior"

    def __init__(self, counts: Counter[str] | None = None, edits: int = 1) -> None:
        self.counts = counts or Counter()
        self.edits = edits

    def score(self, view: SeqView, e: Edit) -> float:
        return self.counts[diff_type(view.lines[e.line - 1], e.text)] + 1e-3 * (-e.line)

    def act(self, view: SeqView) -> Any:
        done = sum(isinstance(h, Edit) for h in view.history)
        if done >= self.edits:
            return SUBMIT
        touched = {h.line for h in view.history if isinstance(h, Edit)}
        options = [e for e in view.edits() if e.line not in touched]
        return max(options, key=lambda e: self.score(view, e)) if options else SUBMIT


class ToolSearch:
    """Propose the best untried edit, RUN it, keep it if the pass count did not fall, else undo.

    ``scorer(view, edit)`` orders proposals. ``initial_run`` spends the first run on the
    unedited program (its ``got`` values feed pattern-reading scorers). This is the careful
    debugger with a test runner: every proposal it keeps was verified by a real RUN, and
    after the runs are spent it submits.
    """

    def __init__(self, scorer: Callable[[SeqView, Edit], float], *, initial_run: bool, name: str) -> None:
        self.scorer, self.initial_run, self.name = scorer, initial_run, name

    def begin(self, view: SeqView) -> None:
        self.best = -1
        self.pending: tuple[Edit, str] | None = None
        self.undo: Edit | None = None
        self.tried: set[tuple[int, str]] = set()
        self.need_run = self.initial_run

    def act(self, view: SeqView) -> Any:
        if self.undo is not None:
            e, self.undo = self.undo, None
            return e
        last = view.last_run()
        if last is not None and last.passed == len(last.results) and self.pending is None:
            return SUBMIT
        if self.need_run and view.runs_left > 0:
            self.need_run = False
            return RUN
        if view.runs_left == 0 or view.steps_left < 2:
            return SUBMIT
        options = [e for e in view.edits() if (e.line, e.text) not in self.tried]
        if not options:
            return SUBMIT
        e = max(options, key=lambda e: self.scorer(view, e))
        self.tried.add((e.line, e.text))
        self.pending = (e, view.lines[e.line - 1])
        self.need_run = True
        return e

    def observe(self, view: SeqView, action: Any, obs: RunObservation | None) -> None:
        if obs is None:
            return
        if self.pending is None:
            self.best = obs.passed
            return
        e, previous = self.pending
        self.pending = None
        if obs.passed == len(obs.results):
            return
        if obs.passed < self.best or (obs.passed == self.best and self.best >= 0):
            self.undo = Edit(e.line, previous)
        else:
            self.best = obs.passed


def analytic_scorer(prior: Prior) -> Callable[[SeqView, Edit], float]:
    """Pattern-reading debugger: if the last RUN shows a constant offset ``d = expected - got``,
    prefer literal edits whose additive effect is ``d``; otherwise fall back to the prior."""

    def score(view: SeqView, e: Edit) -> float:
        base = prior.score(view, e)
        last = view.last_run()
        if last is None:
            return base
        offsets = {
            x_e[1] - r.got
            for x_e, r in zip(view.visible_tests, last.results, strict=True)
            if r.status == "wrong" and isinstance(r.got, int)
        }
        if len(offsets) != 1:
            return base
        d = offsets.pop()
        old, new = view.lines[e.line - 1].split(" "), e.text.split(" ")
        for k, (x, y) in enumerate(zip(old, new, strict=False)):
            if x != y and x.isdigit() and y.isdigit():
                sign = -1 if k > 0 and old[k - 1] == "-" else 1
                if k > 0 and old[k - 1] in ("+", "-", "=") and sign * (int(y) - int(x)) == d:
                    return 1e6 + base
        return base

    return score


class OraclePlanner:
    """CEILING: plans with the real executor on visible tests (a perfect visible-consequence model).

    Search: every single edit and every ordered pair of edits on distinct lines from the
    current program, scored by the visible pass count (ties broken by ``tiebreak``).
    Applies the best plan, then submits. It uses no RUN, because its imagination is already exact.
    """

    name = "oracle_planner"

    def __init__(self, tiebreak: Callable[[SeqView, Edit], float] | None = None, depth: int = 2) -> None:
        self.tiebreak = tiebreak or (lambda v, e: 0.0)
        self.depth = depth

    def begin(self, view: SeqView) -> None:
        self.plan: list[Edit] = []
        tests = view.visible_tests
        base = sum(r.status == "pass" for r in run_visible(view.lines, tests))
        best: tuple[float, list[Edit]] = (base, [])
        for e in view.edits():
            p = sum(r.status == "pass" for r in run_visible(_apply(view.lines, [e]), tests))
            cand = (p + 1e-3 * self.tiebreak(view, e) - 1e-6, [e])
            if cand[0] > best[0]:
                best = cand
            if self.depth >= 2 and p < len(tests):
                for e2 in _second_edits(view.lines, e):
                    p2 = sum(r.status == "pass" for r in run_visible(_apply(view.lines, [e, e2]), tests))
                    cand2 = (p2 + 1e-3 * (self.tiebreak(view, e) + self.tiebreak(view, e2)) - 2e-6, [e, e2])
                    if cand2[0] > best[0]:
                        best = cand2
        self.plan = list(best[1])

    def act(self, view: SeqView) -> Any:
        return self.plan.pop(0) if self.plan else SUBMIT


def all_pairs_count(view: SeqView) -> int:
    return sum(len(_second_edits(view.lines, e)) for e in view.edits())


def fit_prior(tasks: Sequence[Any]) -> Counter[str]:
    """Edit types that repair training faults (training answer keys only; never evaluation keys)."""

    counts: Counter[str] = Counter()
    for t in tasks:
        for n in t.fault_lines:
            counts[diff_type(t.buggy[n - 1], t.reference[n - 1])] += 1
    return counts


__all__ = [
    "OraclePlanner",
    "Prior",
    "SubmitAsIs",
    "ToolSearch",
    "all_pairs_count",
    "analytic_scorer",
    "diff_type",
    "fit_prior",
    "itertools",
]
