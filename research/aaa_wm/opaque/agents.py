"""Agents for ``aaa.python.opaque.v0``: non-learned baselines, the verified planner, and ceilings.

The :class:`Planner` is shared by every model-based arm. It enumerates plans of up to
``depth`` edits from the current program, scores each resulting program with a *consequence
predictor* ``predict(programs, view) -> P(visible test j passes)``, executes the best plan,
verifies it with a real ``RUN`` when runs remain, and re-plans after a failed run, excluding
programs already observed to fail. Which predictor it uses is the experimental variable:

* :class:`TrueLibraryPredictor` -- the environment's own library (CEILING, not a baseline;
  it is what a perfect world model would predict);
* a learned world model (see ``models.py``);
* a learned value model (value-equivalence control).

Prediction outputs are tagged ``WORLD_MODEL_IMAGINATION`` and never enter a view.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .env import RUN, SUBMIT, RunObservation, View
from .program import Edit, apply, edits, run

IMAGINATION = "WORLD_MODEL_IMAGINATION"
Predictor = Callable[[Sequence[str], View], np.ndarray]  # (n programs x V tests) pass probabilities


def fit_prior(tasks: Sequence[Any]) -> Counter[str]:
    """Kinds of the edits that *undo* training faults (training keys only, never evaluation keys)."""

    counts: Counter[str] = Counter()
    for t in tasks:
        for f in t.faults:
            counts[Edit(f.row, f.col, f.new, f.old).kind()] += 1
    return counts


class Prior:
    def __init__(self, counts: Counter[str]) -> None:
        total = sum(counts.values()) + len(counts) + 1
        self.logp = {k: math.log((v + 1) / total) for k, v in counts.items()}
        self.floor = math.log(1 / total)

    def score(self, e: Edit) -> float:
        return self.logp.get(e.kind(), self.floor)


def plans(source: str, names: tuple[str, ...], depth: int) -> list[tuple[Edit, ...]]:
    out: list[tuple[Edit, ...]] = [()]
    first = edits(source, names)
    out += [(e,) for e in first]
    if depth >= 2:
        for e in first:
            after = apply(source, e)
            for e2 in edits(after, names):
                if (e2.row, e2.col) != (e.row, e.col):
                    out.append((e, e2))
    return out


def result_of(source: str, plan: tuple[Edit, ...]) -> str:
    for e in plan:
        source = apply(source, e)
    return source


class SubmitAsIs:
    name = "submit_asis"

    def act(self, view: View) -> Any:
        return SUBMIT


class PriorOnly:
    """No tool: apply the single most prior-likely edit and submit."""

    name = "prior"

    def __init__(self, prior: Prior) -> None:
        self.prior = prior

    def act(self, view: View) -> Any:
        if any(isinstance(h, Edit) for h in view.history):
            return SUBMIT
        return max(view.edits(), key=self.prior.score)


class ToolSearch:
    """Verified local search: best untried edit by ``scorer``; RUN; keep if pass count did not fall."""

    def __init__(self, scorer: Callable[[View, Edit], float], name: str, *, initial_run: bool = False) -> None:
        self.scorer, self.name, self.initial_run = scorer, name, initial_run

    def begin(self, view: View) -> None:
        self.best = -1
        self.pending: Edit | None = None
        self.undo: Edit | None = None
        self.tried: set[str] = set()
        self.need_run = self.initial_run

    def act(self, view: View) -> Any:
        if self.undo is not None:
            e, self.undo = self.undo, None
            return e
        runs = view.runs()
        if runs and all(runs[-1].passed) and self.pending is None:
            return SUBMIT
        if self.need_run and view.runs_left > 0:
            self.need_run = False
            return RUN
        if view.steps_left < 2:
            return SUBMIT
        options = [e for e in view.edits() if apply(view.source, e) not in self.tried]
        if not options:
            return SUBMIT
        e = max(options, key=lambda e: self.scorer(view, e))
        self.tried.add(apply(view.source, e))
        if view.runs_left > 0:
            self.pending = e
            self.need_run = True
        return e if view.runs_left > 0 or not runs else SUBMIT

    def observe(self, view: View, action: Any, obs: RunObservation | None) -> None:
        if obs is None or self.pending is None:
            if obs is not None:
                self.best = sum(obs.passed)
            return
        e, self.pending = self.pending, None
        n = sum(obs.passed)
        if n == len(obs.passed):
            return
        if n <= self.best:
            self.undo = Edit(e.row, e.col, e.new, e.old)
        else:
            self.best = n


def gap_scorer(prior: Prior) -> Callable[[View, Edit], float]:
    """Pattern-reading: with a constant offset ``d = expected - got`` in the last RUN, prefer literal
    edits after a ``+``/``-``/``=``/``return`` whose additive effect is ``d``; else the prior."""

    def score(view: View, e: Edit) -> float:
        base = prior.score(e)
        runs = view.runs()
        if not runs or not e.old.isdigit():
            return base
        offs = {
            exp[1] - got[1]
            for (_, exp), got in zip(view.visible_tests, runs[-1].results, strict=True)
            if got[0] == "ok" and isinstance(got[1], int) and got != exp
        }
        if len(offs) != 1:
            return base
        d = offs.pop()
        line = view.source.split("\n")[e.row - 1]
        before = line[: e.col].rstrip()
        sign = -1 if before.endswith("-") or before.endswith("-=") else 1
        if (before.endswith(("+", "-", "=", "+=", "-="))) and sign * (int(e.new) - int(e.old)) == d:
            return 100.0 + base
        return base

    return score


class Planner:
    """Model-based verified planner (see module docstring)."""

    def __init__(self, predict: Predictor, prior: Prior, name: str, *, depth: int = 2, prior_weight: float = 1.0) -> None:
        self.predict, self.prior, self.name, self.depth, self.w = predict, prior, name, depth, prior_weight

    def begin(self, view: View) -> None:
        self.queue: list[Edit] = []
        self.failed: set[str] = set()
        self.verify_next = False

    def _choose(self, view: View) -> tuple[Edit, ...]:
        cands = [p for p in plans(view.source, view.api_names, self.depth)]
        progs = [result_of(view.source, p) for p in cands]
        keep = [i for i, s in enumerate(progs) if s not in self.failed]
        if not keep:
            return ()
        probs = self.predict([progs[i] for i in keep], view)
        logp = np.log(np.clip(probs, 1e-6, 1.0)).sum(axis=1)
        scores = []
        for j, i in enumerate(keep):
            plan = cands[i]
            scores.append(logp[j] + self.w * sum(self.prior.score(e) for e in plan) * 0.1)
        best = keep[int(np.argmax(scores))]
        return cands[best]

    def act(self, view: View) -> Any:
        if self.queue:
            return self.queue.pop(0)
        runs = view.runs()
        if runs and runs[-1].source == view.source and all(runs[-1].passed):
            return SUBMIT
        if self.verify_next and view.runs_left > 0:
            self.verify_next = False
            return RUN
        plan = self._choose(view)
        if len(plan) + (1 if view.runs_left > 0 else 0) > view.steps_left:
            plan = plan[: max(0, view.steps_left - (1 if view.runs_left > 0 else 0))]
        if not plan:
            if view.runs_left > 0 and not (runs and runs[-1].source == view.source):
                return RUN
            return SUBMIT
        self.queue = list(plan[1:])
        self.verify_next = view.runs_left > 0
        return plan[0]

    def observe(self, view: View, action: Any, obs: RunObservation | None) -> None:
        if obs is not None and not all(obs.passed):
            self.failed.add(obs.source)
            self.on_observation(obs, view)

    def on_observation(self, obs: RunObservation, view: View) -> None:  # hook for online world models
        pass


class TrueLibraryPredictor:
    """CEILING: the environment's library as a 'perfect world model' (0/1 predictions)."""

    def __init__(self, lib: Any) -> None:
        self.lib = lib

    def __call__(self, programs: Sequence[str], view: View) -> np.ndarray:
        out = np.zeros((len(programs), len(view.visible_tests)))
        for i, src in enumerate(programs):
            for j, (x, expected) in enumerate(view.visible_tests):
                out[i, j] = 1.0 if run(src, self.lib, x) == expected else 0.0
        return np.clip(out, 1e-3, 1 - 1e-3)
