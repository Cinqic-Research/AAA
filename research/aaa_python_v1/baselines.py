"""Baselines for ``aaa.python.v1``: chance, memorization, deliberately stupid rules, and tools.

A learner's number means little without the simplest thing that achieves the
same. Each baseline here sees exactly what a learner sees (a :class:`TaskView`
and post-action feedback) and none of them is CPython answering the task: the
oracle is never a baseline.

``uniform``        chance over the valid labels.
``majority``       the most frequent training target per family; for ``repair``
                   it explores candidate positions uniformly during training so
                   its counts are not an artefact of its own choices (v0's
                   majority only ever tried index 0, ``AAA-193``).
``lookup``         exact normalized-source memory, majority fallback.
``v0_heuristic``   v0's surface rules, unchanged, for continuity.
``rules``          *fitted* stupid rules, strictly stronger than v0's where v0
                   had none (``AAA-193``): the syntax lint; for ``outcome`` the
                   training-majority label for each set of risky constructs
                   present; for ``output`` the training-majority answer for
                   each program shape; the first risky line for ``localize``;
                   and for ``repair`` the candidate whose edit, relative to the
                   current line, most often passed during training.
``medoid``         the repair candidate at minimum total token edit distance
                   from the others, excluding the current line: the v0
                   shortcut (``AAA-192``), kept as a permanent attack.
``visible_tests``  (tool) the first candidate other than the current line that
                   passes every visible test; ``rules`` otherwise.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from research.aaa_python.episode import Action, Feedback, TaskView
from research.aaa_python.generator import source_hash
from research.aaa_python.learners import Agent, Heuristic, _looks_invalid, target_from_feedback, valid_labels
from research.aaa_python.rng import Stream

from .encoders import _diff, renamed_tokens

REPAIR = "repair"
RISKY = {
    "floordiv_paren": re.compile(r"//\s*\("),
    "index_literal_list": re.compile(r"\]\["),
    "int_of_ifexp": re.compile(r"int\(\("),
    "str_in_ifexp": re.compile(r"\+ \(str\("),
    "guarded_undefined": re.compile(r"^\s+[a-z][a-z0-9]* = [a-z][a-z0-9]* \+ 1$"),
}


def _labels(view: TaskView) -> tuple[Any, ...]:
    return tuple(range(len(view.candidates))) if view.family == REPAIR else valid_labels(view)


def _current_line(view: TaskView) -> str:
    assert view.repair_line is not None
    return view.source.rstrip("\n").split("\n")[view.repair_line - 1]


def risky_signature(source: str) -> tuple[str, ...]:
    found = []
    for line in source.rstrip("\n").split("\n"):
        for name, pattern in RISKY.items():
            if pattern.search(line):
                found.append(name)
    return tuple(sorted(set(found)))


def first_risky_line(source: str) -> int | None:
    for number, line in enumerate(source.rstrip("\n").split("\n"), start=1):
        if any(pattern.search(line) for pattern in RISKY.values()):
            return number
    return None


def program_shape(source: str) -> str:
    """Token kinds with identifiers renamed and literals erased: the program's skeleton."""

    return " ".join(kind if kind in ("NAME", "NUMBER") else text for kind, text in renamed_tokens(source))


def _token_distance(a: str, b: str) -> int:
    ta, tb = a.split(" "), b.split(" ")
    if len(ta) != len(tb):
        return max(len(ta), len(tb))
    return sum(x != y for x, y in zip(ta, tb, strict=True))


class Uniform(Agent):
    name = "uniform"

    def __init__(self, seed: int) -> None:
        self.stream = Stream(seed)

    def act(self, view: TaskView) -> Action:
        labels = _labels(view)
        return Action(labels[self.stream.below(len(labels))], 1.0 / len(labels))


class Majority(Agent):
    name = "majority"

    def __init__(self, seed: int = 0) -> None:
        self.counts: dict[str, Counter[Any]] = defaultdict(Counter)
        self.explore = Stream(seed)
        self.training = True

    def learn(self, view: TaskView, action: Action, feedback: Feedback | None) -> bool:
        if feedback is None:
            return False
        if view.family == REPAIR:
            results = feedback.fields.get("chosen_candidate_hidden_results")
            target = action.answer if results is not None and all(results) else None
        else:
            target = target_from_feedback(view, feedback)
        if target is None:
            return False
        self.counts[view.family][target] += 1
        return True

    def majority(self, view: TaskView) -> Action:
        labels = _labels(view)
        counts = self.counts[view.family]
        total = sum(counts[label] for label in labels)
        best = max(labels, key=lambda label: (counts[label], -labels.index(label)))
        return Action(best, counts[best] / total if total else 1.0 / len(labels))

    def act(self, view: TaskView) -> Action:
        if self.training and view.family == REPAIR:
            labels = _labels(view)
            return Action(labels[self.explore.below(len(labels))], 1.0 / len(labels))
        return self.majority(view)


class Lookup(Majority):
    name = "lookup"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.memory: dict[tuple[str, str], Any] = {}
        self.hits = 0
        self.queries = 0

    def learn(self, view: TaskView, action: Action, feedback: Feedback | None) -> bool:
        learned = super().learn(view, action, feedback)
        if learned:
            if view.family == REPAIR:
                target = action.answer
            else:
                target = target_from_feedback(view, feedback)
            self.memory[(view.family, source_hash(view.source, view.candidates))] = target
        return learned

    def act(self, view: TaskView) -> Action:
        if self.training:
            return super().act(view)
        self.queries += 1
        key = (view.family, source_hash(view.source, view.candidates))
        if key in self.memory:
            self.hits += 1
            return Action(self.memory[key], 1.0)
        return self.majority(view)


class V0Heuristic(Heuristic):
    """v0's rules, unchanged (its ``outcome`` branch is majority: ``AAA-193``)."""

    name = "v0_heuristic"

    def __init__(self, seed: int = 0) -> None:
        del seed  # deterministic; accepts a seed only for a uniform constructor
        super().__init__()


class Rules(Majority):
    """Fitted stupid rules; each is a lookup from a surface signature to a training majority."""

    name = "rules"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.table: dict[tuple[str, Any], Counter[Any]] = defaultdict(Counter)
        self.edits: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0])

    def _key(self, view: TaskView) -> Any:
        if view.family == "outcome":
            return risky_signature(view.source)
        if view.family == "output":
            return program_shape(view.source)
        return None

    def learn(self, view: TaskView, action: Action, feedback: Feedback | None) -> bool:
        learned = super().learn(view, action, feedback)
        if feedback is None:
            return learned
        if view.family == REPAIR:
            results = feedback.fields.get("chosen_candidate_hidden_results")
            if results is not None:
                edit = tuple(_diff(_current_line(view), view.candidates[int(action.answer)]))
                self.edits[edit][0] += int(all(results))
                self.edits[edit][1] += 1
            return learned
        key = self._key(view)
        target = target_from_feedback(view, feedback)
        if key is not None and target is not None:
            self.table[(view.family, key)][target] += 1
        return learned

    def act(self, view: TaskView) -> Action:
        if view.family == "syntax":
            return Action("invalid" if _looks_invalid(view.source.rstrip("\n").split("\n")) else "valid", 1.0)
        if view.family == "localize":
            line = first_risky_line(view.source)
            labels = valid_labels(view)
            return Action(line if line in labels else labels[-1], 1.0)
        if view.family == REPAIR:
            if self.training:
                return super().act(view)
            current = _current_line(view)
            scored = []
            for index, candidate in enumerate(view.candidates):
                if candidate == current:
                    continue
                passed, tried = self.edits.get(tuple(_diff(current, candidate)), (0, 0))
                scored.append(((passed + 0.5) / (tried + 1.0), -index, index))
            best = max(scored)
            return Action(best[2], float(best[0]))
        counts = self.table.get((view.family, self._key(view)))
        if counts:
            labels = _labels(view)
            options = [label for label in counts if label in labels]
            if options:
                best = max(options, key=lambda label: counts[label])
                return Action(best, counts[best] / sum(counts[label] for label in options))
        return self.majority(view)


class Medoid(Rules):
    name = "medoid"

    def act(self, view: TaskView) -> Action:
        if view.family != REPAIR or self.training:
            return super().act(view)
        current = _current_line(view)
        options = [i for i, c in enumerate(view.candidates) if c != current]
        best = min(
            options, key=lambda i: (sum(_token_distance(view.candidates[i], o) for o in view.candidates), i)
        )
        return Action(best, 1.0 / len(options))


class VisibleTests(Rules):
    """The tool baseline; the runner supplies ``tool_results`` before :meth:`act`."""

    name = "visible_tests"
    uses_tool = True

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.tool_results: dict[int, Sequence[Sequence[bool]]] = {}

    def act(self, view: TaskView) -> Action:
        if view.family != REPAIR or self.training:
            return super().act(view)
        results = self.tool_results.get(view.task_ref)
        if results is None:
            raise RuntimeError("the visible-test baseline was not given its tool results")
        current = _current_line(view)
        passing = [i for i, r in enumerate(results) if all(r) and view.candidates[i] != current]
        if passing:
            return Action(passing[0], 1.0 / len(passing))
        return super().act(view)


BASELINES = {
    "uniform": Uniform,
    "majority": Majority,
    "lookup": Lookup,
    "v0_heuristic": V0Heuristic,
    "rules": Rules,
    "medoid": Medoid,
    "visible_tests": VisibleTests,
}
