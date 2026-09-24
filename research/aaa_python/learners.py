"""The minimal adaptive candidate, its controls, and deterministic baselines.

Every agent has the same interface -- ``act(view) -> Action`` and
``learn(view, action, feedback)`` -- and learns **only** from post-action
:class:`~.episode.Feedback`. No agent is ever given a task's answer key.

``OnlineLinear`` (the candidate)
    One linear softmax head per family over hashed features
    (:mod:`.representation`); repair uses a single linear candidate scorer
    trained from bandit feedback on the chosen candidate only. Plain SGD: no
    optimizer state. It is deliberately the *smallest* thing that exercises
    representation, causal prediction, online learning, feedback, persistence,
    cloning and checkpointing. It is not an architecture claim.

Controls of the same model
    ``frozen``            identical state, updates disabled;
    ``feedback_disabled`` identical state, the environment withholds feedback;
    ``reset_each_task``   memory disabled: restored to its untrained state before
                          every task.

Baselines (CPython itself is the oracle and is never a baseline)
    ``uniform``    chance over the valid labels;
    ``majority``   most frequent label seen in training feedback;
    ``lookup``     exact-match memory of normalized training source, falling
                   back to majority on a miss (its hit rate is reported);
    ``heuristic``  simple deterministic surface rules per family.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import representation as rep_module
from .episode import Action, Feedback, TaskView
from .generator import source_hash
from .rng import Stream

REPAIR = "repair"
STATE_SCHEMA = "aaa.python.v0.learner_state.v1"


class StateError(ValueError):
    pass


def valid_labels(view: TaskView) -> tuple[Any, ...]:
    if view.family == "localize":
        lines = view.source.rstrip("\n").count("\n") + 1
        return tuple(label for label in view.labels if label <= lines)
    return view.labels


def target_from_feedback(view: TaskView, feedback: Feedback | None) -> Any:
    """The supervised target that the revealed feedback implies, or ``None``."""

    if feedback is None:
        return None
    fields = feedback.fields
    if view.family == "syntax":
        return "valid" if fields.get("compile_status") == "valid" else "invalid"
    if view.family == "outcome":
        return "ok" if fields.get("status") == "ok" else fields.get("exception")
    if view.family == "output":
        text = (fields.get("stdout") or "").strip()
        try:
            value = int(text)
        except ValueError:
            return None
        return value if value in view.labels else None
    if view.family == "localize":
        line = fields.get("error_line")
        return line if line in view.labels else None
    return None


class Agent:
    name = "agent"
    adaptive = False

    def act(self, view: TaskView) -> Action:
        raise NotImplementedError

    def learn(self, view: TaskView, action: Action, feedback: Feedback | None) -> bool:
        del view, action, feedback  # a non-learning agent ignores feedback
        return False

    def begin_task(self) -> None:
        """Called before each task; the memory-disabled control resets here."""


class OnlineLinear(Agent):
    adaptive = True

    def __init__(
        self,
        *,
        seed: int,
        representation: str,
        dimensions: int,
        learning_rate: float,
        init_scale: float,
        abstain_below: float,
        label_spaces: Mapping[str, Sequence[Any]],
        name: str = "online",
    ) -> None:
        if representation not in rep_module.REPRESENTATIONS:
            raise ValueError(f"unknown representation {representation!r}")
        self.name = name
        self.seed = seed
        self.representation = representation
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.abstain_below = abstain_below
        self.update_enabled = True
        self.reset_each_task = False
        self.labels = {family: tuple(labels) for family, labels in label_spaces.items()}
        rng = np.random.default_rng(seed)
        self.weights: dict[str, np.ndarray] = {
            family: rng.normal(0.0, init_scale, size=(len(labels), dimensions))
            for family, labels in sorted(self.labels.items())
            if family != REPAIR
        }
        self.weights[REPAIR] = rng.normal(0.0, init_scale, size=(1, dimensions))
        self.updates = 0
        self._initial: dict[str, np.ndarray] | None = None
        self._initial_updates = 0
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    # ---------------------------------------------------------------- features
    def _vector(self, text: str, family: str) -> np.ndarray:
        representation = rep_module.effective_representation(self.representation, family)
        key = (representation, text)
        cached = self._cache.get(key)
        if cached is None:
            cached = rep_module.vector(text, representation, self.dimensions)
            if len(self._cache) < 50000:
                self._cache[key] = cached
        return cached

    def _candidates(self, view: TaskView) -> np.ndarray:
        """Cached equivalent of :func:`.representation.candidate_features`."""

        assert view.repair_line is not None
        lines = view.source.rstrip("\n").split("\n")
        rows = []
        for candidate in view.candidates:
            patched = "\n".join([*lines[: view.repair_line - 1], candidate, *lines[view.repair_line :]])
            rows.append(0.5 * self._vector(patched, REPAIR) + 0.5 * self._vector(candidate, "fragment"))
        return np.array(rows)

    def _probabilities(self, view: TaskView) -> tuple[tuple[Any, ...], np.ndarray]:
        if view.family == REPAIR:
            scores = self._candidates(view) @ self.weights[REPAIR][0]
            labels = tuple(range(len(view.candidates)))
        else:
            labels = valid_labels(view)
            head = self.weights[view.family]
            rows = [self.labels[view.family].index(label) for label in labels]
            scores = head[rows] @ self._vector(view.source, view.family)
        scores = scores - scores.max()
        probabilities = np.exp(scores)
        return labels, probabilities / probabilities.sum()

    # ---------------------------------------------------------------- interface
    def begin_task(self) -> None:
        if self.reset_each_task:
            if self._initial is None:
                raise StateError("reset_each_task requires a remembered untrained state")
            self.weights = {k: v.copy() for k, v in self._initial.items()}
            self.updates = self._initial_updates

    def remember_initial(self) -> None:
        self._initial = {k: v.copy() for k, v in self.weights.items()}
        self._initial_updates = self.updates

    def act(self, view: TaskView) -> Action:
        labels, probabilities = self._probabilities(view)
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        return Action(labels[best], confidence, abstain=confidence < self.abstain_below)

    def learn(self, view: TaskView, action: Action, feedback: Feedback | None) -> bool:
        if not self.update_enabled or feedback is None:
            return False
        if view.family == REPAIR:
            results = feedback.fields.get("chosen_candidate_hidden_results")
            if results is None or action.abstain:
                return False
            features = self._candidates(view)[int(action.answer)]
            reward = 1.0 if all(results) else 0.0
            score = float(features @ self.weights[REPAIR][0])
            self.weights[REPAIR][0] += (
                self.learning_rate * (reward - 1.0 / (1.0 + math.exp(-score))) * features
            )
        else:
            target = target_from_feedback(view, feedback)
            if target is None:
                return False
            labels, probabilities = self._probabilities(view)
            if target not in labels:
                return False
            gradient = -probabilities
            gradient[labels.index(target)] += 1.0
            rows = [self.labels[view.family].index(label) for label in labels]
            self.weights[view.family][rows] += self.learning_rate * np.outer(
                gradient, self._vector(view.source, view.family)
            )
        self.updates += 1
        return True

    # ---------------------------------------------------------------- state
    def clone(self, name: str | None = None) -> OnlineLinear:
        other = copy.copy(self)
        other.weights = {k: v.copy() for k, v in self.weights.items()}
        other._initial = None if self._initial is None else {k: v.copy() for k, v in self._initial.items()}
        other._cache = self._cache  # features are pure functions of text; sharing them changes no state
        if name is not None:
            other.name = name
        return other

    def parameter_count(self) -> dict[str, Any]:
        per_head = {family: int(w.size) for family, w in sorted(self.weights.items())}
        return {
            "trainable": sum(per_head.values()),
            "per_head": per_head,
            "optimizer_state": 0,
            "non_trainable_persistent": 1,  # the update counter
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "config": self.config(),
            "updates": self.updates,
            "weights": {family: w.tolist() for family, w in sorted(self.weights.items())},
        }

    def config(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "representation": self.representation,
            "dimensions": self.dimensions,
            "learning_rate": self.learning_rate,
            "abstain_below": self.abstain_below,
            "labels": {family: list(labels) for family, labels in sorted(self.labels.items())},
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
            raise StateError("unknown learner state schema")
        if state.get("config") != self.config():
            raise StateError("state was produced by a different learner configuration")
        updates = state.get("updates")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise StateError("update counter must be a non-negative integer")
        loaded = {}
        for family, current in self.weights.items():
            array = np.asarray(state["weights"].get(family), dtype=float)
            if array.shape != current.shape or not np.all(np.isfinite(array)):
                raise StateError(f"head {family}: wrong shape or non-finite values")
            loaded[family] = array
        if set(state["weights"]) != set(self.weights):
            raise StateError("state has missing or extra heads")
        self.weights, self.updates = loaded, updates

    def state_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.config(), sort_keys=True).encode("utf-8"))
        digest.update(str(self.updates).encode("ascii"))
        for family, weights in sorted(self.weights.items()):
            digest.update(family.encode("utf-8"))
            digest.update(np.ascontiguousarray(weights, dtype="<f8").tobytes())
        return digest.hexdigest()


# -------------------------------------------------------------------- baselines
class Uniform(Agent):
    name = "uniform"

    def __init__(self, seed: int) -> None:
        self.stream = Stream(seed)

    def act(self, view: TaskView) -> Action:
        labels = tuple(range(len(view.candidates))) if view.family == REPAIR else valid_labels(view)
        return Action(labels[self.stream.below(len(labels))], 1.0 / len(labels))


class Majority(Agent):
    name = "majority"

    def __init__(self) -> None:
        self.counts: dict[str, Counter[Any]] = {}

    def learn(self, view: TaskView, action: Action, feedback: Feedback | None) -> bool:
        if view.family == REPAIR:
            results = None if feedback is None else feedback.fields.get("chosen_candidate_hidden_results")
            target = action.answer if results is not None and all(results) else None
        else:
            target = target_from_feedback(view, feedback)
        if target is None:
            return False
        self.counts.setdefault(view.family, Counter())[target] += 1
        return True

    def majority(self, view: TaskView) -> Action:
        labels = tuple(range(len(view.candidates))) if view.family == REPAIR else valid_labels(view)
        counts = self.counts.get(view.family, Counter())
        seen = [(counts[label], -position) for position, label in enumerate(labels)]
        best = labels[-max(seen)[1]]
        total = sum(counts.values())
        return Action(best, counts[best] / total if total else 1.0 / len(labels))

    def act(self, view: TaskView) -> Action:
        return self.majority(view)


class Lookup(Majority):
    """Exact-match memory; a first-class memorization control."""

    name = "lookup"

    def __init__(self) -> None:
        super().__init__()
        self.memory: dict[tuple[str, str], Any] = {}  # (family, normalized source hash)
        self.hits = 0
        self.queries = 0

    def learn(self, view: TaskView, action: Action, feedback: Feedback | None) -> bool:
        learned = super().learn(view, action, feedback)
        if view.family == REPAIR:
            results = None if feedback is None else feedback.fields.get("chosen_candidate_hidden_results")
            target = action.answer if results is not None and all(results) else None
        else:
            target = target_from_feedback(view, feedback)
        if target is not None:
            self.memory[(view.family, source_hash(view.source, view.candidates))] = target
        return learned

    def act(self, view: TaskView) -> Action:
        self.queries += 1
        key = (view.family, source_hash(view.source, view.candidates))
        if key in self.memory:
            self.hits += 1
            return Action(self.memory[key], 1.0)
        fallback = self.majority(view)
        return Action(fallback.answer, 0.0)


_RISKY = re.compile(r"//\s*\(|\]\[|int\(\(|str\(|\+ 1$")


class Heuristic(Majority):
    """Deterministic surface rules; falls back to the training majority."""

    name = "heuristic"

    def act(self, view: TaskView) -> Action:
        lines = view.source.rstrip("\n").split("\n")
        if view.family == "syntax":
            return Action("invalid" if _looks_invalid(lines) else "valid", 1.0)
        if view.family == "localize":
            for number, line in enumerate(lines, start=1):
                if _RISKY.search(line.strip()):
                    return Action(number, 1.0)
            return Action(len(lines), 1.0)
        if view.family == "output":
            literals = [int(v) for v in re.findall(r"(?<![\w.])-?\d+", view.source)]
            guess = literals[-1] if literals else 0
            return Action(guess if guess in view.labels else 0, 1.0)
        if view.family == REPAIR:
            assert view.repair_line is not None
            current = lines[view.repair_line - 1]
            for index, candidate in enumerate(view.candidates):
                if candidate != current:
                    return Action(index, 1.0)
        return self.majority(view)


def _looks_invalid(lines: Sequence[str]) -> bool:
    text = "\n".join(lines)
    if text.count("(") != text.count(")") or text.count("[") != text.count("]") or text.count('"') % 2:
        return True
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent % 4:
            return True
        if stripped.startswith(("if ", "for ", "def ", "else")) and not stripped.endswith(":"):
            return True
        if stripped.startswith("if ") and re.search(r"[^=!<>]=[^=]", stripped):
            return True
    return False
