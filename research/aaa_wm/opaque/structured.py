"""WM-S: a structured world model. The agent's own exact interpreter plus a *learned* library model.

The agent can read the visible code, and Python's semantics are public knowledge, so the
only unknown part of the environment's dynamics is the opaque library. WM-S keeps
execution of the visible code exact and learns the library as a table
``T[api, argument] -> result`` from **real program-level observations only**
(``(program, input) -> observed result``). The library's calls are never observed
directly.

Learning is abductive constraint propagation:

1. Run an observation with the current table. If execution reaches an unknown entry, the
   observation's first *hole* is ``(api, argument)``.
2. For an observation whose run hits exactly one unknown entry, enumerate candidate values
   ``h`` for it and keep those that reproduce the observed result. Across observations of the
   same entry, intersect the candidate sets. A unique survivor is learned.
3. Repeat until no progress. Entries that were learned and later contradicted by a real
   observation become unknown again (``forget``).

The learned state is the table (counted as adaptive state). No library implementation, test
answer or reference is read. Predictions are imagination: they rank plans for the planner and
are never observations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .env import RunObservation, View
from .program import _BUILTINS, compiled

CANDIDATES = range(-400, 1001)


class Hole(Exception):
    def __init__(self, key: tuple[str, int]) -> None:
        super().__init__(key)
        self.key = key


class LibraryModel:
    """The learned table plus bookkeeping. ``names`` are the library names (visible)."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(names)
        self.table: dict[tuple[str, int], int] = {}
        self.candidates: dict[tuple[str, int], set[int]] = {}
        self.updates = 0

    # ---------------------------------------------------------------- execution
    def execute(
        self, source: str, argument: int, *, fill: dict[tuple[str, int], int] | None = None, trace: list[tuple[str, int]] | None = None
    ) -> tuple[str, Any]:
        """Run with the learned table. Raises :class:`Hole` at the first unknown entry."""

        code = compiled(source, self.names)
        if code is None:
            return ("invalid", None)
        table, extra = self.table, fill or {}

        def make(name: str) -> Any:
            def call(v: int) -> int:
                key = (name, v)
                if key in extra:
                    out = extra[key]
                elif key in table:
                    out = table[key]
                else:
                    raise Hole(key)
                if trace is not None:
                    trace.append(key)
                return out

            return call

        ns: dict[str, Any] = {"__builtins__": dict(_BUILTINS), **{n: make(n) for n in self.names}}
        try:
            exec(code, ns)
            value = ns["f"](argument)
        except Hole:
            raise
        except Exception as error:
            return ("error", type(error).__name__)
        if isinstance(value, bool) or not isinstance(value, int):
            return ("ok", repr(value))
        if abs(value) > 10**6:
            return ("error", "Overflow")
        return ("ok", value)

    def predict(self, source: str, argument: int) -> tuple[str, Any] | None:
        """The imagined result, or ``None`` if it depends on an unknown entry."""

        try:
            return self.execute(source, argument)
        except Hole:
            return None

    # ---------------------------------------------------------------- learning
    def _solve(self, source: str, argument: int, observed: tuple[str, Any], key: tuple[str, int]) -> set[int] | None:
        """Values of the single unknown ``key`` that reproduce ``observed`` (None: another hole appeared)."""

        ok: set[int] = set()
        for h in CANDIDATES:
            try:
                if self.execute(source, argument, fill={key: h}) == observed:
                    ok.add(h)
            except Hole:
                return None
        return ok

    def learn(self, observations: Iterable[tuple[str, int, tuple[str, Any]]], *, rounds: int = 8, per_key: int = 3) -> dict[str, int]:
        """Abduce table entries from observations; returns counts for the record."""

        pending = list(observations)
        stats = {"observations": len(pending), "learned": 0, "rounds": 0}
        for _ in range(rounds):
            stats["rounds"] += 1
            by_hole: dict[tuple[str, int], list[tuple[str, int, tuple[str, Any]]]] = defaultdict(list)
            rest = []
            for src, x, obs in pending:
                try:
                    self.execute(src, x)
                except Hole as hole:
                    if len(by_hole[hole.key]) < per_key:
                        by_hole[hole.key].append((src, x, obs))
                    rest.append((src, x, obs))
            progress = 0
            for key, group in by_hole.items():
                cand = self.candidates.get(key)
                for src, x, obs in group:
                    solved = self._solve(src, x, obs, key)
                    if solved is None:
                        continue
                    cand = solved if cand is None else cand & solved
                if cand is None:
                    continue
                self.candidates[key] = cand
                if len(cand) == 1:
                    self.table[key] = next(iter(cand))
                    self.updates += 1
                    progress += 1
            stats["learned"] += progress
            pending = rest
            if not progress:
                break
        return stats

    def forget(self, keys: Iterable[tuple[str, int]]) -> None:
        for k in keys:
            self.table.pop(k, None)
            self.candidates.pop(k, None)

    def state_size(self) -> int:
        return len(self.table) + sum(len(v) for v in self.candidates.values())


class StructuredPredictor:
    """Planner predictor: P(visible test passes) from the learned library + exact interpreter.

    Unknown entries give ``unknown`` probability. ``online`` = learn from the agent's own real RUN
    observations during an episode (forgetting entries a real observation contradicts).
    """

    def __init__(self, model: LibraryModel, *, unknown: float = 0.3, online: bool = False) -> None:
        self.model, self.unknown, self.online = model, unknown, online

    def __call__(self, programs: Sequence[str], view: View) -> np.ndarray:
        out = np.zeros((len(programs), len(view.visible_tests)))
        for i, src in enumerate(programs):
            for j, (x, expected) in enumerate(view.visible_tests):
                r = self.model.predict(src, x)
                out[i, j] = self.unknown if r is None else (0.999 if r == expected else 0.001)
        return out

    def observe(self, obs: RunObservation, view: View) -> None:
        if not self.online:
            return
        triples = []
        for (x, _), got in zip(view.visible_tests, obs.results, strict=True):
            trace: list[tuple[str, int]] = []
            try:
                predicted = self.model.execute(obs.source, x, trace=trace)
            except Hole:
                predicted = None
            if predicted is not None and predicted != got:
                self.model.forget(trace)  # a real observation contradicts imagination: distrust what it used
            triples.append((obs.source, x, got))
        self.model.learn(triples, rounds=3)


def observations_from_dataset(ds: Any, domain: Sequence[int], limit: int | None = None, seed: int = 0) -> list[tuple[str, int, tuple[str, Any]]]:
    """Real program-level observations from the shared behavior dataset (result classes decoded)."""

    from . import tokens as tok

    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds.sources))
    if limit is not None and limit < len(idx):
        idx = rng.choice(idx, limit, replace=False)
    out = []
    for i in idx:
        src = ds.sources[i]
        for j, x in enumerate(domain):
            c = int(ds.results[i, j])
            if c < tok.NUM_MAX - tok.NUM_MIN + 1:
                out.append((src, x, ("ok", c + tok.NUM_MIN)))
            elif c == tok.NUM_MAX - tok.NUM_MIN + 3:
                out.append((src, x, ("error", "ZeroDivisionError")))
    return out
