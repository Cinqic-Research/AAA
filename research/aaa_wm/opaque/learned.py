"""Learned agents for opaque.v0: model-free policy search and model-based planners over learned predictors.

Every agent receives only :class:`~.env.View` objects. A learned predictor's output is an
*imagined* quantity (``WORLD_MODEL_IMAGINATION``). It ranks or selects actions, and only a
real ``RUN`` (``REAL_OBSERVATION``) is ever treated as evidence about the current program.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import tokens as tok
from .agents import Planner, Prior, ToolSearch
from .env import View
from .program import Edit, apply


class PolicyScorer:
    """Scores every edit of a state with the policy head (cached per state)."""

    def __init__(self, model: Any, device: str) -> None:
        self.model, self.device = model, device
        self._cache: dict[tuple[str, tuple[Any, ...]], dict[Edit, float]] = {}

    def __call__(self, view: View, e: Edit) -> float:
        from .models import predict_policy

        key = (view.source, view.visible_tests)
        if key not in self._cache:
            es = list(view.edits())
            srcs, marks = [], []
            for x in es:
                s = apply(view.source, x)
                ids, pos = tok.encode_program(s)
                srcs.append(s)
                marks.append(next((k for k, w in enumerate(pos) if w == (x.row, x.col)), -1) + 1)
            p = predict_policy(self.model, srcs, marks, view.visible_tests, self.device)
            self._cache = {key: dict(zip(es, p.tolist(), strict=True))}
        return self._cache[key][e]


def policy_agent(model: Any, device: str, name: str = "policy") -> ToolSearch:
    return ToolSearch(PolicyScorer(model, device), name)


class WMPredictor:
    """``wm``: P(visible test j passes) = P(consequence head's result on x_j == expected_j)."""

    def __init__(self, model: Any, device: str, *, shuffle_queries: bool = False, disabled: bool = False) -> None:
        self.model, self.device, self.shuffle, self.disabled = model, device, shuffle_queries, disabled

    def __call__(self, programs: Sequence[str], view: View) -> np.ndarray:
        from .models import predict_pass

        if self.disabled:
            return np.full((len(programs), len(view.visible_tests)), 0.5)
        return predict_pass(self.model, programs, view.visible_tests, self.device, shuffle_queries=self.shuffle)


class ValuePredictor:
    """``value``: P(program correct) placed in column 0 (other columns 1): same planner, reward-only model."""

    def __init__(self, model: Any, device: str, which: int = 0) -> None:
        self.model, self.device, self.which = model, device, which

    def __call__(self, programs: Sequence[str], view: View) -> np.ndarray:
        from .models import predict_value

        p = predict_value(self.model, programs, view.visible_tests, self.device, self.which)
        out = np.ones((len(programs), len(view.visible_tests)))
        out[:, 0] = p
        return out


def planner(predictor: Any, prior: Prior, name: str, depth: int = 2) -> Planner:
    return Planner(predictor, prior, name, depth=depth)
