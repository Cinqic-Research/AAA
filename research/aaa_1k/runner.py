"""The AAA-1K causal loop, and the matched online/frozen branch construction.

Required temporal order for every scored transition, enforced here and nowhere
else:

1. the agent holds only causally available observations;
2. every agent produces its prediction;
3. predictions are recorded;
4. the stream advances;
5. the permitted observation -- and only that -- is revealed;
6. the pre-reveal prediction is scored against latent truth;
7. only then may an enabled learner update;
8. the revealed observation, or a hold, enters the agent's own state.

``accept_observation`` is the single channel into an agent and it carries a
float or ``None``. There is no parameter through which a regime label, event
flag, hidden speed or future value could reach a learner even by accident.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agents import Agent, NeuralAgent, RLSAgent
from .features import PublicScales
from .streams import Stream

RUN_SCHEMA = "aaa.1k.run.v1"


@dataclass(frozen=True)
class ScoredStep:
    """One scored transition, from the evaluator's side of the boundary."""

    index: int
    target_index: int
    true_next_position: float
    target_observed: bool
    input_observed: bool
    regime: str
    event: str | None
    predictions: dict[str, float]
    absolute_errors: dict[str, float]
    normalized_absolute_errors: dict[str, float]
    signed_errors: dict[str, float]
    error_estimates: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class RunResult:
    """Everything one pass over a stream produced."""

    schema: str
    stream_summary: dict[str, Any]
    agent_names: list[str]
    steps: list[ScoredStep]

    def errors(self, name: str) -> np.ndarray:
        return np.asarray([step.normalized_absolute_errors[name] for step in self.steps], dtype=float)

    def mean_absolute_error(self, name: str) -> float:
        values = self.errors(name)
        return float(np.mean(values)) if values.size else float("nan")

    def slice(self, start: int, stop: int | None = None) -> RunResult:
        """A contiguous window, by position in the scored sequence."""

        return RunResult(
            schema=self.schema,
            stream_summary=self.stream_summary,
            agent_names=list(self.agent_names),
            steps=self.steps[start:stop],
        )

    def where(self, *, regime: str | None = None, target_observed: bool | None = None) -> RunResult:
        selected = [
            step
            for step in self.steps
            if (regime is None or step.regime == regime)
            and (target_observed is None or step.target_observed == target_observed)
        ]
        return RunResult(
            schema=self.schema,
            stream_summary=self.stream_summary,
            agent_names=list(self.agent_names),
            steps=selected,
        )


class LeakageError(RuntimeError):
    """Raised when an agent is offered something it may not have."""


def _reveal(agent: Any, value: float | None) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise LeakageError("a revealed observation must be a finite float")
    agent.accept_observation(value)


def run_stream(
    stream: Stream,
    agents: Sequence[Agent],
    *,
    start: int = 0,
    stop: int | None = None,
    begin_episode: bool = True,
    collect_diagnostics: bool = False,
    scales: PublicScales | None = None,
) -> RunResult:
    """Run every agent over one stream, scoring each prediction before reveal.

    ``start`` and ``stop`` index the stream's steps. When ``begin_episode`` is
    false the agents keep the state they already hold, which is how a branch
    continues a trunk rather than restarting it.
    """

    resolved_scales = scales or PublicScales()
    names = [agent.name for agent in agents]
    if len(set(names)) != len(names):
        raise ValueError("agent names must be unique within a run")
    last = len(stream.steps) - 1 if stop is None else min(int(stop), len(stream.steps) - 1)
    if not 0 <= start < last:
        raise ValueError("start must index a step with at least one scored transition after it")

    if begin_episode:
        for agent in agents:
            agent.begin_episode()
        seed_step = stream.steps[start]
        for agent in agents:
            # The first observation is always given; a stream that hides it is
            # rejected when it is constructed.
            _reveal(agent, seed_step.true_position)

    width = resolved_scales.width
    scored: list[ScoredStep] = []
    for index in range(start, last):
        current = stream.steps[index]
        target = stream.steps[index + 1]

        predictions = {agent.name: float(agent.predict()) for agent in agents}
        for name, value in predictions.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"agent {name} produced a non-finite prediction")

        truth = float(target.true_position)
        absolute = {name: abs(value - truth) for name, value in predictions.items()}
        estimates = {agent.name: float(getattr(agent, "error_estimate", float("nan"))) for agent in agents}
        diagnostics = (
            {agent.name: dict(agent.diagnostics()) for agent in agents if hasattr(agent, "diagnostics")}
            if collect_diagnostics
            else {}
        )
        scored.append(
            ScoredStep(
                index=index,
                target_index=index + 1,
                true_next_position=truth,
                target_observed=target.observed,
                input_observed=current.observed,
                regime=target.regime,
                event=target.event,
                predictions=predictions,
                absolute_errors=absolute,
                normalized_absolute_errors={name: value / width for name, value in absolute.items()},
                signed_errors={name: value - truth for name, value in predictions.items()},
                error_estimates=estimates,
                diagnostics=diagnostics,
            )
        )

        # Scoring is finished. Only now is anything revealed, and only the
        # permitted observation.
        revealed = target.true_position if target.observed else None
        for agent in agents:
            _reveal(agent, revealed)

    return RunResult(
        schema=RUN_SCHEMA,
        stream_summary=stream.to_summary(),
        agent_names=names,
        steps=scored,
    )


@dataclass
class BranchOutcome:
    """A paired online/frozen comparison from one declared branch point."""

    branch_index: int
    trunk: RunResult
    branch: RunResult
    clone_state_hash: str
    online_name: str
    frozen_name: str

    def post_branch_advantage(self) -> dict[str, float]:
        """Frozen minus online mean normalized error: positive means online helped."""

        online = self.branch.mean_absolute_error(self.online_name)
        frozen = self.branch.mean_absolute_error(self.frozen_name)
        return {
            "online_mae": online,
            "frozen_mae": frozen,
            "absolute_advantage": frozen - online,
            "relative_advantage": (frozen - online) / frozen if frozen > 0 else float("nan"),
        }


def run_online_frozen_branch(
    stream: Stream,
    trunk_agent: NeuralAgent | RLSAgent,
    *,
    branch_index: int,
    companions: Sequence[Agent] = (),
    collect_diagnostics: bool = False,
) -> BranchOutcome:
    """Run a shared prefix, clone the whole agent, then compare online to frozen.

    The clone copies weights, hidden state, previous-error state, observation
    tracker and TBPTT buffer. Its hash is recorded so that "both arms started
    identically" is a checkable fact rather than a claim. After the branch the
    frozen arm keeps running its recurrence -- only its weights stop moving.
    """

    if not 0 < branch_index < len(stream.steps) - 1:
        raise ValueError("branch_index must fall strictly inside the stream")
    companion_list = list(companions)
    trunk = run_stream(
        stream,
        [trunk_agent, *companion_list],
        start=0,
        stop=branch_index,
        begin_episode=True,
        collect_diagnostics=collect_diagnostics,
    )
    online = trunk_agent.branch(name=f"{trunk_agent.name}__online", update_enabled=True)
    frozen = trunk_agent.branch(name=f"{trunk_agent.name}__frozen", update_enabled=False)
    if isinstance(online, NeuralAgent) and isinstance(frozen, NeuralAgent):
        clone_hash = online.interaction_state_hash()
        if clone_hash != frozen.interaction_state_hash():
            raise RuntimeError("online and frozen clones did not start from identical interaction state")
    else:
        clone_hash = "n/a"
    branch = run_stream(
        stream,
        [online, frozen, *companion_list],
        start=branch_index,
        stop=None,
        begin_episode=False,
        collect_diagnostics=collect_diagnostics,
    )
    return BranchOutcome(
        branch_index=branch_index,
        trunk=trunk,
        branch=branch,
        clone_state_hash=clone_hash,
        online_name=online.name,
        frozen_name=frozen.name,
    )
