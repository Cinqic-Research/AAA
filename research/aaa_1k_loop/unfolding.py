"""The target-unfolding frame lock (M2, found after the gain and overshoot hypotheses fell).

:class:`research.aaa_1k.agents.NeuralAgent` trains on an *unfolded* target: the
revealed observation is mapped back through the public reflection map onto
the branch nearest the agent's own raw prediction
(:func:`aaa.predictors.unfold_observation`), and the displacement target is
taken from the input position::

    target = (unfold(observed, reference=raw_prediction) - input_position) / scale

At a genuine wall crossing that is right: the raw prediction is past the wall
and so is the pre-image of the observation. But the branch is chosen by the
learner's *own* prediction, so the rule can confirm itself. An exploratory
trace of a diverged cell (iteration 0004, overshoot block, disclosed as
post-hoc) showed the champion's raw prediction left just past the lower wall
after a slow bounce; every later observation was then unfolded onto the
mirrored branch, giving a target of roughly ``-2 x distance-from-wall /
scale`` -- a displacement that grows without bound as the dot moves away --
which pulled the raw prediction further past the wall and kept the mirror
branch selected. The scored (reflected) prediction still tracked the dot, so
the error visible to the scorer stayed moderate while the training target
ran away.

Vocabulary used by the round-4 hypotheses:

mirror step
    a trained step whose unfolded observation is not the observation itself.
lock
    a run of at least :data:`LOCK_RUN` consecutive trained mirror steps. A
    genuine crossing produces one or two; the dynamics cannot produce ten.

:class:`UnfoldTraceAgent` records the branch on every trained step without
changing anything. :class:`DeadReckoningUnfoldAgent` is a *probe*: identical
except that the branch is chosen by a prediction-independent public reference
(input position plus the tracker's velocity estimate), which removes the
self-reference while keeping the unfolding.
"""

from __future__ import annotations

from typing import Any

from aaa.predictors import unfold_observation
from research.aaa_1k.agents import NeuralAgent

LOCK_RUN = 10


def longest_runs(flags: list[bool]) -> list[tuple[int, int]]:
    """``(start, length)`` of every maximal run of ``True``."""

    runs = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(flags) - start))
    return runs


class UnfoldTraceAgent(NeuralAgent):
    """A :class:`NeuralAgent` that records, for each trained step, whether the target was mirrored."""

    def __init__(self, model: Any, *, name: str, **kwargs: Any) -> None:
        super().__init__(model, name=name, **kwargs)
        self.mirror_trace: list[bool] = []
        self.target_trace: list[float] = []

    def _reference(self) -> float:
        assert self._raw_prediction is not None
        return self._raw_prediction

    def _training_target(self, revealed: float) -> float:
        if self._input_position is None:
            raise RuntimeError("no prediction to build a target for")
        effective = revealed
        if self.unfold_target and self._raw_prediction is not None:
            effective = unfold_observation(
                revealed, self._reference(), self.scales.lower_bound, self.scales.upper_bound
            )
        target = (effective - self._input_position) / self.scales.displacement_scale
        self.mirror_trace.append(effective != revealed)
        self.target_trace.append(target)
        return target


class DeadReckoningUnfoldAgent(UnfoldTraceAgent):
    """Probe: choose the unfolding branch by input position + velocity estimate, not by the prediction.

    The tracker has not yet accepted the revealed observation when the target
    is built, so its velocity estimate is the one the prediction used.
    """

    def _reference(self) -> float:
        assert self._input_position is not None
        return self._input_position + self.tracker.velocity_estimate()


class GatedUnfoldAgent(UnfoldTraceAgent):
    """Candidate c9: the champion's unfolding, refusing a physically impossible crossing.

    The branch is still chosen by the learner's own raw prediction, exactly as
    the champion chooses it, so a genuine wall crossing is unfolded exactly as
    before (``AAA-120``). A mirrored branch ``u`` is refused -- the folded
    observation is used instead -- only when the one-step crossing it implies
    is longer than the observed motion can account for::

        |u - p| > |v| + |y - p|

    where ``p`` is the input position, ``y`` the revealed observation and ``v``
    the tracker's public velocity estimate (the previous displacement). For a
    genuine crossing at constant speed ``L``, ``|u - p| = L`` and ``|v| = L``,
    so the gate never fires on one. It fires where the champion mirrors a
    transition that crossed nothing: throughout a frame lock (where the implied
    step is about twice the distance from the wall, from the lock's second
    step) and, on smooth motion, on the step *after* a bounce when the raw
    prediction is still past the wall -- a small frame-mixed target the
    champion makes too. The bound is derived from the reflection geometry, not
    fitted to data.
    """

    def __init__(self, model: Any, *, name: str, **kwargs: Any) -> None:
        super().__init__(model, name=name, **kwargs)
        self.gate_trace: list[bool] = []

    def _training_target(self, revealed: float) -> float:
        if self._input_position is None:
            raise RuntimeError("no prediction to build a target for")
        effective = revealed
        gated = False
        if self.unfold_target and self._raw_prediction is not None:
            effective = unfold_observation(
                revealed, self._raw_prediction, self.scales.lower_bound, self.scales.upper_bound
            )
            if effective != revealed:
                implied = abs(effective - self._input_position)
                bound = abs(self.tracker.velocity_estimate()) + abs(revealed - self._input_position)
                if implied > bound:
                    effective, gated = revealed, True
        target = (effective - self._input_position) / self.scales.displacement_scale
        self.mirror_trace.append(effective != revealed)
        self.target_trace.append(target)
        self.gate_trace.append(gated)
        return target


class ReachGatedUnfoldAgent(UnfoldTraceAgent):
    """Candidate c10: the champion's unfolding, refused only where the wall was out of reach.

    A mirrored target asserts that the dot crossed a wall during this
    transition. That requires the input position ``p`` to lie within one step
    of the crossed wall; the observed motion bounds a step by ``|v| + |y - p|``
    (previous displacement plus observed displacement). So the mirrored branch
    is refused -- the folded observation used instead -- only when::

        distance(p, crossed wall) > |v| + |y - p|

    This is a necessary condition for any one-step crossing, weaker than c9's
    ``|u - p|`` bound: the champion's short mirror runs right after a bounce
    (where ``p`` is still next to the wall, and which iteration 0005 showed are
    useful on smooth motion) pass unchanged, and a frame lock is refused once
    the input has moved beyond reach of the wall.
    """

    def __init__(self, model: Any, *, name: str, **kwargs: Any) -> None:
        super().__init__(model, name=name, **kwargs)
        self.gate_trace: list[bool] = []

    def _training_target(self, revealed: float) -> float:
        if self._input_position is None:
            raise RuntimeError("no prediction to build a target for")
        effective = revealed
        gated = False
        lower, upper = self.scales.lower_bound, self.scales.upper_bound
        if self.unfold_target and self._raw_prediction is not None:
            effective = unfold_observation(revealed, self._raw_prediction, lower, upper)
            if effective != revealed:
                p = self._input_position
                distance = p - lower if effective < lower else upper - p
                reach = abs(self.tracker.velocity_estimate()) + abs(revealed - p)
                if distance > reach:
                    effective, gated = revealed, True
        target = (effective - self._input_position) / self.scales.displacement_scale
        self.mirror_trace.append(effective != revealed)
        self.target_trace.append(target)
        self.gate_trace.append(gated)
        return target
