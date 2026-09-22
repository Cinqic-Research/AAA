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
