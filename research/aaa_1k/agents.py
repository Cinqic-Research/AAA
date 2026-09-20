"""Agents: every arm of every AAA-1K comparison, behind one causal interface.

The interface is deliberately narrow::

    begin_episode()                 start a new independent episode
    accept_observation(value|None)  the only channel into an agent
    predict() -> float              the next-position prediction, pre-reveal

``accept_observation(None)`` means *this step was not shown to you*. That is
the whole vocabulary. No scenario name, regime label, event flag, change
schedule, hidden speed, latent truth or future observation is reachable from
here, and :mod:`research.aaa_1k.runner` has no way to pass one.

Two rules applied identically to every arm
------------------------------------------
**Held observations.** When a step is not shown, the last observed position is
held. The holding pipeline carries no velocity of its own: during a gap the
displacement feature is exactly zero for every arm that uses it. Anything an
agent knows about motion across a gap has to be in its own state. The
dead-reckoning baseline is the arm that carries velocity explicitly, and it is
included precisely so that "the network remembered the velocity" has to beat
something that remembered it perfectly and for free.

**Learning only from genuine one-step transitions.** A learner updates only
when both ends of the transition were shown to it. A revealed position that
follows a gap is a multi-step displacement from the held input, not a sample of
the one-step law, so fitting it would teach the model something false. This is
the same lesson as `AAA-120`, applied to a different cause, and it binds the
neural arms and the RLS arm equally.

Frozen is not brain-dead
------------------------
A frozen neural arm runs exactly the same forward pass as its online twin. Its
hidden state keeps evolving, its previous-error input keeps evolving, and only
the weight update is skipped. Freezing the memory as well would answer a
different and much less interesting question.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from aaa.predictors import OnlineRLSPredictor, reflect_prediction, unfold_observation

from .controls import StatelessMLPControl, VanillaRNNControl
from .features import PublicScales, build_inputs
from .model import AAA1KGRU

AGENT_STATE_VERSION = "aaa.1k.agent.v1"

NeuralModel = AAA1KGRU | StatelessMLPControl | VanillaRNNControl


class Agent(Protocol):
    """Structural interface every AAA-1K arm implements."""

    name: str
    update_enabled: bool

    def begin_episode(self) -> None: ...

    def accept_observation(self, observation: float | None) -> None: ...

    def predict(self) -> float: ...


class ObservationTracker:
    """Zero-order hold plus the step distance between the last two observations.

    ``velocity_estimate`` is the average displacement per simulated step
    between the last two *observed* positions, and is zero while the current
    step is unobserved. Dividing by the elapsed step count matters: without it,
    the first revealed step after a gap of four would hand every rule a
    displacement four times too large and then blame the rule for the overshoot.
    """

    def __init__(self) -> None:
        self.known: float | None = None
        self.previous_known: float | None = None
        self.gap: int = 1
        self.observed_last: bool = False
        self._pending: int = 0
        self.steps_seen: int = 0

    def reset(self) -> None:
        self.known = None
        self.previous_known = None
        self.gap = 1
        self.observed_last = False
        self._pending = 0
        self.steps_seen = 0

    def accept(self, observation: float | None) -> None:
        self.steps_seen += 1
        if observation is None:
            self._pending += 1
            self.observed_last = False
            return
        value = float(observation)
        if not math.isfinite(value):
            raise ValueError("observation must be finite")
        self.previous_known = self.known if self.known is not None else value
        self.gap = self._pending + 1
        self.known = value
        self._pending = 0
        self.observed_last = True

    def require(self) -> float:
        if self.known is None:
            raise RuntimeError("agent has not received its first observation")
        return self.known

    def velocity_estimate(self) -> float:
        """Per-step displacement in position units, or zero while unobserved."""

        if not self.observed_last or self.known is None or self.previous_known is None:
            return 0.0
        return (self.known - self.previous_known) / float(self.gap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "known": self.known,
            "previous_known": self.previous_known,
            "gap": self.gap,
            "observed_last": self.observed_last,
            "pending": self._pending,
            "steps_seen": self.steps_seen,
        }

    def load(self, state: Mapping[str, Any]) -> None:
        expected = {"known", "previous_known", "gap", "observed_last", "pending", "steps_seen"}
        if set(state) != expected:
            raise ValueError("tracker state has invalid fields")
        known = None if state["known"] is None else float(state["known"])
        previous = None if state["previous_known"] is None else float(state["previous_known"])
        if known is not None and not math.isfinite(known):
            raise ValueError("tracker known observation must be finite")
        if previous is not None and not math.isfinite(previous):
            raise ValueError("tracker previous observation must be finite")
        for field in ("gap", "pending", "steps_seen"):
            value = state[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"tracker {field} must be a non-negative integer")
        if not isinstance(state["observed_last"], bool):
            raise ValueError("tracker observed_last must be boolean")
        gap, pending, steps_seen = state["gap"], state["pending"], state["steps_seen"]
        if gap < 1 or pending > steps_seen or (known is None) != (steps_seen == 0):
            raise ValueError("tracker state describes an impossible observation history")
        if state["observed_last"] and (known is None or pending != 0):
            raise ValueError("tracker observed-last state is inconsistent")
        if not state["observed_last"] and steps_seen > 0 and pending == 0:
            raise ValueError("tracker pending-gap state is inconsistent")
        self.known, self.previous_known = known, previous
        self.gap, self.observed_last = gap, state["observed_last"]
        self._pending, self.steps_seen = pending, steps_seen


class NeuralAgent:
    """Wraps a neural core in AAA's causal predict / reveal / error / adapt loop.

    The adapter owns the previous signed prediction error and the observation
    tracker, because those are properties of the agent's interaction history
    rather than of the network weights. They are serialized and branched here.
    """

    def __init__(
        self,
        model: NeuralModel,
        *,
        name: str,
        scales: PublicScales | None = None,
        update_enabled: bool = True,
        reflect: bool = True,
        unfold_target: bool = True,
    ) -> None:
        self.model = model
        self.name = str(name)
        self.scales = scales or PublicScales()
        self.update_enabled = bool(update_enabled)
        self.reflect = bool(reflect)
        self.unfold_target = bool(unfold_target)
        self.tracker = ObservationTracker()
        self.previous_signed_error = 0.0
        self.error_estimate: float = float("nan")
        self.skipped_targets = 0
        self.trained_steps = 0
        self.last_update: dict[str, float] = {}
        self._raw_prediction: float | None = None
        self._scored_prediction: float | None = None
        self._input_observed = False
        self._input_position: float | None = None

    # -- interface ------------------------------------------------------
    def begin_episode(self) -> None:
        self.model.reset_state()
        self.tracker.reset()
        self.previous_signed_error = 0.0
        self.error_estimate = float("nan")
        self._raw_prediction = None
        self._scored_prediction = None
        self._input_observed = False
        self._input_position = None

    def predict(self) -> float:
        known = self.tracker.require()
        inputs = build_inputs(
            self.scales,
            known_position=known,
            previous_known_position=known - self.tracker.velocity_estimate(),
            previous_signed_error=self.previous_signed_error,
        )
        output = self.model.forward(inputs)
        displacement, error_estimate = self.model.split_output(output)
        raw = known + self.scales.displacement_scale * displacement
        if not math.isfinite(raw):
            raise FloatingPointError(f"agent {self.name} produced a non-finite prediction")
        self._raw_prediction = raw
        self._input_observed = self.tracker.observed_last or self.tracker.steps_seen == 1
        self._input_position = known
        self.error_estimate = error_estimate
        scored = (
            reflect_prediction(raw, self.scales.lower_bound, self.scales.upper_bound) if self.reflect else raw
        )
        self._scored_prediction = scored
        return scored

    def accept_observation(self, observation: float | None) -> None:
        if observation is None:
            # An observation nobody showed the agent is never an update target.
            # The recurrent state keeps evolving; the weights learn nothing.
            self.skipped_targets += 1
            self.previous_signed_error = 0.0
            self.tracker.accept(None)
            return

        value = float(observation)
        trainable = self._raw_prediction is not None and self._input_observed
        if trainable:
            if self.update_enabled:
                self.last_update = self.model.learn(self._training_target(value))
                self.trained_steps += 1
        elif self._raw_prediction is not None:
            self.skipped_targets += 1
        if self._scored_prediction is not None:
            self.previous_signed_error = (value - self._scored_prediction) / self.scales.displacement_scale
        self.tracker.accept(value)

    # -- internals ------------------------------------------------------
    def _training_target(self, revealed: float) -> float:
        """Normalized displacement target for the prediction just scored.

        With ``unfold_target`` the public reflection map is inverted around the
        agent's own raw prediction, so a transition that crossed a wall becomes
        an ordinary sample of the underlying motion instead of a folded one.
        This is exactly the mechanism and exactly the public function the AAA
        core already uses for `AAA-120`; no evaluator bounce label is involved.
        """

        if self._input_position is None:
            raise RuntimeError("no prediction to build a target for")
        effective = revealed
        if self.unfold_target and self._raw_prediction is not None:
            effective = unfold_observation(
                revealed, self._raw_prediction, self.scales.lower_bound, self.scales.upper_bound
            )
        return (effective - self._input_position) / self.scales.displacement_scale

    # -- state ----------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "agent_format_version": AGENT_STATE_VERSION,
            "name": self.name,
            "update_enabled": self.update_enabled,
            "reflect": self.reflect,
            "unfold_target": self.unfold_target,
            "scales": self.scales.to_dict(),
            "previous_signed_error": self.previous_signed_error,
            "error_estimate": self.error_estimate if math.isfinite(self.error_estimate) else None,
            "tracker": self.tracker.to_dict(),
            "raw_prediction": self._raw_prediction,
            "scored_prediction": self._scored_prediction,
            "input_observed": self._input_observed,
            "input_position": self._input_position,
            "skipped_targets": self.skipped_targets,
            "trained_steps": self.trained_steps,
            "last_update": dict(self.last_update),
            "model": self.model.state_dict(),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        if state.get("agent_format_version") != AGENT_STATE_VERSION:
            raise ValueError("unsupported agent state format")
        restored_model = type(self.model).from_state_dict(state["model"])
        if state.get("scales") != self.scales.to_dict():
            raise ValueError("checkpoint public scales do not match this agent")
        for field in ("update_enabled", "reflect", "unfold_target", "input_observed"):
            if not isinstance(state.get(field), bool):
                raise ValueError(f"agent state {field} must be boolean")
        finite_fields = (
            "previous_signed_error",
            "error_estimate",
            "raw_prediction",
            "scored_prediction",
            "input_position",
        )
        values: dict[str, float | None] = {}
        for field in finite_fields:
            raw = state[field]
            value = None if raw is None else float(raw)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"agent state {field} must be finite or null")
            values[field] = value
        for field in ("skipped_targets", "trained_steps"):
            value = state[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"agent state {field} must be a non-negative integer")
        if (values["raw_prediction"] is None) != (values["scored_prediction"] is None):
            raise ValueError("agent prediction state is inconsistent")
        if values["previous_signed_error"] is None:
            raise ValueError("agent previous signed error cannot be null")
        if values["raw_prediction"] is not None and values["input_position"] is None:
            raise ValueError("agent prediction is missing its input position")
        last_update = state["last_update"]
        if not isinstance(last_update, Mapping) or not all(
            isinstance(key, str) and math.isfinite(float(value)) for key, value in last_update.items()
        ):
            raise ValueError("agent last-update diagnostics are invalid")
        self.model = restored_model
        previous_signed_error = values["previous_signed_error"]
        assert previous_signed_error is not None
        self.previous_signed_error = float(previous_signed_error)
        self.error_estimate = (
            float("nan") if values["error_estimate"] is None else float(values["error_estimate"])
        )
        self.tracker.load(state["tracker"])
        self._raw_prediction = values["raw_prediction"]
        self._scored_prediction = values["scored_prediction"]
        self._input_observed = state["input_observed"]
        self._input_position = values["input_position"]
        self.skipped_targets = state["skipped_targets"]
        self.trained_steps = state["trained_steps"]
        self.last_update = {str(key): float(value) for key, value in last_update.items()}

    def interaction_state_hash(self) -> str:
        """Hash every branch initial condition, excluding treatment labels.

        ``name`` and ``update_enabled`` intentionally differ between the online
        and frozen arms. Everything that can otherwise affect a future
        prediction or update is included.
        """

        state = self.state_dict()
        state.pop("name")
        state.pop("update_enabled")
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def branch(self, *, name: str, update_enabled: bool) -> NeuralAgent:
        """Clone the *entire* agent at a declared branch point.

        Weights, hidden state, previous-error state, observation tracker, TBPTT
        buffer and every counter are copied, with no shared arrays. Both arms
        then start bitwise identical, which is what makes the online-versus-
        frozen comparison a comparison rather than an anecdote.
        """

        clone = NeuralAgent(
            self.model.clone(),
            name=name,
            scales=self.scales,
            update_enabled=update_enabled,
            reflect=self.reflect,
            unfold_target=self.unfold_target,
        )
        clone.load_state(self.state_dict())
        clone.name = name
        clone.update_enabled = update_enabled
        return clone

    def diagnostics(self) -> dict[str, float]:
        return dict(self.model.diagnostics())


# ----------------------------------------------------------------------
# analytic baselines, on exactly the same observable stream
# ----------------------------------------------------------------------
class _HoldingAgent:
    """Baselines that read the held observation stream and never learn."""

    def __init__(self, name: str, scales: PublicScales | None) -> None:
        self.name = name
        self.update_enabled = False
        self.scales = scales or PublicScales()
        self.tracker = ObservationTracker()

    def begin_episode(self) -> None:
        self.tracker.reset()

    def accept_observation(self, observation: float | None) -> None:
        self.tracker.accept(observation)

    def _finish(self, raw: float, reflect: bool) -> float:
        if not reflect:
            return raw
        return reflect_prediction(raw, self.scales.lower_bound, self.scales.upper_bound)


class PersistenceAgent(_HoldingAgent):
    """``x_hat[t+1] = x_hat[t]``. The floor every other arm has to beat."""

    def __init__(self, *, name: str = "persistence", scales: PublicScales | None = None) -> None:
        super().__init__(name, scales)

    def predict(self) -> float:
        return self.tracker.require()


class ConstantMotionAgent:
    """One step of the last observed per-step velocity, optionally reflected.

    During a gap the velocity estimate is zero and this degrades to
    persistence. That is a real property of an arm with no memory, not a
    handicap: it receives exactly the same observable stream as everything else.
    """

    def __init__(
        self,
        *,
        name: str = "constant_motion",
        scales: PublicScales | None = None,
        reflect: bool = False,
    ) -> None:
        self.name = name
        self.update_enabled = False
        self.scales = scales or PublicScales()
        self.reflect = bool(reflect)
        self.tracker = ObservationTracker()

    def begin_episode(self) -> None:
        self.tracker.reset()

    def accept_observation(self, observation: float | None) -> None:
        self.tracker.accept(observation)

    def predict(self) -> float:
        raw = self.tracker.require() + self.tracker.velocity_estimate()
        if not self.reflect:
            return raw
        return reflect_prediction(raw, self.scales.lower_bound, self.scales.upper_bound)


class DeadReckoningAgent:
    """Extrapolates with the last observed velocity, straight through gaps.

    This is the analytic arm that *does* carry memory, and it carries it
    perfectly. It exists so a recurrent network cannot be credited with
    "remembering the velocity" unless it beats a free rule that remembers it.
    """

    def __init__(
        self,
        *,
        name: str = "dead_reckoning",
        scales: PublicScales | None = None,
        reflect: bool = True,
    ) -> None:
        self.name = name
        self.update_enabled = False
        self.scales = scales or PublicScales()
        self.reflect = bool(reflect)
        self.estimate: float | None = None
        self.velocity: float = 0.0
        self._tracker = ObservationTracker()

    def begin_episode(self) -> None:
        self.estimate = None
        self.velocity = 0.0
        self._tracker.reset()

    def accept_observation(self, observation: float | None) -> None:
        self._tracker.accept(observation)
        if observation is None:
            if self.estimate is not None:
                self.estimate, self.velocity = self._advance(self.estimate, self.velocity)
            return
        value = float(observation)
        estimate = self._tracker.velocity_estimate()
        if self.estimate is not None:
            self.velocity = estimate if estimate != 0.0 else self.velocity
        self.estimate = value

    def _advance(self, position: float, velocity: float) -> tuple[float, float]:
        raw = position + velocity
        if raw > self.scales.upper_bound or raw < self.scales.lower_bound:
            return (
                reflect_prediction(raw, self.scales.lower_bound, self.scales.upper_bound),
                -velocity,
            )
        return raw, velocity

    def predict(self) -> float:
        if self.estimate is None:
            raise RuntimeError("agent has not received its first observation")
        raw = self.estimate + self.velocity
        if not self.reflect:
            return raw
        return reflect_prediction(raw, self.scales.lower_bound, self.scales.upper_bound)


class WindowedLinearFitAgent:
    """Least-squares line through the last ``window`` observed positions.

    The memory baseline for the coarse-observation family: averaging over a
    window is exactly the cheap way to recover a velocity that a single
    quantized transition hides. If a recurrent network cannot beat a straight
    line fitted to eight points, it has not earned its hidden state.
    """

    def __init__(
        self,
        *,
        name: str = "linear_fit",
        scales: PublicScales | None = None,
        window: int = 8,
        reflect: bool = True,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self.name = name
        self.update_enabled = False
        self.scales = scales or PublicScales()
        self.window = int(window)
        self.reflect = bool(reflect)
        self.times: list[int] = []
        self.values: list[float] = []
        self.clock = 0

    def begin_episode(self) -> None:
        self.times = []
        self.values = []
        self.clock = 0

    def accept_observation(self, observation: float | None) -> None:
        if observation is not None:
            self.times.append(self.clock)
            self.values.append(float(observation))
            if len(self.times) > self.window:
                self.times.pop(0)
                self.values.pop(0)
        self.clock += 1

    def predict(self) -> float:
        if not self.values:
            raise RuntimeError("agent has not received its first observation")
        if len(self.values) < 2:
            return float(self.values[-1])
        times = np.asarray(self.times, dtype=float)
        values = np.asarray(self.values, dtype=float)
        slope, intercept = np.polyfit(times, values, 1)
        raw = float(slope * self.clock + intercept)
        if not math.isfinite(raw):
            raw = float(self.values[-1])
        if not self.reflect:
            return raw
        return reflect_prediction(raw, self.scales.lower_bound, self.scales.upper_bound)


class RLSAgent:
    """The existing three-parameter AAA candidate, on the AAA-1K stream.

    The learner itself is unmodified. It is fed the held observation history
    and updated only on genuine one-step transitions between two observed
    positions -- the same rule the neural arms obey.
    """

    HISTORY = 4

    def __init__(
        self,
        *,
        name: str = "rls_online",
        scales: PublicScales | None = None,
        update_enabled: bool = True,
        predictor: OnlineRLSPredictor | None = None,
    ) -> None:
        self.name = name
        self.scales = scales or PublicScales()
        self.update_enabled = bool(update_enabled)
        self.predictor = predictor or self.build_predictor(self.scales, name, self.update_enabled)
        self.history: list[float] = []
        self.observed: list[bool] = []

    @staticmethod
    def build_predictor(scales: PublicScales, name: str, update_enabled: bool) -> OnlineRLSPredictor:
        """The v2.1 candidate's declared mechanism set, copied exactly.

        Every value here is read straight from the frozen specification's
        ``candidate`` block. Weakening the incumbent to flatter a new model is
        the oldest way to manufacture a positive result, so the incumbent is
        reproduced rather than re-tuned.
        """

        return OnlineRLSPredictor(
            lower_bound=scales.lower_bound,
            upper_bound=scales.upper_bound,
            displacement_scale=scales.displacement_scale,
            forgetting=0.3,
            forgetting_mode="exponential",
            ridge=1e-4,
            feature_set="displacement_position",
            reflect=True,
            unfold_target=True,
            skip_after_reflected_prediction=True,
            trace_bound=1e5,
            dead_zone=0.0,
            detector_multiplier=8.0,
            detector_floor=1e-6,
            detector_decay=0.05,
            name=name,
            update_enabled=update_enabled,
        )

    def begin_episode(self) -> None:
        self.history = []
        self.observed = []

    def accept_observation(self, observation: float | None) -> None:
        if observation is None:
            if self.history:
                self.history.append(self.history[-1])
                self.observed.append(False)
            return
        value = float(observation)
        if len(self.history) >= self.HISTORY and self.update_enabled and self.observed and self.observed[-1]:
            self.predictor.update(tuple(self.history[-self.HISTORY :]), value)
        self.history.append(value)
        self.observed.append(True)

    def predict(self) -> float:
        if not self.history:
            raise RuntimeError("agent has not received its first observation")
        if len(self.history) < self.HISTORY:
            return float(self.history[-1])
        return float(self.predictor.predict(tuple(self.history[-self.HISTORY :])))

    def branch(self, *, name: str, update_enabled: bool) -> RLSAgent:
        clone = RLSAgent(
            name=name,
            scales=self.scales,
            update_enabled=update_enabled,
            predictor=self.predictor.clone(name=name, update_enabled=update_enabled),
        )
        clone.history = list(self.history)
        clone.observed = list(self.observed)
        return clone


def baseline_suite(scales: PublicScales | None = None) -> list[Any]:
    """Every analytic arm, constructed fresh. Order is the reporting order."""

    resolved = scales or PublicScales()
    return [
        PersistenceAgent(scales=resolved),
        ConstantMotionAgent(scales=resolved, reflect=False),
        ConstantMotionAgent(name="constant_motion_reflected", scales=resolved, reflect=True),
        DeadReckoningAgent(scales=resolved),
        WindowedLinearFitAgent(scales=resolved),
    ]


def agent_names(agents: Sequence[Any]) -> list[str]:
    return [agent.name for agent in agents]
