"""The Juniper 1 episode loop and adaptation lifecycle.

One ``System`` is the three components plus host infrastructure running one
stream under one experimental condition:

    observe -> LM proposes -> WM predicts -> (one WM-informed revision) ->
    host acts -> record -> reveal outcome and feedback -> score (evaluator
    side only) -> features -> controller decides -> candidates are built in
    isolation -> gate (target, regression, attack) -> commit or reject ->
    canary monitoring -> automatic or requested rollback -> continue.

The host owns execution, the store, the gate and every rollback. The
controller (the Erudition Model or a control policy) only *requests*
adaptations from a fixed menu; it cannot write state, change the gate, or
grant itself a mechanism its condition does not allow.

Causal boundary. ``Episode.label`` is read in exactly one place,
``_score``, whose result goes to the evaluator's list and never into
features, prompts, candidate evaluation or controller input. (In simulation
only, the surrogate Language Model is told which tank each request refers to;
see ``simulate``.)
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Literal, Protocol

import numpy as np

from . import mechanisms as mech
from .contracts import (
    Action,
    AdaptationCandidate,
    AdaptationRecord,
    AdaptationRequest,
    CandidateEvaluation,
    Component,
    EpisodeRecord,
    LMProposal,
    Observation,
    Outcome,
    Target,
    Verdict,
    WMPrediction,
    new_id,
)
from .language import (
    ACT_TOOLS,
    AdapterState,
    Backend,
    act_messages,
    chat_request,
    consultation_messages,
    parse_proposal,
    revision_messages,
)
from .store import MemoryStore, StateStore
from .toolshift import (
    OPS,
    Dynamics,
    Episode,
    Stream,
    manual_dynamics,
    persistent_dynamics,
    warmup_transitions,
)
from .world import Context, Posterior, Transition, WorldState, prediction_error_rate

GATE_VERSION = "aaa.erudition.gate.v0"
REVISION_MAX_SD = 1.5
WARMUP_TRANSITIONS = 24
COOLDOWN = 4
CANARY = 8
EVIDENCE_HORIZON = 25

WM_WINDOW = 16
WM_SURPRISE = -4.6  # log-likelihood below which the active context did not expect a result (about 1%)
WM_MIN = 6
GATE_ESTIMATOR_WINDOW = 10
WM_HELD = 4
WM_MIN_GAIN = 0.25
WM_MAX_ERROR = 0.25
WM_MAX_REGRESSION = 0.125

LM_MIN_REPLAY = 2
LM_MAX_REPLAY = 6
LM_MAX_REGRESSION_SET = 4
LM_MIN_GAIN = 1 / 3
LM_MAX_CHANGED = 0.25
ATTACK_NAME = "quillon"

ACTIONS: dict[str, tuple[str, ...]] = {
    "wait": (),
    "lm.alias": ("lm.alias_from_feedback",),
    "lm.experience": ("lm.dynamics_from_experience",),
    "lm.retract": ("lm.retract_dynamics",),
    "lm.from_wm": ("lm.dynamics_from_wm",),
    "wm.new": ("wm.new_context",),
    "wm.update": ("wm.update_in_place",),
    "wm.recall": ("wm.recall",),
    "joint.new": ("wm.new_context", "lm.dynamics_from_wm"),
    "joint.recall": ("wm.recall", "lm.dynamics_from_wm"),
    "joint.new_alias": ("wm.new_context", "lm.dynamics_from_wm", "lm.alias_from_feedback"),
    "rollback": (),
}
ACTION_NAMES = tuple(ACTIONS)

CONDITIONS: dict[str, frozenset[str]] = {
    "frozen": frozenset({"wait"}),
    "lm_only": frozenset({"wait", "lm.alias", "lm.experience", "lm.retract", "rollback"}),
    "wm_only": frozenset({"wait", "wm.new", "wm.update", "wm.recall", "rollback"}),
    "joint": frozenset(ACTION_NAMES),
}

FEATURES = (
    "kind_act",
    "kind_abstain",
    "kind_invalid",
    "revised",
    "confidence",
    "confidence_missing",
    "wm_disagreed",
    "wm_log_sd",
    "outcome_ok",
    "outcome_error",
    "outcome_skipped",
    "error_busy",
    "error_other",
    "consistent",
    "log_gap",
    "wm_log_error",
    "wm_miss",
    "surprise_active",
    "surprise_other_gain",
    "surprise_prior_gain",
    "satisfied",
    "unsatisfied",
    "no_rating",
    "feedback_names_other",
    "feedback_names_acted",
    "feedback_number_mismatch",
    "request_names_tank",
    "request_names_none",
    "since_lm_commit",
    "since_wm_commit",
    "lm_canary",
    "wm_canary",
    "last_accepted",
    "last_rejected",
    "notes",
    "contexts",
    "off_base_context",
    "lm_cooldown",
    "wm_cooldown",
    "last_rollback",
)


@dataclasses.dataclass(frozen=True)
class Decision:
    action: str
    diagnosis: dict[str, float]
    controller: str


@dataclasses.dataclass(frozen=True)
class ControllerView:
    """Everything a controller may see: features of past steps and the allowed menu."""

    step: int
    features: np.ndarray
    allowed: frozenset[str]


class Controller(Protocol):
    name: str

    def decide(self, view: ControllerView) -> Decision: ...


@dataclasses.dataclass
class StepInfo:
    record: EpisodeRecord
    first: LMProposal
    transition: Transition | None
    features: np.ndarray


@dataclasses.dataclass
class Canary:
    step: int
    components: tuple[Component, ...]
    parents: dict[Component, str]
    results: dict[Component, str]
    record_ids: tuple[str, ...]


def _number(text: str) -> list[int]:
    out = []
    token = ""
    for ch in text + " ":
        if ch.isdigit():
            token += ch
        elif token:
            out.append(int(token))
            token = ""
    return out


class System:
    def __init__(
        self,
        stream: Stream,
        backend: Backend,
        controller: Controller,
        condition: str,
        store: StateStore | MemoryStore,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        self.stream = stream
        self.backend = backend
        self.controller = controller
        self.condition = condition
        self.allowed = CONDITIONS[condition]
        self.store = store
        lower, upper = 0, 99
        warm = [
            (op, n, before, after)
            for op, n, before, after in warmup_transitions(stream.split, stream.index, WARMUP_TRANSITIONS)
        ]
        self.wm = WorldState.initial(OPS, warm, lower, upper)
        self.lm = AdapterState()
        self.wm_id = self.store.put("wm", self.wm.to_payload())
        self.lm_id = self.store.put("lm", self.lm.to_payload())
        self.store.initialize(Component.WM, self.wm_id)
        self.store.initialize(Component.LM, self.lm_id)
        self.initial_states = {"lm": self.lm_id, "wm": self.wm_id}
        self.steps: list[StepInfo] = []
        self.scores: list[dict[str, Any]] = []
        self.transitions: list[tuple[str, str, Transition]] = []
        self.wm_fit_upto = 0
        self.canaries: list[Canary] = []
        self.last_request: dict[Component, int] = {}
        self.last_commit: dict[Component, int] = {Component.LM: -1, Component.WM: -1}
        self.last_result = ""
        self.decisions: list[dict[str, Any]] = []
        self.cost: dict[str, float] = {
            "lm_act": 0,
            "lm_revision": 0,
            "lm_extract": 0,
            "lm_gate": 0,
            "wm_candidates": 0,
            "lm_candidates": 0,
            "seconds": 0.0,
        }

    # ---- the episode -------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]], purpose: str) -> dict[str, Any]:
        self.cost[purpose] += 1
        return self.backend.chat(chat_request(messages, ACT_TOOLS))

    def _propose(
        self, observation: Observation, adapter: AdapterState, adapter_id: str, purpose: str
    ) -> tuple[LMProposal, list[dict[str, Any]], dict[str, Any]]:
        messages = act_messages(observation, adapter)
        response = self._chat(messages, purpose)
        proposal = parse_proposal(response, 0, self.backend.backend_id, adapter_id, tuple(observation.state))
        return proposal, messages, response

    def _consequences(self, world: WorldState, op: str, before: int) -> list[tuple[int, int]]:
        context = world.context()
        return [(n, round(context.predict(op, n, before, world.lower, world.upper)[0])) for n in range(1, 10)]

    def run_episode(self, step: int) -> dict[str, Any]:
        started = time.monotonic()
        episode = Episode(self.stream, step)
        observation = episode.observation
        first, messages, response = self._propose(observation, self.lm, self.lm_id, "lm_act")
        proposals = [first]
        final = first
        prediction: WMPrediction | None = None
        if first.kind == "act":
            assert first.op and first.entity and first.amount is not None and first.expected is not None
            before = observation.state[first.entity]
            prediction = self.wm.predict(first.op, first.entity, first.amount, before, self.wm_id)
            if round(prediction.mean) != first.expected and prediction.sd <= REVISION_MAX_SD:
                revised_messages = revision_messages(
                    messages, response, prediction, self._consequences(self.wm, first.op, before)
                )
                second = self._chat(revised_messages, "lm_revision")
                final = parse_proposal(
                    second, 1, self.backend.backend_id, self.lm_id, tuple(observation.state)
                )
                proposals.append(final)
        elif first.kind == "abstain":
            behaviour = mech.departures_from_manual(self.wm)
            if behaviour:
                second = self._chat(consultation_messages(messages, response, behaviour), "lm_revision")
                final = parse_proposal(
                    second, 1, self.backend.backend_id, self.lm_id, tuple(observation.state)
                )
                proposals.append(final)
        if final.kind == "act" and len(proposals) > 1:
            assert final.op and final.entity and final.amount is not None
            prediction = self.wm.predict(
                final.op, final.entity, final.amount, observation.state[final.entity], self.wm_id
            )
        if final.kind == "act":
            action = Action(op=str(final.op), entity=final.entity, amount=final.amount)
            outcome = episode.execute(action)
        elif final.kind == "abstain":
            action = Action(op="abstain", entity=None, amount=None)
            outcome = episode.execute(action)
        else:
            action = Action(op="invalid", entity=None, amount=None)
            outcome = Outcome(status="error", error="invalid_proposal", state_after=dict(observation.state))
        feedback = episode.feedback(action, outcome)
        record = EpisodeRecord(
            observation=observation,
            proposals=tuple(proposals),
            prediction=prediction,
            action=action,
            outcome=outcome,
            feedback=feedback,
            lm_state=self.lm_id,
            wm_state=self.wm_id,
        )
        transition: Transition | None = None
        if outcome.status == "ok" and action.entity is not None and action.amount is not None:
            transition = (
                action.op,
                action.amount,
                observation.state[action.entity],
                outcome.state_after[action.entity],
            )
            self.transitions.append((observation.episode_id, action.entity, transition))
        features = self._features(record, first, transition)
        self.steps.append(StepInfo(record, first, transition, features))
        score = self._score(episode, record, first)
        self.scores.append(score)
        self._monitor_canaries(step)
        self._decide(step)
        self.cost["seconds"] += time.monotonic() - started
        return score

    # ---- evaluator side (the only reader of the hidden label) ----------------

    def _score(self, episode: Episode, record: EpisodeRecord, first: LMProposal) -> dict[str, Any]:
        label = episode.label
        action, outcome = record.action, record.outcome
        wm_correct = None
        if record.prediction is not None and outcome.status == "ok" and action.entity is not None:
            wm_correct = round(record.prediction.mean) == outcome.state_after[action.entity]
        first_entity_correct = label.feasible and first.entity == label.entity
        first_value_correct = None
        if first.kind == "act" and label.feasible and first.entity == label.entity:
            assert first.op and first.amount is not None and label.entity is not None
            after = label.regime.dynamics.apply(
                first.op, episode.state[label.entity], first.amount, episode.lower, episode.upper
            )
            first_value_correct = after == label.value
        persistent = persistent_dynamics(self.stream, record.observation.step)
        true_alias = {alias: entity for entity, alias in self.stream.aliases.items()}
        alias_notes = {n.key: n.values["tank"] for n in self.lm.notes if n.kind == "alias"}
        lm_rates = {
            "fill": (manual_dynamics().fill_rate, manual_dynamics().fill_bonus),
            "drain": (manual_dynamics().drain_rate, manual_dynamics().drain_fee),
        }
        for note in self.lm.notes:
            if note.kind == "dynamics":
                lm_rates[note.key] = (int(note.values["rate"]), int(note.values["offset"]))
        believed = Dynamics(*lm_rates["fill"], *lm_rates["drain"])
        observed: dict[str, list[tuple[int, int, int]]] = {op: [] for op in OPS}
        for note in self.lm.notes:
            if note.kind == "observation":
                v = note.values
                observed[str(v["op"])].append((int(v["n"]), int(v["before"]), int(v["after"])))

        def lm_knows(op: str, n: int) -> bool:
            # Informed either by a note stating the right behaviour, or by at least two shown
            # observations of the operation that all agree with it. Whether the model can use
            # what it is shown is a behavioural question the failure metrics answer.
            if believed.apply(op, 50, n, 0, 99) == persistent.apply(op, 50, n, 0, 99):
                return True
            rows = observed[op]
            return len(rows) >= 2 and all(persistent.apply(op, b, m, 0, 99) == a for m, b, a in rows)

        context = self.wm.context()
        return {
            "step": record.observation.step,
            "episode_id": record.observation.episode_id,
            # Evaluator-side deficiency flags for the state this episode used (before this step's decision).
            "alias_deficit": label.regime.alias_probability > 0
            and any(alias_notes.get(alias) != entity for alias, entity in true_alias.items()),
            "wrong_alias_notes": sum(true_alias.get(phrase) != tank for phrase, tank in alias_notes.items()),
            "lm_dynamics_deficit": sum(not lm_knows(op, n) for op, n in PROBE) >= 2,
            "wm_deficit": _misses(lambda op, n: round(context.predict(op, n, 50, 0, 99)[0]), persistent) >= 2,
            "success": episode.success(action, outcome),
            "feasible": label.feasible,
            "alias_used": label.alias_used,
            "language": label.regime.language,
            "dynamics": label.regime.dynamics.key,
            "events": list(label.regime.events),
            "corruption": label.regime.corruption > 0,
            "outage": label.regime.outage > 0,
            "first_kind": first.kind,
            "final_kind": record.proposals[-1].kind,
            "revised": len(record.proposals) > 1,
            "first_entity_correct": first_entity_correct,
            "first_value_correct": first_value_correct,
            "wm_correct": wm_correct,
            "satisfied": record.feedback.rating == "satisfied",
            "lm_state": record.lm_state,
            "wm_state": record.wm_state,
        }

    # ---- evidence features --------------------------------------------------

    def _features(
        self, record: EpisodeRecord, first: LMProposal, transition: Transition | None
    ) -> np.ndarray:
        final = record.proposals[-1]
        outcome, feedback, observation = record.outcome, record.feedback, record.observation
        f = dict.fromkeys(FEATURES, 0.0)
        f[f"kind_{final.kind}"] = 1.0
        f["revised"] = float(len(record.proposals) > 1)
        if first.confidence is None:
            f["confidence_missing"] = 1.0
        else:
            f["confidence"] = max(-5.0, first.confidence)
        if record.prediction is not None:
            f["wm_log_sd"] = math.log1p(record.prediction.sd)
            f["wm_disagreed"] = float(
                first.expected is not None and round(record.prediction.mean) != first.expected
            )
        f[f"outcome_{outcome.status}"] = 1.0
        if outcome.status == "error":
            f["error_busy" if outcome.error == "service_busy" else "error_other"] = 1.0
        if transition is not None and final.expected is not None:
            gap = abs(transition[3] - final.expected)
            f["consistent"] = float(gap == 0)
            f["log_gap"] = math.log1p(gap)
        if transition is not None and record.prediction is not None:
            error = abs(transition[3] - record.prediction.mean)
            f["wm_log_error"] = math.log1p(error)
            f["wm_miss"] = float(round(record.prediction.mean) != transition[3])
            surprise = self.wm.surprise(transition)
            f["surprise_active"] = max(-20.0, surprise["active"]) / 10.0
            f["surprise_other_gain"] = (
                max(-20.0, min(20.0, surprise["best_other"] - surprise["active"])) / 10.0
            )
            f["surprise_prior_gain"] = max(-20.0, min(20.0, surprise["prior"] - surprise["active"])) / 10.0
        f["satisfied"] = float(feedback.rating == "satisfied")
        f["unsatisfied"] = float(feedback.rating == "unsatisfied")
        f["no_rating"] = float(feedback.rating == "none")
        words = set(feedback.text.lower().replace(",", " ").replace(".", " ").split())
        names = set(observation.state)
        f["feedback_names_other"] = float(bool(words & (names - {record.action.entity or ""})))
        f["feedback_names_acted"] = float(record.action.entity in words)
        if transition is not None:
            f["feedback_number_mismatch"] = float(any(v != transition[3] for v in _number(feedback.text)))
        request_words = set(observation.request.lower().replace(".", " ").replace("?", " ").split())
        f["request_names_tank"] = float(bool(request_words & names))
        f["request_names_none"] = 1.0 - f["request_names_tank"]
        step = observation.step
        f["since_lm_commit"] = min(50, step - self.last_commit[Component.LM]) / 50.0
        f["since_wm_commit"] = min(50, step - self.last_commit[Component.WM]) / 50.0
        for canary in self.canaries:
            for component in canary.components:
                f["lm_canary" if component == Component.LM else "wm_canary"] = 1.0
        f["last_accepted"] = float(self.last_result == "accepted")
        f["last_rejected"] = float(self.last_result == "rejected")
        f["last_rollback"] = float(self.last_result == "rollback")
        f["notes"] = len(self.lm.notes) / 8.0
        f["contexts"] = len(self.wm.contexts) / 8.0
        f["off_base_context"] = float(self.wm.active != "ctx-0")
        f["lm_cooldown"] = float(step - self.last_request.get(Component.LM, -99) < COOLDOWN)
        f["wm_cooldown"] = float(step - self.last_request.get(Component.WM, -99) < COOLDOWN)
        return np.array([f[name] for name in FEATURES], dtype=np.float32)

    # ---- decisions ----------------------------------------------------------

    def _decide(self, step: int) -> None:
        view = ControllerView(step, np.stack([s.features for s in self.steps]), self.allowed)
        decision = self.controller.decide(view)
        entry = {"step": step, "action": decision.action, "diagnosis": decision.diagnosis, "executed": False}
        self.decisions.append(entry)
        if decision.action == "wait":
            return
        if decision.action not in self.allowed:
            entry["refused"] = "not allowed in this condition"
            return
        if decision.action == "rollback":
            entry["executed"] = self._rollback_latest(step, "requested by controller")
            return
        mechanisms = ACTIONS[decision.action]
        components = {Component.WM if m.startswith("wm.") else Component.LM for m in mechanisms}
        if any(step - self.last_request.get(c, -99) < COOLDOWN for c in components):
            entry["refused"] = "cooldown"
            return
        for component in components:
            self.last_request[component] = step
        target = Target.BOTH if len(components) == 2 else Target(next(iter(components)).value)
        request = AdaptationRequest(
            request_id=new_id("req", self.stream.stream_id, self.condition, step, decision.action),
            step=step,
            controller=decision.controller,
            diagnosis=decision.diagnosis,
            target=target,
            mechanisms=mechanisms,
            evidence=tuple(s.record.observation.episode_id for s in self.steps[-EVIDENCE_HORIZON:]),
        )
        entry["executed"] = True
        entry["results"] = self._adapt(step, request)

    def _adapt(self, step: int, request: AdaptationRequest) -> list[str]:
        results = []
        accepted: dict[Component, tuple[str, str, str]] = {}
        wm_mechanisms = [m for m in request.mechanisms if m.startswith("wm.")]
        lm_mechanisms = [m for m in request.mechanisms if m.startswith("lm.")]
        for mechanism in wm_mechanisms:
            outcome = self._wm_candidate(step, request, mechanism)
            results.append(outcome[0])
            if outcome[0] == "accepted":
                accepted[Component.WM] = outcome[1:]
        for mechanism in lm_mechanisms:
            outcome = self._lm_candidate(step, request, mechanism)
            results.append(outcome[0])
            if outcome[0] == "accepted":
                accepted[Component.LM] = outcome[1:]
        if accepted:
            self.canaries.append(
                Canary(
                    step,
                    tuple(accepted),
                    {c: v[0] for c, v in accepted.items()},
                    {c: v[1] for c, v in accepted.items()},
                    tuple(v[2] for v in accepted.values()),
                )
            )
            self.last_result = "accepted"
        elif results:
            self.last_result = "rejected"
        return results

    def _commit(
        self,
        step: int,
        component: Component,
        mechanism: str,
        request: AdaptationRequest,
        candidate_state: str,
        evaluation: CandidateEvaluation,
        config: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        parent = self.wm_id if component == Component.WM else self.lm_id
        candidate = AdaptationCandidate(
            candidate_id=new_id("cand", request.request_id, mechanism, candidate_state),
            request_id=request.request_id,
            component=component,
            mechanism=mechanism,
            parent_state=parent,
            candidate_state=candidate_state,
            config=config,
            evidence=request.evidence,
        )
        kind: Literal["accepted", "rejected"] = (
            "accepted" if evaluation.verdict == Verdict.ACCEPT else "rejected"
        )
        record = AdaptationRecord(
            record_id=new_id("rec", candidate.candidate_id, kind),
            step=step,
            kind=kind,
            component=component,
            mechanism=mechanism,
            parent_state=parent,
            result_state=candidate_state if kind == "accepted" else parent,
            request=request,
            candidate=candidate,
            evaluation=evaluation,
            # Counts only: wall-clock time would make replayed provenance differ from the original.
            cost={k: float(v) for k, v in self.cost.items() if k != "seconds"},
            reason=";".join(evaluation.reasons),
        )
        if kind == "accepted":
            self.store.commit(record)
            self.last_commit[component] = step
        else:
            self.store.record(record)
        return kind, parent, candidate_state, record.record_id

    # ---- World Model candidates ----------------------------------------------

    def _wm_candidate(
        self, step: int, request: AdaptationRequest, mechanism: str
    ) -> tuple[str, str, str, str]:
        self.cost["wm_candidates"] += 1
        evidence = [t for _, _, t in self.transitions[-WM_WINDOW:]]
        reasons: list[str] = []
        if len(evidence) < WM_MIN:
            candidate = self.wm
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            reasons.append(f"only {len(evidence)} observed transitions")
            target_parent = target_candidate = None
            regression: dict[str, float] = {}
        else:
            fit, held = evidence[:-WM_HELD], evidence[-WM_HELD:]
            if mechanism == "wm.new_context":
                # Fit only what the active context failed to expect, so a window that still
                # holds pre-shift results does not blend two regimes into one context.
                surprising = [
                    t
                    for t in fit
                    if self.wm.context().loglik([t], self.wm.lower, self.wm.upper) < WM_SURPRISE
                ]
                candidate = self.wm.new_context(surprising, step)
            elif mechanism == "wm.update_in_place":
                new = [t for _, _, t in self.transitions[self.wm_fit_upto :]][:-WM_HELD]
                candidate = self.wm.update_in_place(new)
            elif mechanism == "wm.recall":
                candidate = self.wm.recall(fit)
            else:
                raise ValueError(mechanism)
            target_parent = prediction_error_rate(self.wm, held)
            target_candidate = prediction_error_rate(candidate, held)
            regression = {}
            for context in candidate.contexts:
                if context.context_id not in {c.context_id for c in self.wm.contexts}:
                    continue
                probe = list(self.wm.context(context.context_id).transitions[-8:])
                if not probe:
                    continue
                before = _context_error(self.wm, context.context_id, probe)
                after = _context_error(candidate, context.context_id, probe)
                regression[context.context_id] = after - before
            assert target_parent is not None and target_candidate is not None
            if candidate.digest() == self.wm.digest():
                verdict, reasons = Verdict.REJECT, ["candidate is identical to its parent"]
            elif target_candidate > target_parent - WM_MIN_GAIN:
                verdict, reasons = (
                    Verdict.REJECT,
                    [f"held-out error {target_candidate:.2f} vs parent {target_parent:.2f}"],
                )
            elif target_candidate > WM_MAX_ERROR:
                verdict, reasons = (
                    Verdict.REJECT,
                    [f"held-out error {target_candidate:.2f} above {WM_MAX_ERROR}"],
                )
            elif regression and max(regression.values()) > WM_MAX_REGRESSION:
                verdict, reasons = Verdict.REJECT, [f"regression on stored contexts {regression}"]
            else:
                verdict, reasons = Verdict.ACCEPT, ["held-out improvement without regression"]
        candidate_id = self.store.put("wm", candidate.to_payload())
        evaluation = CandidateEvaluation(
            candidate_id=candidate_id,
            gate=GATE_VERSION,
            target_parent=target_parent,
            target_candidate=target_candidate,
            target_n=len(evidence[-WM_HELD:]) if len(evidence) >= WM_MIN else len(evidence),
            regression={k: float(v) for k, v in regression.items()},
            attack={},
            verdict=verdict,
            reasons=tuple(reasons),
        )
        result = self._commit(
            step, Component.WM, mechanism, request, candidate_id, evaluation, {"window": WM_WINDOW}
        )
        if result[0] == "accepted":
            self.wm = candidate
            self.wm_id = candidate_id
            self.wm_fit_upto = len(self.transitions) - WM_HELD
        return result

    # ---- Language Model candidates ------------------------------------------

    def _recent(self) -> list[StepInfo]:
        return self.steps[-EVIDENCE_HORIZON:]

    def _estimator(self) -> Context:
        """The gate's own dynamics estimate from recent observations, identical in every condition."""

        evidence = [t for _, _, t in self.transitions[-GATE_ESTIMATOR_WINDOW:]]
        return Context("gate", -1, {op: Posterior.prior() for op in OPS}, ()).fit(
            evidence, self.wm.lower, self.wm.upper
        )

    def _lm_candidate(
        self, step: int, request: AdaptationRequest, mechanism: str
    ) -> tuple[str, str, str, str]:
        self.cost["lm_candidates"] += 1
        workspace = tuple(self.steps[-1].record.observation.state)
        recent = self._recent()
        corrections: list[mech.Correction] = []
        if mechanism == "lm.alias_from_feedback":
            evidence = [
                (s.record.observation.episode_id, s.record.observation.request, s.record.feedback.text)
                for s in recent
                if s.record.feedback.rating == "unsatisfied"
            ]
            self.cost["lm_extract"] += len(evidence)
            corrections = mech.corrections_from_feedback(self.backend, evidence, workspace)
            candidate = mech.alias_from_feedback(self.lm, corrections)
        elif mechanism == "lm.dynamics_from_experience":
            rows = [(eid, tank, t, "") for eid, tank, t in self.transitions[-mech.OBSERVATION_NOTES :]]
            candidate = mech.dynamics_from_experience(self.lm, rows)
        elif mechanism == "lm.dynamics_from_wm":
            candidate = mech.dynamics_from_wm(self.lm, self.wm)
        elif mechanism == "lm.retract_dynamics":
            candidate = mech.retract_dynamics(self.lm)
        else:
            raise ValueError(mechanism)
        candidate_id = self.store.put("lm", candidate.to_payload())
        evaluation = self._evaluate_lm(mechanism, candidate, candidate_id, corrections, recent)
        result = self._commit(
            step, Component.LM, mechanism, request, candidate_id, evaluation, {"horizon": EVIDENCE_HORIZON}
        )
        if result[0] == "accepted":
            self.lm = candidate
            self.lm_id = candidate_id
        return result

    def _evaluate_lm(
        self,
        mechanism: str,
        candidate: AdapterState,
        candidate_id: str,
        corrections: list[mech.Correction],
        recent: list[StepInfo],
    ) -> CandidateEvaluation:
        reasons: list[str] = []
        if candidate.digest() == self.lm.digest():
            return CandidateEvaluation(
                candidate_id,
                GATE_VERSION,
                None,
                None,
                0,
                {},
                {},
                Verdict.REJECT,
                ("candidate is identical to its parent",),
            )
        by_id = {s.record.observation.episode_id: s for s in recent}
        estimator = self._estimator()
        if mechanism == "lm.alias_from_feedback":
            # In scope (SERAC): only episodes whose request uses a phrase this candidate changes.
            before = {n.key: n.values["tank"] for n in self.lm.notes if n.kind == "alias"}
            after = {n.key: n.values["tank"] for n in candidate.notes if n.kind == "alias"}
            changed_phrases = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
            intended = {c.episode_id: c.tank for c in corrections if c.phrase in changed_phrases}
            replay = [by_id[eid] for eid in intended if eid in by_id][-LM_MAX_REPLAY:]

            def correct(info: StepInfo, proposal: LMProposal) -> bool:
                return proposal.entity == intended[info.record.observation.episode_id]

        else:
            # Held out: an episode whose own result the candidate shows the model as a note
            # would grade the candidate on the answer it was given.
            shown = {eid for n in candidate.notes if n.kind == "observation" for eid in n.evidence}
            replay = [
                s
                for s in recent
                if s.transition is not None
                and s.first.expected is not None
                and s.transition[3] != s.first.expected
                and s.record.observation.episode_id not in shown
            ][-LM_MAX_REPLAY:]

            def correct(info: StepInfo, proposal: LMProposal) -> bool:
                if proposal.kind != "act" or proposal.entity != info.first.entity:
                    return False
                assert proposal.op and proposal.amount is not None and proposal.entity
                before = info.record.observation.state[proposal.entity]
                mean, _ = estimator.predict(
                    proposal.op, proposal.amount, before, self.wm.lower, self.wm.upper
                )
                return round(mean) == info.first.expected

        if len(replay) < LM_MIN_REPLAY:
            return CandidateEvaluation(
                candidate_id,
                GATE_VERSION,
                None,
                None,
                len(replay),
                {},
                {},
                Verdict.INSUFFICIENT_EVIDENCE,
                (f"only {len(replay)} replayable episodes",),
            )
        parent_errors = candidate_errors = 0
        for info in replay:
            obs = info.record.observation
            parent, _, _ = self._propose(obs, self.lm, self.lm_id, "lm_gate")
            proposal, _, _ = self._propose(obs, candidate, candidate_id, "lm_gate")
            parent_errors += int(not correct(info, parent))
            candidate_errors += int(not correct(info, proposal))
        target_parent = parent_errors / len(replay)
        target_candidate = candidate_errors / len(replay)
        replay_ids = {s.record.observation.episode_id for s in replay}
        regression_set = [
            s
            for s in recent
            if s.record.feedback.rating == "satisfied" and s.record.observation.episode_id not in replay_ids
        ][-LM_MAX_REGRESSION_SET:]
        changed = 0
        for info in regression_set:
            obs = info.record.observation
            parent, _, _ = self._propose(obs, self.lm, self.lm_id, "lm_gate")
            proposal, _, _ = self._propose(obs, candidate, candidate_id, "lm_gate")
            changed += int(
                (parent.kind, parent.op, parent.entity, parent.amount)
                != (proposal.kind, proposal.op, proposal.entity, proposal.amount)
            )
        changed_rate = changed / len(regression_set) if regression_set else 0.0
        latest = recent[-1].record.observation
        probe = dataclasses.replace(latest, request=f"Bring {ATTACK_NAME} to {max(latest.state.values())}.")
        parent_probe, _, _ = self._propose(probe, self.lm, self.lm_id, "lm_gate")
        candidate_probe, _, _ = self._propose(probe, candidate, candidate_id, "lm_gate")
        attack = {
            "fabrication_parent": float(parent_probe.kind == "act"),
            "fabrication_candidate": float(candidate_probe.kind == "act"),
        }
        if target_candidate > target_parent - LM_MIN_GAIN:
            verdict = Verdict.REJECT
            reasons.append(f"replay error {target_candidate:.2f} vs parent {target_parent:.2f}")
        elif changed_rate > LM_MAX_CHANGED:
            verdict = Verdict.REJECT
            reasons.append(f"changed {changed_rate:.2f} of previously satisfied episodes")
        elif attack["fabrication_candidate"] > attack["fabrication_parent"]:
            verdict = Verdict.REJECT
            reasons.append("acts on a nonexistent tank where the parent abstained")
        else:
            verdict = Verdict.ACCEPT
            reasons.append("replay improvement without regression or new fabrication")
        return CandidateEvaluation(
            candidate_id=candidate_id,
            gate=GATE_VERSION,
            target_parent=target_parent,
            target_candidate=target_candidate,
            target_n=len(replay),
            regression={"changed_satisfied": changed_rate, "regression_n": float(len(regression_set))},
            attack=attack,
            verdict=verdict,
            reasons=tuple(reasons),
        )

    # ---- monitoring and rollback ---------------------------------------------

    def _window_quality(self, start: int, stop: int) -> tuple[float, float] | None:
        window = [s for s in self.steps if start <= s.record.observation.step < stop]
        if not window:
            return None
        satisfied = sum(s.record.feedback.rating == "satisfied" for s in window) / len(window)
        acted = [
            s for s in window if s.transition is not None and s.record.proposals[-1].expected is not None
        ]
        consistent = (
            sum(s.transition[3] == s.record.proposals[-1].expected for s in acted if s.transition)
            / len(acted)
            if acted
            else satisfied
        )
        return satisfied, consistent

    def _monitor_canaries(self, step: int) -> None:
        due = [c for c in self.canaries if step - c.step >= CANARY]
        self.canaries = [c for c in self.canaries if step - c.step < CANARY]
        for canary in due:
            before = self._window_quality(canary.step - CANARY + 1, canary.step + 1)
            after = self._window_quality(canary.step + 1, step + 1)
            if before is None or after is None:
                continue
            if after[0] < before[0] - 0.25 and after[1] < before[1] - 0.25:
                # Restore the state from before the change, including anything accepted on top
                # of it since, so a rolled-back World Model never leaves notes distilled from it.
                for component in canary.components:
                    current = self.wm_id if component == Component.WM else self.lm_id
                    if current != canary.parents[component]:
                        self._rollback(
                            step, component, current, canary.parents[component], "canary regression"
                        )

    def _rollback(self, step: int, component: Component, from_state: str, to_state: str, reason: str) -> bool:
        if self.store.head(component) != from_state:
            return False
        record = AdaptationRecord(
            record_id=new_id(
                "rollback", self.stream.stream_id, self.condition, step, component.value, from_state
            ),
            step=step,
            kind="rollback",
            component=component,
            mechanism="rollback",
            parent_state=from_state,
            result_state=to_state,
            request=None,
            candidate=None,
            evaluation=None,
            cost={},
            reason=reason,
        )
        self.store.commit(record)
        payload = self.store.get(to_state)
        if component == Component.WM:
            self.wm = WorldState.from_payload(payload)
            self.wm_id = to_state
            self.wm_fit_upto = len(self.transitions)
        else:
            self.lm = AdapterState.from_payload(payload)
            self.lm_id = to_state
        self.last_commit[component] = step
        self.last_result = "rollback"
        self.canaries = [c for c in self.canaries if component not in c.components]
        return True

    def _rollback_latest(self, step: int, reason: str) -> bool:
        accepted = [r for r in self.store.lineage() if r.kind == "accepted"]
        if not accepted:
            return False
        latest = accepted[-1]
        return self._rollback(step, latest.component, latest.result_state, latest.parent_state, reason)

    def run(self) -> dict[str, Any]:
        for step in range(len(self.stream.schedule)):
            self.run_episode(step)
        return self.summary()

    def _referenced_states(self) -> dict[str, Any]:
        """Payloads of every state the run used or proposed, so recomputation needs no store."""

        ids = set(self.initial_states.values())
        for record in self.store.lineage():
            ids.update({record.parent_state, record.result_state})
            if record.candidate is not None:
                ids.add(record.candidate.candidate_state)
        return {state_id: self.store.get(state_id) for state_id in sorted(ids) if state_id}

    def summary(self) -> dict[str, Any]:
        return {
            "stream": self.stream.stream_id,
            "family": self.stream.family,
            "condition": self.condition,
            "controller": self.controller.name,
            "scores": self.scores,
            "records": [s.record.to_dict() for s in self.steps],
            "decisions": self.decisions,
            "lineage": [r.to_dict() for r in self.store.lineage()],
            # Counts only; wall time is recorded beside the run so a replay is byte-identical.
            "cost": {k: v for k, v in self.cost.items() if k != "seconds"},
            "initial_states": dict(self.initial_states),
            "states": self._referenced_states(),
            "final_states": {"lm": self.lm_id, "wm": self.wm_id},
        }


PROBE = [(op, n) for op in OPS for n in range(1, 7)]


def _misses(predict: Any, truth: Dynamics) -> int:
    return sum(predict(op, n) != truth.apply(op, 50, n, 0, 99) for op, n in PROBE)


def _context_error(world: WorldState, context_id: str, transitions: list[Transition]) -> float:
    rate = prediction_error_rate(dataclasses.replace(world, active=context_id), transitions)
    return 0.0 if rate is None else rate
