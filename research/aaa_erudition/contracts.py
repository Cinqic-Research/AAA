"""Typed, versioned contracts between Juniper 1's three model components.

Every message that crosses a component boundary is one of these records:
the Language Model's proposal, the World Model's prediction, the host's
action and outcome, user feedback, and every step of the adaptation
lifecycle. Records are frozen, serialize to canonical JSON, and are
validated field by field when read back, so a malformed or forged message
fails at the boundary instead of propagating.

Evaluator-only facts (the user's true intent, the scheduled regime, the
hidden dynamics) are deliberately absent. They live in
``env.toolshift.EvaluatorLabel`` and are consumed only by the scorer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
import typing
from enum import Enum
from typing import Any, TypeVar, Union

CONTRACT_VERSION = "juniper1.contracts.v1"


class ContractError(ValueError):
    """A record failed validation at a component boundary."""


class Component(str, Enum):
    LM = "lm"
    WM = "wm"
    ERUDITION = "erudition"


class Target(str, Enum):
    """Which model component(s) an adaptation request addresses."""

    NONE = "none"
    LM = "lm"
    WM = "wm"
    BOTH = "both"


class Diagnosis(str, Enum):
    STABLE = "stable"
    TRANSIENT = "transient"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FEEDBACK_UNRELIABLE = "feedback_unreliable"
    TOOL_FAILURE = "tool_failure"
    LM_DEFICIENCY = "lm_deficiency"
    WM_DEFICIENCY = "wm_deficiency"
    JOINT_DEFICIENCY = "joint_deficiency"


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def canonical_json(value: Any) -> bytes:
    """The single serialization used for every hash in the package."""

    def refuse(item: Any) -> Any:
        raise ContractError(f"value is not canonical JSON: {type(item).__name__}")

    def check(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            raise ContractError("non-finite number in a contract record")
        if isinstance(item, dict):
            for key, inner in item.items():
                if not isinstance(key, str):
                    raise ContractError("contract mappings must have string keys")
                check(inner)
        elif isinstance(item, (list, tuple)):
            for inner in item:
                check(inner)
        return item

    return json.dumps(check(value), sort_keys=True, separators=(",", ":"), default=refuse).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


R = TypeVar("R", bound="Record")


class Record:
    """Base for frozen contract records.

    Subclasses are frozen dataclasses with a ``SCHEMA`` class attribute.
    ``from_dict`` validates every field against its annotation, so a record
    read from disk or from another component is exactly as typed.
    """

    SCHEMA: typing.ClassVar[str] = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {"schema": self.SCHEMA}
        for field in dataclasses.fields(self):  # type: ignore[arg-type]
            payload[field.name] = _encode(getattr(self, field.name))
        return payload

    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls: type[R], payload: Any) -> R:
        if not isinstance(payload, dict):
            raise ContractError(f"{cls.__name__}: expected an object")
        if payload.get("schema") != cls.SCHEMA:
            raise ContractError(f"{cls.__name__}: schema {payload.get('schema')!r} is not {cls.SCHEMA!r}")
        hints = typing.get_type_hints(cls)
        names = [field.name for field in dataclasses.fields(cls)]  # type: ignore[arg-type]
        extra = set(payload) - set(names) - {"schema"}
        if extra:
            raise ContractError(f"{cls.__name__}: unexpected fields {sorted(extra)}")
        values = {}
        for name in names:
            if name not in payload:
                raise ContractError(f"{cls.__name__}: missing field {name!r}")
            values[name] = _decode(hints[name], payload[name], f"{cls.__name__}.{name}")
        return cls(**values)


def _encode(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(hint: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    if origin in (Union, types.UnionType):
        options = typing.get_args(hint)
        if value is None and type(None) in options:
            return None
        errors = []
        for option in options:
            if option is type(None):
                continue
            try:
                return _decode(option, value, where)
            except ContractError as error:
                errors.append(str(error))
        raise ContractError(f"{where}: no union member accepted the value ({'; '.join(errors)})")
    if origin is typing.Literal:
        if value not in typing.get_args(hint):
            raise ContractError(f"{where}: {value!r} is not one of {typing.get_args(hint)}")
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise ContractError(f"{where}: expected a list")
        (item_hint, _ellipsis) = typing.get_args(hint)
        return tuple(_decode(item_hint, item, f"{where}[]") for item in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ContractError(f"{where}: expected an object")
        key_hint, item_hint = typing.get_args(hint)
        if key_hint is not str:
            raise ContractError(f"{where}: only string-keyed mappings are supported")
        return {str(key): _decode(item_hint, item, f"{where}.{key}") for key, item in value.items()}
    if isinstance(hint, type) and issubclass(hint, Record):
        return hint.from_dict(value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        try:
            return hint(value)
        except ValueError as error:
            raise ContractError(f"{where}: {value!r} is not a {hint.__name__}") from error
    if hint is bool:
        if not isinstance(value, bool):
            raise ContractError(f"{where}: expected a boolean")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{where}: expected an integer")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError(f"{where}: expected a finite number")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ContractError(f"{where}: expected a string")
        return value
    if hint is Any:
        canonical_json(value)
        return value
    raise ContractError(f"{where}: unsupported annotation {hint!r}")


# ---- Episode-level messages -------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Observation(Record):
    """What every component may see before acting."""

    SCHEMA = "juniper1.observation.v1"
    episode_id: str
    step: int
    workspace: str
    state: dict[str, int]
    lower: int
    upper: int
    request: str


@dataclasses.dataclass(frozen=True)
class LMProposal(Record):
    """The Language Model's structured proposal, parsed and validated by the host.

    ``confidence`` is the mean token log-probability of the emitted tool call
    when the backend exposes it; it is evidence, not a calibrated probability.
    """

    SCHEMA = "juniper1.lm_proposal.v1"
    kind: typing.Literal["act", "abstain", "invalid"]
    op: str | None
    entity: str | None
    amount: int | None
    expected: int | None
    confidence: float | None
    revision: int
    transcript_sha256: str
    backend: str
    adapter_state: str


@dataclasses.dataclass(frozen=True)
class WMPrediction(Record):
    """The World Model's prediction for one proposed action, made before it runs."""

    SCHEMA = "juniper1.wm_prediction.v1"
    op: str
    entity: str
    amount: int
    before: int
    mean: float
    sd: float
    context: str
    wm_state: str


@dataclasses.dataclass(frozen=True)
class Action(Record):
    """What the host executed. ``op == "abstain"`` executes nothing."""

    SCHEMA = "juniper1.action.v1"
    op: str
    entity: str | None
    amount: int | None


@dataclasses.dataclass(frozen=True)
class Outcome(Record):
    """Host-observed result of executing an action."""

    SCHEMA = "juniper1.outcome.v1"
    status: typing.Literal["ok", "error", "skipped"]
    error: str | None
    state_after: dict[str, int]


@dataclasses.dataclass(frozen=True)
class Feedback(Record):
    """User feedback, revealed only after the outcome. It may be wrong."""

    SCHEMA = "juniper1.feedback.v1"
    rating: typing.Literal["satisfied", "unsatisfied", "none"]
    text: str


@dataclasses.dataclass(frozen=True)
class EpisodeRecord(Record):
    """Everything the system observed about one episode, in causal order."""

    SCHEMA = "juniper1.episode.v1"
    observation: Observation
    proposals: tuple[LMProposal, ...]
    prediction: WMPrediction | None
    action: Action
    outcome: Outcome
    feedback: Feedback
    lm_state: str
    wm_state: str


# ---- Adaptation lifecycle ----------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AdaptationRequest(Record):
    SCHEMA = "juniper1.adaptation_request.v1"
    request_id: str
    step: int
    controller: str
    diagnosis: dict[str, float]
    target: Target
    mechanisms: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AdaptationCandidate(Record):
    SCHEMA = "juniper1.adaptation_candidate.v1"
    candidate_id: str
    request_id: str
    component: Component
    mechanism: str
    parent_state: str
    candidate_state: str
    config: dict[str, Any]
    evidence: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CandidateEvaluation(Record):
    """The isolated evaluation of one candidate against its parent.

    ``target_*`` are error rates on the deficiency the candidate addresses;
    ``regression`` and ``attack`` hold parent/candidate error rates on probe
    banks it must not damage.
    """

    SCHEMA = "juniper1.candidate_evaluation.v1"
    candidate_id: str
    gate: str
    target_parent: float | None
    target_candidate: float | None
    target_n: int
    regression: dict[str, float]
    attack: dict[str, float]
    verdict: Verdict
    reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AdaptationRecord(Record):
    """Provenance for one accepted, rejected or rolled-back adaptation."""

    SCHEMA = "juniper1.adaptation_record.v1"
    record_id: str
    step: int
    kind: typing.Literal["accepted", "rejected", "rollback"]
    component: Component
    mechanism: str
    parent_state: str
    result_state: str
    request: AdaptationRequest | None
    candidate: AdaptationCandidate | None
    evaluation: CandidateEvaluation | None
    cost: dict[str, float]
    reason: str


def new_id(prefix: str, *parts: Any) -> str:
    """Deterministic identifier from its defining parts."""

    return f"{prefix}-{sha256_json([prefix, *parts])[:16]}"
