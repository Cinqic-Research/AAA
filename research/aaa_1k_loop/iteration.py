"""The iteration record, its state model, and a validator that fails closed.

A loop iteration is one JSON record. It names its parent champion, its
problem, its hypotheses and candidates, and a ``history`` of states it has
entered, each with the artifacts that justify entering it. The validator does
not believe the record. It checks:

* every transition is allowed by :data:`TRANSITIONS`, from ``OBSERVED``;
* every state reached has the artifact roles :data:`REQUIRED_ARTIFACTS` demands;
* every artifact exists, is strict JSON when it is JSON, and still has the
  SHA-256 the record claims (a rewritten or deleted artifact is caught);
* every rejected or abandoned candidate is still listed with a reason;
* a ``DECIDED`` outcome equals the outcome *recomputed* from the confirmation
  artifact by :func:`research.aaa_1k_loop.decision.decide` -- a stored
  ``"PROMOTE"`` is never evidence that the criteria held;
* a ``PROMOTE`` outcome has a committed freeze, a spent confirmation block and
  an unchanged parameter accounting;
* a record may not claim ``PRESERVED`` while any of the above fails.

States are deliberately few. This is a guard against ambiguous state, not a
workflow engine.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from .evidence import EvidenceError, read_strict_json

ITERATION_SCHEMA = "aaa.loop.iteration.v1"

STATES = (
    "OBSERVED",
    "CLASSIFIED",
    "DIAGNOSED",
    "TEST_DEFINED",
    "CHALLENGER_CREATED",
    "CHALLENGER_ATTACKED",
    "FROZEN",
    "CONFIRMED",
    "DECIDED",
    "REJECTED",
    "NO_INTERVENTION",
    "PRESERVED",
)
OUTCOMES = ("PROMOTE", "REJECT", "INCONCLUSIVE")

TRANSITIONS: dict[str, tuple[str, ...]] = {
    "OBSERVED": ("CLASSIFIED",),
    "CLASSIFIED": ("DIAGNOSED",),
    "DIAGNOSED": ("TEST_DEFINED",),
    "TEST_DEFINED": ("CHALLENGER_CREATED", "NO_INTERVENTION"),
    # development screening may reject every candidate
    "CHALLENGER_CREATED": ("CHALLENGER_ATTACKED", "REJECTED"),
    "CHALLENGER_ATTACKED": ("FROZEN", "REJECTED"),
    "FROZEN": ("CONFIRMED",),
    "CONFIRMED": ("DECIDED",),
    "DECIDED": ("PRESERVED",),
    "REJECTED": ("PRESERVED",),
    "NO_INTERVENTION": ("PRESERVED",),
    "PRESERVED": (),
}

REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "OBSERVED": ("champion", "observation"),
    "CLASSIFIED": (),
    "DIAGNOSED": ("diagnosis",),
    "TEST_DEFINED": ("diagnosis",),
    "CHALLENGER_CREATED": ("development",),
    "CHALLENGER_ATTACKED": ("attack",),
    "FROZEN": ("freeze",),
    "CONFIRMED": ("confirmation",),
    "DECIDED": ("confirmation",),
    "REJECTED": (),
    "NO_INTERVENTION": (),
    "PRESERVED": (),
}

TERMINAL_OUTCOME = {"REJECTED": "REJECT", "NO_INTERVENTION": "REJECT"}


class IterationError(RuntimeError):
    """Raised when an iteration record is invalid; the loop fails closed."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_entry(root: Path, path: str, role: str) -> dict[str, str]:
    """Build a record entry from a file that exists now."""

    full = root / path
    if not full.is_file():
        raise IterationError(f"artifact {path} does not exist")
    return {"path": path, "role": role, "sha256": sha256_file(full)}


def _states_reached(record: Mapping[str, Any]) -> list[str]:
    history = record.get("history")
    if not isinstance(history, list) or not history:
        raise IterationError("an iteration needs a non-empty history")
    states = []
    for entry in history:
        if not isinstance(entry, Mapping) or entry.get("state") not in STATES:
            raise IterationError(f"invalid history entry {entry!r}")
        if not isinstance(entry.get("at"), str) or not entry["at"]:
            raise IterationError(f"history entry {entry['state']} has no timestamp")
        states.append(str(entry["state"]))
    return states


def validate_transitions(states: list[str]) -> None:
    if states[0] != "OBSERVED":
        raise IterationError("an iteration must begin at OBSERVED")
    if len(set(states)) != len(states):
        raise IterationError("a state may not be entered twice")
    for previous, current in pairwise(states):
        if current not in TRANSITIONS[previous]:
            raise IterationError(f"illegal transition {previous} -> {current}")


def validate_iteration(
    record: Mapping[str, Any],
    root: Path,
    *,
    decide: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate ``record`` against the repository at ``root``; raise on any failure."""

    if record.get("schema") != ITERATION_SCHEMA:
        raise IterationError("unknown iteration schema")
    for field in ("iteration_id", "parent_champion", "problem", "classification", "hypotheses", "candidates"):
        if field not in record:
            raise IterationError(f"iteration record lacks {field!r}")
    states = _states_reached(record)
    validate_transitions(states)

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise IterationError("artifacts must be a list")
    roles: dict[str, list[Mapping[str, Any]]] = {}
    for artifact in artifacts:
        path = root / str(artifact["path"])
        if not path.is_file():
            raise IterationError(f"artifact {artifact['path']} is missing")
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise IterationError(
                f"artifact {artifact['path']} changed: recorded {artifact['sha256']}, now {actual}"
            )
        if path.suffix == ".json":
            try:
                read_strict_json(path)
            except EvidenceError as error:
                raise IterationError(str(error)) from error
        roles.setdefault(str(artifact["role"]), []).append(artifact)
    for state in states:
        missing = [role for role in REQUIRED_ARTIFACTS[state] if role not in roles]
        if missing:
            raise IterationError(f"state {state} is claimed but artifacts {missing} are absent")

    candidates = record["candidates"]
    if not isinstance(candidates, list) or (not candidates and "CHALLENGER_CREATED" in states):
        raise IterationError("a created challenger must be listed among the candidates")
    for candidate in candidates:
        if candidate.get("status") not in (
            "REJECTED_IN_DEVELOPMENT",
            "REJECTED_IN_ATTACK",
            "FROZEN",
            "NOT_ADVANCED",
        ):
            raise IterationError(f"candidate {candidate.get('candidate_id')} has invalid status")
        if candidate["status"].startswith("REJECTED") and not candidate.get("reason"):
            raise IterationError(f"rejected candidate {candidate.get('candidate_id')} lacks a reason")

    # a rejected attempt cannot be dropped from the record: every candidate an
    # artifact declares or attacks must still be listed
    listed = {candidate.get("candidate_id") for candidate in candidates}
    for role in ("development", "attack"):
        for artifact in roles.get(role, []):
            content = read_strict_json(root / str(artifact["path"]))
            named = [entry["candidate_id"] for entry in content.get("candidates_declared", [])]
            named += [content[key] for key in ("challenger_id", "claim_id") if key in content]
            dropped = sorted(set(named) - listed)
            if dropped:
                raise IterationError(
                    f"{artifact['path']} names candidates missing from the record: {dropped}"
                )

    terminal = states[-2] if states[-1] == "PRESERVED" and len(states) > 1 else states[-1]
    outcome = record.get("outcome")
    if terminal in TERMINAL_OUTCOME:
        if outcome != TERMINAL_OUTCOME[terminal]:
            raise IterationError(f"a {terminal} iteration must record outcome {TERMINAL_OUTCOME[terminal]}")
        if any(candidate["status"] == "FROZEN" for candidate in candidates):
            raise IterationError("a candidate cannot be FROZEN in an iteration that stopped before a freeze")
    recomputed: Mapping[str, Any] | None = None
    if "DECIDED" in states:
        if outcome not in OUTCOMES:
            raise IterationError(f"decided iteration has invalid outcome {outcome!r}")
        if decide is None:
            raise IterationError("a decided iteration can only be validated with a decision recomputation")
        confirmation = read_strict_json(root / str(roles["confirmation"][0]["path"]))
        recomputed = decide(confirmation)
        if recomputed["outcome"] != outcome:
            raise IterationError(
                f"recorded outcome {outcome} but criteria recompute to {recomputed['outcome']}"
            )
        frozen = [candidate for candidate in candidates if candidate["status"] == "FROZEN"]
        if len(frozen) != 1:
            raise IterationError("a decided iteration must have exactly one frozen challenger")
    elif outcome == "PROMOTE":
        raise IterationError("PROMOTE requires a decided confirmation")
    if outcome == "PROMOTE":
        accounting = record.get("accounting", {})
        if accounting.get("parameters_after") != accounting.get("parameters_counted_from_arrays"):
            raise IterationError("parameter accounting disagrees with the counted arrays")
    return {
        "iteration_id": record["iteration_id"],
        "states": states,
        "outcome": outcome,
        "artifacts_verified": len(artifacts),
        "recomputed_outcome": None if recomputed is None else recomputed["outcome"],
        "status": "VALID",
    }
