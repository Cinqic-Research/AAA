"""Deterministic stream allocation and the confirmation batch registry.

Two ideas are kept strictly apart:

``purpose``
    *What the stream is for*: the literal ``development`` purpose, or a frozen
    confirmation batch identity. The purpose, not a free-text attempt label,
    is what makes a confirmation stream fresh. Renaming an attempt no longer
    silently re-runs the same numbers.
``lineage``
    *Which training history produced a checkpoint*. Confirmation A and B share
    the ``selected`` lineage on purpose, so the pair measures generalization of
    one frozen selected model set rather than training randomness. The
    ``high_replication`` role uses independent training lineages instead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEVELOPMENT_PURPOSE = "development"
REGISTRY_SCHEMA = "aaa.confirmation_batches.v1"

BATCH_STATUSES = ("planned", "consumed", "retired")
CONFIRMATION_ROLES = ("confirmation_a", "confirmation_b")
ROLES = ("development", "high_replication", *CONFIRMATION_ROLES)


def stable_label(label: str) -> int:
    """Deterministic 32-bit label. ``hash()`` is salted per process; this is not."""

    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def trial_seed(root_seed: int, purpose: str, family: str, replica: int, episode: int) -> int:
    """Stream identity for one evaluation trial.

    Depends only on trial identity, so parallel scheduling cannot change which
    numbers a trial sees.
    """

    for name, value in (("replica", replica), ("episode", episode)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    sequence = np.random.SeedSequence(
        [int(root_seed), stable_label(purpose), stable_label(family), int(replica), int(episode)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def training_seed(root_seed: int, lineage: str, replica: int, episode: int) -> int:
    """Stream identity for one training episode of one replica."""

    return trial_seed(root_seed, f"training::{lineage}", "training", replica, episode)


def probe_seed(root_seed: int, offset: int, episode: int) -> int:
    """Fixed development probe bank, identical at every learning checkpoint."""

    return trial_seed(root_seed, "learning_probe", "constant_velocity", 0, int(offset) + int(episode))


def purpose_for(role: str, batch_id: str | None) -> str:
    """Resolve the stream purpose for a role."""

    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if role in CONFIRMATION_ROLES:
        if not batch_id:
            raise ValueError(f"role {role} requires a declared confirmation batch id")
        return f"confirmation::{batch_id}"
    if role == "high_replication":
        return "high_replication"
    return DEVELOPMENT_PURPOSE


def lineage_for(role: str, spec_ab_relationship: str) -> str:
    """Resolve the training lineage for a role."""

    if role in CONFIRMATION_ROLES:
        if spec_ab_relationship == "shared_frozen_checkpoints":
            return "selected"
        return f"independent::{role}"
    if role == "high_replication":
        return "high_replication"
    return "development"


# ---------------------------------------------------------------------------
# confirmation batch registry
# ---------------------------------------------------------------------------


@dataclass
class ConfirmationBatch:
    batch_id: str
    role: str
    status: str
    declared_at: str
    spec_hash: str
    notes: str = ""
    consumed_by: list[str] = field(default_factory=list)
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "role": self.role,
            "status": self.status,
            "declared_at": self.declared_at,
            "spec_hash": self.spec_hash,
            "notes": self.notes,
            "consumed_by": list(self.consumed_by),
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConfirmationBatch:
        status = str(value["status"])
        if status not in BATCH_STATUSES:
            raise ValueError(f"batch status must be one of {BATCH_STATUSES}, got {status!r}")
        role = str(value["role"])
        if role not in CONFIRMATION_ROLES:
            raise ValueError(f"batch role must be one of {CONFIRMATION_ROLES}, got {role!r}")
        return cls(
            batch_id=str(value["batch_id"]),
            role=role,
            status=status,
            declared_at=str(value["declared_at"]),
            spec_hash=str(value["spec_hash"]),
            notes=str(value.get("notes", "")),
            consumed_by=[str(item) for item in value.get("consumed_by", [])],
            outcome=None if value.get("outcome") is None else str(value["outcome"]),
        )


class BatchRegistryError(RuntimeError):
    """Raised when confirmation batch discipline would be violated."""


class ConfirmationBatchRegistry:
    """Tracks which confirmation batches exist and which are already spent."""

    def __init__(self, path: str | Path, batches: Sequence[ConfirmationBatch] | None = None) -> None:
        self.path = Path(path)
        self._batches: dict[str, ConfirmationBatch] = {batch.batch_id: batch for batch in (batches or ())}

    # -- io --------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> ConfirmationBatchRegistry:
        source = Path(path)
        if not source.exists():
            return cls(source)
        value = json.loads(source.read_text(encoding="utf-8"))
        if value.get("schema_version") != REGISTRY_SCHEMA:
            raise BatchRegistryError(
                f"unsupported confirmation batch registry schema {value.get('schema_version')!r}"
            )
        batches = [ConfirmationBatch.from_dict(item) for item in value.get("batches", [])]
        identifiers = [batch.batch_id for batch in batches]
        if len(set(identifiers)) != len(identifiers):
            raise BatchRegistryError("confirmation batch ids must be unique")
        return cls(source, batches)

    def save(self) -> None:
        payload = {
            "schema_version": REGISTRY_SCHEMA,
            "batches": [
                batch.to_dict() for batch in sorted(self._batches.values(), key=lambda b: b.batch_id)
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    # -- queries ---------------------------------------------------------
    def __contains__(self, batch_id: object) -> bool:
        return batch_id in self._batches

    def get(self, batch_id: str) -> ConfirmationBatch:
        if batch_id not in self._batches:
            raise BatchRegistryError(
                f"confirmation batch {batch_id!r} is not declared; a fresh confirmation must use a "
                "batch that was predeclared and committed before any result was observed"
            )
        return self._batches[batch_id]

    def batches(self) -> list[ConfirmationBatch]:
        return sorted(self._batches.values(), key=lambda batch: batch.batch_id)

    # -- transitions -----------------------------------------------------
    def declare(self, batch_id: str, role: str, spec_hash: str, *, notes: str = "") -> ConfirmationBatch:
        if batch_id in self._batches:
            raise BatchRegistryError(f"confirmation batch {batch_id!r} is already declared")
        if role not in CONFIRMATION_ROLES:
            raise BatchRegistryError(f"role must be one of {CONFIRMATION_ROLES}")
        batch = ConfirmationBatch(
            batch_id=batch_id,
            role=role,
            status="planned",
            declared_at=datetime.now(timezone.utc).isoformat(),
            spec_hash=spec_hash,
            notes=notes,
        )
        self._batches[batch_id] = batch
        return batch

    def claim(
        self, batch_id: str, role: str, spec_hash: str, *, reproduction: bool = False
    ) -> ConfirmationBatch:
        """Check that a batch may be used now, before any result is produced."""

        batch = self.get(batch_id)
        if batch.role != role:
            raise BatchRegistryError(
                f"confirmation batch {batch_id!r} was declared for role {batch.role!r}, not {role!r}"
            )
        if batch.spec_hash != spec_hash:
            raise BatchRegistryError(
                f"confirmation batch {batch_id!r} was declared against specification {batch.spec_hash[:12]}..., "
                f"but the resolved specification is {spec_hash[:12]}...; declare a new batch instead"
            )
        if reproduction:
            if batch.status == "planned":
                raise BatchRegistryError(
                    f"confirmation batch {batch_id!r} has never been run; reproduction mode requires a consumed batch"
                )
            return batch
        if batch.status != "planned":
            raise BatchRegistryError(
                f"confirmation batch {batch_id!r} has status {batch.status!r}. A fresh confirmation requires an "
                "unused batch. Use --reproduce to re-run an existing batch, or declare a new batch."
            )
        return batch

    def record_outcome(self, batch_id: str, run_id: str, *, passed: bool) -> ConfirmationBatch:
        """Record a completed confirmation. A failed batch is retired for good."""

        batch = self.get(batch_id)
        if run_id not in batch.consumed_by:
            batch.consumed_by.append(run_id)
        batch.outcome = "all_required_gates_pass" if passed else "required_gate_failure"
        batch.status = "consumed" if passed else "retired"
        return batch


def golden_seed_fixture(root_seed: int, cases: Iterable[tuple[str, str, int, int]]) -> dict[str, int]:
    """Build a committed fixture mapping known trial identities to seeds."""

    return {
        f"{purpose}|{family}|{replica}|{episode}": trial_seed(root_seed, purpose, family, replica, episode)
        for purpose, family, replica, episode in cases
    }


DEFAULT_GOLDEN_CASES: tuple[tuple[str, str, int, int], ...] = (
    (DEVELOPMENT_PURPOSE, "constant_velocity", 0, 0),
    (DEVELOPMENT_PURPOSE, "bouncing", 0, 0),
    (DEVELOPMENT_PURPOSE, "bouncing", 4, 99),
    (DEVELOPMENT_PURPOSE, "speed_change", 2, 17),
    (DEVELOPMENT_PURPOSE, "changed_law", 1, 5),
    ("confirmation::aaa-v2_1-confirmation-a-0001", "bouncing", 0, 0),
    ("confirmation::aaa-v2_1-confirmation-b-0001", "bouncing", 0, 0),
    ("training::selected", "training", 0, 0),
    ("training::selected", "training", 4, 23),
    ("learning_probe", "constant_velocity", 0, 900000),
)
