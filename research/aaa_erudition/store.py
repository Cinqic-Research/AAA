"""Content-addressed, append-only store for adaptable component state.

Invariants:

* A state is immutable once written. Its identity is the SHA-256 of its
  canonical JSON, and every read re-verifies that hash, so silent
  corruption is an error rather than a different model.
* The only mutable things are the per-component *heads*. A head moves only
  through ``commit`` or ``rollback``, each of which first appends a
  provenance record to the lineage log and then replaces ``heads.json``
  atomically. A crash between the two leaves the previous head in force;
  ``open`` detects the dangling record and reports it.
* Nothing is ever deleted. Rolling back moves a head to an earlier state
  and records why; the rejected state stays addressable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .contracts import AdaptationRecord, Component, ContractError, canonical_json, sha256_json

STORE_SCHEMA = "aaa.erudition.store.v1"


class StoreError(RuntimeError):
    pass


class StoreCorruption(StoreError):
    """Stored bytes do not match their recorded identity."""


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` so that ``path`` holds either the old bytes or all new bytes."""

    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.lineage_path = self.root / "lineage.jsonl"
        self.heads_path = self.root / "heads.json"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.recovered: list[str] = []
        if not self.heads_path.exists():
            atomic_write(self.heads_path, canonical_json({"schema": STORE_SCHEMA, "heads": {}}))
        self._heads = self._read_heads()
        self._check_tail()

    # ---- objects -------------------------------------------------------------

    def put(self, kind: str, payload: dict[str, Any]) -> str:
        body = {"kind": kind, "payload": payload}
        digest = sha256_json(body)
        path = self.objects / f"{digest}.json"
        if not path.exists():
            atomic_write(path, canonical_json(body))
        return f"{kind}:{digest}"

    def get(self, state_id: str) -> dict[str, Any]:
        kind, _, digest = state_id.partition(":")
        if not kind or len(digest) != 64:
            raise StoreError(f"malformed state id {state_id!r}")
        path = self.objects / f"{digest}.json"
        try:
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise StoreError(f"unknown state {state_id}") from error
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as error:
            raise StoreCorruption(f"state {state_id} is not valid JSON") from error
        if sha256_json(body) != digest or canonical_json(body) != raw:
            raise StoreCorruption(f"state {state_id} does not match its identity")
        if body.get("kind") != kind:
            raise StoreCorruption(f"state {state_id} has kind {body.get('kind')!r}")
        payload: dict[str, Any] = body["payload"]
        return payload

    # ---- heads and lineage ---------------------------------------------------

    def _read_heads(self) -> dict[str, str]:
        try:
            body = json.loads(self.heads_path.read_bytes())
        except json.JSONDecodeError as error:
            raise StoreCorruption("heads.json is not valid JSON") from error
        if body.get("schema") != STORE_SCHEMA or not isinstance(body.get("heads"), dict):
            raise StoreCorruption("heads.json has the wrong schema")
        heads: dict[str, str] = body["heads"]
        for state_id in heads.values():
            self.get(state_id)
        return heads

    def _check_tail(self) -> None:
        records = self.lineage(strict=False)
        if self.recovered:
            # Rewrite the log without the torn line so later appends stay parseable.
            atomic_write(
                self.lineage_path, b"".join(canonical_json(record.to_dict()) + b"\n" for record in records)
            )
        if not records or records[-1].kind == "rejected":
            return
        last = records[-1]
        head = self._heads.get(last.component.value)
        if head == last.result_state:
            return
        # The process died between appending provenance and moving the head.
        # Record, explicitly, that the head never moved.
        self.recovered.append(
            f"lineage record {last.record_id} was written but its head update was not; "
            "the previous head remains in force"
        )
        self._append(
            AdaptationRecord(
                record_id=f"recovery-{last.record_id}",
                step=last.step,
                kind="rollback",
                component=last.component,
                mechanism="interrupted_commit_recovery",
                parent_state=last.result_state,
                result_state=head or "",
                request=None,
                candidate=None,
                evaluation=None,
                cost={},
                reason="interrupted commit: head update did not complete",
            )
        )

    def head(self, component: Component) -> str | None:
        return self._heads.get(component.value)

    def lineage(self, *, strict: bool = True) -> list[AdaptationRecord]:
        if not self.lineage_path.exists():
            return []
        records = []
        lines = self.lineage_path.read_bytes().split(b"\n")
        for index, line in enumerate(lines):
            if not line:
                continue
            try:
                records.append(AdaptationRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ContractError) as error:
                torn_tail = index == len(lines) - 1 or (index == len(lines) - 2 and not lines[-1])
                if strict or not torn_tail:
                    raise StoreCorruption(f"lineage line {index + 1} is unreadable") from error
                self.recovered.append(f"lineage line {index + 1} is a torn write and is ignored")
        return records

    def _append(self, record: AdaptationRecord) -> None:
        with self.lineage_path.open("ab") as stream:
            stream.write(canonical_json(record.to_dict()) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _set_head(self, component: Component, state_id: str) -> None:
        heads = dict(self._heads)
        heads[component.value] = state_id
        atomic_write(self.heads_path, canonical_json({"schema": STORE_SCHEMA, "heads": heads}))
        self._heads = heads

    def initialize(self, component: Component, state_id: str) -> None:
        """Set the first head of a component. Refuses to overwrite an existing head."""

        if component.value in self._heads:
            raise StoreError(f"{component.value} already has a head")
        self.get(state_id)
        self._set_head(component, state_id)

    def record(self, record: AdaptationRecord) -> None:
        """Append provenance for a decision that does not move a head (a rejection)."""

        if record.kind != "rejected":
            raise StoreError("only rejections are recorded without a head change")
        self._append(record)

    def commit(self, record: AdaptationRecord) -> None:
        if record.kind not in ("accepted", "rollback"):
            raise StoreError("commit requires an accepted or rollback record")
        current = self._heads.get(record.component.value)
        if current != record.parent_state:
            raise StoreError(
                f"{record.component.value} head is {current}, but the record's parent is {record.parent_state}"
            )
        self.get(record.result_state)
        self._append(record)
        self._set_head(record.component, record.result_state)

    def history(self, component: Component) -> list[str]:
        """States the component's head has held, oldest first."""

        states: list[str] = []
        for record in self.lineage():
            if record.component == component and record.kind in ("accepted", "rollback"):
                if not states:
                    states.append(record.parent_state)
                states.append(record.result_state)
        return states

    def verify(self) -> list[str]:
        """Re-hash every object and check lineage and heads. Returns problems found."""

        problems = []
        for path in sorted(self.objects.glob("*.json")):
            try:
                body = json.loads(path.read_bytes())
                if sha256_json(body) != path.stem:
                    problems.append(f"object {path.name} does not match its name")
            except json.JSONDecodeError:
                problems.append(f"object {path.name} is not valid JSON")
        try:
            for record in self.lineage():
                for state in (record.parent_state, record.result_state):
                    if state:
                        self.get(state)
        except StoreError as error:
            problems.append(str(error))
        return problems


class MemoryStore:
    """The same contract held in memory, for simulation branches that are discarded.

    Used only to generate the Erudition Model's training data, where thousands of
    counterfactual branches are copied and thrown away. Evidence runs use
    ``StateStore``.
    """

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self._heads: dict[str, str] = {}
        self._lineage: list[AdaptationRecord] = []
        self.recovered: list[str] = []

    def put(self, kind: str, payload: dict[str, Any]) -> str:
        state_id = f"{kind}:{sha256_json({'kind': kind, 'payload': payload})}"
        self.objects.setdefault(state_id, payload)
        return state_id

    def get(self, state_id: str) -> dict[str, Any]:
        try:
            return self.objects[state_id]
        except KeyError as error:
            raise StoreError(f"unknown state {state_id}") from error

    def head(self, component: Component) -> str | None:
        return self._heads.get(component.value)

    def initialize(self, component: Component, state_id: str) -> None:
        if component.value in self._heads:
            raise StoreError(f"{component.value} already has a head")
        self.get(state_id)
        self._heads[component.value] = state_id

    def record(self, record: AdaptationRecord) -> None:
        if record.kind != "rejected":
            raise StoreError("only rejections are recorded without a head change")
        self._lineage.append(record)

    def commit(self, record: AdaptationRecord) -> None:
        if record.kind not in ("accepted", "rollback"):
            raise StoreError("commit requires an accepted or rollback record")
        if self._heads.get(record.component.value) != record.parent_state:
            raise StoreError("the record's parent is not the current head")
        self.get(record.result_state)
        self._lineage.append(record)
        self._heads[record.component.value] = record.result_state

    def lineage(self) -> list[AdaptationRecord]:
        return list(self._lineage)

    def verify(self) -> list[str]:
        return []
