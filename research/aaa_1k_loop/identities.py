"""Evidence identities for the loop, and the mechanical proof that they are fresh.

AAA-1K derives every stream and initialization seed as
``sha256("aaa.1k.v1:<namespace>:<index>")``. The loop uses the same
construction with a *different salt per iteration*, so its identities cannot be
an offset of an AAA-1K identity or of an earlier loop iteration's identity
except through a hash collision -- and a collision is not assumed away, it is
checked.

Freshness is proven, not asserted
---------------------------------
A block of loop identities is declared fresh only if its seed values are
disjoint from

1. every AAA-1K seed derivable from any of its namespaces at any index below
   :data:`AAA1K_INDEX_HORIZON` -- a superset of every index the phase declares
   (round 1 used 0-191, round 2 10,000+, round 3 40,000-60,011, development
   0-11, 500+ and 800+, the probe bank 1000-1007);
2. every integer seed recorded anywhere in the tracked AAA-1K evidence files,
   so an identity the code no longer declares but the evidence used is still
   caught;
3. every other block in the loop identity ledger, whatever its status.

"These numbers look different" is not a disjointness argument; a set
intersection is.

A spent identity stays spent
----------------------------
The ledger records each block's role and status. A confirmation block becomes
``spent`` the moment its result is observed and can never be reserved again
for anything. A block cannot be re-reserved under another role. The ledger is
append-only in the sense that matters: :func:`validate_ledger` refuses a
ledger whose blocks overlap, whose spent blocks lost their observation record,
or whose roles are unknown.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from research.aaa_1k.seeds import NAMESPACES as AAA1K_NAMESPACES
from research.aaa_1k.seeds import derive_seed as aaa1k_derive_seed

from . import PILOT_ITERATION_ID

LEDGER_SCHEMA = "aaa.loop.identity_ledger.v1"
LEDGER_PATH = "benchmarks/aaa1k_loop_identity_ledger.json"

_SEED_MODULUS = 2**31 - 1

AAA1K_INDEX_HORIZON = 100_000
"""Every AAA-1K namespace is enumerated at indices ``[0, horizon)``."""

AAA1K_EVIDENCE_GLOB = "docs/evidence/aaa_1k_*.json"

ROLES = (
    "diagnostic",
    "development",
    "attack",
    "confirmation",
)
"""What a block may be used for. Only ``confirmation`` blocks may decide a promotion."""

STATUSES = ("reserved", "used_for_development", "spent")

SELECTION_ROLES = ("diagnostic", "development", "attack")
"""Roles whose evidence may inform model selection. ``confirmation`` never may."""


class IdentityError(RuntimeError):
    """Raised when an identity would be reused, overlapped or misattributed."""


def loop_salt(iteration_id: str = PILOT_ITERATION_ID) -> str:
    return f"aaa.loop:{iteration_id}"


def derive_loop_seed(namespace: str, index: int, *, iteration_id: str = PILOT_ITERATION_ID) -> int:
    """``sha256("aaa.loop:<iteration>:<namespace>:<index>")`` reduced like AAA-1K's seeds."""

    if not namespace or ":" in namespace:
        raise IdentityError("a loop namespace must be a non-empty string without ':'")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise IdentityError("seed index must be a non-negative integer")
    digest = hashlib.sha256(f"{loop_salt(iteration_id)}:{namespace}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def block_seeds(block: Mapping[str, Any]) -> list[int]:
    """The concrete seed values a ledger block names."""

    return [
        derive_loop_seed(
            str(block["namespace"]), int(block["start"]) + offset, iteration_id=block["iteration"]
        )
        for offset in range(int(block["count"]))
    ]


# ----------------------------------------------------------------------
# what AAA-1K has already consumed
# ----------------------------------------------------------------------
def aaa1k_declared_seeds(horizon: int = AAA1K_INDEX_HORIZON) -> set[int]:
    """Every AAA-1K seed at every index below ``horizon`` in every namespace."""

    return {aaa1k_derive_seed(namespace, index) for namespace in AAA1K_NAMESPACES for index in range(horizon)}


_SEED_KEYS = {"seed", "model_seed", "initialization_seed", "environment_seed"}
_STREAM_ID = re.compile(r"^[a-z_0-9]+:(?:[a-z_]+:)?(\d+)(?::[a-z]+)?$")


def _walk_seeds(node: Any, out: set[int]) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in _SEED_KEYS and isinstance(value, int) and not isinstance(value, bool):
                out.add(int(value))
            elif key == "stream_id" and isinstance(value, str):
                match = _STREAM_ID.match(value)
                if match:
                    out.add(int(match.group(1)))
            _walk_seeds(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_seeds(value, out)


def evidence_seeds(root: Path, patterns: Iterable[str] = (AAA1K_EVIDENCE_GLOB,)) -> dict[str, Any]:
    """Every integer seed recorded in the tracked AAA-1K evidence."""

    files = sorted({path for pattern in patterns for path in root.glob(pattern)})
    if not files:
        raise IdentityError("no AAA-1K evidence found; freshness cannot be proven against nothing")
    seeds: set[int] = set()
    for path in files:
        _walk_seeds(json.loads(path.read_text(encoding="utf-8")), seeds)
    return {"files": [str(path.relative_to(root)) for path in files], "seeds": seeds}


# ----------------------------------------------------------------------
# the ledger
# ----------------------------------------------------------------------
def empty_ledger() -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "rule": (
            "blocks never overlap; a confirmation block becomes spent when its result is "
            "observed and may never be reserved again; selection may use only diagnostic, "
            "development and attack blocks"
        ),
        "blocks": [],
    }


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_ledger()
    ledger = json.loads(path.read_text(encoding="utf-8"))
    validate_ledger(ledger)
    return dict(ledger)


def validate_ledger(ledger: Mapping[str, Any]) -> dict[int, str]:
    """Check every structural invariant; return the seed -> block-id map."""

    if ledger.get("schema") != LEDGER_SCHEMA:
        raise IdentityError("unknown identity ledger schema")
    blocks = ledger.get("blocks")
    if not isinstance(blocks, list):
        raise IdentityError("ledger blocks must be a list")
    owner: dict[int, str] = {}
    ids: set[str] = set()
    for block in blocks:
        required = {"block_id", "iteration", "role", "namespace", "start", "count", "status", "purpose"}
        if not isinstance(block, Mapping) or not required <= set(block):
            raise IdentityError(f"ledger block is missing fields: {block!r}")
        block_id = str(block["block_id"])
        if block_id in ids:
            raise IdentityError(f"duplicate ledger block id {block_id}")
        ids.add(block_id)
        if block["role"] not in ROLES:
            raise IdentityError(f"block {block_id} has unknown role {block['role']!r}")
        if block["status"] not in STATUSES:
            raise IdentityError(f"block {block_id} has unknown status {block['status']!r}")
        if block["role"] == "confirmation" and block["status"] == "used_for_development":
            raise IdentityError(f"confirmation block {block_id} cannot be used for development")
        if block["status"] == "spent" and not block.get("observed_by"):
            raise IdentityError(f"spent block {block_id} must record what observed it")
        count = block["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise IdentityError(f"block {block_id} count must be a positive integer")
        for seed in block_seeds(block):
            if seed in owner:
                raise IdentityError(f"block {block_id} overlaps block {owner[seed]} at seed {seed}")
            owner[seed] = block_id
    return owner


def prove_fresh(
    ledger: Mapping[str, Any],
    root: Path,
    *,
    horizon: int = AAA1K_INDEX_HORIZON,
    declared: set[int] | None = None,
) -> dict[str, Any]:
    """Mechanical disjointness proof for every block in the ledger.

    Returns a strict-JSON record of what was checked and fails closed on any
    intersection.
    """

    owner = validate_ledger(ledger)
    declared_set = aaa1k_declared_seeds(horizon) if declared is None else declared
    evidence = evidence_seeds(root)
    loop_seeds = set(owner)
    against_declared = sorted(loop_seeds & declared_set)
    against_evidence = sorted(loop_seeds & evidence["seeds"])
    if against_declared or against_evidence:
        raise IdentityError(
            "loop identities collide with AAA-1K identities: "
            f"declared={against_declared[:5]} evidence={against_evidence[:5]}"
        )
    return {
        "status": "DISJOINT",
        "loop_blocks": len(ledger["blocks"]),
        "loop_seed_count": len(loop_seeds),
        "aaa1k_namespaces": list(AAA1K_NAMESPACES),
        "aaa1k_index_horizon": horizon,
        "aaa1k_declared_seed_count": len(declared_set),
        "aaa1k_evidence_files": evidence["files"],
        "aaa1k_evidence_seed_count": len(evidence["seeds"]),
        "intersections": {"declared": 0, "evidence": 0, "within_ledger": 0},
    }


def reserve(
    ledger: Mapping[str, Any],
    *,
    block_id: str,
    role: str,
    namespace: str,
    count: int,
    purpose: str,
    start: int = 0,
    iteration_id: str = PILOT_ITERATION_ID,
) -> dict[str, Any]:
    """Return a new ledger with one more block, or refuse."""

    candidate = {
        "block_id": block_id,
        "iteration": iteration_id,
        "role": role,
        "namespace": namespace,
        "start": start,
        "count": count,
        "status": "reserved",
        "purpose": purpose,
    }
    updated = {**ledger, "blocks": [*ledger["blocks"], candidate]}
    validate_ledger(updated)
    return updated


def find_block(ledger: Mapping[str, Any], block_id: str) -> dict[str, Any]:
    for block in ledger["blocks"]:
        if block["block_id"] == block_id:
            return dict(block)
    raise IdentityError(f"no ledger block {block_id!r}")


def mark(ledger: Mapping[str, Any], block_id: str, *, status: str, observed_by: str) -> dict[str, Any]:
    """Advance a block's status. Status never moves backwards."""

    order = {name: rank for rank, name in enumerate(STATUSES)}
    if status not in order:
        raise IdentityError(f"unknown status {status!r}")
    blocks = []
    found = False
    for block in ledger["blocks"]:
        if block["block_id"] == block_id:
            found = True
            if order[status] < order[block["status"]]:
                raise IdentityError(f"block {block_id} cannot move from {block['status']} back to {status}")
            if block["status"] == "spent":
                raise IdentityError(f"block {block_id} is already spent by {block.get('observed_by')}")
            block = {**block, "status": status, "observed_by": observed_by}
        blocks.append(block)
    if not found:
        raise IdentityError(f"no ledger block {block_id!r}")
    updated = {**ledger, "blocks": blocks}
    validate_ledger(updated)
    return updated


def require_usable(ledger: Mapping[str, Any], block_id: str, *, purpose: str) -> dict[str, Any]:
    """Fail unless ``block_id`` may be used for ``purpose`` (``selection`` or ``confirmation``)."""

    block = find_block(ledger, block_id)
    if purpose == "selection":
        if block["role"] not in SELECTION_ROLES:
            raise IdentityError(f"block {block_id} has role {block['role']} and may not inform selection")
        return block
    if purpose == "confirmation":
        if block["role"] != "confirmation":
            raise IdentityError(f"block {block_id} is not a confirmation block")
        if block["status"] != "reserved":
            raise IdentityError(f"confirmation block {block_id} is {block['status']}, not fresh")
        return block
    raise IdentityError(f"unknown purpose {purpose!r}")


def write_ledger(path: Path, ledger: Mapping[str, Any]) -> None:
    validate_ledger(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
