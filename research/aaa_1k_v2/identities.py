"""Evidence identities for ``aaa.1k.v2``, with disjointness proven rather than claimed.

``AAA-163`` recorded that AAA-1K's docstring claimed its SHA-derived seeds
"cannot collide", while seeds reduced modulo ``2^31 - 1`` do collide (73
cross-index collisions below index 100,000). This phase therefore

* derives each seed from ``sha256("aaa.1k.v2:<role>:<namespace>:<index>")``
  using the full leading 64 bits (no modular reduction), which makes an
  accidental collision improbable but *not impossible*; and
* never relies on improbability: :func:`prove_disjoint` enumerates every seed
  of every registered block and intersects it with every other block, every
  AAA-1K seed declared below index 100,000 in every namespace, every seed
  recorded in AAA-1K evidence, and every block of the loop's identity ledger.
  Any intersection raises. A collision is detected, reported and refused.

Roles
-----
``development``, ``attack``, ``confirmation``, ``diagnostic``, ``capacity``,
``qualification`` and ``scratch``. Only ``development``, ``attack`` and
``diagnostic`` blocks may inform selection; ``confirmation`` blocks may be
observed once, after a committed freeze, and are marked spent *before* the
first cell runs; ``capacity`` blocks serve only the post-confirmation capacity
diagnostic; ``qualification`` blocks serve compute benchmarking and backend
parity; ``scratch`` blocks serve shakedowns and may satisfy nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PHASE_SALT = "aaa.1k.v2"
REGISTRY_SCHEMA = "aaa.1k.v2.identity_registry.v1"
REGISTRY_PATH = "benchmarks/aaa1k_v2_identity_registry.json"

ROLES = ("development", "attack", "confirmation", "diagnostic", "capacity", "qualification", "scratch")
SELECTION_ROLES = ("development", "attack", "diagnostic")
STATUSES = ("reserved", "used", "spent")


class IdentityError(RuntimeError):
    """An identity would be reused, overlap another, or be used for the wrong purpose."""


def derive_seed(role: str, namespace: str, index: int) -> int:
    """64-bit seed for ``(role, namespace, index)``; never reduced, never assumed collision-free."""

    if role not in ROLES:
        raise IdentityError(f"unknown role {role!r}")
    if not namespace or ":" in namespace:
        raise IdentityError("a namespace must be a non-empty string without ':'")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise IdentityError("an identity index must be a non-negative integer")
    digest = hashlib.sha256(f"{PHASE_SALT}:{role}:{namespace}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def block_seeds(block: Mapping[str, Any]) -> list[int]:
    return [
        derive_seed(str(block["role"]), str(block["namespace"]), int(block["start"]) + offset)
        for offset in range(int(block["count"]))
    ]


def empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "phase": PHASE_SALT,
        "rule": (
            "blocks never overlap each other or any AAA-1K / loop identity (checked by set intersection); "
            "confirmation blocks are observed once, after a committed freeze, and marked spent first; "
            "only development, attack and diagnostic blocks may inform selection"
        ),
        "blocks": [],
    }


def validate_registry(registry: Mapping[str, Any]) -> dict[int, str]:
    """Structural checks plus within-registry disjointness; returns seed -> block id."""

    if registry.get("schema") != REGISTRY_SCHEMA:
        raise IdentityError("unknown v2 identity registry schema")
    owner: dict[int, str] = {}
    ids: set[str] = set()
    for block in registry.get("blocks", []):
        required = {"block_id", "role", "namespace", "start", "count", "status", "purpose"}
        if not required <= set(block):
            raise IdentityError(f"block is missing fields: {block!r}")
        block_id = str(block["block_id"])
        if block_id in ids:
            raise IdentityError(f"duplicate block id {block_id}")
        ids.add(block_id)
        if block["role"] not in ROLES or block["status"] not in STATUSES:
            raise IdentityError(f"block {block_id} has an unknown role or status")
        if block["role"] == "confirmation" and block["status"] == "used":
            raise IdentityError(f"confirmation block {block_id} cannot be marked used for selection")
        if block["status"] == "spent" and not block.get("observed_by"):
            raise IdentityError(f"spent block {block_id} must record what observed it")
        count = block["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise IdentityError(f"block {block_id} count must be a positive integer")
        for seed in block_seeds(block):
            if seed in owner:
                raise IdentityError(f"block {block_id} collides with block {owner[seed]} at seed {seed}")
            owner[seed] = block_id
    return owner


def historical_seeds(root: Path) -> dict[str, set[int]]:
    """Every identity consumed or declarable by ``aaa.1k.v1`` and the loop, by source."""

    from research.aaa_1k_loop.identities import (
        AAA1K_INDEX_HORIZON,
        LEDGER_PATH,
        aaa1k_declared_seeds,
        evidence_seeds,
        load_ledger,
        validate_ledger,
    )

    loop_evidence = evidence_seeds(root, ("docs/evidence/aaa1k_loop_*/*.json",))["seeds"]
    return {
        f"aaa1k_declared_below_{AAA1K_INDEX_HORIZON}": aaa1k_declared_seeds(),
        "aaa1k_evidence": evidence_seeds(root)["seeds"],
        "loop_ledger": set(validate_ledger(load_ledger(root / LEDGER_PATH))),
        "loop_evidence": loop_evidence,
    }


def prove_disjoint(
    registry: Mapping[str, Any], root: Path, *, history: Mapping[str, set[int]] | None = None
) -> dict[str, Any]:
    """Mechanical disjointness proof; raises :class:`IdentityError` on any intersection."""

    owner = validate_registry(registry)
    sources = historical_seeds(root) if history is None else history
    seeds = set(owner)
    collisions = {name: sorted(seeds & values) for name, values in sources.items()}
    found = {name: values for name, values in collisions.items() if values}
    if found:
        detail = {name: [(seed, owner[seed]) for seed in values[:5]] for name, values in found.items()}
        raise IdentityError(f"v2 identities collide with historical identities: {detail}")
    return {
        "status": "DISJOINT",
        "v2_blocks": len(registry["blocks"]),
        "v2_seed_count": len(seeds),
        "seed_bits": 64,
        "compared_against": {name: len(values) for name, values in sources.items()},
        "intersections": dict.fromkeys(sources, 0),
        "within_registry": 0,
    }


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_registry()
    registry = json.loads(path.read_text(encoding="utf-8"))
    validate_registry(registry)
    return dict(registry)


def write_registry(path: Path, registry: Mapping[str, Any]) -> None:
    validate_registry(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def reserve(registry: Mapping[str, Any], blocks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """A new registry with ``blocks`` appended (status ``reserved``), or refuse."""

    known = {block["block_id"] for block in registry["blocks"]}
    added = [{**dict(block), "status": "reserved"} for block in blocks if block["block_id"] not in known]
    updated = {**registry, "blocks": [*registry["blocks"], *added]}
    validate_registry(updated)
    return updated


def find(registry: Mapping[str, Any], block_id: str) -> dict[str, Any]:
    for block in registry["blocks"]:
        if block["block_id"] == block_id:
            return dict(block)
    raise IdentityError(f"no block {block_id!r}")


def require(registry: Mapping[str, Any], block_id: str, *, purpose: str) -> dict[str, Any]:
    """Refuse a block used for the wrong purpose.

    ``purpose``: ``selection`` (development/attack/diagnostic), ``confirmation``
    (must still be reserved), ``capacity``, ``qualification`` or ``scratch``.
    """

    block = find(registry, block_id)
    role = block["role"]
    if purpose == "selection":
        if role not in SELECTION_ROLES:
            raise IdentityError(f"block {block_id} ({role}) may not inform selection")
    elif purpose == "confirmation":
        if role != "confirmation":
            raise IdentityError(f"block {block_id} is not a confirmation block")
        if block["status"] != "reserved":
            raise IdentityError(f"confirmation block {block_id} is {block['status']}, not fresh")
    elif purpose in ("capacity", "qualification", "scratch"):
        if role != purpose:
            raise IdentityError(f"block {block_id} is {role}, not {purpose}")
    else:
        raise IdentityError(f"unknown purpose {purpose!r}")
    return block


def mark(registry: Mapping[str, Any], block_id: str, *, status: str, observed_by: str) -> dict[str, Any]:
    order = {name: rank for rank, name in enumerate(STATUSES)}
    if status not in order:
        raise IdentityError(f"unknown status {status!r}")
    blocks = []
    found = False
    for block in registry["blocks"]:
        if block["block_id"] == block_id:
            found = True
            if block["status"] == "spent":
                raise IdentityError(f"block {block_id} is already spent by {block.get('observed_by')}")
            if order[status] < order[block["status"]]:
                raise IdentityError(f"block {block_id} cannot move from {block['status']} back to {status}")
            block = {**block, "status": status, "observed_by": observed_by}
        blocks.append(block)
    if not found:
        raise IdentityError(f"no block {block_id!r}")
    updated = {**registry, "blocks": blocks}
    validate_registry(updated)
    return updated


def seeds_of(registry: Mapping[str, Any], block_id: str) -> list[int]:
    return block_seeds(find(registry, block_id))
