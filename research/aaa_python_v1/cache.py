"""A content-addressed disk cache for generated ``aaa.python.v1`` pools.

Generating a full pool runs thousands of sandboxed CPython processes. The cache
stores a pool only under a key derived from **every byte that can change a
task**: the v1 generator and specification, the frozen v0 generator, oracle,
sandbox child, subset validator and random stream, and the interpreter's
``major.minor`` version. Editing any of them selects a different directory, so
a stale pool can never be returned. Each file also records its own key and a
digest of its content, and loading re-checks both.

The cache is used only when ``AAA_DATA_ROOT`` is set (it lives under
``$AAA_DATA_ROOT/cache/aaa_python_v1``). Confirmation pools are never cached:
they are generated once, under an admitted freeze, inside the run that uses them.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.aaa_python.generator import Task

from . import spec as spec_module

SCHEMA = "aaa.python.v1.pool_cache.v1"
_SOURCES = (
    "research/aaa_python_v1/generator.py",
    "research/aaa_python/generator.py",
    "research/aaa_python/oracle.py",
    "research/aaa_python/_sandbox_child.py",
    "research/aaa_python/subset.py",
    "research/aaa_python/rng.py",
)
_ROOT = Path(__file__).resolve().parents[2]


def cache_key() -> str:
    digest = hashlib.sha256()
    for name in _SOURCES:
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256((_ROOT / name).read_bytes()).digest())
    digest.update(spec_module.spec_hash().encode("ascii"))
    digest.update(f"{sys.version_info[0]}.{sys.version_info[1]}".encode("ascii"))
    return digest.hexdigest()


def directory() -> Path | None:
    root = os.environ.get("AAA_DATA_ROOT")
    if not root:
        return None
    return Path(root) / "cache" / "aaa_python_v1" / cache_key()[:32]


def _to_json(task: Task) -> dict[str, Any]:
    payload = asdict(task)
    payload["oracle"] = dict(task.oracle)
    payload["notes"] = dict(task.notes)
    return payload


def _from_json(payload: dict[str, Any]) -> Task:
    tuples = ("candidates",)
    nested = ("visible_tests", "hidden_tests", "candidate_hidden_results")
    fields = dict(payload)
    for name in tuples:
        fields[name] = tuple(fields[name])
    for name in nested:
        fields[name] = tuple(tuple(item) for item in fields[name])
    return Task(**fields)


def _content_digest(tasks: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(tasks, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load(split: str, family: str) -> tuple[Task, ...] | None:
    folder = directory()
    if folder is None or split == "confirmation":
        return None
    path = folder / f"{split}-{family}.json.gz"
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return None
    if (
        not isinstance(stored, dict)
        or stored.get("schema") != SCHEMA
        or stored.get("key") != cache_key()
        or stored.get("digest") != _content_digest(stored.get("tasks", []))
    ):
        return None
    return tuple(_from_json(item) for item in stored["tasks"])


def store(split: str, family: str, tasks: tuple[Task, ...]) -> None:
    folder = directory()
    if folder is None or split == "confirmation":
        return
    folder.mkdir(parents=True, exist_ok=True)
    payload = [_to_json(task) for task in tasks]
    record = {"schema": SCHEMA, "key": cache_key(), "digest": _content_digest(payload), "tasks": payload}
    path = folder / f"{split}-{family}.json.gz"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
    temporary.replace(path)


def v0_hashes(split: str, family: str) -> set[str]:
    """Normalized source hashes of a v0 pool, cached under the same content key."""

    from research.aaa_python import generator as v0_generator

    folder = directory()
    path = None if folder is None else folder / f"v0-{split}-{family}.hashes.json"
    if path is not None and path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("key") == cache_key() and isinstance(stored.get("hashes"), list):
                return set(stored["hashes"])
        except (OSError, ValueError):
            pass
    hashes = v0_generator.pool_hashes(split, family)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"key": cache_key(), "hashes": sorted(hashes)}), encoding="utf-8")
        temporary.replace(path)
    return hashes
