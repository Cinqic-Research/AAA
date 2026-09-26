"""Cross-interpreter golden keys for ``aaa.python.gen.v1``.

A task must be the same on every supported CPython (3.10-3.13). The packaged
keys hold, per family: the normalized source hash and oracle answer of the first
ten *training* tasks (built through the sandboxed oracle), and the source hash
of the attempt-0 *development* draft for indices 0-11, which covers every
slice twice (novel names and compositions included) without generating the
earlier pools a full development build needs. ``python -m research.aaa_python_v1
golden`` and the unit suite compare every CI interpreter against them.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from . import GENERATOR_VERSION, PROTOCOL_VERSION, generator
from . import spec as spec_module

RESOURCE = "golden_answers_v1.json"
SCHEMA = "aaa.python.v1.golden_answers.v1"


def payload() -> dict[str, Any]:
    spec = spec_module.load()
    keys: dict[str, Any] = {}
    for family in spec["families"]:
        for task in generator.build("train", family, list(range(10)), spec):
            keys[task.task_id] = {"source_sha256": task.source_sha256, "answer": task.answer}
        for index in range(12):
            d = generator.draft("development", family, index, 0, spec)
            identity = generator.task_id("development", family, index) + ":draft0"
            keys[identity] = (
                None
                if d is None
                else {
                    "source_sha256": generator.source_hash(
                        d.source, d.repair["candidates"] if d.repair else ()
                    )
                }
            )
    return {"schema": SCHEMA, "protocol": PROTOCOL_VERSION, "generator": GENERATOR_VERSION, "keys": keys}


def path() -> Path:
    return Path(str(resources.files("research.aaa_python_v1").joinpath("data", RESOURCE)))


def differences() -> list[str]:
    expected = json.loads(path().read_text(encoding="utf-8"))
    live = payload()
    if expected.get("schema") != SCHEMA or set(expected["keys"]) != set(live["keys"]):
        return ["golden key set or schema differs"]
    return [
        f"{identity}: expected {record}, this interpreter produced {live['keys'][identity]}"
        for identity, record in expected["keys"].items()
        if live["keys"][identity] != record
    ]
