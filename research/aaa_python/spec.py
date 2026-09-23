"""The packaged ``aaa.python.v0`` protocol: strict loading and a canonical hash."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources
from typing import Any

from . import PROTOCOL_VERSION

SPEC_RESOURCE = "aaa_python_v0.json"
REQUIRED = (
    "protocol",
    "status",
    "generator_version",
    "question",
    "interpreter",
    "subset",
    "sandbox",
    "families",
    "templates",
    "splits",
    "learner",
    "development",
    "metrics",
    "statistics",
    "promotion",
)
FAMILIES = ("syntax", "outcome", "output", "localize", "repair")


class SpecError(ValueError):
    pass


def _refuse_constant(token: str) -> Any:
    raise SpecError(f"non-standard JSON constant {token}")


def spec_bytes() -> bytes:
    return resources.files("research.aaa_python").joinpath("data", SPEC_RESOURCE).read_bytes()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise SpecError("the specification must be a JSON object")
    missing = [key for key in REQUIRED if key not in spec]
    extra = sorted(set(spec) - set(REQUIRED))
    if missing or extra:
        raise SpecError(f"specification keys: missing {missing}, unknown {extra}")
    if spec["protocol"] != PROTOCOL_VERSION:
        raise SpecError(f"protocol {spec['protocol']!r} is not {PROTOCOL_VERSION}")
    if tuple(spec["families"]) != FAMILIES:
        raise SpecError(f"families must be exactly {FAMILIES} in that order")
    order = spec["splits"]["order"]
    if order != ["train", "development", "probe", "attack", "confirmation"]:
        raise SpecError("split order is fixed")
    if "confirmation" in spec["splits"]["pool_per_family"]:
        raise SpecError("aaa.python.v0 declares no confirmation pool")
    if spec["learner"]["default_representation"] not in spec["learner"]["representations"]:
        raise SpecError("default representation is not a declared representation")
    limits = spec["subset"]["limits"]
    for key, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SpecError(f"subset limit {key} must be a positive integer")
    return spec


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    return validate(json.loads(spec_bytes().decode("utf-8"), parse_constant=_refuse_constant))


def spec_hash() -> str:
    return canonical_hash(load())
