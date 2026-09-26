"""The packaged ``aaa.python.v1`` protocol: strict loading and a canonical hash.

v1 executes programs through v0's frozen oracle and subset validator, which
read only the ``subset`` and ``sandbox`` sections of the specification they
are given. Those two sections must therefore stay byte-for-byte equal (as
canonical JSON) to v0's; :func:`validate` refuses a v1 specification whose
execution environment silently diverged from the one v0's golden keys pin.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources
from typing import Any

from research.aaa_python import spec as v0_spec

from . import GENERATOR_VERSION, PROTOCOL_VERSION

SPEC_RESOURCE = "aaa_python_v1.json"
REQUIRED = (
    "protocol",
    "status",
    "generator_version",
    "predecessor",
    "question",
    "interpreter",
    "subset",
    "sandbox",
    "families",
    "templates",
    "slices",
    "splits",
    "tools",
    "statistics",
    "promotion",
)
FAMILIES = ("syntax", "outcome", "output", "localize", "repair")
SPLITS = ("train", "development", "probe", "attack", "confirmation")
SLICES = ("in_distribution", "novel_literals", "novel_names", "novel_structure", "novel_composition")


class SpecError(ValueError):
    pass


def _refuse_constant(token: str) -> Any:
    raise SpecError(f"non-standard JSON constant {token}")


def spec_bytes() -> bytes:
    return resources.files("research.aaa_python_v1").joinpath("data", SPEC_RESOURCE).read_bytes()


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
    if spec["protocol"] != PROTOCOL_VERSION or spec["generator_version"] != GENERATOR_VERSION:
        raise SpecError("protocol or generator version does not match this package")
    if tuple(spec["families"]) != FAMILIES:
        raise SpecError(f"families must be exactly {FAMILIES} in that order")
    if tuple(spec["splits"]["order"]) != SPLITS:
        raise SpecError("split order is fixed")
    if set(spec["splits"]["pool_per_family"]) != set(SPLITS):
        raise SpecError("every split, including confirmation, declares its pool size")
    for split, size in spec["splits"]["pool_per_family"].items():
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SpecError(f"pool size of {split} must be a positive integer")
    cycle = spec["slices"]["cycle"]
    if set(cycle) != set(SLICES) or set(spec["slices"]["meaning"]) != set(SLICES):
        raise SpecError(f"the slice cycle must use exactly {SLICES}")
    v0 = v0_spec.load()
    for section in ("subset", "sandbox", "interpreter"):
        if canonical_hash(spec[section]) != canonical_hash(v0[section]):
            raise SpecError(f"the {section} section must equal aaa.python.v0's (the oracle is shared)")
    if spec["predecessor"]["spec_sha256"] != v0_spec.spec_hash():
        raise SpecError("the recorded predecessor hash is not the packaged aaa.python.v0 specification")
    names = spec["templates"]["names"]
    pools = [set(names[key]["variables"]) for key in ("in_distribution", "novel_names")]
    if pools[0] & pools[1]:
        raise SpecError("novel variable names must be disjoint from training names")
    return spec


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    return validate(json.loads(spec_bytes().decode("utf-8"), parse_constant=_refuse_constant))


def spec_hash() -> str:
    return canonical_hash(load())
