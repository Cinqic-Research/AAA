"""Disjoint deterministic seed namespaces for the AAA-1K phase.

Seeds are derived by hashing a namespace label together with an index, so no
stream is ever an obvious offset of another and model initialization cannot
accidentally share a realization with an environment. The namespaces are
disjoint by construction: two different labels cannot collide except by a
SHA-256 collision.

Pairing, where the protocol requires it (an online arm and its frozen clone on
the *same* environment realization), is expressed by using the same seed
deliberately and saying so, never by reusing a seed silently.
"""

from __future__ import annotations

import hashlib

PHASE_SALT = "aaa.1k.v1"

NAMESPACES = (
    "model_init",
    "development_env",
    "evaluation_env",
    "benchmark_generation",
    "bootstrap",
)

_SEED_MODULUS = 2**31 - 1


def derive_seed(namespace: str, index: int) -> int:
    """Return a deterministic non-negative seed for ``(namespace, index)``."""

    if namespace not in NAMESPACES:
        raise ValueError(f"unknown seed namespace {namespace!r}; expected one of {NAMESPACES}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("seed index must be a non-negative integer")
    digest = hashlib.sha256(f"{PHASE_SALT}:{namespace}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def seed_block(namespace: str, count: int, *, start: int = 0) -> list[int]:
    """Return ``count`` consecutive derived seeds from one namespace."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    return [derive_seed(namespace, start + offset) for offset in range(count)]
