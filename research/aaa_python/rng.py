"""A counter-mode SHA-256 random stream, identical on every interpreter.

The task generator must produce byte-identical programs on every supported
CPython version, so it does not use :mod:`random`, whose algorithms are an
implementation detail that has changed between releases. Every draw here is a
pure function of a byte seed and a counter.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def derive_seed(*parts: str | int) -> int:
    """A 64-bit identity seed over ``aaa.python.v0`` label parts (stable, collision-checked elsewhere)."""

    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class Stream:
    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a 64-bit non-negative integer")
        self._key = seed.to_bytes(8, "big")
        self._counter = 0

    def _word(self) -> int:
        block = hashlib.sha256(self._key + self._counter.to_bytes(8, "big")).digest()
        self._counter += 1
        return int.from_bytes(block[:8], "big")

    def below(self, bound: int) -> int:
        """Uniform integer in ``[0, bound)`` by rejection sampling (no modulo bias)."""

        if bound <= 0:
            raise ValueError("bound must be positive")
        limit = (2**64 // bound) * bound
        while True:
            word = self._word()
            if word < limit:
                return word % bound

    def between(self, low: int, high: int) -> int:
        """Uniform integer in ``[low, high]``."""

        return low + self.below(high - low + 1)

    def choice(self, options: Sequence[T]) -> T:
        return options[self.below(len(options))]

    def chance(self, numerator: int, denominator: int) -> bool:
        return self.below(denominator) < numerator

    def shuffled(self, items: Sequence[T]) -> list[T]:
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.below(i + 1)
            out[i], out[j] = out[j], out[i]
        return out
