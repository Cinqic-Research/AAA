"""The opaque library of ``aaa.python.opaque.v0``: environment-owned functions the agent cannot see.

A *library version* assigns each name ``api0 .. api{M-1}`` a unary integer function from a
small family with drawn parameters. The agent sees only the names in programs; the
implementations live here, in the environment, and are injected into the execution
namespace at run time. Versions are pure functions of their identity string, so every
interpreter reproduces them exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from research.aaa_python.rng import Stream, derive_seed

FAMILIES = ("affine", "clip_hi", "clip_lo", "mod", "step", "absdist")


@dataclass(frozen=True)
class ApiSpec:
    name: str
    family: str
    params: tuple[int, ...]

    def fn(self) -> Callable[[int], int]:
        f, a = self.family, self.params
        if f == "affine":
            return lambda x: a[0] * x + a[1]
        if f == "clip_hi":
            return lambda x: min(x, a[0])
        if f == "clip_lo":
            return lambda x: max(x, a[0])
        if f == "mod":
            return lambda x: x % a[0] + a[1]
        if f == "step":
            return lambda x: a[1] if x > a[0] else a[2]
        if f == "absdist":
            return lambda x: abs(x - a[0])
        raise ValueError(f)

    def describe(self) -> str:  # evaluator-side documentation only
        return f"{self.name}: {self.family}{self.params}"


def _draw(s: Stream, name: str, family: str) -> ApiSpec:
    if family == "affine":
        return ApiSpec(name, family, (s.between(1, 3), s.between(-4, 4)))
    if family in ("clip_hi", "clip_lo"):
        return ApiSpec(name, family, (s.between(2, 10),))
    if family == "mod":
        return ApiSpec(name, family, (s.between(3, 7), s.between(0, 3)))
    if family == "step":
        return ApiSpec(name, family, (s.between(2, 10), s.between(-3, 8), s.between(-3, 8)))
    if family == "absdist":
        return ApiSpec(name, family, (s.between(2, 10),))
    raise ValueError(family)


@lru_cache(maxsize=16)
def version(identity: str, count: int = 6, *, changed_from: str | None = None, changes: int = 0) -> tuple[ApiSpec, ...]:
    """A library version. With ``changed_from``, redraw ``changes`` functions of that version."""

    s = Stream(derive_seed("aaa.python.opaque.v0", "library", identity))
    if changed_from is None:
        families = s.shuffled(list(FAMILIES))[:count]
        return tuple(_draw(s, f"api{k}", families[k % len(families)]) for k in range(count))
    base = list(version(changed_from, count))
    targets = s.shuffled(list(range(count)))[:changes]
    for k in targets:
        spec = base[k]
        for _ in range(100):
            new = _draw(s, spec.name, s.choice(FAMILIES))
            if new != spec:
                break
        base[k] = new
    return tuple(base)


def namespace(lib: tuple[ApiSpec, ...]) -> dict[str, Any]:
    return {spec.name: spec.fn() for spec in lib}
