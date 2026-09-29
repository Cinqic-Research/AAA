"""Fast in-process execution of generated seq.v0 functions, cross-checked against the sandbox.

Sequential episodes and planning ceilings execute many small programs. The v0
sandbox starts one CPython process per program (tens of milliseconds); this
module runs a validated safe-subset function in-process with restricted builtins
(microseconds). It is only ever applied to programs the seq.v0 generator or a
mutation of one produced, and those programs are validated against the v1 subset
before first use. Their loops are bounded by literal ``range`` arguments, so no
infinite loop is possible. :func:`sandbox_agreement` re-runs a declared sample through
the sandboxed oracle and requires identical results. A test enforces this, and every
benchmark build records it.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from research.aaa_python.subset import SubsetError, validate
from research.aaa_python_v1 import spec as v1_spec

_BUILTINS = {
    "range": range,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
    "str": str,
    "bool": bool,
    "sum": sum,
}
_MAX_RANGE = 64


def _safe_range(*args: int) -> range:
    r = range(*args)
    if len(r) > _MAX_RANGE:
        raise RuntimeError("range too long")
    return r


@lru_cache(maxsize=500000)
def _compiled(source: str) -> Any:
    try:
        validate(source, v1_spec.load())
    except SubsetError:
        return None
    try:
        return compile(source, "<seq>", "exec")
    except SyntaxError:
        return None


@lru_cache(maxsize=2000000)
def run_function(source: str, argument: int) -> tuple[str, Any]:
    """``("ok", value)``, ``("error", ExceptionName)`` or ``("invalid", None)`` for ``f(argument)``."""

    code = _compiled(source)
    if code is None:
        return ("invalid", None)
    namespace: dict[str, Any] = {"__builtins__": {**_BUILTINS, "range": _safe_range}}
    try:
        exec(code, namespace)
        fn = next(v for k, v in namespace.items() if k != "__builtins__" and callable(v))
        value = fn(argument)
    except Exception as error:
        return ("error", type(error).__name__)
    if isinstance(value, bool) or not isinstance(value, int):
        return ("ok", repr(value))
    return ("ok", value)


def sandbox_agreement(programs: Sequence[tuple[str, int]]) -> dict[str, Any]:
    """Run ``(source, argument)`` pairs through the v0 sandboxed oracle and compare."""

    from research.aaa_python.oracle import run_many

    jobs = []
    for source, argument in programs:
        name = source.split("\n", 1)[0][4:].split("(")[0]
        jobs.append(("exec", source + f"print({name}({argument}))\n"))
    outcomes = run_many(jobs, v1_spec.load())
    disagreements = []
    for (source, argument), outcome in zip(programs, outcomes, strict=True):
        fast = run_function(source, argument)
        if outcome.status == "ok":
            slow: tuple[str, Any] = ("ok", int(outcome.stdout.strip()))
        elif outcome.status == "runtime_error":
            slow = ("error", outcome.exception)
        else:
            slow = (outcome.status, None)
        if fast != slow:
            disagreements.append({"source": source, "argument": argument, "fast": fast, "sandbox": slow})
    return {"checked": len(programs), "disagreements": disagreements}
