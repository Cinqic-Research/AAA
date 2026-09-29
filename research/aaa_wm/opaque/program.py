"""Programs, edit sites, the symmetric mutation relation and execution for ``aaa.python.opaque.v0``.

Programs are canonical text in the v1 safe subset (single spaces between tokens, four-space
indentation) that call library names ``apiK``. Validation prepends stub definitions of every
library name, so the v1 subset validator accepts the calls. Execution happens in-process with
restricted builtins *plus the environment's library*. This module is the environment's
interpreter, and no agent receives the library.
"""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from research.aaa_python.subset import SubsetError, validate
from research.aaa_python_v1 import spec as v1_spec

from .library import ApiSpec, namespace

_OPS: dict[str, tuple[str, ...]] = {
    "+": ("-", "*"),
    "-": ("+",),
    "*": ("+",),
    "//": ("%",),
    "%": ("//",),
    ">": (">=",),
    ">=": (">",),
    "<": ("<=",),
    "<=": ("<",),
    "==": ("!=",),
    "!=": ("==",),
    "+=": ("-=",),
    "-=": ("+=",),
}
_DELTAS = (-2, -1, 1, 2)
_BUILTINS = {"range": range, "len": len, "abs": abs, "min": min, "max": max, "int": int, "bool": bool, "sum": sum}


@dataclass(frozen=True)
class Edit:
    """Replace the token at ``(row, col)`` (1-based row, 0-based column) ``old`` -> ``new``."""

    row: int
    col: int
    old: str
    new: str

    def kind(self) -> str:
        if self.old.isdigit():
            return f"lit{int(self.new) - int(self.old):+d}"
        if self.old.startswith("api"):
            return "api"
        return f"op:{self.old}>{self.new}"


def tokens(source: str) -> list[tokenize.TokenInfo]:
    return [
        t
        for t in tokenize.generate_tokens(io.StringIO(source).readline)
        if t.type in (tokenize.NAME, tokenize.NUMBER, tokenize.OP)
    ]


def mutations_of(token: str, api_names: tuple[str, ...]) -> tuple[str, ...]:
    if token.isdigit():
        return tuple(str(int(token) + d) for d in _DELTAS if int(token) + d >= 0)
    if token in _OPS:
        return _OPS[token]
    if token in api_names:
        return tuple(n for n in api_names if n != token)
    return ()


@lru_cache(maxsize=200000)
def edits(source: str, api_names: tuple[str, ...]) -> tuple[Edit, ...]:
    out = []
    for t in tokens(source):
        if t.string == "def" or (t.type == tokenize.NAME and not t.string.startswith("api")):
            continue
        for new in mutations_of(t.string, api_names):
            out.append(Edit(t.start[0], t.start[1], t.string, new))
    return tuple(out)


def apply(source: str, edit: Edit) -> str:
    lines = source.split("\n")
    line = lines[edit.row - 1]
    if line[edit.col : edit.col + len(edit.old)] != edit.old:
        raise ValueError("edit does not match the program")
    lines[edit.row - 1] = line[: edit.col] + edit.new + line[edit.col + len(edit.old) :]
    return "\n".join(lines)


def undo(edit: Edit) -> Edit:
    return Edit(edit.row, edit.col, edit.new, edit.old)


def _stubs(api_names: tuple[str, ...]) -> str:
    return "".join(f"def {n}(x):\n    return x\n" for n in api_names)


_NUMBER = re.compile(r"\b\d+\b")


@lru_cache(maxsize=100000)
def _valid_skeleton(skeleton: str, api_names: tuple[str, ...]) -> bool:
    try:
        validate(_stubs(api_names) + skeleton, v1_spec.load())
        return True
    except (SubsetError, SyntaxError):
        return False


def valid(source: str, api_names: tuple[str, ...]) -> bool:
    """Subset validity. The v1 validator's verdict on these programs depends on literal values only
    through the integer-literal and ``range``-bound limits. So when every literal is within the smaller
    of the two limits, the verdict equals that of the skeleton with every literal set to 0, which is
    cached. Otherwise the full validator runs. ``tests/test_aaa_wm_opaque.py`` checks the equivalence."""

    limits = v1_spec.load()["subset"]["limits"]
    bound = min(int(limits["max_int_literal"]), int(limits["max_range_bound"]))
    if all(int(m) <= bound for m in _NUMBER.findall(source)):
        return _valid_skeleton(_NUMBER.sub("0", source), api_names)
    try:
        validate(_stubs(api_names) + source, v1_spec.load())
        return True
    except (SubsetError, SyntaxError):
        return False


@lru_cache(maxsize=100000)
def compiled(source: str, api_names: tuple[str, ...]) -> Any:
    if not valid(source, api_names):
        return None
    try:
        return compile(source, "<opaque>", "exec")
    except SyntaxError:
        return None


@lru_cache(maxsize=400000)
def run(source: str, lib: tuple[ApiSpec, ...], argument: int) -> tuple[str, Any]:
    """Environment execution: ``("ok", int)``, ``("error", ExceptionName)`` or ``("invalid", None)``."""

    names = tuple(s.name for s in lib)
    code = compiled(source, names)
    if code is None:
        return ("invalid", None)
    ns: dict[str, Any] = {"__builtins__": dict(_BUILTINS), **namespace(lib)}
    try:
        exec(code, ns)
        value = ns["f"](argument) if "f" in ns else ns["g"](argument)
    except Exception as error:  # the exception class is the observation
        return ("error", type(error).__name__)
    if isinstance(value, bool) or not isinstance(value, int):
        return ("ok", repr(value))
    if abs(value) > 10**6:
        return ("error", "Overflow")
    return ("ok", value)


def signature(source: str, lib: tuple[ApiSpec, ...], domain: tuple[int, ...]) -> tuple[tuple[str, Any], ...]:
    return tuple(run(source, lib, x) for x in domain)
