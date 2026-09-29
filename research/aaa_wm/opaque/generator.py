"""Task generation for ``aaa.python.opaque.v0``: a random grammar, symmetric faults, domain success.

Everything is a pure function of ``(split, index, attempt)`` through the counter-mode
SHA-256 stream. Acceptance rules:

1. the reference executes without error on every visible input;
2. the buggy program fails at least one visible test;
3. every fault is individually necessary: undoing any single fault is not domain-equivalent
   to the reference;
4. the program has at least one library call;
5. no normalized program hash of an earlier split is reused (checked by :func:`pool`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from research.aaa_python.rng import Stream, derive_seed

from .library import ApiSpec, version
from .program import Edit, apply, edits, run, signature

PROTOCOL = "aaa.python.opaque.v0"
_SPEC_PATH = Path(__file__).with_name("data") / "aaa_python_opaque_v0.json"


@lru_cache(maxsize=1)
def load_spec() -> dict[str, Any]:
    return json.loads(_SPEC_PATH.read_text())


def spec_hash() -> str:
    return hashlib.sha256(_SPEC_PATH.read_bytes()).hexdigest()


def library(name: str = "A") -> tuple[ApiSpec, ...]:
    spec = load_spec()["library"]
    if name == "A":
        return version(spec["A"], spec["count"])
    b = spec["B"]
    return version(b["identity"], spec["count"], changed_from=spec["A"], changes=b["changes"])


def domain() -> tuple[int, ...]:
    lo, hi = load_spec()["domain"]
    return tuple(range(lo, hi + 1))


class ConfirmationNotAdmitted(RuntimeError):
    pass


@dataclass(frozen=True)
class OpaqueTask:
    task_id: str
    split: str
    index: int
    slice: str
    library: str
    reference: str  # hidden
    buggy: str  # visible starting state
    faults: tuple[Edit, ...]  # hidden: the edits that injected the faults (applied to the reference)
    visible_tests: tuple[tuple[int, Any], ...]  # (input, expected result); expected is ("ok", v) or ("error", cls)
    k: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- grammar
class _G:
    def __init__(self, s: Stream, lit: tuple[int, int], apis: tuple[str, ...], novel: bool) -> None:
        self.s, self.lo, self.hi, self.apis, self.novel = s, lit[0], lit[1], apis, novel
        self.calls = 0

    def L(self) -> str:
        return str(self.s.between(self.lo, self.hi))

    def atom(self, names: Sequence[str]) -> str:
        return self.s.choice([*names, self.L()])

    def api(self, arg: str) -> str:
        self.calls += 1
        return f"{self.s.choice(self.apis)}({arg})"

    def expr(self, names: Sequence[str]) -> str:
        r = self.s.below(7)
        a = self.s.choice(names)
        if r == 0:
            return f"{a} + {self.L()}"
        if r == 1:
            return f"{a} - {self.L()}"
        if r == 2:
            return f"{a} * {self.s.between(2, 3)}"
        if r == 3:
            return self.api(a)
        if r == 4:
            return f"{self.api(a)} + {self.atom(names)}"
        if r == 5:
            return f"{a} // {self.s.between(2, 4)}"
        return f"{a} % {self.s.between(3, 6)}"

    def cond(self, names: Sequence[str]) -> str:
        op = self.s.choice((">", ">=", "<", "<=", "==", "!="))
        lhs = self.api(self.s.choice(names)) if self.s.chance(1, 3) else self.s.choice(names)
        return f"{lhs} {op} {self.L()}"

    def statement(self, indent: str) -> list[str]:
        names = ["p", "q"]
        kinds = ["assign", "aug", "if", "for"] + (["for_if", "if_else_aug"] if self.novel else [])
        kind = self.s.choice(kinds)
        if self.novel and self.s.chance(1, 2):
            kind = self.s.choice(["for_if", "if_else_aug"])
        if kind == "assign":
            return [f"{indent}q = {self.expr(names)}"]
        if kind == "aug":
            return [f"{indent}q {self.s.choice(('+=', '-='))} {self.expr(names)}"]
        if kind == "if":
            body = [f"{indent}if {self.cond(names)}:", f"{indent}    q = {self.expr(names)}"]
            if self.s.chance(1, 2):
                body += [f"{indent}else:", f"{indent}    q = {self.expr(names)}"]
            return body
        if kind == "for":
            n = self.s.between(2, 4)
            inner = self.s.choice([f"q += {self.api('i')}", "q += i", f"q += {self.expr(names)}"])
            return [f"{indent}for i in range({n}):", f"{indent}    {inner}"]
        if kind == "for_if":
            n = self.s.between(2, 4)
            return [
                f"{indent}for i in range({n}):",
                f"{indent}    if {self.cond(['i', 'p'])}:",
                f"{indent}        q += {self.expr(['i', 'p'])}",
            ]
        return [
            f"{indent}if {self.cond(names)}:",
            f"{indent}    q += {self.expr(names)}",
            f"{indent}else:",
            f"{indent}    q -= {self.expr(names)}",
        ]


def _program(s: Stream, lit: tuple[int, int], apis: tuple[str, ...], novel: bool, length: tuple[int, int]) -> str:
    g = _G(s, lit, apis, novel)
    lines = ["def f(p):", f"    q = {g.expr(['p'])}"]
    for _ in range(s.between(length[0], length[1])):
        lines += g.statement("    ")
    lines.append(f"    return {s.choice(['q', 'q + p', g.api('q')])}")
    text = "\n".join(lines) + "\n"
    if "api" not in text:  # rule 4 (checked on the text: ``choice`` evaluates every option eagerly)
        lines[1] = f"    q = {g.api('p')}"
        text = "\n".join(lines) + "\n"
    return text


def slice_of(split: str, index: int) -> str:
    table = load_spec()["slices"]
    if split in table["pure"]:
        return str(table["pure"][split])
    cycle = table["blocks"]
    block = table["block_size"]
    return str(cycle[(index // block) % len(cycle)])


def draft(split: str, index: int, attempt: int) -> OpaqueTask | None:
    spec = load_spec()
    s = Stream(derive_seed(PROTOCOL, split, index, attempt))
    slice_name = slice_of(split, index)
    lib_name = "B" if slice_name == "library_B" else "A"
    lib = library(lib_name)
    names = tuple(a.name for a in lib)
    lit_key = "novel_literals" if slice_name == "novel_literals" else "in_distribution"
    lit = (int(spec["literals"][lit_key][0]), int(spec["literals"][lit_key][1]))
    reference = _program(s, lit, names, slice_name == "novel_grammar", tuple(spec["statements"]))  # type: ignore[arg-type]
    dom = domain()
    ref_sig = signature(reference, lib, dom)
    k = 2 if slice_name == "two_fault" else s.choice(spec["faults"]["k"])
    faults: list[Edit] = []
    buggy = reference
    for _ in range(k):
        options = [e for e in edits(buggy, names) if all((e.row, e.col) != (f.row, f.col) for f in faults)]
        if not options:
            return None
        e = s.choice(options)
        faults.append(e)
        buggy = apply(buggy, e)
    vis_inputs = s.shuffled(list(dom))[: spec["tests"]["visible"]]
    visible = tuple((x, run(reference, lib, x)) for x in vis_inputs)
    task = OpaqueTask(f"{PROTOCOL}:{split}:{index}", split, index, slice_name, lib_name, reference, buggy, tuple(faults), visible, k)
    return task if accept(task, lib, ref_sig) else None


def accept(task: OpaqueTask, lib: tuple[ApiSpec, ...], ref_sig: tuple[Any, ...]) -> bool:
    if "api" not in task.reference:
        return False
    if any(expected[0] != "ok" for _, expected in task.visible_tests):
        return False
    if all(run(task.buggy, lib, x) == expected for x, expected in task.visible_tests):
        return False
    if signature(task.buggy, lib, domain()) == ref_sig:
        return False
    if task.k > 1:
        # undo each fault alone (in reverse application order the positions are stable: distinct sites)
        for f in task.faults:
            back = Edit(f.row, f.col, f.new, f.old)
            try:
                partial = apply(task.buggy, back)
            except ValueError:
                return False
            if signature(partial, lib, domain()) == ref_sig:
                return False
    return True


def build(split: str, index: int, *, admission: object | None = None) -> OpaqueTask:
    if split == "confirmation" and admission is None:
        raise ConfirmationNotAdmitted("opaque.v0 confirmation tasks exist only under an admitted freeze")
    for attempt in range(500):
        task = draft(split, index, attempt)
        if task is not None:
            return task
    raise RuntimeError(f"no acceptable task for {split}:{index}")


def program_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def pool(split: str, count: int | None = None, start: int = 0, *, exclude: set[str] | None = None) -> list[OpaqueTask]:
    """Tasks ``[start, start+count)`` of ``split``; tasks whose reference or buggy hash is in ``exclude`` are
    replaced by later attempts (disjointness across splits)."""

    if split == "confirmation":
        raise ConfirmationNotAdmitted("use build() with an admission")
    n = load_spec()["splits"][split] if count is None else count
    out = []
    for i in range(start, start + n):
        for attempt in range(500):
            t = draft(split, i, attempt)
            if t is None:
                continue
            if exclude and (program_hash(t.reference) in exclude or program_hash(t.buggy) in exclude):
                continue
            out.append(t)
            break
        else:
            raise RuntimeError(f"no acceptable task for {split}:{i}")
    return out
