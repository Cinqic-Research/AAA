"""Task generation for ``aaa.python.seq.v0``: multi-fault functions for sequential debugging.

A task is a small function ``f(p)`` in the v1 safe subset, a reference version,
``k`` injected faults (each one symmetric v1 mutation on a distinct line), and
visible and hidden input/expected-output tests. Generation is a pure function of
``(split, index)`` through the counter-mode SHA-256 stream, so identities are
reproducible on every interpreter.

Acceptance rules (every one is checked by :func:`accept`):

1. the reference passes every test (by construction: expected values are its outputs);
2. the buggy program fails at least one **visible** test (a test run reveals a defect);
3. every fault matters: undoing any single fault while keeping the others still fails
   at least one hidden test, so a ``k``-fault task needs ``k`` edits;
4. the hidden tests separate the reference from every program one edit away from it
   that is not behaviourally equivalent to it on the declared input domain.
   (Rule 4 bounds lucky submissions; see ``AAA-seq-01`` in the design notes.)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from research.aaa_python.rng import Stream, derive_seed
from research.aaa_python_v1.generator import mutations

from .execute import run_function

PROTOCOL = "aaa.python.seq.v0"
_SPEC_PATH = Path(__file__).with_name("data") / "aaa_python_seq_v0.json"


@lru_cache(maxsize=1)
def load_spec() -> dict[str, Any]:
    return json.loads(_SPEC_PATH.read_text())


def spec_hash() -> str:
    return hashlib.sha256(_SPEC_PATH.read_bytes()).hexdigest()


class ConfirmationNotAdmitted(RuntimeError):
    pass


@dataclass(frozen=True)
class SeqTask:
    task_id: str
    split: str
    index: int
    slice: str
    template: str
    function: str
    reference: tuple[str, ...]  # hidden
    buggy: tuple[str, ...]  # visible (the starting state)
    fault_lines: tuple[int, ...]  # hidden, 1-based
    visible_tests: tuple[tuple[int, int], ...]
    hidden_tests: tuple[tuple[int, int], ...]  # hidden
    k: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ templates
def _body(template: str, s: Stream, lit: tuple[int, int], nm: Mapping[str, Any]) -> list[str]:
    f, p, q, i, j = nm["function"], nm["parameter"], nm["local"], nm["loops"][0], nm["loops"][1]
    lo, hi = lit

    def L() -> int:
        return s.between(lo, hi)

    def small() -> int:  # loop bounds and multipliers stay small in every slice
        return s.between(2, 4)

    if template == "affine3":
        return [
            f"def {f}({p}):",
            f"    {q} = {p} * {small()} + {L()}",
            f"    {q} = {q} - {L()}",
            f"    return {q} + {L()}",
        ]
    if template == "piecewise":
        return [
            f"def {f}({p}):",
            f"    if {p} > {L()}:",
            f"        return {p} - {L()}",
            f"    return {p} + {L()}",
        ]
    if template == "clamp":
        return [
            f"def {f}({p}):",
            f"    {q} = {p} + {L()}",
            f"    if {q} > {L()}:",
            f"        {q} = {q} - {L()}",
            f"    return {q} * {small()}",
        ]
    if template == "accumulate":
        return [
            f"def {f}({p}):",
            f"    {q} = {L()}",
            f"    for {i} in range({small()}):",
            f"        {q} += {p} - {L()}",
            f"    return {q}",
        ]
    if template == "count":
        return [
            f"def {f}({p}):",
            f"    {q} = 0",
            f"    for {i} in range({small() + 3}):",
            f"        if {i} < {p} - {L()}:",
            f"            {q} += {small()}",
            f"    return {q} + {L()}",
        ]
    if template == "branch_else":
        return [
            f"def {f}({p}):",
            f"    {q} = {p} * {small()}",
            f"    if {q} >= {L()}:",
            f"        {q} = {q} - {L()}",
            "    else:",
            f"        {q} = {q} + {L()}",
            f"    return {q}",
        ]
    if template == "double_loop":
        return [
            f"def {f}({p}):",
            f"    {q} = {L()}",
            f"    for {i} in range({small()}):",
            f"        for {j} in range({small()}):",
            f"            {q} += {p} - {L()}",
            f"    return {q}",
        ]
    raise ValueError(template)


def mutable_lines(lines: Sequence[str]) -> list[int]:
    return [n for n, line in enumerate(lines, start=1) if mutations(line)]


def outputs(lines: Sequence[str], inputs: Sequence[int]) -> tuple[Any, ...]:
    return tuple(run_function("\n".join(lines) + "\n", x) for x in inputs)


def passes(lines: Sequence[str], tests: Sequence[tuple[int, int]]) -> tuple[bool, ...]:
    return tuple(
        r == ("ok", e) for r, (_, e) in zip(outputs(lines, [x for x, _ in tests]), tests, strict=True)
    )


def slice_of(split: str, index: int) -> str:
    cycle = load_spec()["slices"]["cycle"][split]
    return str(cycle[index % len(cycle)])


def draft(split: str, index: int, attempt: int) -> SeqTask | None:
    spec = load_spec()
    s = Stream(derive_seed(PROTOCOL, split, index, attempt))
    slice_name = slice_of(split, index)
    nm = spec["names"]["novel_names" if slice_name == "novel_names" else "in_distribution"]
    lit = tuple(spec["literals"]["novel_literals" if slice_name == "novel_literals" else "in_distribution"])
    templates = spec["templates"]["novel_structure" if slice_name == "novel_structure" else "in_distribution"]
    cycle = len(spec["slices"]["cycle"][split])
    template = templates[(index // cycle) % len(templates)]  # balanced, and independent of the slice
    reference = _body(template, s, (int(lit[0]), int(lit[1])), nm)
    k = s.choice(spec["faults"]["k_by_slice"].get(slice_name, spec["faults"]["k"]))
    candidates = mutable_lines(reference)
    if len(candidates) < k:
        return None
    lines_ = sorted(s.shuffled(candidates)[:k])
    buggy = list(reference)
    for n in lines_:
        buggy[n - 1] = s.choice(mutations(reference[n - 1]))
    lo, hi = int(lit[0]), int(lit[1])
    domain = list(range(lo - 3, hi + 9))
    inputs = s.shuffled(domain)[: spec["tests"]["visible"] + spec["tests"]["hidden"]]
    ref_out = outputs(reference, inputs)
    if any(status != "ok" for status, _ in ref_out):
        return None
    tests = [(x, int(v)) for x, (_, v) in zip(inputs, ref_out, strict=True)]
    nv = spec["tests"]["visible"]
    task = SeqTask(
        task_id=f"{PROTOCOL}:{split}:{index}",
        split=split,
        index=index,
        slice=slice_name,
        template=template,
        function=nm["function"],
        reference=tuple(reference),
        buggy=tuple(buggy),
        fault_lines=tuple(lines_),
        visible_tests=tuple(tests[:nv]),
        hidden_tests=tuple(tests[nv:]),
        k=k,
    )
    return task if accept(task, domain) else None


def accept(task: SeqTask, domain: Sequence[int]) -> bool:
    if all(passes(task.buggy, task.visible_tests)):
        return False  # rule 2
    for n in task.fault_lines:  # rule 3
        partial = list(task.buggy)
        partial[n - 1] = task.reference[n - 1]
        if task.k > 1 and all(passes(partial, task.hidden_tests)):
            return False
    ref_domain = outputs(task.reference, domain)
    for n in mutable_lines(task.reference):  # rule 4
        for m in mutations(task.reference[n - 1]):
            near = list(task.reference)
            near[n - 1] = m
            if outputs(near, domain) != ref_domain and all(passes(near, task.hidden_tests)):
                return False
    return True


def build(split: str, index: int, *, admission: object | None = None) -> SeqTask:
    if split == "confirmation" and admission is None:
        raise ConfirmationNotAdmitted("seq.v0 confirmation tasks exist only under an admitted freeze")
    for attempt in range(200):
        task = draft(split, index, attempt)
        if task is not None:
            return task
    raise RuntimeError(f"no acceptable task for {split}:{index}")


def normalized(task: SeqTask) -> str:
    return hashlib.sha256(("\n".join(task.buggy) + repr(task.visible_tests)).encode()).hexdigest()


def pool(split: str, count: int | None = None, start: int = 0) -> list[SeqTask]:
    if split == "confirmation":
        raise ConfirmationNotAdmitted("use build() with an admission")
    n = load_spec()["splits"][split] if count is None else count
    return [build(split, i) for i in range(start, start + n)]
