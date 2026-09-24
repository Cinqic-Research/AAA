"""Deterministic, auditable task generation for ``aaa.python.v1`` (``aaa.python.gen.v1``).

This is a versioned fork of ``research.aaa_python.generator`` (v0), which is
left byte-for-byte unchanged. It keeps everything that made v0 auditable:

* a task is a pure function of ``aaa.python.v1:<split>:<family>:<index>`` and
  an attempt counter, drawn from v0's version-independent SHA-256 stream;
* answer keys come from CPython through v0's sandboxed oracle, never from the
  generator's intent;
* oracle-dependent acceptance retries the same identity with the next attempt;
* each split excludes the normalized source hashes of all earlier v1 splits.

What changed, and why:

1. **Repair candidates are symmetric** (``AAA-192``). In v0 every candidate
   was a single-token mutation *of the correct line*, so the correct line was
   the edit-distance medoid of the candidate set: a rule that never executes
   anything picked it 90-93% of the time. Here all four candidates are single
   mutations of a hidden base line that is never shown, so candidate geometry
   no longer identifies the answer. The rest of the family is unchanged.
2. **Identifier pools are parameters.** The ``novel_names`` slice renames
   every identifier -- variables, the function, its parameter, loop
   variables, locals and the undefined names that raise ``NameError`` -- from
   a pool disjoint from training, so name memorization can be separated from
   structure.
3. **A composition slice.** ``novel_composition`` joins two *training*
   templates in one program, with one value flowing from the first into the
   second. It asks for recombination of familiar parts, not new parts.
4. **Larger pools and v0 exclusion.** Pools are sized for the replication the
   pre-scale questions need. Every split after ``train`` also excludes the
   normalized source hashes of v0's train, development and probe pools, which
   were observed during v0.
5. **Confirmation is generated only under an admitted freeze**
   (:mod:`.freeze`): a committed manifest whose phase fingerprint matches the
   running source at a clean commit.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache as memoize
from typing import Any

from research.aaa_python import generator as v0_generator
from research.aaa_python.generator import GenerationError, Task, answer_from, source_hash
from research.aaa_python.oracle import Outcome, run_many
from research.aaa_python.rng import Stream, derive_seed
from research.aaa_python.subset import SubsetError, validate

from . import GENERATOR_VERSION, PROTOCOL_VERSION, cache
from . import spec as spec_module

MAX_ATTEMPTS = 60
V0_EXCLUDED_SPLITS = ("train", "development", "probe")

__all__ = [
    "ConfirmationNotAdmitted",
    "GenerationError",
    "Names",
    "Task",
    "build",
    "pool",
    "pool_hashes",
    "slice_of",
    "source_hash",
    "task_id",
]


class ConfirmationNotAdmitted(RuntimeError):
    """Confirmation identities exist only under an admitted, committed freeze."""


def task_id(split: str, family: str, index: int) -> str:
    return f"{PROTOCOL_VERSION}:{split}:{family}:{index}"


def slice_of(split: str, index: int, spec: Mapping[str, Any] | None = None) -> str:
    spec = spec or spec_module.load()
    if split in ("train", "probe"):
        return "in_distribution"
    cycle = spec["slices"]["cycle"]
    return str(cycle[index % len(cycle)])


@dataclass(frozen=True)
class Names:
    """Every identifier a generated program may bind or reference."""

    variables: tuple[str, ...]
    function: str
    parameter: str
    loops: tuple[str, str]
    local: str
    undefined: tuple[str, ...]

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], slice_name: str) -> Names:
        key = "novel_names" if slice_name == "novel_names" else "in_distribution"
        raw = spec["templates"]["names"][key]
        loops = tuple(raw["loops"])
        if len(loops) != 2:
            raise GenerationError("two loop variable names are required")
        return cls(
            variables=tuple(raw["variables"]),
            function=str(raw["function"]),
            parameter=str(raw["parameter"]),
            loops=(loops[0], loops[1]),
            local=str(raw["local"]),
            undefined=tuple(raw["undefined"]),
        )


# ----------------------------------------------------------------- templates
class Builder:
    """Draws the pieces of one program from a stream, within a literal range and a name pool."""

    def __init__(self, stream: Stream, literals: tuple[int, int], names: Names) -> None:
        self.s = stream
        self.low, self.high = literals
        self.nm = names
        self.taken: set[str] = set()

    def lit(self) -> int:
        return self.s.between(self.low, self.high)

    def names(self, count: int) -> list[str]:
        free = [name for name in self.nm.variables if name not in self.taken]
        chosen = self.s.shuffled(free)[:count]
        self.taken.update(chosen)
        return chosen

    def expr(self, operands: Sequence[str], allow_literal: bool = True) -> str:
        left = self.s.choice(operands)
        form = self.s.below(5)
        right = (
            str(self.lit())
            if (allow_literal and self.s.chance(1, 2)) or len(operands) < 2
            else self.s.choice(operands)
        )
        if form == 0:
            return f"{left} + {right}"
        if form == 1:
            return f"{left} - {right}"
        if form == 2:
            return f"{left} * {self.s.between(0, 3)}"
        if form == 3:
            return f"{left} // {self.s.between(1, 4)}"
        return f"{left} % {self.s.between(2, 5)}"

    def cond(self, operands: Sequence[str]) -> str:
        op = self.s.choice((">", "<", ">=", "<=", "==", "!="))
        return f"{self.s.choice(operands)} {op} {self.lit()}"


Block = list[str]


def straight_line(b: Builder) -> tuple[list[Block], str]:
    names = b.names(4)
    count = b.s.between(2, 4)
    blocks: list[Block] = [[f"{names[0]} = {b.lit()}"], [f"{names[1]} = {b.lit()}"]]
    defined = names[:2]
    for k in range(2, count + 1):
        target = names[k] if k < len(names) else b.s.choice(defined)
        blocks.append([f"{target} = {b.expr(defined)}"])
        if target not in defined:
            defined.append(target)
    if b.s.chance(1, 2):
        blocks.append([f"{b.s.choice(defined)} += {b.lit()}"])
    return blocks, b.s.choice(defined)


def conditional(b: Builder) -> tuple[list[Block], str]:
    first, second, out = b.names(3)
    blocks: list[Block] = [[f"{first} = {b.lit()}"], [f"{second} = {b.lit()}"]]
    blocks.append(
        [
            f"if {b.cond([first, second])}:",
            f"    {out} = {b.expr([first, second])}",
            "else:",
            f"    {out} = {b.expr([first, second])}",
        ]
    )
    return blocks, out


def loop_accumulate(b: Builder) -> tuple[list[Block], str]:
    base, acc = b.names(2)
    i = b.nm.loops[0]
    blocks: list[Block] = [[f"{base} = {b.lit()}"], [f"{acc} = {b.s.between(0, 3)}"]]
    step = b.s.choice(("+=", "-="))
    blocks.append([f"for {i} in range({b.s.between(2, 5)}):", f"    {acc} {step} {b.expr([i, base])}"])
    return blocks, acc


def function_call(b: Builder) -> tuple[list[Block], str]:
    arg, out, second = b.names(3)
    f, p = b.nm.function, b.nm.parameter
    blocks: list[Block] = [[f"def {f}({p}):", f"    return {b.expr([p])}"], [f"{arg} = {b.lit()}"]]
    blocks.append([f"{out} = {f}({arg})"])
    if b.s.chance(1, 2):
        blocks.append([f"{second} = {f}({out}) - {arg}"])
        return blocks, second
    return blocks, out


def nested_loop(b: Builder) -> tuple[list[Block], str]:
    (acc,) = b.names(1)
    i, j = b.nm.loops
    blocks: list[Block] = [[f"{acc} = 0"]]
    blocks.append(
        [
            f"for {i} in range({b.s.between(2, 3)}):",
            f"    for {j} in range({b.s.between(2, 3)}):",
            f"        {acc} += {b.s.choice((i, j))} {b.s.choice(('+', '-', '*'))} {b.s.choice((i, j, '1'))}",
        ]
    )
    return blocks, acc


def list_walk(b: Builder) -> tuple[list[Block], str]:
    items, acc = b.names(2)
    i = b.nm.loops[0]
    size = b.s.between(3, 5)
    values = ", ".join(str(b.lit()) for _ in range(size))
    blocks: list[Block] = [[f"{items} = [{values}]"], [f"{acc} = 0"]]
    blocks.append([f"for {i} in range({b.s.between(1, size)}):", f"    {acc} += {items}[{i}] - {i}"])
    return blocks, acc


TEMPLATES: Mapping[str, Callable[[Builder], tuple[list[Block], str]]] = {
    "straight_line": straight_line,
    "conditional": conditional,
    "loop_accumulate": loop_accumulate,
    "function_call": function_call,
    "nested_loop": nested_loop,
    "list_walk": list_walk,
}
_LITERAL_ASSIGNMENT = re.compile(r"^([a-z][a-z0-9]*) = (\d+)$")


def composition(b: Builder, spec: Mapping[str, Any]) -> tuple[list[Block], str, str]:
    """Two distinct training templates; the first's result seeds a literal of the second."""

    first, second = b.s.shuffled(list(spec["templates"]["in_distribution"]))[:2]
    blocks_a, printed_a = TEMPLATES[first](b)
    blocks_b, printed_b = TEMPLATES[second](b)
    for block in blocks_b:
        match = _LITERAL_ASSIGNMENT.match(block[0]) if len(block) == 1 else None
        if match:
            block[0] = f"{match.group(1)} = {printed_a} % {b.s.between(3, 9)}"
            break
    return blocks_a + blocks_b, printed_b, f"{first}+{second}"


def _literals(spec: Mapping[str, Any], slice_name: str) -> tuple[int, int]:
    literals = spec["templates"]["literals"]
    low, high = literals["novel_literals" if slice_name == "novel_literals" else "in_distribution"]
    return int(low), int(high)


def _templates(spec: Mapping[str, Any], slice_name: str) -> list[str]:
    key = "novel_structure" if slice_name == "novel_structure" else "in_distribution"
    return list(spec["templates"][key])


def _defined_names(blocks: Sequence[Block], variables: Sequence[str]) -> list[str]:
    names = []
    for block in blocks:
        head = block[0]
        if " = " in head and not head.startswith(("def ", "for ", "if ")):
            names.append(head.split(" = ", 1)[0].strip())
    for block in blocks:
        for line in block[1:]:
            stripped = line.strip()
            for op in (" += ", " -= ", " = "):
                if op in stripped:
                    names.append(stripped.split(op, 1)[0])
                    break
    return [n for n in dict.fromkeys(names) if n in variables]


def fault_block(b: Builder, names: Sequence[str], avoid: Sequence[str]) -> tuple[str, Block]:
    """A value-dependent fault; the assigned name is one the program never uses (see v0)."""

    kind = b.s.choice(("zero_division", "index", "name", "type", "value"))
    v, u = b.s.choice(names), b.s.choice(names)
    t = b.s.choice([name for name in b.nm.variables if name not in avoid])
    k = b.lit()
    if kind == "zero_division":
        return kind, [f"{t} = {v} // ({u} - {k})"]
    if kind == "index":
        return kind, [f"{t} = [{b.lit()}, {b.lit()}, {b.lit()}][{v} - {k}]"]
    if kind == "name":
        return kind, [f"if {v} > {k}:", f"    {t} = {b.s.choice(b.nm.undefined)} + 1"]
    if kind == "type":
        return kind, [f"{t} = {v} + (str({u}) if {u} > {k} else {u})"]
    return kind, [f'{t} = int(("{b.lit()}" if {v} > {k} else "z"))']


def semantic_program(
    stream: Stream, spec: Mapping[str, Any], slice_name: str, *, fault: bool, second_fault: str = "maybe"
) -> tuple[str, str, dict[str, Any]]:
    """A template program, optionally with value-dependent faults.

    ``second_fault`` is ``"maybe"`` (v0's rule: a second fault with probability
    one half) or ``"always"``. ``localize`` uses ``"always"`` (``AAA-194``): in
    v0 half the localization programs had a single fault, and because the
    program must fail, "the only risky-looking line" was the answer 97-99% of
    the time. With two value-dependent faults the question is which fires first.
    """
    b = Builder(stream, _literals(spec, slice_name), Names.from_spec(spec, slice_name))
    if slice_name == "novel_composition":
        blocks, printed, template = composition(b, spec)
    else:
        template = b.s.choice(_templates(spec, slice_name))
        blocks, printed = TEMPLATES[template](b)
    notes: dict[str, Any] = {}
    if fault:
        position = b.s.between(1, len(blocks))
        names = _defined_names(blocks[:position], b.nm.variables)
        if not names:
            position, names = len(blocks), _defined_names(blocks, b.nm.variables) or [printed]
        used = set(re.findall(r"[a-z][a-z0-9]*", "\n".join(line for block in blocks for line in block)))
        kind, block = fault_block(b, names, sorted(used))
        blocks.insert(position, block)
        notes["fault"] = [kind]
        if second_fault == "always" or b.s.chance(1, 2):
            used |= set(re.findall(r"[a-z][a-z0-9]*", "\n".join(block)))
            later = b.s.between(position + 1, len(blocks))
            names = _defined_names(blocks[:later], b.nm.variables) or names
            kind, block = fault_block(b, names, sorted(used))
            blocks.insert(later, block)
            notes["fault"].append(kind)
    blocks.append([f"print({printed})"])
    return "\n".join(line for block in blocks for line in block) + "\n", template, notes


# ----------------------------------------------------------------- repair
_SWAPS = ((" + ", " - "), (" > ", " >= "), (" < ", " <= "), (" * ", " + "))
_DELTAS = (-2, -1, 1, 2)


def mutations(line: str) -> list[str]:
    """Single-token mutations under a *symmetric* relation: ``m in mutations(x)`` iff ``x in mutations(m)``.

    v0's relation was asymmetric (digit deltas +1, -1, +2 with clamping at 0;
    ``*`` -> ``+`` without the reverse; ``<`` -> ``<=`` without the reverse), which
    would let candidate geometry reveal the correct line even around a hidden
    base. Operators swap in both directions, digit deltas are symmetric, and a
    mutation that would make a literal negative is skipped rather than clamped.
    Each operator occurrence and each literal is mutated independently.
    """

    tokens = line.split(" ")
    out: list[str] = []
    for position, token in enumerate(tokens):
        for a, b in _SWAPS:
            for x, y in ((a.strip(), b.strip()), (b.strip(), a.strip())):
                if token == x:
                    changed = list(tokens)
                    changed[position] = y
                    out.append(" ".join(changed))
        if token.isdigit():
            for delta in _DELTAS:
                value = int(token) + delta
                if value >= 0:
                    changed = list(tokens)
                    changed[position] = str(value)
                    out.append(" ".join(changed))
    return [m for m in dict.fromkeys(out) if m != line]


def repair_candidate(stream: Stream, spec: Mapping[str, Any], slice_name: str) -> dict[str, Any] | None:
    """A buggy function, its line to repair, and four *symmetric* candidates (``AAA-192``)."""

    names = Names.from_spec(spec, slice_name)
    b = Builder(stream, _literals(spec, slice_name), names)
    kind = "novel_structure" if slice_name == "novel_structure" else "in_distribution"
    template = b.s.choice(spec["templates"]["repair"][kind])
    f, p, q, i = names.function, names.parameter, names.local, names.loops[0]
    lo, hi = b.lit(), b.lit()
    if template == "affine":
        body = [
            f"def {f}({p}):",
            f"    {q} = {p} * {b.s.between(1, 3)} + {lo}",
            f"    return {q} - " + str(b.s.between(0, 3)),
        ]
        target = 1
    elif template == "piecewise":
        body = [
            f"def {f}({p}):",
            f"    if {p} > {lo}:",
            f"        return {p} - {hi}",
            f"    return {p} + {b.s.between(0, 4)}",
        ]
        target = b.s.choice((1, 2, 3))
    else:
        body = [
            f"def {f}({p}):",
            f"    {q} = 0",
            f"    for {i} in range({b.s.between(2, 4)}):",
            f"        {q} += {p} - {lo}",
            f"    return {q}",
        ]
        target = 3
    correct = body[target]
    family = spec["families"]["repair"]
    n_candidates, n_visible, n_hidden = family["candidates"], family["visible_tests"], family["hidden_tests"]
    others: list[str] | None = None
    # A hidden base line one mutation from the correct line; every candidate is one mutation
    # from that base, and the base itself is never shown.
    for base in stream.shuffled(mutations(correct)):
        around = mutations(base)
        if correct not in around:
            continue
        pool_ = [m for m in around if m != correct]
        if len(pool_) >= n_candidates - 1:
            others = stream.shuffled(pool_)[: n_candidates - 1]
            break
    if others is None:
        return None
    buggy = others[0]
    candidates = stream.shuffled([correct, *others])
    inputs = stream.shuffled(list(range(b.low - 2, b.high + 6)))[: n_visible + n_hidden]
    return {
        "template": template,
        "reference": "\n".join(body) + "\n",
        "buggy": "\n".join([*body[:target], buggy, *body[target + 1 :]]) + "\n",
        "line": target + 1,
        "candidates": tuple(candidates),
        "correct": candidates.index(correct),
        "visible_inputs": tuple(inputs[:n_visible]),
        "hidden_inputs": tuple(inputs[n_visible : n_visible + n_hidden]),
        "body": body,
        "target": target,
        "function": f,
    }


def _repair_program(
    body: Sequence[str], target: int, replacement: str, inputs: Sequence[int], function: str
) -> str:
    lines = [*list(body[:target]), replacement, *list(body[target + 1 :])]
    return "\n".join(lines + [f"print({function}({value}))" for value in inputs]) + "\n"


# ----------------------------------------------------------------- building
@dataclass
class Draft:
    task_id: str
    split: str
    family: str
    index: int
    attempt: int
    slice: str
    template: str
    source: str
    jobs: list[tuple[str, str]]
    notes: dict[str, Any]
    repair: dict[str, Any] | None = None


def draft(split: str, family: str, index: int, attempt: int, spec: Mapping[str, Any]) -> Draft | None:
    stream = Stream(derive_seed(PROTOCOL_VERSION, GENERATOR_VERSION, split, family, index, attempt))
    slice_name = slice_of(split, index, spec)
    identity = task_id(split, family, index)
    if family == "repair":
        repair = repair_candidate(stream, spec, slice_name)
        if repair is None:
            return None
        inputs = repair["visible_inputs"] + repair["hidden_inputs"]
        fn = repair["function"]
        jobs = [
            (
                "exec",
                _repair_program(
                    repair["body"], repair["target"], repair["body"][repair["target"]], inputs, fn
                ),
            )
        ]
        jobs += [
            ("exec", _repair_program(repair["body"], repair["target"], c, inputs, fn))
            for c in repair["candidates"]
        ]
        return Draft(
            identity,
            split,
            family,
            index,
            attempt,
            slice_name,
            repair["template"],
            repair["buggy"],
            jobs,
            {},
            repair,
        )
    fault = (
        family == "localize"
        or (family == "outcome" and stream.chance(2, 3))
        or (family == "syntax" and stream.chance(1, 3))
    )
    source, template, notes = semantic_program(
        stream, spec, slice_name, fault=fault, second_fault="always" if family == "localize" else "maybe"
    )
    if family == "syntax":
        if stream.chance(1, 2):
            source, notes["corruption"] = v0_generator.corrupt(stream, source)
        return Draft(
            identity,
            split,
            family,
            index,
            attempt,
            slice_name,
            template,
            source,
            [("compile", source)],
            notes,
        )
    return Draft(
        identity, split, family, index, attempt, slice_name, template, source, [("exec", source)], notes
    )


def _within_limits(d: Draft, spec: Mapping[str, Any]) -> bool:
    """Size limits for every job, and the full subset validator for every executed job.

    v0's oracle raises :class:`SubsetError` rather than returning a status, so a
    composed program over the node limit must be rejected here, as an attempt.
    """

    limits = spec["subset"]["limits"]
    for mode, source in d.jobs:
        if (
            len(source.encode("utf-8")) > limits["max_source_bytes"]
            or source.count("\n") > limits["max_lines"]
        ):
            return False
        if mode == "exec":
            try:
                validate(source, spec)
            except SubsetError:
                return False
    return True


def _accept(d: Draft, outcomes: Sequence[Outcome], spec: Mapping[str, Any]) -> bool:
    if not all(o.is_program_outcome for o in outcomes):
        return False
    first = outcomes[0]
    if d.family == "output":
        text = first.stdout.strip()
        family = spec["families"]["output"]
        return (
            first.status == "ok"
            and text.lstrip("-").isdigit()
            and text.count("-") <= 1
            and family["min"] <= int(text) <= family["max"]
        )
    if d.family == "localize":
        return (
            first.status == "exception"
            and first.line is not None
            and first.line <= spec["families"]["localize"]["max_line"]
        )
    if d.family == "outcome":
        return first.status == "ok" or (
            first.status == "exception" and first.exception in spec["families"]["outcome"]["labels"]
        )
    if d.family == "repair":
        assert d.repair is not None
        visible = len(d.repair["visible_inputs"])
        total = visible + len(d.repair["hidden_inputs"])
        reference = first.stdout.split()
        if first.status != "ok" or len(reference) != total:
            return False
        hidden = reference[visible:]
        passing = [
            k for k, o in enumerate(outcomes[1:]) if o.status == "ok" and o.stdout.split()[visible:] == hidden
        ]
        return passing == [d.repair["correct"]]
    return True


def _finish(d: Draft, outcomes: Sequence[Outcome]) -> Task:
    first = outcomes[0]
    oracle = {
        "status": first.status,
        "exception": first.exception,
        "line": first.line,
        "stdout": first.stdout,
    }
    if d.family != "repair":
        return Task(
            d.task_id,
            d.split,
            d.family,
            d.index,
            d.attempt,
            d.slice,
            d.template,
            d.source,
            answer=answer_from(d.family, first),
            oracle=oracle,
            notes=d.notes,
        )
    assert d.repair is not None
    reference = [int(v) for v in first.stdout.split()]
    inputs = d.repair["visible_inputs"] + d.repair["hidden_inputs"]
    tests = tuple(zip(inputs, reference, strict=True))
    visible = len(d.repair["visible_inputs"])
    hidden = [str(v) for v in reference[visible:]]
    results = tuple(
        tuple(
            o.status == "ok"
            and len(o.stdout.split()) == len(reference)
            and o.stdout.split()[visible + k] == hidden[k]
            for k in range(len(hidden))
        )
        for o in outcomes[1:]
    )
    return Task(
        d.task_id,
        d.split,
        d.family,
        d.index,
        d.attempt,
        d.slice,
        d.template,
        d.source,
        repair_line=d.repair["line"],
        candidates=d.repair["candidates"],
        visible_tests=tests[:visible],
        hidden_tests=tests[visible:],
        answer=d.repair["correct"],
        oracle=oracle,
        candidate_hidden_results=results,
        notes={"function": d.repair["function"]},
    )


def _excluded(split: str, family: str, spec: Mapping[str, Any]) -> set[str]:
    order = list(spec["splits"]["order"])
    excluded: set[str] = set()
    for earlier in order[: order.index(split)]:
        excluded |= pool_hashes(earlier, family)
    if split != "train":
        for v0_split in V0_EXCLUDED_SPLITS:
            excluded |= cache.v0_hashes(v0_split, family)
    return excluded


def build(
    split: str,
    family: str,
    indices: Sequence[int],
    spec: Mapping[str, Any] | None = None,
    *,
    admission: object | None = None,
) -> list[Task]:
    """Tasks for ``indices`` of one split and family, oracle-accepted and disjoint from earlier pools."""

    spec = spec or spec_module.load()
    if split == "confirmation":
        from .freeze import Admission

        if not isinstance(admission, Admission) or not admission.verify():
            raise ConfirmationNotAdmitted(
                "aaa.python.v1 confirmation identities are generated only under an admitted, committed freeze"
            )
    order = spec["splits"]["order"]
    if split not in order or family not in spec["families"]:
        raise GenerationError(f"unknown split or family: {split}/{family}")
    size = spec["splits"]["pool_per_family"][split]
    outside = [i for i in indices if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < size]
    if outside:
        raise GenerationError(
            f"{split}/{family}: indices {outside[:5]} are outside the declared pool of {size}"
        )
    if len(set(indices)) != len(indices):
        raise GenerationError(f"{split}/{family}: duplicate indices")
    excluded = _excluded(split, family, spec)
    attempts = dict.fromkeys(indices, 0)
    done: dict[int, Task] = {}
    while attempts:
        drafts: list[Draft] = []
        for index, attempt in list(attempts.items()):
            if attempt >= MAX_ATTEMPTS:
                raise GenerationError(
                    f"{task_id(split, family, index)}: no acceptable task in {MAX_ATTEMPTS} attempts"
                )
            d = draft(split, family, index, attempt, spec)
            if (
                d is None
                or not _within_limits(d, spec)
                or source_hash(d.source, d.repair["candidates"] if d.repair else ()) in excluded
            ):
                attempts[index] += 1
                continue
            drafts.append(d)
        jobs = [job for d in drafts for job in d.jobs]
        outcomes = run_many(jobs, spec) if jobs else []
        cursor = 0
        for d in drafts:
            mine = outcomes[cursor : cursor + len(d.jobs)]
            cursor += len(d.jobs)
            if _accept(d, mine, spec):
                done[d.index] = _finish(d, mine)
                del attempts[d.index]
            else:
                attempts[d.index] += 1
    return [done[i] for i in indices]


@memoize
def _pool(split: str, family: str) -> tuple[Task, ...]:
    if split == "confirmation":
        raise ConfirmationNotAdmitted("the confirmation pool is never built implicitly")
    cached = cache.load(split, family)
    if cached is not None:
        return cached
    size = spec_module.load()["splits"]["pool_per_family"][split]
    built = tuple(build(split, family, range(size)))
    cache.store(split, family, built)
    return built


def pool(split: str, family: str) -> tuple[Task, ...]:
    """The complete declared pool of a non-confirmation split (cached per process)."""

    return _pool(split, family)


def pool_hashes(split: str, family: str) -> set[str]:
    return {task.source_sha256 for task in pool(split, family)}
