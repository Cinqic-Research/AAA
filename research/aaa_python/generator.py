"""Deterministic, auditable task generation for ``aaa.python.v0``.

Every task is a pure function of its identity ``aaa.python.v0:<split>:<family>:<index>``:

* programs come from a small set of frozen templates, drawn by a
  version-independent SHA-256 stream (:mod:`.rng`), so the same identity yields
  byte-identical source on every supported interpreter;
* **answer keys come from CPython** through the sandboxed oracle, never from the
  generator's intent. A corruption meant to break syntax that happens to
  leave valid code is labelled valid, because that is what the interpreter says;
* acceptance rules that need the oracle (an output in the declared range, a
  program that actually fails, a repair with exactly one passing candidate)
  retry with the next attempt counter of that same identity, so the result
  does not depend on batching or order;
* a split excludes every normalized source hash that occurs in the declared
  pools of all earlier splits (train < development < probe < attack), so an
  exact-match lookup cannot find an evaluation program in the training pool;
* confirmation identities are reserved and **refused**: v0 has no admitted
  freeze.

Slices separate what generalization is being asked for. Training and probe
items are ``in_distribution``. Development and attack items cycle through
``in_distribution``, ``novel_literals`` (constants from a disjoint range) and
``novel_structure`` (templates never used for training).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from . import GENERATOR_VERSION, PROTOCOL_VERSION
from . import spec as spec_module
from .oracle import Outcome, check_syntax, run_many
from .rng import Stream, derive_seed

SLICE_CYCLE = ("in_distribution", "in_distribution", "novel_literals", "novel_structure")
MAX_ATTEMPTS = 40
VARIABLES = ("a", "b", "c", "d", "n", "m", "x", "y", "val", "acc", "res", "total")
UNDEFINED = ("q", "w", "k2", "tmp9")


class ConfirmationNotAdmitted(RuntimeError):
    """aaa.python.v0 reserves confirmation identities and admits no freeze that could spend them."""


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Task:
    """Evaluator-side task. Learners never receive this object (see :mod:`.episode`)."""

    task_id: str
    split: str
    family: str
    index: int
    attempt: int
    slice: str
    template: str
    source: str
    repair_line: int | None = None
    candidates: tuple[str, ...] = ()
    visible_tests: tuple[tuple[int, int], ...] = ()
    hidden_tests: tuple[tuple[int, int], ...] = ()
    answer: Any = None
    oracle: Mapping[str, Any] = field(default_factory=dict)
    candidate_hidden_results: tuple[tuple[bool, ...], ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_sha256(self) -> str:
        return source_hash(self.source, self.candidates)


def source_hash(source: str, candidates: Sequence[str] = ()) -> str:
    normalized = "\n".join(line.rstrip() for line in source.strip("\n").splitlines())
    payload = normalized + "\0" + "\0".join(candidates)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_id(split: str, family: str, index: int) -> str:
    return f"{PROTOCOL_VERSION}:{split}:{family}:{index}"


def slice_of(split: str, index: int) -> str:
    return "in_distribution" if split in ("train", "probe") else SLICE_CYCLE[index % len(SLICE_CYCLE)]


# ----------------------------------------------------------------- templates
class Builder:
    """Draws the pieces of one program from a stream, within a literal range."""

    def __init__(self, stream: Stream, literals: tuple[int, int]) -> None:
        self.s = stream
        self.low, self.high = literals

    def lit(self) -> int:
        return self.s.between(self.low, self.high)

    def names(self, count: int) -> list[str]:
        return self.s.shuffled(VARIABLES)[:count]

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
    blocks: list[Block] = [[f"{base} = {b.lit()}"], [f"{acc} = {b.s.between(0, 3)}"]]
    step = b.s.choice(("+=", "-="))
    blocks.append([f"for i in range({b.s.between(2, 5)}):", f"    {acc} {step} {b.expr(['i', base])}"])
    return blocks, acc


def function_call(b: Builder) -> tuple[list[Block], str]:
    arg, out, second = b.names(3)
    blocks: list[Block] = [["def f(p):", f"    return {b.expr(['p'])}"], [f"{arg} = {b.lit()}"]]
    blocks.append([f"{out} = f({arg})"])
    if b.s.chance(1, 2):
        blocks.append([f"{second} = f({out}) - {arg}"])
        return blocks, second
    return blocks, out


def nested_loop(b: Builder) -> tuple[list[Block], str]:
    (acc,) = b.names(1)
    blocks: list[Block] = [[f"{acc} = 0"]]
    blocks.append(
        [
            f"for i in range({b.s.between(2, 3)}):",
            f"    for j in range({b.s.between(2, 3)}):",
            f"        {acc} += {b.s.choice(('i', 'j'))} {b.s.choice(('+', '-', '*'))} {b.s.choice(('i', 'j', '1'))}",
        ]
    )
    return blocks, acc


def list_walk(b: Builder) -> tuple[list[Block], str]:
    items, acc = b.names(2)
    size = b.s.between(3, 5)
    values = ", ".join(str(b.lit()) for _ in range(size))
    blocks: list[Block] = [[f"{items} = [{values}]"], [f"{acc} = 0"]]
    blocks.append([f"for i in range({b.s.between(1, size)}):", f"    {acc} += {items}[i] - i"])
    return blocks, acc


TEMPLATES: Mapping[str, Callable[[Builder], tuple[list[Block], str]]] = {
    "straight_line": straight_line,
    "conditional": conditional,
    "loop_accumulate": loop_accumulate,
    "function_call": function_call,
    "nested_loop": nested_loop,
    "list_walk": list_walk,
}


def _literals(spec: Mapping[str, Any], slice_name: str) -> tuple[int, int]:
    literals = spec["templates"]["literals"]
    low, high = literals["novel_literals" if slice_name == "novel_literals" else "in_distribution"]
    return int(low), int(high)


def _templates(spec: Mapping[str, Any], slice_name: str) -> list[str]:
    key = "novel_structure" if slice_name == "novel_structure" else "in_distribution"
    return list(spec["templates"][key])


def _defined_names(blocks: Sequence[Block]) -> list[str]:
    names = []
    for block in blocks:
        head = block[0]
        if " = " in head and not head.startswith(("def ", "for ", "if ")):
            names.append(head.split(" = ", 1)[0].strip())
        elif " += " in head or " -= " in head:
            continue
    for block in blocks:
        for line in block[1:]:
            stripped = line.strip()
            for op in (" += ", " -= ", " = "):
                if op in stripped:
                    names.append(stripped.split(op, 1)[0])
                    break
    return [n for n in dict.fromkeys(names) if n in VARIABLES]


def fault_block(b: Builder, names: Sequence[str], avoid: Sequence[str]) -> tuple[str, Block]:
    """A value-dependent fault: whether it fires depends on the program's state, not its surface.

    The assigned name is drawn from variables the program never uses, so the
    fault line carries no fixed marker and changes nothing downstream.
    """

    kind = b.s.choice(("zero_division", "index", "name", "type", "value"))
    v, u = b.s.choice(names), b.s.choice(names)
    t = b.s.choice([name for name in VARIABLES if name not in avoid])
    k = b.lit()
    if kind == "zero_division":
        return kind, [f"{t} = {v} // ({u} - {k})"]
    if kind == "index":
        return kind, [f"{t} = [{b.lit()}, {b.lit()}, {b.lit()}][{v} - {k}]"]
    if kind == "name":
        return kind, [f"if {v} > {k}:", f"    {t} = {b.s.choice(UNDEFINED)} + 1"]
    if kind == "type":
        return kind, [f"{t} = {v} + (str({u}) if {u} > {k} else {u})"]
    return kind, [f'{t} = int(("{b.lit()}" if {v} > {k} else "z"))']


def semantic_program(
    stream: Stream, spec: Mapping[str, Any], slice_name: str, *, fault: bool
) -> tuple[str, str, dict[str, Any]]:
    b = Builder(stream, _literals(spec, slice_name))
    template = b.s.choice(_templates(spec, slice_name))
    blocks, printed = TEMPLATES[template](b)
    notes: dict[str, Any] = {}
    if fault:
        # The fault references only names bound by earlier top-level blocks, so
        # whether it fires depends on program state, not on an unbound name.
        position = b.s.between(1, len(blocks))
        names = _defined_names(blocks[:position])
        if not names:
            position, names = len(blocks), _defined_names(blocks) or [printed]
        used = set(re.findall(r"[a-z][a-z0-9]*", "\n".join(line for block in blocks for line in block)))
        kind, block = fault_block(b, names, sorted(used))
        blocks.insert(position, block)
        notes["fault"] = [kind]
        # Half the faulty programs get a second value-dependent fault as a decoy:
        # a risky-looking line then no longer identifies the failing one.
        if b.s.chance(1, 2):
            used |= set(re.findall(r"[a-z][a-z0-9]*", "\n".join(block)))
            later = b.s.between(position + 1, len(blocks))
            names = _defined_names(blocks[:later]) or names
            kind, block = fault_block(b, names, sorted(used))
            blocks.insert(later, block)
            notes["fault"].append(kind)
    blocks.append([f"print({printed})"])
    return "\n".join(line for block in blocks for line in block) + "\n", template, notes


CORRUPTIONS = ("drop_colon", "drop_paren", "bad_indent", "assign_in_test", "open_string")


def corrupt(stream: Stream, source: str) -> tuple[str, str]:
    lines = source.rstrip("\n").split("\n")
    kind = stream.choice(CORRUPTIONS)
    colon_lines = [i for i, line in enumerate(lines) if line.rstrip().endswith(":")]
    paren_lines = [i for i, line in enumerate(lines) if line.rstrip().endswith(")")]
    if kind == "drop_colon" and colon_lines:
        i = stream.choice(colon_lines)
        lines[i] = lines[i].rstrip()[:-1]
    elif kind == "drop_paren" and paren_lines:
        i = stream.choice(paren_lines)
        lines[i] = lines[i].rstrip()[:-1]
    elif kind == "assign_in_test" and any(" == " in line for line in lines):
        i = stream.choice([i for i, line in enumerate(lines) if " == " in line])
        lines[i] = lines[i].replace(" == ", " = ", 1)
    elif kind == "open_string":
        i = stream.below(len(lines))
        lines[i] = lines[i] + ' + "'
    else:
        kind = "bad_indent"
        i = stream.below(len(lines))
        lines[i] = "  " + lines[i] if not lines[i].startswith(" ") else lines[i][1:]
    return "\n".join(lines) + "\n", kind


# ----------------------------------------------------------------- repair
REPAIR_TEMPLATES = {
    "in_distribution": ("affine", "piecewise"),
    "novel_structure": ("loop_sum",),
}


def _mutations(line: str) -> list[str]:
    swaps = [
        (" + ", " - "),
        (" - ", " + "),
        (" > ", " >= "),
        (" >= ", " > "),
        (" < ", " <= "),
        (" * ", " + "),
    ]
    out = [line.replace(a, b, 1) for a, b in swaps if a in line]
    tokens = line.split(" ")
    for position, token in enumerate(tokens):
        if token.isdigit():
            for delta in (1, -1, 2):
                changed = list(tokens)
                changed[position] = str(max(0, int(token) + delta))
                out.append(" ".join(changed))
    return [m for m in dict.fromkeys(out) if m != line]


def repair_candidate(stream: Stream, spec: Mapping[str, Any], slice_name: str) -> dict[str, Any] | None:
    b = Builder(stream, _literals(spec, slice_name))
    template = b.s.choice(
        REPAIR_TEMPLATES["novel_structure" if slice_name == "novel_structure" else "in_distribution"]
    )
    lo, hi = b.lit(), b.lit()
    if template == "affine":
        body = [
            "def f(p):",
            f"    q = p * {b.s.between(1, 3)} + {lo}",
            "    return q - " + str(b.s.between(0, 3)),
        ]
        target = 1
    elif template == "piecewise":
        body = [
            "def f(p):",
            f"    if p > {lo}:",
            f"        return p - {hi}",
            f"    return p + {b.s.between(0, 4)}",
        ]
        target = b.s.choice((1, 2, 3))
    else:
        body = [
            "def f(p):",
            "    q = 0",
            f"    for i in range({b.s.between(2, 4)}):",
            f"        q += p - {lo}",
            "    return q",
        ]
        target = 3
    correct = body[target]
    mutations = _mutations(correct)
    if len(mutations) < 4:
        return None
    mutations = stream.shuffled(mutations)
    buggy = mutations[0]
    distractors = mutations[1:3]
    candidates = stream.shuffled([correct, buggy, *distractors])
    inputs = stream.shuffled(list(range(b.low - 2, b.high + 6)))[:6]
    return {
        "template": template,
        "reference": "\n".join(body) + "\n",
        "buggy": "\n".join([*body[:target], buggy, *body[target + 1 :]]) + "\n",
        "line": target + 1,
        "candidates": tuple(candidates),
        "correct": candidates.index(correct),
        "visible_inputs": tuple(inputs[:2]),
        "hidden_inputs": tuple(inputs[2:6]),
        "body": body,
        "target": target,
    }


def _repair_program(body: Sequence[str], target: int, replacement: str, inputs: Sequence[int]) -> str:
    lines = [*list(body[:target]), replacement, *list(body[target + 1 :])]
    return "\n".join(lines + [f"print(f({value}))" for value in inputs]) + "\n"


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
    slice_name = slice_of(split, index)
    identity = task_id(split, family, index)
    if family == "repair":
        repair = repair_candidate(stream, spec, slice_name)
        if repair is None:
            return None
        inputs = repair["visible_inputs"] + repair["hidden_inputs"]
        jobs = [
            (
                "exec",
                _repair_program(repair["body"], repair["target"], repair["body"][repair["target"]], inputs),
            )
        ]
        jobs += [
            ("exec", _repair_program(repair["body"], repair["target"], c, inputs))
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
    source, template, notes = semantic_program(stream, spec, slice_name, fault=fault)
    if family == "syntax":
        if stream.chance(1, 2):
            source, notes["corruption"] = corrupt(stream, source)
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


def _accept(d: Draft, outcomes: Sequence[Outcome], spec: Mapping[str, Any]) -> bool:
    if not all(o.is_program_outcome for o in outcomes):
        return False
    first = outcomes[0]
    if d.family == "output":
        text = first.stdout.strip()
        family = spec["families"]["output"]
        return first.status == "ok" and _is_int(text) and family["min"] <= int(text) <= family["max"]
    if d.family == "localize":
        return first.status == "exception" and first.line is not None
    if d.family == "outcome":
        return first.status == "ok" or (
            first.status == "exception" and first.exception in spec["families"]["outcome"]["labels"]
        )
    if d.family == "repair":
        assert d.repair is not None
        reference = first.stdout.split()
        if first.status != "ok" or len(reference) != 6:
            return False
        hidden = reference[2:]
        passing = [
            k for k, o in enumerate(outcomes[1:]) if o.status == "ok" and o.stdout.split()[2:] == hidden
        ]
        return passing == [d.repair["correct"]]
    return True


def _is_int(text: str) -> bool:
    return text.lstrip("-").isdigit() and text.count("-") <= 1


def answer_from(family: str, outcome: Outcome) -> Any:
    """The canonical answer a task's oracle outcome implies (repair is handled by its candidates)."""

    if family == "syntax":
        return "valid" if outcome.status == "valid" else "invalid"
    if family == "outcome":
        return "ok" if outcome.status == "ok" else outcome.exception
    if family == "output":
        return int(outcome.stdout.strip())
    if family == "localize":
        return outcome.line
    raise GenerationError(f"no single-outcome answer for {family}")


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
    hidden = [str(v) for v in reference[2:]]
    results = tuple(
        tuple(
            o.status == "ok" and len(o.stdout.split()) == 6 and o.stdout.split()[2 + k] == hidden[k]
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
        visible_tests=tests[:2],
        hidden_tests=tests[2:],
        answer=d.repair["correct"],
        oracle=oracle,
        candidate_hidden_results=results,
    )


def build(
    split: str, family: str, indices: Sequence[int], spec: Mapping[str, Any] | None = None
) -> list[Task]:
    """Tasks for ``indices`` of one split and family, oracle-accepted and disjoint from earlier pools."""

    spec = spec or spec_module.load()
    if split == "confirmation":
        raise ConfirmationNotAdmitted(
            "aaa.python.v0 reserves confirmation identities; no freeze admits them, so none is generated"
        )
    order = spec["splits"]["order"]
    if split not in order or family not in spec["families"]:
        raise GenerationError(f"unknown split or family: {split}/{family}")
    excluded: set[str] = set()
    for earlier in order[: order.index(split)]:
        excluded |= pool_hashes(earlier, family)
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
            if d is None or source_hash(d.source, d.repair["candidates"] if d.repair else ()) in excluded:
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


@cache
def _pool(split: str, family: str) -> tuple[Task, ...]:
    size = spec_module.load()["splits"]["pool_per_family"][split]
    return tuple(build(split, family, range(size)))


def pool(split: str, family: str) -> tuple[Task, ...]:
    """The complete declared pool of a split (cached per process)."""

    return _pool(split, family)


def pool_hashes(split: str, family: str) -> set[str]:
    return {task.source_sha256 for task in pool(split, family)}


def syntax_answer(source: str, spec: Mapping[str, Any] | None = None) -> str:
    return "valid" if check_syntax(source, spec).status == "valid" else "invalid"
