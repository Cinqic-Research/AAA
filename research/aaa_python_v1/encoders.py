"""Fixed, parameter-free encoders for ``aaa.python.v1``: an explicit, versioned interface.

An encoder maps a :class:`~research.aaa_python.episode.TaskView` -- the only
thing a learner may see before acting -- to sparse, signed-hashed feature
vectors in ``R^D``:

* ``program``: the whole source (for ``syntax``, ``outcome``, ``output`` and
  the one-hot ``localize`` head);
* ``lines``: one vector per source line (for a per-line pointer head);
* ``candidates``: one vector per repair candidate.

Every encoder here has **zero trainable parameters and no adaptive state**, so
at a fixed downstream model the encoders are compared at exactly matched
trainable capacity. What an encoder *does* cost is recorded in
:func:`accounting`: its fixed lookup tables and the features it computes.

``e0`` -- minimal control
    Token unigrams (type and lexeme) of the v0 lexer: a bag of tokens with no
    order at all.
``e1`` -- the v0-tied lexical encoder
    v0's ``lexical`` representation (token unigrams and bigrams, type and
    lexeme) with one correction justified by v1's hashing diagnostic: signed
    hashing, which improved v0's own learner at small ``D`` and was neutral at
    ``D >= 1024``. The model owns its bias, so no bias feature is hashed.
``e2`` -- structural
    Built from measured v0 weaknesses and the mechanism review
    (``docs/aaa_python_v1_literature_review.md``). Four channels:

    * ``alpha``: e1's n-grams after alpha-renaming every identifier to its
      first-occurrence role (``V0``, ``V1``, ...) and bucketing literals >= 10;
    * ``line``: line-anchored structure: first/last token shape, indentation
      and its change after a header, running bracket depth and quote parity;
    * ``flow`` (never for ``syntax``): static def-use and operator/operand
      structure from CPython's AST: how each used name was last defined
      (literal, expression, loop variable, parameter, list, undefined), what
      each operator is applied to, and statement nesting;
    * ``position`` (per-line vectors only, and shared by all encoders): the
      line's index from the start and from the end.

    ``flow`` performs **no evaluation**: it never propagates values or runs
    anything, because constant propagation over these programs would be the
    oracle in disguise.

Leakage rules enforced here and attacked in the tests:

1. ``syntax`` never reaches the parser. Whether ``ast.parse`` succeeds *is* the
   answer key (``AAA-184``); the lexer and line channels are parser-free.
2. For other families every program is valid by construction; a patched
   repair program that failed to parse would make *every* candidate of that
   task fall back to parser-free features, so parse success cannot single one
   candidate out.
3. Encoders read only ``TaskView`` fields. They cannot reach answer keys,
   oracle observations or hidden tests (the view does not contain them).
"""

from __future__ import annotations

import ast
import hashlib
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from research.aaa_python.episode import TaskView
from research.aaa_python.representation import KEYWORDS, tokens

ENCODERS = ("e0", "e1", "e2")
E2_CHANNELS = ("alpha", "line", "flow")
BUILTINS = frozenset({"print", "range", "len", "abs", "min", "max", "int", "str", "bool", "sum"})
ENCODER_VERSION = "aaa.python.v1.encoders.v1"


@dataclass(frozen=True)
class Sparse:
    """A sparse vector: parallel index and value arrays; indices are unique and sorted."""

    index: np.ndarray
    value: np.ndarray

    def dense(self, dimensions: int) -> np.ndarray:
        out = np.zeros(dimensions)
        out[self.index] = self.value
        return out


@dataclass(frozen=True)
class Encoding:
    program: Sparse
    lines: tuple[Sparse, ...]
    candidates: tuple[Sparse, ...]


def _hash(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


def sparse(features: Iterable[str], dimensions: int) -> Sparse:
    """Signed feature hashing (Weinberger et al. 2009), then L2 normalization."""

    acc: dict[int, float] = {}
    for feature in features:
        h = _hash(feature)
        slot = h % dimensions
        acc[slot] = acc.get(slot, 0.0) + (1.0 if (h >> 63) == 0 else -1.0)
    keys = sorted(k for k, v in acc.items() if v != 0.0)
    value = np.array([acc[k] for k in keys], dtype=float)
    norm = float(np.linalg.norm(value))
    if norm > 0:
        value = value / norm
    index = np.array(keys, dtype=np.int64)
    index.setflags(write=False)
    value.setflags(write=False)
    return Sparse(index, value)


def blend(parts: Sequence[tuple[float, Sparse]]) -> Sparse:
    acc: dict[int, float] = {}
    for weight, part in parts:
        for i, v in zip(part.index.tolist(), part.value.tolist(), strict=True):
            acc[i] = acc.get(i, 0.0) + weight * v
    keys = sorted(k for k, v in acc.items() if v != 0.0)
    index = np.array(keys, dtype=np.int64)
    value = np.array([acc[k] for k in keys], dtype=float)
    index.setflags(write=False)
    value.setflags(write=False)
    return Sparse(index, value)


# --------------------------------------------------------------------- lexical
def _lexical(source: str, bigrams: bool) -> list[str]:
    stream = [f"{k}={t}" for k, t in tokens(source)]
    out = [f"t1:{t}" for t in stream]
    if bigrams:
        out += [f"t2:{a}|{b}" for a, b in itertools.pairwise(stream)]
    return out


def _literal(text: str) -> str:
    value = int(text)
    return text if value < 10 else f"{value // 10}x"


def renamed_tokens(source: str, mapping: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """Tokens with identifiers replaced by first-occurrence roles and literals >= 10 bucketed."""

    mapping = {} if mapping is None else mapping
    out = []
    for kind, text in tokens(source):
        if kind == "NAME" and text not in BUILTINS:
            if text not in mapping:
                mapping[text] = f"V{len(mapping)}"
            out.append((kind, mapping[text]))
        elif kind == "NUMBER":
            out.append((kind, _literal(text)))
        else:
            out.append((kind, text))
    return out


def _alpha(source: str, mapping: dict[str, str] | None = None) -> list[str]:
    stream = [f"{k}={t}" for k, t in renamed_tokens(source, mapping)]
    return [f"a1:{t}" for t in stream] + [f"a2:{a}|{b}" for a, b in itertools.pairwise(stream)]


# --------------------------------------------------------------------- line structure
def _shape(kind: str, text: str) -> str:
    if kind in ("KEYWORD", "OP"):
        return f"{kind}:{text}"
    return kind


def line_structure(source: str) -> list[list[str]]:
    """Parser-free, line-anchored features, one list per source line."""

    lines = source.rstrip("\n").split("\n")
    depth = 0
    previous_colon = False
    previous_indent = 0
    per_line: list[list[str]] = []
    for line in lines:
        features: list[str] = []
        indent = len(line) - len(line.lstrip(" "))
        toks = [(k, t) for k, t in tokens(line.strip()) if k != "NEWLINE"]
        first = _shape(*toks[0]) if toks else "EMPTY"
        last = _shape(*toks[-1]) if toks else "EMPTY"
        features.append(f"l:shape:{first}|{last}")
        features.append(f"l:indent_mod4:{indent % 4}")
        delta = max(-2, min(2, (indent - previous_indent) // 2))
        features.append(f"l:indent_delta:{delta}:after_colon:{previous_colon}")
        opened = False
        for _kind, text in toks:
            if text in "([":
                depth += 1
                opened = True
            elif text in ")]":
                depth -= 1
            if depth < 0:
                features.append("l:depth_negative")
        features.append(f"l:depth_end:{max(-2, min(2, depth))}:opened:{opened}")
        if line.count('"') % 2 or line.count("'") % 2:
            features.append("l:quote_odd")
        previous_colon = line.rstrip().endswith(":")
        previous_indent = indent
        per_line.append(features)
    if per_line:
        per_line[-1].append(f"l:final_depth:{max(-2, min(2, depth))}")
    return per_line


# --------------------------------------------------------------------- static flow
def _kind(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return f"Const_{type(node.value).__name__}"
    return type(node).__name__


class _Flow(ast.NodeVisitor):
    """Static def-use and operator structure per line; no values are ever computed."""

    def __init__(self) -> None:
        self.defs: dict[str, tuple[str, int]] = {}
        self.features: dict[int, list[str]] = {}
        self.depth = 0
        self.in_function = False
        self.parameters: set[str] = set()

    def _add(self, line: int, feature: str) -> None:
        self.features.setdefault(line, []).append(feature)

    def _define(self, target: ast.AST, kind: str, line: int) -> None:
        if isinstance(target, ast.Name):
            self.defs[target.id] = (kind, line)

    @staticmethod
    def _value_kind(value: ast.AST) -> str:
        if isinstance(value, ast.Constant):
            return "lit"
        if isinstance(value, ast.List):
            return "list"
        if isinstance(value, ast.Call):
            return "call"
        return "expr"

    def _uses(self, node: ast.AST, line: int) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id not in BUILTINS:
                if child.id in self.parameters:
                    kind, distance = "param", 0
                elif child.id in self.defs:
                    kind, at = self.defs[child.id]
                    distance = min(3, line - at)
                else:
                    kind, distance = "undefined", 0
                self._add(line, f"f:use:{kind}:{distance}")
            elif isinstance(child, ast.BinOp):
                self._add(line, f"f:bin:{type(child.op).__name__}:{_kind(child.left)}:{_kind(child.right)}")
            elif isinstance(child, ast.Subscript):
                self._add(line, f"f:sub:{_kind(child.value)}:{_kind(child.slice)}")
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                name = child.func.id if child.func.id in BUILTINS else "user"
                self._add(line, f"f:call:{name}:{'_'.join(_kind(a) for a in child.args)}")
            elif isinstance(child, ast.IfExp):
                self._add(line, f"f:ifexp:{_kind(child.body)}:{_kind(child.orelse)}")
            elif isinstance(child, ast.Compare):
                self._add(line, f"f:cmp:{'_'.join(type(o).__name__ for o in child.ops)}:{_kind(child.left)}")

    def _statement(self, node: ast.stmt) -> None:
        line = node.lineno
        self._add(line, f"f:stmt:{type(node).__name__}:depth:{min(self.depth, 3)}:fn:{self.in_function}")

    def visit_Assign(self, node: ast.Assign) -> None:
        self._statement(node)
        self._uses(node.value, node.lineno)
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                self._uses(target, node.lineno)
            self._define(target, self._value_kind(node.value), node.lineno)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._statement(node)
        self._uses(node.value, node.lineno)
        self._uses(ast.Name(id=getattr(node.target, "id", ""), ctx=ast.Load()), node.lineno)
        self._define(node.target, "aug", node.lineno)

    def visit_Expr(self, node: ast.Expr) -> None:
        self._statement(node)
        self._uses(node.value, node.lineno)

    def visit_Return(self, node: ast.Return) -> None:
        self._statement(node)
        if node.value is not None:
            self._uses(node.value, node.lineno)

    def visit_If(self, node: ast.If) -> None:
        self._statement(node)
        self._uses(node.test, node.lineno)
        self.depth += 1
        for child in [*node.body, *node.orelse]:
            self.visit(child)
        self.depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._statement(node)
        self._uses(node.iter, node.lineno)
        self._define(node.target, "loop", node.lineno)
        self.depth += 1
        for child in node.body:
            self.visit(child)
        self.depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._statement(node)
        saved = (self.in_function, set(self.parameters), dict(self.defs))
        self.in_function = True
        self.parameters = {a.arg for a in node.args.args}
        self.depth += 1
        for child in node.body:
            self.visit(child)
        self.depth -= 1
        self.in_function, self.parameters, self.defs = saved
        self.defs[node.name] = ("function", node.lineno)


def flow_features(source: str) -> dict[int, list[str]] | None:
    """Per-line static flow features, or ``None`` if the source does not parse."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    visitor = _Flow()
    for statement in tree.body:
        visitor.visit(statement)
    return visitor.features


def position_features(index: int, count: int) -> list[str]:
    return [f"p:from_start:{min(index, 12)}", f"p:from_end:{min(count - 1 - index, 12)}"]


# --------------------------------------------------------------------- encoders
@dataclass(frozen=True)
class Encoder:
    """A named, fixed encoder; ``channels`` selects ``e2``'s channels (ablations drop one)."""

    name: str
    dimensions: int
    channels: tuple[str, ...] = E2_CHANNELS

    def __post_init__(self) -> None:
        if self.name not in ENCODERS:
            raise ValueError(f"unknown encoder {self.name!r}")
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions < 8:
            raise ValueError("dimensions must be an integer >= 8")
        if self.name == "e2" and (not self.channels or set(self.channels) - set(E2_CHANNELS)):
            raise ValueError(f"e2 channels must be a non-empty subset of {E2_CHANNELS}")

    @property
    def identity(self) -> str:
        suffix = "" if self.name != "e2" or self.channels == E2_CHANNELS else "-" + "+".join(self.channels)
        return f"{ENCODER_VERSION}:{self.name}{suffix}:D{self.dimensions}"

    # -- whole-source features ----------------------------------------------------
    def _source_features(self, source: str, family: str, mapping: dict[str, str] | None = None) -> list[str]:
        if self.name == "e0":
            return _lexical(source, bigrams=False)
        if self.name == "e1":
            return _lexical(source, bigrams=True)
        features: list[str] = []
        if "alpha" in self.channels:
            features += _alpha(source, mapping)
        else:
            features += _lexical(source, bigrams=True)
        if "line" in self.channels:
            features += [f for line in line_structure(source) for f in line]
        if "flow" in self.channels and family != "syntax":
            flow = flow_features(source)
            if flow is not None:
                features += [f for line in sorted(flow) for f in flow[line]]
        return features

    def _line_features(self, source: str, family: str) -> list[list[str]]:
        lines = source.rstrip("\n").split("\n")
        count = len(lines)
        per_line: list[list[str]] = []
        if self.name == "e2":
            mapping: dict[str, str] = {}
            renamed = [_alpha(line, mapping) for line in lines] if "alpha" in self.channels else None
            structure = line_structure(source) if "line" in self.channels else None
            flow = flow_features(source) if "flow" in self.channels and family != "syntax" else None
            for i, line in enumerate(lines):
                features = list(renamed[i]) if renamed is not None else _lexical(line, bigrams=True)
                if structure is not None:
                    features += structure[i]
                if flow is not None:
                    features += flow.get(i + 1, [])
                per_line.append(features + position_features(i, count))
        else:
            for i, line in enumerate(lines):
                per_line.append(_lexical(line, bigrams=self.name == "e1") + position_features(i, count))
        return per_line

    def encode(self, view: TaskView) -> Encoding:
        return _encode_cached(self, view.family, view.source, view.repair_line, view.candidates)

    def _encode(
        self, family: str, source: str, repair_line: int | None, candidates: tuple[str, ...]
    ) -> Encoding:
        D = self.dimensions
        if family == "repair":
            assert repair_line is not None
            source_lines = source.rstrip("\n").split("\n")
            patched = [
                "\n".join([*source_lines[: repair_line - 1], c, *source_lines[repair_line:]]) + "\n"
                for c in candidates
            ]
            # Rule 2: parse failure of any patched program removes flow features for all of them.
            parse_all = self.name != "e2" or all(flow_features(p) is not None for p in patched)
            encoder = (
                self
                if parse_all
                else Encoder(self.name, D, tuple(c for c in self.channels if c != "flow") or ("alpha",))
            )
            current = source_lines[repair_line - 1]
            rows = []
            for candidate, patched_source in zip(candidates, patched, strict=True):
                mapping: dict[str, str] = {}
                whole = sparse(encoder._source_features(patched_source, family, mapping), D)
                line_feats = (
                    _alpha(candidate, mapping)
                    if self.name == "e2" and "alpha" in self.channels
                    else _lexical(candidate, bigrams=self.name != "e0")
                )
                if self.name == "e2":
                    # The edit relative to the visible current line is structure every e2 variant keeps.
                    line_feats += _diff(current, candidate)
                rows.append(blend([(0.5, whole), (0.5, sparse(line_feats, D))]))
            return Encoding(sparse(encoder._source_features(source, family), D), (), tuple(rows))
        program = sparse(self._source_features(source, family), D)
        lines_ = (
            tuple(sparse(f, D) for f in self._line_features(source, family)) if family == "localize" else ()
        )
        return Encoding(program, lines_, ())


def _diff(current: str, candidate: str) -> list[str]:
    """How a candidate line differs from the line it would replace (both are visible)."""

    a, b = current.split(" "), candidate.split(" ")
    if len(a) != len(b):
        return ["d:shape_changed"]
    out = []
    for x, y in zip(a, b, strict=True):
        if x == y:
            continue
        if x.isdigit() and y.isdigit():
            out.append(f"d:lit:{max(-3, min(3, int(y) - int(x)))}")
        else:
            out.append(f"d:op:{x}>{y}")
    return out or ["d:identical"]


@lru_cache(maxsize=200000)
def _encode_cached(
    encoder: Encoder, family: str, source: str, repair_line: int | None, candidates: tuple[str, ...]
) -> Encoding:
    return encoder._encode(family, source, repair_line, candidates)


def accounting(encoder: Encoder) -> dict[str, Any]:
    """What an encoder costs: nothing trainable, no adaptive state; its fixed tables are listed."""

    return {
        "identity": encoder.identity,
        "trainable_parameters": 0,
        "adaptive_state": 0,
        "fixed_tables": {
            "lexer_keywords": len(KEYWORDS),
            "builtin_names": len(BUILTINS) if encoder.name == "e2" else 0,
        },
        "output_dimensions": encoder.dimensions,
        "hashing": "BLAKE2b-64, signed (top bit), modulo D, L2-normalized",
        "uses_parser": encoder.name == "e2" and "flow" in encoder.channels,
        "parser_refused_for": ["syntax"],
        "evaluates_programs": False,
    }
