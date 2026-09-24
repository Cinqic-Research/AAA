"""Candidate input representations, treated as an experimental variable.

All four hash sparse features into a fixed ``D``-dimensional space with a
stable hash (BLAKE2b), so no vocabulary is fitted and unknown identifiers need
no out-of-vocabulary mechanism:

``bytes``          UTF-8 byte 1-3 grams. Deterministic vocabulary; represents
                   anything, including later natural-language text.
``lexical``        token type + lexeme, unigrams and bigrams.
``lexical_types``  token types only (an ablation of ``lexical``: no lexemes).
``ast_nodes``      parent/child AST node-type pairs, from CPython's own parser.

``ast_nodes`` is refused for the ``syntax`` family and falls back to
``lexical`` there (:func:`effective_representation`). Whether ``ast.parse``
succeeds *is* the interpreter's syntax verdict, the family's answer key; the
first development smoke scored 1.0 through exactly that channel before this
rule existed. For the other families every program parses, so the
representation carries structure, not the answer.

The lexer is a small regular-expression tokenizer written here rather than
:mod:`tokenize`, whose behaviour changed in CPython 3.12 (PEP 701); a
representation must not silently differ between supported interpreters.
Repair tasks add features of each candidate line in context (see
:func:`candidate_features`).
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import re
from collections.abc import Iterable, Sequence

import numpy as np

REPRESENTATIONS = ("bytes", "lexical", "lexical_types", "ast_nodes")
_TOKEN = re.compile(
    r"(?P<NUMBER>\d+)|(?P<NAME>[A-Za-z_][A-Za-z0-9_]*)|(?P<STRING>\"[^\"\n]*\"?|'[^'\n]*'?)"
    r"|(?P<OP>//|==|!=|<=|>=|\+=|-=|\*=|[-+*/%<>=()\[\]:,])|(?P<NEWLINE>\n)|(?P<SPACE>[ \t]+)|(?P<OTHER>.)"
)
KEYWORDS = frozenset({"if", "else", "elif", "for", "in", "def", "return", "and", "or", "not", "pass"})


def tokens(source: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    at_line_start = True
    for match in _TOKEN.finditer(source):
        kind = match.lastgroup or "OTHER"
        text = match.group()
        if kind == "SPACE":
            if at_line_start:
                out.append(("INDENT", str(len(text))))
            continue
        at_line_start = kind == "NEWLINE"
        if kind == "NAME" and text in KEYWORDS:
            kind = "KEYWORD"
        out.append((kind, text))
    return out


def effective_representation(representation: str, family: str) -> str:
    """The representation actually used for ``family`` (see the module docstring)."""

    if representation == "ast_nodes" and family in ("syntax", "fragment"):
        # "fragment": a lone repair-candidate line, which is not a parsable program.
        return "lexical"
    return representation


def _stable(feature: str, dimensions: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def _features(source: str, representation: str) -> Iterable[str]:
    if representation == "bytes":
        data = source.encode("utf-8")
        for n in (1, 2, 3):
            for i in range(len(data) - n + 1):
                yield f"b{n}:{data[i : i + n].hex()}"
    elif representation in ("lexical", "lexical_types"):
        stream = [f"{k}" if representation == "lexical_types" else f"{k}={t}" for k, t in tokens(source)]
        yield from (f"t1:{t}" for t in stream)
        yield from (f"t2:{a}|{b}" for a, b in itertools.pairwise(stream))
    elif representation == "ast_nodes":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Only reachable if a caller bypasses effective_representation.
            raise ValueError(
                "ast_nodes cannot represent unparsable source without leaking the syntax verdict"
            ) from None
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                yield f"a:{type(parent).__name__}>{type(child).__name__}"
    else:
        raise ValueError(f"unknown representation {representation!r}")


def vector(source: str, representation: str, dimensions: int) -> np.ndarray:
    """A dense, L2-normalized hashed feature vector (with a constant bias feature)."""

    out = np.zeros(dimensions)
    for feature in _features(source, representation):
        out[_stable(feature, dimensions)] += 1.0
    norm = float(np.linalg.norm(out))
    if norm > 0:
        out /= norm
    out[_stable("<bias>", dimensions)] += 1.0
    return out


def candidate_features(
    source: str, line: int, candidates: Sequence[str], representation: str, dimensions: int
) -> np.ndarray:
    """One row per repair candidate: the program with that line substituted, plus the line itself."""

    lines = source.rstrip("\n").split("\n")
    rows = []
    for candidate in candidates:
        patched = "\n".join([*lines[: line - 1], candidate, *lines[line:]])
        rows.append(
            0.5 * vector(patched, representation, dimensions)
            + 0.5 * vector(candidate, representation, dimensions)
        )
    return np.array(rows)
