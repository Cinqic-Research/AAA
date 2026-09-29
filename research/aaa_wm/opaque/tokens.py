"""A fixed, versioned tokenizer for opaque.v0 programs and test values (no learned state).

Program text becomes structural tokens (keywords, names, library names, operators,
NEWLINE / INDENT / DEDENT) plus numeric tokens. Every integer, whether a code literal, a test
input or an expected value, uses one shared numeric vocabulary ``[-128, 511]`` with
``<NUM_LO>`` / ``<NUM_HI>`` overflow tokens. A per-position *type* id says which role a
number plays. Results (the world model's targets) use the same numeric range plus
exception classes.
"""

from __future__ import annotations

import hashlib
import io
import tokenize
from functools import lru_cache

import numpy as np

VERSION = "aaa.wm.opaque.tokens.v1"
NUM_MIN, NUM_MAX = -128, 511
SPECIALS = ["<PAD>", "<CLS>", "<SEP>", "<NL>", "<IND>", "<DED>", "<Q>", "<NUM_LO>", "<NUM_HI>", "<UNK>"]
WORDS = ["def", "if", "else", "for", "in", "return", "range", "f", "g", "p", "q", "i", "j", "r", "h"]
WORDS += [f"api{k}" for k in range(8)]
OPS = ["+", "-", "*", "//", "%", ">", ">=", "<", "<=", "==", "!=", "+=", "-=", "=", "(", ")", ":", ","]
VOCAB = SPECIALS + WORDS + OPS + [str(v) for v in range(NUM_MIN, NUM_MAX + 1)]
INDEX = {t: i for i, t in enumerate(VOCAB)}
PAD, CLS, SEP, Q = INDEX["<PAD>"], INDEX["<CLS>"], INDEX["<SEP>"], INDEX["<Q>"]

# token types
T_CODE, T_INPUT, T_EXPECTED, T_QUERY, T_MARK = 0, 1, 2, 3, 4

# result classes: numeric range, below, above, ZeroDivisionError, other error
RESULT_ERRORS = ["<LO>", "<HI>", "ZeroDivisionError", "<ERR>"]
N_RESULTS = (NUM_MAX - NUM_MIN + 1) + len(RESULT_ERRORS)


def vocab_hash() -> str:
    return hashlib.sha256(("\n".join([VERSION, *VOCAB, *RESULT_ERRORS])).encode()).hexdigest()


def number(v: int) -> int:
    if v < NUM_MIN:
        return INDEX["<NUM_LO>"]
    if v > NUM_MAX:
        return INDEX["<NUM_HI>"]
    return INDEX[str(v)]


NUM_BASE = INDEX[str(NUM_MIN)]


def numbers(values: np.ndarray) -> np.ndarray:
    """Vectorized :func:`number`."""

    v = np.asarray(values, dtype=np.int64)
    out = NUM_BASE + np.clip(v, NUM_MIN, NUM_MAX) - NUM_MIN
    out = np.where(v < NUM_MIN, INDEX["<NUM_LO>"], out)
    return np.where(v > NUM_MAX, INDEX["<NUM_HI>"], out)


def result_class(result: tuple[str, object]) -> int:
    kind, value = result
    if kind == "ok" and isinstance(value, int):
        if value < NUM_MIN:
            return NUM_MAX - NUM_MIN + 1
        if value > NUM_MAX:
            return NUM_MAX - NUM_MIN + 2
        return value - NUM_MIN
    if kind == "error" and value == "ZeroDivisionError":
        return NUM_MAX - NUM_MIN + 3
    return NUM_MAX - NUM_MIN + 4


@lru_cache(maxsize=100000)
def encode_program(source: str) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Token ids, and for every id its ``(row, col)`` in the source (``(0, 0)`` for layout tokens)."""

    ids: list[int] = []
    pos: list[tuple[int, int]] = []
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type == tokenize.NEWLINE:
            ids.append(INDEX["<NL>"])
        elif t.type == tokenize.INDENT:
            ids.append(INDEX["<IND>"])
        elif t.type == tokenize.DEDENT:
            ids.append(INDEX["<DED>"])
        elif t.type == tokenize.NUMBER:
            ids.append(number(int(t.string)))
        elif t.type in (tokenize.NAME, tokenize.OP):
            ids.append(INDEX.get(t.string, INDEX["<UNK>"]))
        else:
            continue
        pos.append(t.start if t.type in (tokenize.NUMBER, tokenize.NAME, tokenize.OP) else (0, 0))
    return tuple(ids), tuple(pos)


def sequence(
    source: str,
    *,
    tests: tuple[tuple[int, int], ...] = (),
    queries: tuple[int, ...] = (),
    mark: tuple[int, int] | None = None,
    length: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """``[CLS] program [SEP] (x e [SEP])* (<Q> x)*`` as ids and types, padded to ``length``."""

    code, where = encode_program(source)
    ids = [CLS, *code, SEP]
    types = [T_CODE] * len(ids)
    if mark is not None:
        for k, w in enumerate(where):
            if w == mark:
                types[1 + k] = T_MARK
    for x, e in tests:
        ids += [number(x), number(e), SEP]
        types += [T_INPUT, T_EXPECTED, T_CODE]
    for x in queries:
        ids += [Q, number(x)]
        types += [T_QUERY, T_INPUT]
    if len(ids) > length:
        raise ValueError(f"sequence of {len(ids)} tokens exceeds {length}")
    pad = length - len(ids)
    return np.array(ids + [PAD] * pad, dtype=np.int16), np.array(types + [0] * pad, dtype=np.int8)
