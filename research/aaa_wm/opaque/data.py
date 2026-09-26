"""The shared behavior dataset: one set of training programs and real executions for every arm.

For each ``train`` task, the programs are:

* the reference and the buggy program;
* every single edit of each *state* (the buggy program, and for two-fault tasks the two
  one-fault-undone programs);
* ``pairs`` random two-edit programs from the buggy program;
* ``near`` random single edits of the reference.

Every program is executed by the environment on the **whole** training domain. Training-time
practice with the real library is the declared training interaction, and it is identical for
every arm. The arms then read different labels from the same records:

* world model: the result class per (program, input);
* value model: whether the program is domain-equivalent to the reference, and whether it passes
  the task's visible tests;
* policy: for each (state, edit), whether the edit undoes a remaining fault or produces a
  domain-equivalent program.

Records are cached under ``$AAA_DATA_ROOT/opaque`` with a key over the generator, the
program module, the tokenizer, the specification and these parameters.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.aaa_python.rng import Stream, derive_seed

from . import generator as gen
from . import tokens as tok
from .program import Edit, apply, edits, signature

CODE_LEN = 144  # [CLS] + code (<= 142) + [SEP]; batches are trimmed to their longest row
_SOURCES = ("generator.py", "program.py", "tokens.py", "data.py", "library.py")


@dataclass
class Dataset:
    code: np.ndarray  # [P, CODE_LEN] int16 ids ([CLS] code [SEP] PAD...)
    positions: list[tuple[tuple[int, int], ...]]  # per program: (row, col) of each code token
    results: np.ndarray  # [P, |domain|] int16 result classes
    task_of: np.ndarray  # [P] int32
    equivalent: np.ndarray  # [P] bool
    visible_pass: np.ndarray  # [P, V] bool
    tests: np.ndarray  # [T, V, 2] int32 (input, expected value) per task
    policy: np.ndarray  # [E, 4] int32: program index, marked code index (+1 for CLS), label, state kind
    sources: list[str]


def cache_key(split: str, count: int, pairs: int, near: int, library: str) -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in _SOURCES:
        h.update((here / name).read_bytes())
    h.update(gen.spec_hash().encode())
    h.update(tok.vocab_hash().encode())
    h.update(f"{split}:{count}:{pairs}:{near}:{library}".encode())
    return h.hexdigest()[:24]


def _code_row(source: str) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    ids, pos = tok.encode_program(source)
    row = np.full(CODE_LEN, tok.PAD, dtype=np.int16)
    body = [tok.CLS, *ids, tok.SEP]
    if len(body) > CODE_LEN:
        raise ValueError("program too long")
    row[: len(body)] = body
    return row, pos


def build(split: str = "train", count: int = 6000, *, pairs: int = 32, near: int = 16, library: str = "A") -> Dataset:
    root = os.environ.get("AAA_DATA_ROOT")
    path = Path(root) / "opaque" / f"dataset_{cache_key(split, count, pairs, near, library)}.npz" if root else None
    if path is not None and path.exists():
        return _load(path)
    lib = gen.library(library)
    names = tuple(a.name for a in lib)
    dom = gen.domain()
    tasks = gen.pool(split, count)
    sources: list[str] = []
    index: dict[str, int] = {}
    task_of: list[int] = []
    equivalent: list[bool] = []
    visible: list[list[bool]] = []
    results: list[list[int]] = []
    policy: list[tuple[int, int, int, int]] = []
    tests = np.zeros((len(tasks), 3, 2), dtype=np.int32)

    for ti, t in enumerate(tasks):
        ref_sig = signature(t.reference, lib, dom)
        tests[ti] = [[x, e[1]] for x, e in t.visible_tests]
        local: dict[str, int] = {}

        def add(src: str) -> int:
            key = f"{ti}|{src}"
            if key in index:
                return index[key]
            sig = signature(src, lib, dom)
            index[key] = len(sources)
            sources.append(src)
            task_of.append(ti)
            equivalent.append(sig == ref_sig)
            by_x = dict(zip(dom, sig, strict=True))
            visible.append([by_x[x] == e for x, e in t.visible_tests])
            results.append([tok.result_class(r) for r in sig])
            local[src] = index[key]
            return index[key]

        add(t.reference)
        add(t.buggy)
        # states: the buggy program and, for k = 2, each one-fault-undone program
        states = [(t.buggy, t.faults)]
        if t.k == 2:
            for f in t.faults:
                back = Edit(f.row, f.col, f.new, f.old)
                try:
                    states.append((apply(t.buggy, back), tuple(g for g in t.faults if g != f)))
                except ValueError:
                    pass
        for kind, (state, remaining) in enumerate(states):
            fixes = {(f.row, f.col, f.new, f.old) for f in remaining}
            for e in edits(state, names):
                src = apply(state, e)
                pi = add(src)
                good = (e.row, e.col, e.old, e.new) in fixes or equivalent[pi]
                ids, pos = tok.encode_program(src)
                mark = next((k for k, w in enumerate(pos) if w == (e.row, e.col)), -1)
                policy.append((pi, mark + 1, int(good), min(kind, 1)))
        s = Stream(derive_seed("aaa.wm.opaque.data", split, ti))
        first = edits(t.buggy, names)
        for _ in range(pairs):
            e1 = s.choice(first)
            mid = apply(t.buggy, e1)
            second = [e for e in edits(mid, names) if (e.row, e.col) != (e1.row, e1.col)]
            if second:
                add(apply(mid, s.choice(second)))
        near_edits = edits(t.reference, names)
        for _ in range(min(near, len(near_edits))):
            add(apply(t.reference, s.choice(near_edits)))

    code = np.zeros((len(sources), CODE_LEN), dtype=np.int16)
    positions = []
    for i, src in enumerate(sources):
        code[i], pos = _code_row(src)
        positions.append(pos)
    ds = Dataset(
        code,
        positions,
        np.array(results, dtype=np.int16),
        np.array(task_of, dtype=np.int32),
        np.array(equivalent, dtype=bool),
        np.array(visible, dtype=bool),
        tests,
        np.array(policy, dtype=np.int32),
        sources,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".tmp.npz"),
            code=ds.code,
            results=ds.results,
            task_of=ds.task_of,
            equivalent=ds.equivalent,
            visible_pass=ds.visible_pass,
            tests=ds.tests,
            policy=ds.policy,
            sources=np.array(ds.sources, dtype=object),
        )
        path.with_suffix(".tmp.npz").replace(path)
    return ds


def _load(path: Path) -> Dataset:
    z = np.load(path, allow_pickle=True)
    sources = list(z["sources"])
    return Dataset(
        z["code"],
        [],  # token positions are recomputed on demand (tok.encode_program); training uses precomputed marks
        z["results"],
        z["task_of"],
        z["equivalent"],
        z["visible_pass"],
        z["tests"],
        z["policy"],
        sources,
    )


def summary(ds: Dataset) -> dict[str, Any]:
    return {
        "programs": int(len(ds.sources)),
        "tasks": int(ds.tests.shape[0]),
        "executions": int(ds.results.size),
        "equivalent_fraction": float(ds.equivalent.mean()),
        "visible_all_pass_fraction": float(ds.visible_pass.all(axis=1).mean()),
        "policy_records": int(len(ds.policy)),
        "policy_positive_fraction": float(ds.policy[:, 2].mean()),
    }
