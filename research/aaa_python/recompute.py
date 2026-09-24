"""Independent recomputation of an ``aaa.python.v0`` development run.

Three checks, each failing closed:

1. **Records.** The per-action records match the digest and count the evidence
   declares.
2. **Evaluator.** Every scored task is regenerated from its identity, its
   program (and, for repair, every candidate against the reference) is
   re-executed by CPython, and the answer is derived by this module's own
   mapping. It does not use the generator's ``answer_from``. Each record's
   ``truth`` must equal it, and each record's ``correct`` must equal
   ``answer == truth and not abstain``.
3. **Primitives.** Cell counts are re-aggregated from the records by a separate
   loop and compared with the stored cells; the summary is recomputed from the
   stored cells and compared with the stored summary.

A stored value is never trusted because it is stored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import generator
from . import spec as spec_module
from .experiment import Plan, records_sha256, summarize
from .oracle import check_syntax, execute


class RecomputeError(RuntimeError):
    pass


def _refuse_constant(token: str) -> Any:
    raise RecomputeError(f"non-standard JSON constant {token}")


def load_evidence(text: str) -> dict[str, Any]:
    payload = json.loads(text, parse_constant=_refuse_constant)
    if not isinstance(payload, dict) or payload.get("schema") != "aaa.python.v0.development.v1":
        raise RecomputeError("not an aaa.python.v0 development evidence document")
    return payload


FLOAT_REL_TOLERANCE = 1e-12


def summary_differences(stored: Any, recomputed: Any, path: str = "summary") -> list[str]:
    """Every difference between two summaries, with floats compared to a relative 1e-12.

    Counts, statuses, resolved signs and structure must match exactly. Floats
    may differ in their final bits across interpreters: CPython 3.12 changed
    the built-in float ``sum()`` to compensated summation, which moves derived
    values such as the Brier score by about 1e-16 relative (``AAA-186``).
    Non-finite values never match anything.
    """

    if isinstance(stored, dict) and isinstance(recomputed, dict):
        if set(stored) != set(recomputed):
            return [f"{path}: keys differ"]
        return [
            d
            for key in sorted(stored)
            for d in summary_differences(stored[key], recomputed[key], f"{path}.{key}")
        ]
    if isinstance(stored, list) and isinstance(recomputed, list):
        if len(stored) != len(recomputed):
            return [f"{path}: lengths differ"]
        return [
            d
            for i, (a, b) in enumerate(zip(stored, recomputed, strict=True))
            for d in summary_differences(a, b, f"{path}[{i}]")
        ]
    if isinstance(stored, float) or isinstance(recomputed, float):
        numeric = all(isinstance(v, int | float) and not isinstance(v, bool) for v in (stored, recomputed))
        if not numeric or not (math.isfinite(stored) and math.isfinite(recomputed)):
            return [f"{path}: {stored!r} != {recomputed!r}"]
        if not math.isclose(stored, recomputed, rel_tol=FLOAT_REL_TOLERANCE, abs_tol=1e-15):
            return [f"{path}: {stored!r} != {recomputed!r}"]
        return []
    if type(stored) is not type(recomputed) or stored != recomputed:
        return [f"{path}: {stored!r} != {recomputed!r}"]
    return []


def _oracle_truth(task: generator.Task) -> Any:
    """The answer, re-derived from fresh CPython executions (independent of the generator's mapping)."""

    if task.family == "syntax":
        status = check_syntax(task.source).status
        if status not in ("valid", "syntax_error"):
            raise RecomputeError(f"{task.task_id}: sandbox returned {status}")
        return "valid" if status == "valid" else "invalid"
    if task.family == "repair":
        assert task.repair_line is not None
        lines = task.source.rstrip("\n").split("\n")
        inputs = [i for i, _ in (*task.visible_tests, *task.hidden_tests)]
        passing = []
        for index, candidate in enumerate(task.candidates):
            patched = [*lines[: task.repair_line - 1], candidate, *lines[task.repair_line :]]
            program = "\n".join(patched + [f"print(f({value}))" for value in inputs]) + "\n"
            outcome = execute(program)
            observed = outcome.stdout.split() if outcome.status == "ok" else []
            expected = [str(o) for _, o in task.hidden_tests]
            if observed[len(task.visible_tests) :] == expected:
                passing.append(index)
        if len(passing) != 1:
            raise RecomputeError(f"{task.task_id}: {len(passing)} candidates pass the hidden tests")
        return passing[0]
    outcome = execute(task.source)
    if task.family == "outcome":
        if outcome.status == "ok":
            return "ok"
        if outcome.status != "exception":
            raise RecomputeError(f"{task.task_id}: sandbox returned {outcome.status}")
        return outcome.exception
    if task.family == "output":
        if outcome.status != "ok":
            raise RecomputeError(f"{task.task_id}: output program did not run cleanly")
        return int(outcome.stdout.strip())
    if task.family == "localize":
        if outcome.status != "exception" or outcome.line is None:
            raise RecomputeError(f"{task.task_id}: localization program did not fail")
        return outcome.line
    raise RecomputeError(f"unknown family {task.family}")


def _tasks_by_id(task_ids: set[str]) -> dict[str, generator.Task]:
    wanted: dict[tuple[str, str], list[int]] = {}
    for identity in task_ids:
        protocol, split, family, index = identity.split(":")
        if protocol != "aaa.python.v0" or split not in ("development", "probe"):
            raise RecomputeError(f"record names a task outside development/probe: {identity}")
        wanted.setdefault((split, family), []).append(int(index))
    tasks: dict[str, generator.Task] = {}
    for (split, family), indices in wanted.items():
        for task in generator.build(split, family, sorted(indices)):
            tasks[task.task_id] = task
    return tasks


def verify(evidence: Mapping[str, Any], records: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    spec = spec_module.load()
    problems: list[str] = []
    if evidence.get("spec_sha256") != spec_module.canonical_hash(spec):
        problems.append("the evidence was produced under a different specification")
    plan = Plan(
        **{
            **evidence["plan"],
            "representations": tuple(evidence["plan"]["representations"]),
            "adaptation_families": tuple(evidence["plan"]["adaptation_families"]),
        }
    )
    recomputed_summary = json.loads(json.dumps(summarize(evidence["cells"], plan, spec)))
    differences = summary_differences(evidence["summary"], recomputed_summary)
    if differences:
        problems.append(f"the summary does not recompute from the stored cells: {differences[:5]}")
    result: dict[str, Any] = {"summary_recomputed": not problems}
    if records is None:
        result.update(
            records_checked=False,
            problems=problems,
            verdict="FAIL" if problems else "SUMMARY_ONLY_NOT_VERIFIED",
        )
        return result
    if (
        len(records) != evidence["records"]["count"]
        or records_sha256(records) != evidence["records"]["sha256"]
    ):
        problems.append("the records do not match the declared count and digest")
    bins = spec["statistics"]["calibration_bins"]
    totals: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for record in records:
        key = (record["arm"], record["family"], record["init"], record["stream"])
        row = totals.setdefault(
            key,
            {
                "n": 0,
                "correct": 0,
                "abstained": 0,
                "updates": 0,
                "brier_sum": 0.0,
                "bins": {"n": [0] * bins, "confidence_sum": [0.0] * bins, "correct": [0] * bins},
                "per_class": {},
            },
        )
        row["n"] += 1
        row["correct"] += int(record["correct"])
        row["abstained"] += int(record["abstain"])
        row["updates"] += int(record["updated"])
        expected_correct = (not record["abstain"]) and record["answer"] == record["truth"]
        if record["correct"] != expected_correct:
            problems.append(f"{record['task_id']} ({record['arm']}): stored score disagrees with its answer")
        confidence = record["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not 0.0 <= confidence <= 1.0
        ):
            problems.append(f"{record['task_id']} ({record['arm']}): malformed confidence")
        elif not math.isfinite(confidence):
            problems.append(f"{record['task_id']}: non-finite confidence")
        else:
            hit = int(record["correct"])
            row["brier_sum"] += (confidence - float(hit)) ** 2
            index = min(int(confidence * bins), bins - 1)
            row["bins"]["n"][index] += 1
            row["bins"]["confidence_sum"][index] += confidence
            row["bins"]["correct"][index] += hit
            label = json.dumps(record["truth"])
            counts = row["per_class"].setdefault(label, [0, 0])
            counts[0] += 1
            counts[1] += hit
    stored = {
        (c["arm"], c["family"], c["init"], c["stream"]): {
            field: c[field]
            for field in ("n", "correct", "abstained", "updates", "brier_sum", "bins", "per_class")
        }
        for c in evidence["cells"]
    }
    if len(stored) != len(evidence["cells"]):
        problems.append("duplicate stored cell identities")
    cell_differences = summary_differences(stored, totals, "cells")
    if cell_differences:
        problems.append(f"stored cell primitives differ from records: {cell_differences[:5]}")
    tasks = _tasks_by_id({r["task_id"] for r in records})
    truths = {identity: _oracle_truth(task) for identity, task in sorted(tasks.items())}
    mismatched = sorted({r["task_id"] for r in records if truths[r["task_id"]] != r["truth"]})
    if mismatched:
        problems.append(
            f"{len(mismatched)} tasks' recorded truth differs from fresh CPython execution: {mismatched[:5]}"
        )
    result.update(
        records_checked=True,
        tasks_reexecuted=len(truths),
        problems=problems,
        verdict="FAIL" if problems else "PASS",
    )
    return result
