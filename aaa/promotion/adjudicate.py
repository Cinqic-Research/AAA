"""Fail-closed adjudication under ``aaa.promotion.crossed.v1``.

A promotion verdict exists only when the primary evaluator and the
independent recomputation agree on every declared quantity:

* the point value (relative tolerance from the contract);
* the interval status;
* both interval bounds, within the contract's Monte Carlo tolerance expressed
  in interval widths;
* every criterion status;
* the verdict.

Malformed, incomplete, extra, non-finite, duplicated or mismatched primitives
give ``INVALID_EVIDENCE``; any disagreement gives ``DISAGREEMENT``. Neither can
promote, and neither is resolved in favour of either implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import independent, primary
from .contract import Contract, PrimitiveError, require_admissible

Evaluator = Callable[[Contract, Sequence[Mapping[str, Any]]], dict[str, Any]]


def _finite_result(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_result(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_result(item) for item in value)
    return True


def compare(contract: Contract, first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not math.isclose(
        first["geometric_ratio"], second["geometric_ratio"], rel_tol=contract.point_rel_tolerance, abs_tol=0.0
    ):
        problems.append(f"point value {first['geometric_ratio']!r} != {second['geometric_ratio']!r}")
    if first["interval_status"] != second["interval_status"]:
        problems.append(f"interval status {first['interval_status']} != {second['interval_status']}")
    elif first["interval_status"] == "MEASURED":
        width = max(first["upper"] - first["lower"], second["upper"] - second["lower"])
        allowed = contract.bound_tolerance_widths * width
        for bound in ("lower", "upper"):
            gap = abs(first[bound] - second[bound])
            if not gap <= allowed:
                problems.append(f"{bound} bound differs by {gap:.3g} (> {allowed:.3g})")
    for name, status in first["criteria"].items():
        if second["criteria"].get(name) != status:
            problems.append(f"criterion {name}: {status} != {second['criteria'].get(name)}")
    if set(first["criteria"]) != set(second["criteria"]):
        problems.append("the two implementations report different criteria")
    if first["verdict"] != second["verdict"]:
        problems.append(f"verdict {first['verdict']} != {second['verdict']}")
    return problems


def adjudicate(
    contract: Contract,
    records: Sequence[Mapping[str, Any]],
    *,
    primary_evaluator: Evaluator = primary.evaluate,
    independent_evaluator: Evaluator = independent.recompute,
) -> dict[str, Any]:
    """Both implementations, their comparison, and the single fail-closed verdict."""

    require_admissible(contract.contract_id)
    outcome: dict[str, Any] = {"contract": contract.to_dict()}
    try:
        first = primary_evaluator(contract, records)
    except (PrimitiveError, ArithmeticError, ValueError) as error:
        return {**outcome, "verdict": "INVALID_EVIDENCE", "reason": f"primary: {error}"}
    if not _finite_result(first):
        return {**outcome, "verdict": "DISAGREEMENT", "reason": "primary: non-finite result"}
    try:
        second = independent_evaluator(contract, records)
    except (PrimitiveError, ArithmeticError, ValueError) as error:
        return {**outcome, "verdict": "INVALID_EVIDENCE", "reason": f"independent: {error}", "primary": first}
    if not _finite_result(second):
        return {
            **outcome,
            "verdict": "DISAGREEMENT",
            "reason": "independent: non-finite result",
        }
    problems = compare(contract, first, second)
    outcome.update(primary=first, independent=second, agreement_problems=problems)
    outcome["verdict"] = "DISAGREEMENT" if problems else first["verdict"]
    return outcome
