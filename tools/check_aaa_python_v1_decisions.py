#!/usr/bin/env python3
"""Post-freeze check of stored v1 decisions against retained primitives.

This audit tool sits outside the frozen aaa.python.v1 source fingerprint. It
checks decision fields that the original recompute command does not inspect.
It does not create new confirmation evidence or change the frozen protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aaa.promotion import adjudicate
from research.aaa_python_v1.confirm import contract_from, smoothed_errors
from research.aaa_python_v1.recompute import verify_document
from research.aaa_python_v1.stages import capacity_verdict
from research.aaa_python_v1.summarize import summarize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/evidence/aaa_python_v1/confirmation.json"


def same_adjudication(
    stored: Any, fresh: Any, *, bound_width: float = 0.0, tolerance_widths: float = 0.05
) -> bool:
    """Require exact decisions and contract, allowing cross-platform float drift.

    Recomputed bootstrap bounds may differ within the frozen promotion
    contract's own width tolerance. All verdicts, criteria and status fields
    must still agree exactly. Point values use a much tighter tolerance.
    """

    if isinstance(stored, Mapping) and isinstance(fresh, Mapping):
        if stored.keys() != fresh.keys():
            return False
        if all(key in stored for key in ("lower", "upper")):
            bound_width = max(
                float(stored["upper"]) - float(stored["lower"]),
                float(fresh["upper"]) - float(fresh["lower"]),
                1e-12,
            )
        return all(
            (
                math.isclose(float(stored[key]), float(fresh[key]), abs_tol=tolerance_widths * bound_width)
                if key in ("lower", "upper") and bound_width > 0
                else same_adjudication(
                    stored[key], fresh[key], bound_width=bound_width, tolerance_widths=tolerance_widths
                )
            )
            for key in stored
        )
    if isinstance(stored, Sequence) and not isinstance(stored, (str, bytes)):
        return (
            isinstance(fresh, Sequence)
            and not isinstance(fresh, (str, bytes))
            and len(stored) == len(fresh)
            and all(
                same_adjudication(a, b, bound_width=bound_width, tolerance_widths=tolerance_widths)
                for a, b in zip(stored, fresh, strict=True)
            )
        )
    if isinstance(stored, bool) or isinstance(fresh, bool):
        return stored is fresh
    if isinstance(stored, (int, float)) and isinstance(fresh, (int, float)):
        return (
            math.isfinite(stored)
            and math.isfinite(fresh)
            and math.isclose(stored, fresh, rel_tol=1e-12, abs_tol=1e-12)
        )
    return stored == fresh


def verify(document: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if document.get("stage", {}).get("stage") != "confirmation":
        return ["expected confirmation evidence"]
    stage = document["stage"]
    freeze = document["freeze"]
    if freeze != json.loads((ROOT / "docs/evidence/aaa_python_v1/freeze.json").read_text()):
        problems.append("embedded freeze differs from the committed manifest")
    if stage["primary"] != freeze["design"]["primary"] or stage["secondary"] != []:
        problems.append("stored contrasts differ from the frozen plan")
    frozen_arms = {row["name"]: row for row in freeze["arms"]}
    if set(stage["arms"]) != set(frozen_arms) or any(
        {key: value for key, value in row.items() if key != "parameters"} != frozen_arms[name]
        for name, row in stage["arms"].items()
        if name in frozen_arms
    ):
        problems.append("stored arms differ from the frozen plan")
    if {row["arm"] for row in stage["baselines"]} != set(freeze["baselines"]):
        problems.append("stored baselines differ from the frozen plan")
    if problems:
        return problems
    recounted = verify_document(document)
    if recounted["verdict"] != "PASS":
        problems.extend(recounted["problems"])
    primary = [tuple(c) for c in stage["primary"]]
    secondary = [tuple(c) for c in stage["secondary"]]
    fresh_summary = summarize(stage, primary, secondary, split="confirmation")
    if fresh_summary != stage["summary"]:
        problems.append("stored summary, including Holm fields, differs from primitives")
    encoder = freeze["design"]["encoder"]
    if capacity_verdict(fresh_summary, encoder) != stage["capacity_verdict"]:
        problems.append("stored capacity verdict differs from the declared rule")
    declared = {row["name"]: row for row in freeze["contracts"]}
    if set(stage["adjudications"]) != set(declared):
        problems.append("stored adjudication names differ from frozen contracts")
    for name, row in declared.items():
        contract = contract_from(row)
        records = smoothed_errors(stage["evaluations"], contract.reference)
        records += smoothed_errors(stage["evaluations"], contract.challenger)
        fresh = adjudicate(contract, records)
        stored = stage["adjudications"].get(name, {})
        if stored.get("contract") != fresh["contract"] or not same_adjudication(
            stored, fresh, tolerance_widths=float(fresh["contract"]["bound_tolerance_widths"])
        ):
            problems.append(
                f"{name}: stored adjudication differs from primitives and frozen contract "
                f"(stored verdict={stored.get('verdict')}, fresh verdict={fresh['verdict']}; "
                f"stored primary={stored.get('primary')}, fresh primary={fresh['primary']}; "
                f"stored independent={stored.get('independent')}, fresh independent={fresh['independent']})"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        document = json.loads(
            args.evidence.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
        )
        problems = verify(document)
    except (KeyError, TypeError, ValueError, OSError) as error:
        problems = [f"invalid evidence: {error}"]
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print("v1 decision audit: PASS (summary, Holm, capacity rule, four adjudications)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
