#!/usr/bin/env python3
"""Post-freeze check of stored v1 decisions against retained primitives.

This audit tool sits outside the frozen aaa.python.v1 source fingerprint. It
checks decision fields that the original recompute command does not inspect.
It does not create new confirmation evidence or change the frozen protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aaa.promotion import adjudicate
from research.aaa_python_v1.confirm import contract_from, smoothed_errors
from research.aaa_python_v1.recompute import verify_document
from research.aaa_python_v1.stages import capacity_verdict
from research.aaa_python_v1.summarize import summarize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/evidence/aaa_python_v1/confirmation.json"


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
        if stage["adjudications"].get(name) != fresh:
            problems.append(f"{name}: stored adjudication differs from primitives and frozen contract")
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
