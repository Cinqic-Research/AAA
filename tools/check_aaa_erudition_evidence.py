#!/usr/bin/env python3
"""Recompute every retained ``aaa.erudition.v0`` result from its committed primitives.

* the fitted surrogate parameters re-fit exactly from the committed characterization;
* the committed Erudition weights hash to their recorded digest and parameter count;
* every committed run re-executes against the environment (``recompute``);
* every committed evaluation recomputes to the same metrics, and confirmation
  decisions re-adjudicate from the freeze's own contracts over every declared run;
* confirmation evidence exists only with a freeze committed *before* it, and
  only for the streams and arms that freeze declared.

    python tools/check_aaa_erudition_evidence.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_erudition.characterize import fit_surrogate  # noqa: E402
from research.aaa_erudition.contracts import canonical_json  # noqa: E402
from research.aaa_erudition.evaluate import collect, decide, describe  # noqa: E402

EVIDENCE = ROOT / "docs/evidence/aaa_erudition_v0"
SURROGATE = ROOT / "research/aaa_erudition/data/lm_surrogate.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def _first_commit(path: Path) -> str | None:
    out = _git("log", "--diff-filter=A", "--format=%H", "--", str(path.relative_to(ROOT))).split()
    return out[-1] if out else None


def check_surrogate(problems: list[str]) -> None:
    files = sorted((EVIDENCE / "characterization").glob("characterization*.json"))
    committed = json.loads(SURROGATE.read_text("utf-8"))
    if not files:
        problems.append("no committed characterization behind the surrogate")
        return
    recorded = {entry["file"]: entry["sha256"] for entry in committed["source"]}
    for path in files:
        if recorded.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            problems.append(f"surrogate source {path.name} does not match its recorded digest")
    refit = fit_surrogate([json.loads(p.read_text("utf-8")) for p in files], committed["rates"])
    if refit["rates"] != committed["rates"]:
        problems.append("surrogate rates do not re-fit from the committed characterization")


def check_model(problems: list[str]) -> None:
    for meta_path in sorted((EVIDENCE / "model").glob("*.json")):
        meta = json.loads(meta_path.read_text("utf-8"))
        weights = meta_path.with_suffix(".pt")
        if hashlib.sha256(weights.read_bytes()).hexdigest() != meta["weights_sha256"]:
            problems.append(f"{weights.name} does not match its recorded digest")
        if meta["trainable_parameters"] < 1_000_000:
            problems.append(f"{weights.name} is below the 1M trainable-parameter floor")


def check_stage(stage: Path, problems: list[str], *, freeze: Path | None = None) -> list[dict]:
    runs = sorted((stage / "records").glob("*.json.gz"))
    if not runs:
        problems.append(f"{stage.name}: no runs")
        return []
    try:
        rows = collect(runs, freeze=freeze)
    except Exception as error:  # every failure of recomputation is a finding, whatever its type
        problems.append(f"{stage.name}: {error}")
        return []
    evaluation_path = stage / "evaluation.json"
    if not evaluation_path.exists():
        problems.append(f"{stage.name}: no committed evaluation")
        return rows
    committed = json.loads(evaluation_path.read_text("utf-8"))
    if canonical_json(describe(rows)) != canonical_json(committed["descriptive"]):
        problems.append(f"{stage.name}: descriptive metrics do not recompute")
    return rows


def check_confirmation(problems: list[str]) -> None:
    stage = EVIDENCE / "confirmation"
    freeze_path = EVIDENCE / "freeze.json"
    if not stage.exists():
        print("confirmation: not executed")
        return
    if not freeze_path.exists():
        problems.append("confirmation evidence exists without a freeze")
        return
    freeze_commit = _first_commit(freeze_path)
    if freeze_commit is None:
        problems.append("the freeze is not committed")
        return
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        commit = _first_commit(path)
        if commit is None:
            problems.append(f"{path.name} is not committed")
            continue
        ancestry = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", freeze_commit, commit], check=False
        )
        if ancestry.returncode != 0 or commit == freeze_commit:
            problems.append(f"{path.name} was not added after the freeze commit")
    freeze = json.loads(freeze_path.read_text("utf-8"))
    plan = freeze["confirmation"]
    for run in (stage / "records").glob("*.json.gz"):
        _, index, condition, controller = run.name.removesuffix(".json.gz").split("-", 3)
        if int(index) not in plan["indices"] or f"{condition}/{controller}" not in plan["arms"]:
            problems.append(f"{run.name} is not declared in the freeze")
    rows = check_stage(stage, problems, freeze=freeze_path)
    if not rows:
        return
    try:
        decisions = decide(freeze, rows)
    except Exception as error:  # a missing declared run is missing evidence
        problems.append(f"confirmation: {error}")
        return
    committed = json.loads((stage / "evaluation.json").read_text("utf-8"))
    for name, decision in decisions.items():
        if committed.get("decisions", {}).get(name, {}).get("verdict") != decision["verdict"]:
            problems.append(f"confirmation: {name} recomputes to {decision['verdict']}")


def main() -> int:
    problems: list[str] = []
    if not EVIDENCE.exists():
        print("no aaa.erudition.v0 evidence is committed")
        return 1
    check_surrogate(problems)
    check_model(problems)
    for name in ("development", "attack"):
        if (EVIDENCE / name).exists():
            check_stage(EVIDENCE / name, problems)
    check_confirmation(problems)
    for problem in problems:
        print(f"FAILED: {problem}", file=sys.stderr)
    print("aaa.erudition.v0 evidence:", "FAILED" if problems else "recomputes")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
