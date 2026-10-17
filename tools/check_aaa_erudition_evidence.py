#!/usr/bin/env python3
"""Recompute every retained ``aaa.erudition.v0`` result from its committed primitives.

Always checked:

* the fitted surrogate parameters re-fit exactly from the committed characterization;
* the committed Erudition weights hash to their recorded digest;
* every committed run re-executes against the environment (``recompute``) and every
  stage's committed metrics recompute;
* confirmation:
  - the freeze is byte-identical to the version first committed;
  - the live phase fingerprint equals the frozen one, since a frozen phase's source
    never changes and a change is a successor identity;
  - every confirmation file was added at its path after the freeze, exactly
    once, and never modified;
  - every declared stream and arm is present;
  - the decisions re-adjudicate from the freeze's own contracts.

With ``--replay`` (needs PyTorch for the Erudition controller), every committed
run is also re-executed from its stage's recorded model exchanges, so the
model's parsed proposals, the controller's decisions and the gate's verdicts are
reproduced, not only the environment. The regenerated run must equal the
committed one exactly, with one exception: the Erudition Model's float32
diagnosis probabilities (values under a ``diagnosis`` key) may differ by a
relative ``DIAGNOSIS_REL_TOLERANCE``, because CPU math kernels round
differently across processors. Every action, state, identifier, score and
verdict must still be identical, and the tolerated differences are counted and
reported.

    python tools/check_aaa_erudition_evidence.py [--replay]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_erudition.characterize import fit_surrogate  # noqa: E402
from research.aaa_erudition.contracts import canonical_json  # noqa: E402
from research.aaa_erudition.evaluate import collect, decide, describe  # noqa: E402
from research.aaa_erudition.identity import fingerprint  # noqa: E402

EVIDENCE = ROOT / "docs/evidence/aaa_erudition_v0"
SURROGATE = ROOT / "research/aaa_erudition/data/lm_surrogate.json"
STAGES = ("development", "attack", "confirmation")
# Observed cross-CPU drift in the diagnosis probabilities is below 4e-6 relative;
# float32 epsilon is 1.2e-7.
DIAGNOSIS_REL_TOLERANCE = 1e-4


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def _commits_touching(path: Path) -> list[str]:
    return _git("log", "--format=%H", "--follow", "--", str(path.relative_to(ROOT))).split()


def _history_at(path: Path) -> list[tuple[str, str]]:
    """Each commit touching exactly this path, with its status; no rename or copy detection.

    Following renames would attribute a file's history to any earlier file it
    resembles (a cache-hash log resembles the progress note that mirrored it).
    A file moved into place shows here as an addition at the move.
    """

    output = _git("log", "--format=%H", "--no-renames", "--name-status", "--", str(path.relative_to(ROOT)))
    history: list[tuple[str, str]] = []
    commit = ""
    for line in output.splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            history.append((commit, line.split("\t", 1)[0]))
        else:
            commit = line.strip()
    return history


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


def check_stage(stage: Path, problems: list[str], *, freeze: Path | None = None) -> list[dict[str, Any]]:
    runs = sorted((stage / "records").glob("*.json.gz"))
    if not runs:
        problems.append(f"{stage.name}: no records")
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
    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        problems.append("the checkout is shallow; confirmation ordering needs full history (fetch-depth: 0)")
        return
    freeze_commits = _commits_touching(freeze_path)
    if len(freeze_commits) != 1:
        problems.append("the freeze is uncommitted or was modified after it was first committed")
        return
    freeze_commit = freeze_commits[0]
    relative = str(freeze_path.relative_to(ROOT))
    if _git("show", f"{freeze_commit}:{relative}").encode() != freeze_path.read_bytes():
        problems.append("the freeze differs from its committed version")
        return
    freeze = json.loads(freeze_path.read_text("utf-8"))
    if fingerprint()["sha256"] != freeze["fingerprint"]["sha256"]:
        problems.append(
            "the phase source differs from the frozen source; a change needs a successor identity"
        )
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        history = _history_at(path)
        if len(history) != 1 or history[0][1] != "A":
            problems.append(f"{path.name} is uncommitted or was modified after it was added")
            continue
        added = history[0][0]
        ancestry = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", freeze_commit, added], check=False
        )
        if ancestry.returncode != 0 or added == freeze_commit:
            problems.append(f"{path.name} was not added after the freeze commit")
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
    if set(committed.get("decisions", {})) != set(decisions):
        problems.append("confirmation: committed decisions do not cover exactly the frozen contracts")
    for name, decision in decisions.items():
        if committed.get("decisions", {}).get(name, {}).get("verdict") != decision["verdict"]:
            problems.append(f"confirmation: {name} recomputes to {decision['verdict']}")


def _differences(committed: Any, regenerated: Any, tolerated: list[float], path: str = "") -> list[str]:
    """Where two run documents differ, as JSON paths with both values (first few only).

    A float under a ``diagnosis`` key that differs by at most a relative
    ``DIAGNOSIS_REL_TOLERANCE`` is not a difference; its relative size is
    appended to ``tolerated``.
    """

    if isinstance(committed, float) and isinstance(regenerated, float) and "/diagnosis/" in path:
        if committed == regenerated:
            return []
        relative = abs(committed - regenerated) / max(abs(committed), abs(regenerated))
        if relative <= DIAGNOSIS_REL_TOLERANCE:
            tolerated.append(relative)
            return []
        return [f"{path}: {committed!r} != {regenerated!r} (relative {relative:.2e})"]
    if type(committed) is not type(regenerated):
        return [f"{path or '/'}: {committed!r:.80} != {regenerated!r:.80}"]
    if isinstance(committed, dict):
        out: list[str] = []
        for key in sorted(set(committed) | set(regenerated)):
            if key not in committed or key not in regenerated:
                out.append(f"{path}/{key}: present in only one")
            else:
                out += _differences(committed[key], regenerated[key], tolerated, f"{path}/{key}")
            if len(out) >= 3:
                break
        return out
    if isinstance(committed, list):
        if len(committed) != len(regenerated):
            return [f"{path}: length {len(committed)} != {len(regenerated)}"]
        out = []
        for index, (a, b) in enumerate(zip(committed, regenerated, strict=True)):
            out += _differences(a, b, tolerated, f"{path}[{index}]")
            if len(out) >= 3:
                break
        return out
    return [] if committed == regenerated else [f"{path}: {committed!r:.80} != {regenerated!r:.80}"]


def replay_stage(stage: Path, problems: list[str]) -> None:
    """Re-execute every committed run from the stage's recorded exchanges; require identical runs."""

    from research.aaa_erudition.experiment import controller_for, run_one
    from research.aaa_erudition.language import CachedBackend, CallCache

    model = EVIDENCE / "model/erudition.pt"
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "calls.jsonl"
        cache_path.write_bytes(gzip.decompress((stage / "calls.jsonl.gz").read_bytes()))
        cache = CallCache(cache_path)
        tolerated: list[float] = []
        for path in sorted((stage / "records").glob("*.json.gz")):
            committed = json.loads(gzip.decompress(path.read_bytes()))
            split, index, condition, controller_name = path.name.removesuffix(".json.gz").split("-", 3)
            backend = CachedBackend(cache, None, backend_id=committed["backend"])
            out = Path(tmp) / path.name.removesuffix(".json.gz")
            try:
                regenerated = run_one(
                    split,
                    int(index),
                    condition,
                    controller_for(controller_name, model),
                    backend,
                    out,
                    admitted=split == "confirmation",
                )
            except Exception as error:  # a replay miss or crash is a finding
                problems.append(f"replay {path.name}: {error}")
                continue
            regenerated.pop("wall_seconds", None)
            if canonical_json(regenerated) == canonical_json(committed):
                continue
            where = _differences(committed, json.loads(canonical_json(regenerated)), tolerated)
            if where:
                problems.append(
                    f"replay {path.name}: the regenerated run differs from the committed one ({'; '.join(where)})"
                )
        count = len(list((stage / "records").glob("*.json.gz")))
        if tolerated:
            print(
                f"{stage.name}: replayed {count} runs; {len(tolerated)} diagnosis probabilities differ "
                f"within tolerance (largest relative {max(tolerated):.2e}), everything else identical"
            )
        else:
            print(f"{stage.name}: replayed {count} runs, byte-identical")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--replay", action="store_true", help="re-execute every run from its recorded exchanges"
    )
    args = parser.parse_args()
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
    if args.replay and not problems:
        for name in STAGES:
            if (EVIDENCE / name).exists():
                replay_stage(EVIDENCE / name, problems)
    for problem in problems:
        print(f"FAILED: {problem}", file=sys.stderr)
    print("aaa.erudition.v0 evidence:", "FAILED" if problems else "recomputes")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
