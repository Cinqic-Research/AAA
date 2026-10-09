#!/usr/bin/env python3
"""Regenerate a stage's runs by replay at the current commit and write them as retained evidence.

Run from a clean checkout. Every run is re-executed against the recorded
Language Model exchanges (``--backend replay``: a missing exchange is an
error, never a new model call), so the retained runs carry exactly this
commit's code identity. Writes ``records/``, the compressed exchange record,
``evaluation.json`` and ``manifest.json`` under
``docs/evidence/aaa_erudition_v0/<stage>/``.

    python tools/package_aaa_erudition_stage.py development --cache <calls.jsonl> \\
        --plan "0,1,2,3,4,5:frozen,joint,lm_only,wm_only:heuristic" ...
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/aaa_erudition_v0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("development", "attack", "confirmation"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--plan", action="append", help="indices:conditions:controller (development and attack)"
    )
    parser.add_argument(
        "--runs", type=Path, help="confirmation: the directory the confirmation runs were written to"
    )
    parser.add_argument(
        "--attempts", type=Path, help="confirmation: the attempt and cache-hash log kept while it ran"
    )
    parser.add_argument("--model", type=Path, default=EVIDENCE / "model/erudition.pt")
    args = parser.parse_args()
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        print("refusing: the tree has uncommitted changes", file=sys.stderr)
        return 2
    target = EVIDENCE / args.stage
    if target.exists():
        print(f"refusing: {target} exists", file=sys.stderr)
        return 2
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if args.stage == "confirmation":
        return package_confirmation(args, target, head)
    if not args.plan:
        print("--plan is required for development and attack", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "runs"
        for plan in args.plan:
            indices, conditions, controller = plan.split(":")
            command = [
                sys.executable,
                "-m",
                "research.aaa_erudition.experiment",
                "--split",
                args.stage,
                "--indices",
                indices,
                "--conditions",
                conditions,
                "--controller",
                controller,
                "--backend",
                "replay",
                "--cache",
                str(args.cache),
                "--out",
                str(out),
            ]
            if controller == "erudition":
                command += ["--model", str(args.model)]
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        runs = target / "records"
        runs.mkdir(parents=True)
        for path in sorted(out.glob("*.json.gz")):
            shutil.copy2(path, runs / path.name)
        calls = args.cache.read_bytes()
        (target / "calls.jsonl.gz").write_bytes(gzip.compress(calls, compresslevel=9, mtime=0))
        evaluation = target / "evaluation.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.aaa_erudition.evaluate",
                *map(str, sorted(runs.glob("*.json.gz"))),
                "--output",
                str(evaluation),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    manifest = {
        "schema": "aaa.erudition.stage_manifest.v1",
        "stage": args.stage,
        "commit": head,
        "regenerated_by": "replay of the recorded exchanges; no model call",
        "plans": args.plan,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "calls_sha256": hashlib.sha256(calls).hexdigest(),
        "calls": sum(1 for line in calls.splitlines() if line.strip()),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"packaged {args.stage} at {head[:12]}: {len(list(runs.glob('*.json.gz')))} runs")
    return 0


def package_confirmation(args: argparse.Namespace, target: Path, head: str) -> int:
    """Copy confirmation runs exactly as the frozen code produced them; never regenerate them."""

    if args.runs is None or args.attempts is None:
        print("confirmation needs --runs and --attempts", file=sys.stderr)
        return 2
    freeze = EVIDENCE / "freeze.json"
    records = target / "records"
    records.mkdir(parents=True)
    for path in sorted(args.runs.glob("*.json.gz")):
        shutil.copy2(path, records / path.name)
    calls = args.cache.read_bytes()
    (target / "calls.jsonl.gz").write_bytes(gzip.compress(calls, compresslevel=9, mtime=0))
    shutil.copy2(args.attempts, target / "attempt_log.txt")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.aaa_erudition.evaluate",
            *map(str, sorted(records.glob("*.json.gz"))),
            "--freeze",
            str(freeze),
            "--output",
            str(target / "evaluation.json"),
        ],
        cwd=ROOT,
        check=True,
    )
    manifest = {
        "schema": "aaa.erudition.stage_manifest.v1",
        "stage": "confirmation",
        "packaged_at_commit": head,
        "produced_by": "the frozen code at the freeze commit, live against the real model; copied unchanged",
        "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "calls_sha256": hashlib.sha256(calls).hexdigest(),
        "calls": sum(1 for line in calls.splitlines() if line.strip()),
        "server_manifests": sorted(p.name for p in args.runs.glob("manifest*.json")),
    }
    for path in sorted(args.runs.glob("manifest*.json")):
        shutil.copy2(path, target / f"server_{path.name}")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"packaged confirmation: {len(list(records.glob('*.json.gz')))} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
