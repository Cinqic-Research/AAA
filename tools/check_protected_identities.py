#!/usr/bin/env python3
"""Prove that AAA's protected historical identities have not moved.

The dot-era research is retained as evidence and as benchmark lineage. Its
scientific identities must survive every later change, including the ones
that move the repository's centre of gravity elsewhere. This tool recomputes
each identity from the live tree and compares it with the committed
expectation in ``benchmarks/protected_identities.json``:

* the benchmark v2.1 specification hash and the observation-noise v1.1
  protocol hash;
* the ``aaa.1k.v1`` and ``aaa.1k.v2`` phase fingerprints (Champion 0,
  Champion 1 and every AAA-1K/v2 artifact carry them);
* the SHA-256 of every tracked retained-evidence byte (``docs/evidence/``,
  ``results/``, the frozen registries and ledgers under ``benchmarks/``) and of
  the historical reports, reviews, decision logs and handoffs.

The repository-wide observation-noise *source* fingerprint is deliberately not
listed: it covers every tracked file and moves with any new file by design
(``AAA-152``).

    python tools/check_protected_identities.py            # exit 1 on any drift
    python tools/check_protected_identities.py --record   # write the expectation

``--record`` exists for the first capture only. Changing a recorded value is a
change to historical identity and needs a recorded decision in the issue
ledger, never an edit made to turn a check green.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED = "benchmarks/protected_identities.json"
SCHEMA = "aaa.protected_identities.v1"

EVIDENCE_PREFIXES = ("docs/evidence/", "results/")
FROZEN_BENCHMARK_FILES = (
    "benchmarks/aaa1k_loop_identity_ledger.json",
    "benchmarks/aaa1k_v2_identity_registry.json",
    "benchmarks/benchmark_v2.json",
    "benchmarks/confirmation_batches.json",
    "benchmarks/freeze_manifest.json",
    "benchmarks/golden_seeds.json",
    "benchmarks/hardware/flowbox.json",
    "benchmarks/observation_noise_candidate_ledger.json",
    "benchmarks/observation_noise_development_selection.json",
    "benchmarks/observation_noise_freeze.json",
    "benchmarks/observation_noise_registry.json",
    "benchmarks/observation_noise_source_freeze.json",
)
HISTORICAL_DOCUMENTS = (
    "docs/aaa_1k_handoff.md",
    "docs/aaa_1k_report.md",
    "docs/aaa_1k_report_round1_superseded.md",
    "docs/aaa_1k_report_round2_superseded.md",
    "docs/aaa_1k_self_review.md",
    "docs/aaa_1k_sol_review.md",
    "docs/aaa_1k_v2_compute_report.md",
    "docs/aaa_1k_v2_compute_strategy.md",
    "docs/aaa_1k_v2_decisions.md",
    "docs/aaa_1k_v2_handoff.md",
    "docs/aaa_1k_v2_literature_review.md",
    "docs/aaa_1k_v2_report.md",
    "docs/aaa_1k_v2_self_review.md",
    "docs/benchmark_protocol.md",
    "docs/benchmark_v2_superseded.md",
    "docs/candidate_selection.md",
    "docs/diagnosis.md",
    "docs/experiment_registry.md",
    "docs/handoff_sol.md",
    "docs/independent_review_2026-09-22.md",
    "docs/loop_0004_0006_handoff.md",
    "docs/loop_pilot_handoff.md",
    "docs/loop_pilot_report.md",
    "docs/loop_report_0004_0006.md",
    "docs/observation_noise_protocol.md",
    "docs/phase_closure_decisions.md",
    "docs/pr20_independent_review.md",
    "docs/pre_scale_review_2026-09-23.md",
    "docs/self_review.md",
    "docs/sol_review.md",
)


def _tracked() -> list[str]:
    output = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z", "--cached"])
    return sorted(name for name in output.decode("utf-8").split("\0") if name)


def _sha256(relative: str) -> str:
    path = ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"protected path is missing, not a regular file, or a symlink: {relative}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files(tracked: list[str]) -> list[str]:
    selected = {name for name in tracked if name.startswith(EVIDENCE_PREFIXES)}
    selected.update(FROZEN_BENCHMARK_FILES)
    selected.update(HISTORICAL_DOCUMENTS)
    return sorted(selected)


def live_identities() -> dict[str, Any]:
    from aaa.benchmark.spec import load_spec, spec_hash
    from aaa.noise.spec import canonical_protocol_hash
    from research.aaa_1k.identity import phase_fingerprint
    from research.aaa_1k_v2.identity import fingerprint as v2_fingerprint

    tracked = _tracked()
    return {
        "schema": SCHEMA,
        "identities": {
            "benchmark_v2_1_specification": spec_hash(load_spec()),
            "observation_noise_v1_1_protocol": canonical_protocol_hash(),
            "aaa_1k_v1_phase_fingerprint": phase_fingerprint(ROOT)["sha256"],
            "aaa_1k_v2_phase_fingerprint": v2_fingerprint(ROOT)["sha256"],
        },
        "files": {name: _sha256(name) for name in protected_files(tracked)},
    }


def compare(expected: dict[str, Any], live: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if expected.get("schema") != SCHEMA:
        return [f"unknown expectation schema {expected.get('schema')!r}"]
    for key, value in expected["identities"].items():
        if live["identities"].get(key) != value:
            problems.append(f"identity {key}: expected {value}, live {live['identities'].get(key)}")
    for name, digest in expected["files"].items():
        observed = live["files"].get(name)
        if observed is None:
            problems.append(f"protected file no longer tracked: {name}")
        elif observed != digest:
            problems.append(f"protected file changed: {name}")
    # A newly tracked evidence file is allowed (new phases add evidence); an
    # expected one that disappears or changes is not.
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true", help="write the expectation (first capture only)")
    parser.add_argument("--force", action="store_true", help="with --record, overwrite an existing file")
    args = parser.parse_args(argv)
    live = live_identities()
    target = ROOT / EXPECTED
    if args.record:
        if target.exists() and not args.force:
            print(f"refusing: {EXPECTED} exists; a change to it needs a recorded decision", file=sys.stderr)
            return 2
        target.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded {len(live['identities'])} identities and {len(live['files'])} files")
        return 0
    expected = json.loads(target.read_text(encoding="utf-8"))
    problems = compare(expected, live)
    for key, value in live["identities"].items():
        print(f"{key}: {value}")
    print(f"protected files checked: {len(expected['files'])}")
    if problems:
        for problem in problems:
            print(f"DRIFT: {problem}", file=sys.stderr)
        return 1
    print("protected identities: UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
