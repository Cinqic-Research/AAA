#!/usr/bin/env python3
"""Generate the current AAA phase-closure handoff from retained evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "handoff_sol.md"


def load(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{relative} must contain a JSON object")
    return value


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main() -> None:
    protocol = load("aaa/noise/data/observation_noise_v1.json")
    ledger = load("benchmarks/observation_noise_candidate_ledger.json")
    registry = load("benchmarks/observation_noise_registry.json")
    source_freeze = load("benchmarks/observation_noise_source_freeze.json")
    storage = load("docs/evidence/phase_closure_storage_profile.json")
    validation_path = ROOT / "docs/evidence/phase_closure_validation.json"
    validation = load("docs/evidence/phase_closure_validation.json") if validation_path.is_file() else {}
    selected = str(ledger["selected_candidate"])
    selected_entry = next(row for row in ledger["entries"] if row["candidate_id"] == selected)
    batches = [
        row
        for row in registry["batches"]
        if row.get("batch_id") in {"observation-noise-a-0002", "observation-noise-b-0002"}
    ]
    lines = [
        "# AAA engineering phase-closure handoff",
        "",
        "This is the active GPT-5.6 Sol self-audit handoff for the integrated",
        "observation-noise engineering closure. Sol's initial review of inherited",
        "work was independent; post-remediation verification is a self-audit backed",
        "by adversarial tests, independent recomputation, clean-checkout execution,",
        "and exact hosted CI. It is not human review or product-release approval.",
        "",
        "## Final identities",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| generated from local head | `{git('rev-parse', 'HEAD')}` |",
        f"| protocol | `{protocol['protocol_version']}` |",
        f"| protocol hash | `{source_freeze['protocol_hash']}` |",
        f"| scientific fingerprint | `{source_freeze['scientific_fingerprint_sha256']}` |",
        f"| dependency lock | `{source_freeze['dependency_lock']['sha256']}` |",
        f"| v2.1 reference commit | `{source_freeze['reference']['commit']}` |",
        f"| selected candidate | `{selected}` |",
        f"| candidate configuration | `{selected_entry['configuration_hash']}` |",
        "| formal observation-noise outcome | `NOT EXECUTED` |",
        "| external durable archival | `NOT VERIFIED`; not required for this internal phase |",
        "",
        "## Decisions",
        "",
        "The complete option analysis is retained in `phase_closure_decisions.md`.",
        "FontTools 4.65.0 was integrated before the final freeze. PR #11's",
        "infrastructure is retained with portable schedule storage IDs and a 100 GB",
        "formal-output preflight. PR #12 and PR #13 remain exploratory history and",
        "are not part of the maintained scientific tree. Broad fingerprint coverage",
        "and deterministic sharding remain in place.",
        "",
        "Formal A/B was deliberately deferred. The v1.1 batches are:",
        "",
        "| Batch | Role | Status |",
        "|---|---|---|",
        *[f"| `{row['batch_id']}` | `{row['role']}` | `{row['status']}` |" for row in batches],
        "",
        "They were never observed and may never be reactivated. A successor must",
        "receive a new protocol version, precision justification, freeze, and fresh",
        "batch identities before observation.",
        "",
        "## Storage evidence",
        "",
        f"The dedicated ext4 HDD had {storage['mount']['available_bytes_before_profile']:,}",
        "available bytes before profiling. POSIX permissions, symlinks, ordinary",
        "writes, and same-filesystem atomic rename passed. SMART health is",
        "`NOT_VERIFIED` because `smartctl` is unavailable.",
        "",
        f"The retained quick run wrote {storage['attempt']['total_files']:,} files and",
        f"{storage['attempt']['total_bytes']:,} bytes in {storage['attempt']['wall_seconds']:.2f}",
        "seconds, then independently recomputed `PASS`. The linear formal projection",
        f"is {storage['formal_projection']['projected_wall_hours_per_batch']:.1f} wall",
        f"hours and {storage['formal_projection']['projected_attempt_bytes_per_batch'] / 1e9:.1f}",
        "GB per batch, about",
        f"{storage['formal_projection']['projected_file_count_per_batch']:,.0f} files per batch.",
        "That measurement supports local execution capacity and retaining sharding;",
        "it does not justify the replication budget or establish external durability.",
        "",
        "## Validation",
        "",
        *(
            [
                f"- Complete suite: {validation['tests']['count']} tests, exit 0, "
                f"{validation['tests']['seconds']:.3f} seconds.",
                f"- Coverage: {validation['coverage']['percent']}% against the configured "
                f"{validation['coverage']['required_percent']}% floor.",
                "- Lock, Ruff lint/format, mypy, exit-code contract, strict JSON parse,",
                "  package build/payload, clean wheel install, installed CLI smoke,",
                "  v2.1 smoke/recomputation, observation-noise smoke/recomputation, and",
                "  zero-noise pinned-reference replay passed.",
                f"- Final-head hosted CI runs: {validation.get('github_ci_runs', 'pending')}.",
                f"- Exact final remote main: {validation.get('final_main', 'pending')}.",
            ]
            if validation
            else ["Final clean-checkout and hosted-CI validation is pending generation."]
        ),
        "",
        "## Approval matrix",
        "",
        "| Gate | Verdict |",
        "|---|---|",
        "| Repository engineering | APPROVED, subject to exact final-head CI when marked pending above |",
        "| Test and CI integrity | APPROVED only when final-head hosted jobs are recorded green |",
        "| Packaging and clean-install integrity | APPROVED when final validation record is present |",
        "| Evidence integrity | APPROVED for the retained engineering/development evidence |",
        "| Benchmark v2.1 historical state | LIMITED: valid confirmed results, historical raw-archive limitations retained |",
        "| Observation-noise implementation | APPROVED when final-head artifact upload is green |",
        "| Observation-noise scientific outcome | NOT EXECUTED |",
        "| Local evidence retention | VERIFIED on the dedicated HDD |",
        "| External durable archival | NOT VERIFIED; not required for current internal phase |",
        "| Repository hygiene | APPROVED after final PR/branch closure |",
        "| Ready for next AAA phase | APPROVED after exact final main and hosted CI verification |",
        "",
        "The scientific `NOT EXECUTED` status is intentional and is not represented as",
        "a negative, inconclusive, or successful experiment.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
