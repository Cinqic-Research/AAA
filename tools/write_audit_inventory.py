#!/usr/bin/env python3
"""Generate the stable, exhaustive final-review inventory for every tracked path."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "final_audit.md"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


V2_PATHS = (
    "research/aaa_1k_v2/",
    "aaa/compute/",
    "docs/aaa_1k_v2_",
    "docs/evidence/aaa_1k_v2/",
    "benchmarks/aaa1k_v2_",
    "benchmarks/hardware/",
    "tests/test_aaa_1k_v2",
    "tests/test_compute.py",
    "tools/write_aaa_1k_v2_report.py",
    "tools/reproduce_aaa_1k_v2_confirmation.py",
    "tools/aaa_1k_v2_dysts_validation.py",
    "requirements-cuda-lock.txt",
)
"""Paths added or owned by the aaa.1k.v2 phase and the FLOWBOX compute layer."""


def is_v2(path: str) -> bool:
    return path.startswith(V2_PATHS)


PYTHON_PATHS = (
    "research/aaa_python/",
    "aaa/promotion/",
    "docs/aaa_python_",
    "docs/evidence/aaa_python_v0/",
    "docs/promotion_contract.md",
    "tests/test_aaa_python",
    "tests/test_promotion.py",
    "tools/write_aaa_python_report.py",
    "tools/check_promotion_successor.py",
)
"""Paths added by the Python-first transition: aaa.python.v0 and the AAA-180 successor."""


def purpose(path: str) -> str:
    if path.startswith("research/aaa_python_v1/"):
        return "frozen aaa.python.v1 implementation or packaged protocol data"
    if path.startswith("docs/evidence/aaa_python_v1/"):
        return "retained aaa.python.v1 development, freeze or confirmation evidence"
    if path.startswith("research/aaa_python/"):
        return "active aaa.python.v0 research implementation or packaged protocol data"
    if path.startswith("aaa/promotion/"):
        return "active versioned promotion contract (aaa.promotion.crossed.v1, AAA-180 successor)"
    if path.startswith("docs/evidence/aaa_python_v0/"):
        return "current retained aaa.python.v0 development evidence"
    if path == "benchmarks/protected_identities.json":
        return "active expectation of protected historical identities and evidence bytes"
    if path.startswith("research/aaa_1k_v2/"):
        return "active AAA-1K v2 research implementation (frozen by the v2 fingerprint)"
    if path.startswith("aaa/compute/"):
        return "active compute layer: device selection, backend provenance, hardware probe"
    if path.startswith("docs/evidence/aaa_1k_v2/"):
        return "current retained AAA-1K v2 evidence"
    if path.startswith("research/aaa_1k_loop/"):
        return "active AAA-1K improvement-loop research and protocol implementation"
    if path.startswith("research/aaa_1k/"):
        return "active isolated AAA-1K research implementation"
    if path.startswith("aaa/"):
        return "active implementation or packaged protocol data"
    if path.startswith("tests/"):
        return "active regression and protocol verification"
    if path.startswith(".github/"):
        return "active CI, governance, or dependency automation"
    if path.startswith("benchmarks/"):
        return "active frozen protocol identity or seed registry"
    if path.startswith("results/"):
        return "historical or current immutable experiment evidence"
    if path == "docs/evidence/aaa_1k_evaluation_round3.json":
        return "current retained AAA-1K evaluation evidence"
    if path.startswith("docs/evidence/aaa_1k_evaluation_round"):
        return "superseded retained AAA-1K evaluation evidence"
    if path.startswith("docs/evidence/aaa_1k_"):
        return "retained AAA-1K development or characterization evidence"
    if path.startswith("docs/evidence/"):
        return "retained diagnostic or selection evidence"
    if path.startswith("docs/"):
        return "active documentation or review record"
    if path.startswith("tools/"):
        return "active validation or deterministic document generator"
    return "project metadata, packaging, license, or operator entry point"


def category(path: str) -> str:
    if path in {
        "docs/evidence/phase_closure_validation.json",
        "docs/aaa_python_development_report.md",
        "docs/aaa_python_v1_development_report.md",
        "docs/final_audit.md",
        "docs/handoff_sol.md",
        "docs/sol_review.md",
    }:
        return "generated"
    if path.startswith("results/final/") or path.startswith("results/benchmark_v2/"):
        return "historical"
    if path in {
        "docs/evidence/aaa_1k_evaluation_round1_superseded.json",
        "docs/evidence/aaa_1k_evaluation_round2_superseded.json",
    }:
        return "superseded evidence"
    if path == "docs/evidence/aaa_1k_evaluation_round3.json":
        return "current evidence"
    if path.startswith(("docs/evidence/aaa_1k_v2/", "docs/evidence/aaa_python_v0/")):
        return "current evidence"
    if path.startswith("results/benchmark_v2_1/") or path.startswith("docs/evidence/"):
        return "retained evidence"
    return "active"


def verification(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "read; syntax/tests/lint as applicable"
    if suffix == ".json":
        return "read; strict JSON parsed; semantic checks as applicable"
    if suffix in {".yml", ".yaml", ".toml"}:
        return "read; parser or CI validation as applicable"
    if suffix == ".png":
        return "opened at original resolution; historical labeling checked"
    if suffix == ".gz":
        return "decompressed; strict JSON lines; digest and recomputation checked"
    if suffix in {".md", ".txt"} or Path(path).name in {"LICENSE", ".gitignore"}:
        return "read; references and claims audited"
    return "read as bytes; retained provenance checked"


REVIEW_2026_09_22 = {
    ".github/dependabot.yml": "AAA-175",
    "CHANGELOG.md": "AAA-174 through AAA-178",
    "README.md": "AAA-176",
    "SECURITY.md": "AAA-175",
    "docs/dependencies.md": "AAA-175",
    "docs/evidence_policy.md": "AAA-178",
    "docs/independent_review_2026-09-22.md": "AAA-174 through AAA-178",
    "docs/issue_ledger.md": "AAA-174 through AAA-178",
    "docs/loop_protocol.md": "AAA-176",
    "research/aaa_1k_loop/champion1.py": "AAA-174",
    "tests/test_aaa_1k_loop_iteration4.py": "AAA-174",
    "tests/test_repository_automation.py": "AAA-175",
    "tools/write_audit_inventory.py": "AAA-174 through AAA-178",
}
"""Paths changed by the 2026-09-22 independent review, and the findings that changed them."""

REVIEW_2026_09_23 = {
    ".github/workflows/ci.yml": "Q2 measurement parity",
    "README.md": "Q2 non-replication; current research direction",
    "docs/aaa_1k_v2_handoff.md": "Q2 measurement parity; AAA-180",
    "docs/aaa_1k_v2_report.md": "AAA-181",
    "docs/hardware.md": "105M planning decision",
    "docs/issue_ledger.md": "AAA-180, AAA-181",
    "docs/limitations.md": "Q2 measurement parity; AAA-180, AAA-181",
    "docs/pre_scale_review_2026-09-23.md": "AAA-180, AAA-181; Q2 measurement parity",
    "docs/research_direction.md": "Python-first coding decision",
    "docs/scaling_readiness.md": "AAA-180; 105M planning decision",
    "tools/check_adaptation_parity.py": "Q2 measurement parity",
    "tools/write_aaa_1k_v2_report.py": "AAA-181",
}


TRANSITION_2026_09_23 = {
    ".github/workflows/ci.yml": "AAA-180 successor and aaa.python.v0 verification; protected identities",
    "CHANGELOG.md": "AAA-180, AAA-182 through AAA-186",
    "CITATION.cff": "AAA-183",
    "CONTRIBUTING.md": "AAA-183, AAA-184",
    "README.md": "AAA-183",
    "SECURITY.md": "AAA-183",
    "pyproject.toml": "aaa.python.v0 package data",
    "benchmarks/protected_identities.json": "protected historical identities",
    "docs/dot_benchmark_archive.md": "AAA-183",
    "docs/errata.md": "AAA-180, AAA-182, AAA-183",
    "docs/evidence_policy.md": "aaa.python.v0 evidence",
    "docs/hardware.md": "AAA-183",
    "docs/issue_ledger.md": "AAA-180 prospective repair; AAA-182 through AAA-186",
    "docs/limitations.md": "AAA-180; aaa.python.v0 limitations",
    "docs/reproduction.md": "aaa.python.v0 and AAA-180 successor reproduction",
    "docs/research_direction.md": "Python-first capability ladder; English as a future direction",
    "docs/scaling_readiness.md": "AAA-180 prospective repair; PYTHON_PHASE_IN_DEVELOPMENT",
    "tests/test_protected_identities.py": "protected historical identities",
    "tools/check_protected_identities.py": "protected historical identities",
    "tools/write_audit_inventory.py": "Python-first transition classification",
    "docs/aaa_python_self_review.md": "implementer self-review of the transition (not independent)",
}
"""Paths changed by the 2026-09-23 Python-first transition, and why."""

REVIEW_2026_09_24 = {
    "AGENTS.md": "FLOWBOX HDD-first agent entry point",
    "CLAUDE.md": "FLOWBOX HDD-first agent entry point",
    "CHANGELOG.md": "AAA-187 through AAA-191; storage and backup policy",
    "CONTRIBUTING.md": "FLOWBOX HDD-first policy",
    "README.md": "FLOWBOX HDD-first policy; AAA-191 claim scope",
    "SECURITY.md": "AAA-191; specification and path claim scope",
    "aaa/promotion/adjudicate.py": "AAA-189",
    "docs/aaa_python_architecture.md": "runtime state accounting clarification",
    "docs/agent_work_policy.md": "FLOWBOX HDD-first and scientific work policy",
    "docs/backup_policy.md": "canonical GitHub and local/off-device backup scope",
    "docs/dot_benchmark_archive.md": "historical phase wording",
    "docs/errata.md": "historical charter wording",
    "docs/evidence_policy.md": "Git bundle versus omitted raw evidence",
    "docs/hardware.md": "FLOWBOX HDD-first policy",
    "docs/issue_ledger.md": "AAA-187 through AAA-191",
    "docs/pr28_independent_review_2026-09-24.md": "independent PR #28 review",
    "docs/reproduction.md": "FLOWBOX HDD-first policy",
    "research/aaa_python/checks.py": "AAA-191",
    "research/aaa_python/cli.py": "AAA-191",
    "research/aaa_python/episode.py": "AAA-190",
    "research/aaa_python/experiment.py": "AAA-188; strict record JSON",
    "research/aaa_python/oracle.py": "AAA-191",
    "research/aaa_python/recompute.py": "AAA-187",
    "research/aaa_python/subset.py": "AAA-191",
    "tests/test_review_regressions.py": "AAA-187 through AAA-190",
    "tests/test_storage_preflight.py": "FLOWBOX HDD-first policy",
    "tools/backup_repository.py": "commit-addressed Git bundle and restore test",
    "tools/storage_preflight.py": "FLOWBOX HDD mount and path gate",
    "tools/write_audit_inventory.py": "independent review inventory semantics",
}


def findings(path: str) -> str:
    notes = [
        inherited_findings(path),
        REVIEW_2026_09_22.get(path),
        REVIEW_2026_09_23.get(path),
        TRANSITION_2026_09_23.get(path),
        REVIEW_2026_09_24.get(path),
    ]
    relevant = [note for note in notes if note and note != "none"]
    return "; ".join(relevant) if relevant else "none"


def inherited_findings(path: str) -> str:
    if path.startswith(
        (
            "research/aaa_python_v1/",
            "docs/aaa_python_v1_",
            "docs/evidence/aaa_python_v1/",
            "tests/test_aaa_python_v1",
            "tools/write_aaa_python_v1_",
            "tools/benchmark_aaa_python_v1_",
        )
    ):
        return "aaa.python.v1 evidence-gated pre-scale phase; retained confirmation at a73765b freeze"
    if path.startswith(PYTHON_PATHS):
        if (
            path.startswith(
                ("aaa/promotion/", "tests/test_promotion.py", "tools/check_promotion_successor.py")
            )
            or path == "docs/promotion_contract.md"
        ):
            return "AAA-180, AAA-182 (prospective successor)"
        return "AAA-184 through AAA-186; aaa.python.v0 development only, confirmation not admitted"
    if is_v2(path):
        return "AAA-179, AAA-180; AAA-162, AAA-163 and AAA-172 revisited in aaa.1k.v2"
    loop_path = (
        path.startswith("research/aaa_1k_loop/")
        or path.startswith("docs/evidence/aaa1k_loop_")
        or path.startswith("docs/loop_")
        or path == "benchmarks/aaa1k_loop_identity_ledger.json"
        or path
        in {
            "tests/test_aaa_1k_loop.py",
            "tests/test_aaa_1k_loop_iteration4.py",
            "tests/test_aaa_1k_loop_tbptt.py",
        }
    )
    if loop_path:
        return "AAA-162 through AAA-173; v0 artifacts retained under current aaa.loop.v1 governance"
    aaa_1k_path = (
        path.startswith("research/aaa_1k/")
        or path.startswith("docs/aaa_1k_")
        or path.startswith("docs/evidence/aaa_1k_")
        or path == "tests/test_aaa_1k.py"
    )
    if aaa_1k_path:
        return "AAA-152 through AAA-161; rounds 1 and 2 retained as superseded"
    noise_path = (
        path.startswith("aaa/noise/")
        or path.startswith("docs/evidence/observation_noise")
        or path.startswith("docs/evidence/sol_observation_noise")
        or "observation_noise" in path
        or path
        in {
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            ".github/workflows/ci.yml",
            "docs/evidence_policy.md",
            "docs/experiment_registry.md",
            "docs/issue_ledger.md",
            "docs/limitations.md",
            "docs/reproduction.md",
            "docs/self_review.md",
            "docs/sol_review.md",
            "docs/handoff_sol.md",
            "tools/write_handoff.py",
        }
    )
    if noise_path:
        return "AAA-135 through AAA-151; AAA-134/144 retained as publication-grade limitations"
    ids: list[str] = []
    if path.startswith(("aaa/", "tests/", "tools/", "benchmarks/")):
        ids.extend(["AAA-121", "AAA-122", "AAA-123"])
    if path.startswith("results/benchmark_v2_1/") or path in {
        "docs/evidence_policy.md",
        "docs/reproduction.md",
    }:
        ids.append("AAA-124")
    if path in {"aaa/benchmark/seeds.py", ".github/workflows/benchmark.yml"}:
        ids.append("AAA-125")
    return ", ".join(dict.fromkeys(ids)) or "none"


def main() -> None:
    import sys

    check = "--check" in sys.argv[1:]
    paths = [line for line in git("ls-files", "--cached").splitlines() if line]
    rows = []
    for name in paths:
        target = ROOT / name
        if not target.is_file() or target.is_symlink():
            raise SystemExit(f"tracked path is absent or not a regular file: {name}")
        digest = (
            "generated-self-reference"
            if target == OUTPUT
            else hashlib.sha256(target.read_bytes()).hexdigest()
        )
        rows.append(
            f"| `{name}` | `{digest}` | {purpose(name)} | {category(name)} | "
            f"{verification(name)} | {findings(name)} | retain candidate content |"
        )
    text = "\n".join(
        [
            "# Final tracked-file audit",
            "",
            "Generated deterministically by `tools/write_audit_inventory.py` from the staged review candidate.",
            f"It inventories **{len(rows)} tracked regular files**; the count and path set must equal `git ls-files --cached`.",
            "Hashes are SHA-256 of the tracked worktree bytes, not Git blob IDs.",
            "Rows retain inspection classifications across documented reviews; they do not prove a fresh reread in this review or upgrade historical evidence.",
            "`active` denotes a maintained tracked path, not a current scientific result; historical claims inside active documents retain their original dates and evidence scope.",
            "The Verification column names an appropriate inspection method, not proof that every listed method was executed anew for this review; see the independent review record for actual coverage.",
            "",
            "| Path | Tracked SHA-256 | Purpose | Category | Verification | Related findings | Disposition |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )
    if check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            raise SystemExit(f"{OUTPUT.relative_to(ROOT)} does not match the tracked tree; regenerate it")
        print(f"{OUTPUT.relative_to(ROOT)} matches all {len(rows)} tracked files")
        return
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(rows)} rows")


if __name__ == "__main__":
    main()
