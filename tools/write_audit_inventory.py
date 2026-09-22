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


def purpose(path: str) -> str:
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


def findings(path: str) -> str:
    inherited = inherited_findings(path)
    review = REVIEW_2026_09_22.get(path)
    if review is None:
        return inherited
    return review if inherited == "none" else f"{inherited}; {review}"


def inherited_findings(path: str) -> str:
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
            "Hashes are SHA-256 of the reviewed worktree bytes, not Git blob IDs.",
            "A row records inspection coverage; it does not upgrade historical evidence into fresh verification.",
            "",
            "| Path | Reviewed SHA-256 | Purpose | Category | Verification | Related findings | Disposition |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(rows)} rows")


if __name__ == "__main__":
    main()
