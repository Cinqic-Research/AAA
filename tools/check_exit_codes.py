"""Prove the confirmation exit-code contract from a real process.

A confirmation that records a failed, unverified or insufficient required gate
must leave a non-zero process status behind, and CI must be able to see it.
This runs the actual CLI in subprocesses rather than trusting a unit test's
in-process return value.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "aaa.cli", *argv], cwd=ROOT, capture_output=True, text=True)


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        runs = Path(directory) / "runs"

        # 1. A development attempt too small to satisfy coverage records the
        #    shortfall and still exits 0.
        development = run(
            [
                "benchmark",
                "--role",
                "development",
                "--attempt-label",
                "exit-contract-dev",
                "--replicas",
                "1",
                "--episodes",
                "1",
                "--output-root",
                str(runs),
            ]
        )
        if development.returncode != 0:
            failures.append(f"development exited {development.returncode}, expected 0")
        if "INSUFFICIENT_EVIDENCE" not in development.stdout:
            failures.append("development attempt did not record any unmet gate to check against")

        # 2. A confirmation with an undeclared batch is refused before it runs.
        refused = run(
            [
                "benchmark",
                "--role",
                "confirmation_a",
                "--batch-id",
                "never-declared-batch",
                "--output-root",
                str(runs),
            ]
        )
        if refused.returncode == 0:
            failures.append("a confirmation against an undeclared batch exited 0")

        # 3. A confirmation may not weaken the declared minimums.
        weakened = run(
            [
                "benchmark",
                "--role",
                "confirmation_a",
                "--batch-id",
                "never-declared-batch",
                "--replicas",
                "1",
                "--episodes",
                "1",
                "--output-root",
                str(runs),
            ]
        )
        if weakened.returncode == 0:
            failures.append("a confirmation with weakened minimums exited 0")

    if failures:
        print("exit-code contract violated:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("confirmation exit-code contract holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
