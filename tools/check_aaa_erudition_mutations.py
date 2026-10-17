#!/usr/bin/env python3
"""Failure injection for ``aaa.erudition.v0``: each critical invariant's tests must catch a deliberate break.

Every mutation below is applied to a scratch copy of the repository, one at a
time, and the named tests are run against it. The check passes only if every
mutated copy *fails* its tests and the unmodified copy passes them. A test
suite that never sees a broken implementation is weak evidence that it
protects anything.

    python tools/check_aaa_erudition_mutations.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "research/aaa_erudition"

# (name, file, original text, mutated text, tests that must fail)
MUTATIONS = (
    (
        "a component reads the hidden label",
        "lifecycle.py",
        "        features = self._features(record, first, transition)\n",
        "        features = self._features(record, first, transition)\n        _ = episode.label.entity\n",
        "tests.test_aaa_erudition.LifecycleTests.test_hidden_label_is_read_only_by_the_scorer",
    ),
    (
        "one correction is enough to write an alias note",
        "mechanisms.py",
        "ALIAS_MIN_SUPPORT = 2\n",
        "ALIAS_MIN_SUPPORT = 1\n",
        "tests.test_aaa_erudition.MechanismTests",
    ),
    (
        "the store accepts a commit whose parent is not the head",
        "store.py",
        "        if current != record.parent_state:\n",
        "        if False and current != record.parent_state:\n",
        "tests.test_aaa_erudition.StoreTests",
    ),
    (
        "the store trusts object bytes without re-hashing",
        "store.py",
        "        if sha256_json(body) != digest or canonical_json(body) != raw:\n",
        "        if False:\n",
        "tests.test_aaa_erudition.StoreTests",
    ),
    (
        "a new World Model context overwrites the active one",
        "world.py",
        "        return dataclasses.replace(self, contexts=(*self.contexts, fresh), active=context_id)\n",
        "        return dataclasses.replace(self, contexts=(fresh,), active=context_id)\n",
        "tests.test_aaa_erudition.WorldModelTests",
    ),
    (
        "confirmation streams are generated without admission",
        "toolshift.py",
        '    if split == "confirmation" and not admitted:\n',
        '    if False and split == "confirmation" and not admitted:\n',
        "tests.test_aaa_erudition.EnvironmentTests",
    ),
    (
        "recomputation trusts the recorded outcome",
        "recompute.py",
        "            if outcome != record.outcome:\n",
        "            if False:\n",
        "tests.test_aaa_erudition.RecomputeTests",
    ),
    (
        "the primary promotion path passes superiority on a straddling interval",
        "promotion.py",
        '        if upper < threshold:\n            return "PASS"\n',
        '        if lower < threshold:\n            return "PASS"\n',
        "tests.test_aaa_erudition_promotion",
    ),
    (
        "the World Model gate accepts any candidate",
        "lifecycle.py",
        "            elif target_candidate > target_parent - WM_MIN_GAIN:\n",
        "            elif False:\n",
        "tests.test_aaa_erudition.LifecycleTests",
    ),
    (
        "an adapter loads against a different base model",
        "language.py",
        '        if payload.get("base") != base_artifact():\n',
        "        if False:\n",
        "tests.test_aaa_erudition.LanguageTests",
    ),
    (
        "host validation accepts a nonexistent tank",
        "language.py",
        "        and tank.strip().lower() in workspace\n",
        "        and bool(tank)\n",
        "tests.test_aaa_erudition.LanguageTests",
    ),
)


def run_tests(root: Path, tests: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "-q", tests], cwd=root, capture_output=True, text=True, check=False
    )
    return result.returncode == 0


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "repo"
        shutil.copytree(
            ROOT, scratch, ignore=shutil.ignore_patterns(".git", "*.pyc", "__pycache__", ".venv*")
        )
        for name, file, original, mutated, tests in MUTATIONS:
            path = scratch / PACKAGE / file
            source = path.read_text("utf-8")
            if source.count(original) != 1:
                failures.append(f"{name}: the mutation site is not unique in {file}")
                continue
            if not run_tests(scratch, tests):
                failures.append(f"{name}: tests fail even without the mutation")
                continue
            path.write_text(source.replace(original, mutated), "utf-8")
            try:
                caught = not run_tests(scratch, tests)
            finally:
                path.write_text(source, "utf-8")
            print(f"{'caught' if caught else 'MISSED'}: {name}")
            if not caught:
                failures.append(f"{name}: survived {tests}")
    for failure in failures:
        print(f"FAILURE: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
