"""Regressions for aaa.python.v0 integrity repairs made during the v1 pre-scale phase.

AAA-195: recompute accepted evidence whose plan or cell grid differed from the
declared design. AAA-198: a checkpoint could resume under changed learner code.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research.aaa_python import experiment, recompute
from research.aaa_python import spec as spec_module

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/aaa_python_v0/development.json"


class RecomputeDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        self.spec = spec_module.load()

    def test_retained_evidence_matches_the_declared_design(self) -> None:
        self.assertEqual(recompute.design_problems(self.evidence, self.spec), [])

    def test_changed_draws_is_refused(self) -> None:
        tampered = copy.deepcopy(self.evidence)
        tampered["plan"]["draws"] = 50
        result = recompute.verify(tampered, None)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("draws", " ".join(result["problems"]))

    def test_dropped_initialization_is_refused(self) -> None:
        tampered = copy.deepcopy(self.evidence)
        tampered["cells"] = [c for c in tampered["cells"] if c["init"] != 2]
        problems = recompute.design_problems(tampered, self.spec)
        self.assertTrue(any("missing" in p for p in problems), problems)

    def test_wrong_task_count_is_refused(self) -> None:
        tampered = copy.deepcopy(self.evidence)
        tampered["cells"][0]["n"] += 1
        self.assertTrue(recompute.design_problems(tampered, self.spec))

    def test_malformed_evidence_is_a_refusal_not_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "research.aaa_python", "recompute", "--evidence", str(path)],
                capture_output=True,
                text=True,
                cwd=ROOT,
                check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertNotIn("Traceback", completed.stderr)


class CheckpointCodeBindingTests(unittest.TestCase):
    def test_a_checkpoint_from_other_code_is_refused(self) -> None:
        spec = spec_module.load()
        plan = experiment.Plan.from_spec(spec, quick=True)
        tasks = {"train": {family: [] for family in spec["families"]}}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            experiment.trained_learner(spec, plan, tasks, 0, "lexical", directory)
            with (
                mock.patch.object(experiment, "code_identity", return_value="0" * 64),
                self.assertRaises(experiment.RunError),
            ):
                experiment.trained_learner(spec, plan, tasks, 0, "lexical", directory)
            resumed = experiment.trained_learner(spec, plan, tasks, 0, "lexical", directory)
            self.assertEqual(resumed.updates, 0)


if __name__ == "__main__":
    unittest.main()
