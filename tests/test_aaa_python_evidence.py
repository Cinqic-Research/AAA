"""Retained aaa.python.v0 development evidence stays consistent, recomputable and correctly labelled."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from research.aaa_python import experiment, recompute

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/aaa_python_v0/development.json"
RECORDS = ROOT / "docs/evidence/aaa_python_v0/development_records.jsonl.gz"


class RetainedDevelopmentEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = recompute.load_evidence(EVIDENCE.read_text(encoding="utf-8"))

    def test_it_is_development_evidence_from_a_clean_commit(self) -> None:
        self.assertIn("not confirmation", self.evidence["status"])
        self.assertFalse(self.evidence["plan"]["quick"])
        source = self.evidence["provenance"]["source"]
        self.assertFalse(source["dirty"])
        self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(self.evidence["provenance"]["captured_before_run"])

    def test_no_confirmation_or_attack_identity_was_observed(self) -> None:
        for record in experiment.read_records(RECORDS):
            self.assertIn(record["task_id"].split(":")[1], ("development", "probe"))

    def test_records_cells_truths_and_summary_recompute(self) -> None:
        result = recompute.verify(self.evidence, experiment.read_records(RECORDS))
        self.assertEqual(result["verdict"], "PASS", result["problems"])

    def test_the_report_matches_the_evidence(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/write_aaa_python_report.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_the_evidence_is_strict_json(self) -> None:
        json.loads(EVIDENCE.read_text(encoding="utf-8"), parse_constant=recompute._refuse_constant)


if __name__ == "__main__":
    unittest.main()
