"""Smoke coverage for the development-only diagnosis and selection drivers.

Neither produces acceptance evidence. These tests check that both run, write
the artifacts the documentation promises, and keep their development-only
labelling, so a later reviewer cannot mistake them for confirmation results.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aaa.diagnosis import run_diagnosis
from aaa.selection import run_candidate_selection


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.output = run_diagnosis(Path(cls._directory.name) / "diagnosis", quick=True)
        cls.result = json.loads((cls.output / "diagnosis.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_it_writes_both_artifacts(self):
        self.assertTrue((self.output / "diagnosis.json").is_file())
        self.assertTrue((self.output / "diagnosis.md").is_file())

    def test_the_result_is_labelled_development_only(self):
        self.assertIn("development", json.dumps(self.result).lower())
        self.assertNotIn("confirmation_a", json.dumps(self.result))

    def test_it_compares_more_than_one_feature_basis(self):
        self.assertGreater(len({row["features"] for row in self.result["rows"]}), 1)

    def test_it_compares_more_than_one_estimator(self):
        self.assertGreater(len({row["estimator"] for row in self.result["rows"]}), 1)

    def test_it_reports_conditioning_for_every_basis(self):
        for regime, bases in self.result["conditioning"].items():
            self.assertTrue(bases, regime)
            for name, entry in bases.items():
                if not isinstance(entry, dict):
                    continue
                for key in (
                    "condition_number",
                    "effective_rank",
                    "singular_values",
                    "column_scales",
                    "column_means",
                    "off_bias_correlation",
                ):
                    self.assertIn(key, entry, f"{regime}/{name}")

    def test_divergence_is_recorded_rather_than_dropped(self):
        self.assertTrue(all("diverged" in row for row in self.result["rows"]))

    def test_order_sensitivity_is_reported(self):
        self.assertIn("order_sensitivity", self.result)

    def test_the_trace_bound_ablation_is_present_on_both_arms(self):
        bounds = {row.get("trace_bound") for row in self.result["rows"] if "rls" in row["estimator"]}
        self.assertIn(None, bounds)
        self.assertTrue([value for value in bounds if value is not None])

    def test_unbounded_forgetting_is_shown_to_lose_the_covariance(self):
        unbounded = [
            row
            for row in self.result["rows"]
            if row.get("trace_bound") is None
            and row.get("hyperparameter") == 0.9
            and row["estimator"] == "sqrt_rls"
            and row["features"] == "legacy_positions"
        ]
        self.assertTrue(unbounded)
        self.assertGreater(max(row["covariance_trace"] for row in unbounded), 1e12)

    def test_the_trace_bound_keeps_the_covariance_valid(self):
        bounded = [row for row in self.result["rows"] if row["estimator"] == "sqrt_rls_trace_bounded"]
        self.assertTrue(bounded)
        for row in bounded:
            self.assertGreater(row["covariance_min_eigenvalue"], 0.0)
            self.assertLessEqual(row["covariance_trace"], row["trace_bound"] * 1.05)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.path = run_candidate_selection(Path(cls._directory.name) / "candidate_selection.json", quick=True)
        cls.result = json.loads(Path(cls.path).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_it_records_more_than_one_variant(self):
        self.assertGreater(len(self.result["variants"]), 1)

    def test_unsuccessful_variants_are_retained(self):
        selected = self.result["selected"]["label"]
        others = [row for row in self.result["variants"] if row["label"] != selected]
        self.assertTrue(others)

    def test_the_selection_rule_is_stated(self):
        self.assertIn("selection_rule", self.result)
        self.assertTrue(self.result["selection_rule"])

    def test_selection_uses_development_streams_only(self):
        self.assertNotIn("confirmation", json.dumps(self.result["seeds"]))

    def test_every_variant_reports_a_stability_verdict(self):
        for row in self.result["variants"]:
            self.assertIn("numerically_stable", row)
            self.assertIn("stress_failures", row)
            self.assertIn("covariance_diagnostics", row)


if __name__ == "__main__":
    unittest.main()
