"""Independent recomputation from raw evidence, and freeze-manifest discipline.

These two capabilities are what let a reviewer audit an attempt without
trusting the runtime that produced it, and what stop a confirmation from
quietly drifting away from what was frozen before any result was seen.
"""

from __future__ import annotations

import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from aaa.benchmark.evidence import json_dump, sha256_file
from aaa.benchmark.manifest import (
    FreezeMismatch,
    build_manifest,
    check_manifest,
    load_manifest,
    save_manifest,
)
from aaa.benchmark.recompute import compare_results, recompute_run
from aaa.benchmark.runner import run_benchmark
from aaa.benchmark.spec import canonical_spec_hash, load_spec

PROJECT = Path(__file__).resolve().parents[1]


def _rewrite_checksums(run_dir: Path) -> None:
    json_dump(
        run_dir / "checksums.json",
        {
            str(path.relative_to(run_dir)): sha256_file(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        },
    )


class RecomputeTests(unittest.TestCase):
    """One small real attempt, reused by every case in this class."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.outcome = run_benchmark(
            role="development",
            output_root=root / "runs",
            attempt_label="recompute-fixture",
            replicas=2,
            episodes=2,
            project_root=PROJECT,
        )
        cls.run_dir = cls.outcome.directory

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def _copy(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        holder = tempfile.TemporaryDirectory()
        target = Path(holder.name) / "attempt"
        shutil.copytree(self.run_dir, target)
        return holder, target

    def test_every_stored_gate_status_is_reproduced_from_raw_evidence(self):
        result = recompute_run(self.run_dir)
        stored = {gate["name"]: gate["status"] for gate in self.outcome.summary["gates"]["gates"]}
        again = {gate["name"]: gate["status"] for gate in result["gates"]["gates"]}
        self.assertEqual(stored, again)

    def test_recomputed_family_results_match_the_stored_summary(self):
        result = recompute_run(self.run_dir)
        spec = load_spec()
        comparison = compare_results(
            self.outcome.summary["results"], result["results"], tolerance=spec.tolerances.recompute_absolute
        )
        self.assertTrue(comparison["equivalent"], comparison["differences"])

    def test_recomputation_verifies_checksums_first(self):
        result = recompute_run(self.run_dir)
        self.assertTrue(result["checksums"]["ok"])
        self.assertGreater(result["checksums"]["files"], 0)

    def test_a_corrupted_raw_file_stops_recomputation(self):
        holder, target = self._copy()
        try:
            victim = next(iter(sorted((target / "raw").rglob("*.jsonl.gz"))))
            with gzip.open(victim, "wt", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError) as caught:
                recompute_run(target)
            self.assertIn("checksum verification failed", str(caught.exception))
        finally:
            holder.cleanup()

    def test_a_deleted_raw_file_stops_recomputation(self):
        holder, target = self._copy()
        try:
            next(iter(sorted((target / "raw").rglob("*.jsonl.gz")))).unlink()
            with self.assertRaises(ValueError):
                recompute_run(target)
        finally:
            holder.cleanup()

    def test_a_specification_hash_mismatch_is_refused(self):
        holder, target = self._copy()
        try:
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            summary["spec_hash"] = "0" * 64
            json_dump(target / "summary.json", summary)
            _rewrite_checksums(target)
            with self.assertRaises(ValueError) as caught:
                recompute_run(target)
            self.assertIn("specification hash mismatch", str(caught.exception))
        finally:
            holder.cleanup()

    def test_a_run_without_a_summary_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                recompute_run(Path(directory))

    def test_checksum_verification_can_be_skipped_explicitly(self):
        result = recompute_run(self.run_dir, verify=False)
        self.assertTrue(result["checksums"].get("skipped"))

    def test_the_attempt_retains_the_oscillator_intervention_evidence(self):
        interventions = json.loads(
            (self.run_dir / "metrics" / "changed_law_interventions.json").read_text(encoding="utf-8")
        )
        self.assertTrue(interventions)
        first = interventions[0]
        for key in ("branch_state_hash", "prefix_history", "start_position", "start_velocity"):
            self.assertIn(key, first)

    def test_the_registry_records_every_completed_trial(self):
        registry = json.loads((self.run_dir / "experiment_registry.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["counts"]["FAILED"], 0)
        self.assertEqual(registry["counts"]["RUNNING"], 0)
        self.assertGreater(registry["counts"]["COMPLETE"], 0)


class CompareResultsTests(unittest.TestCase):
    def test_identical_trees_are_equivalent(self):
        tree = {"a": {"b": [1.0, 2.0]}, "c": True, "d": None}
        self.assertTrue(compare_results(tree, tree, tolerance=0.0)["equivalent"])

    def test_a_numeric_difference_inside_tolerance_is_equivalent(self):
        self.assertTrue(
            compare_results({"a": 1.0}, {"a": 1.0 + 1e-12}, tolerance=1e-9)["equivalent"]
        )

    def test_a_numeric_difference_outside_tolerance_is_reported(self):
        result = compare_results({"a": 1.0}, {"a": 1.1}, tolerance=1e-9)
        self.assertFalse(result["equivalent"])
        self.assertEqual(result["differences"][0]["path"], "results.a")

    def test_a_boolean_flip_is_never_absorbed_by_a_numeric_tolerance(self):
        result = compare_results({"ok": True}, {"ok": False}, tolerance=1e9)
        self.assertFalse(result["equivalent"])

    def test_a_missing_key_is_reported(self):
        result = compare_results({"a": 1, "b": 2}, {"a": 1}, tolerance=0.0)
        self.assertFalse(result["equivalent"])

    def test_a_length_difference_is_reported(self):
        result = compare_results({"a": [1, 2]}, {"a": [1]}, tolerance=0.0)
        self.assertFalse(result["equivalent"])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec()
        self.hashes = ["a" * 64, "b" * 64]
        self.seeds = {0: [1, 2, 3], 1: [4, 5, 6]}
        self.manifest = build_manifest(
            self.spec,
            project_root=PROJECT,
            checkpoint_hashes=self.hashes,
            training_seeds=self.seeds,
            planned_batches=["batch-a", "batch-b"],
            notes="unit test",
        )

    def _check(self, **overrides):
        arguments = dict(
            project_root=PROJECT,
            checkpoint_hashes=self.hashes,
            batch_id="batch-a",
            training_seeds=self.seeds,
        )
        arguments.update(overrides)
        check_manifest(self.manifest, self.spec, **arguments)

    def test_a_matching_attempt_is_accepted(self):
        self._check()

    def test_the_manifest_records_the_frozen_identity(self):
        self.assertEqual(self.manifest.spec_hash, canonical_spec_hash())
        self.assertEqual(self.manifest.checkpoint_hashes, self.hashes)
        self.assertEqual(self.manifest.planned_batches, ["batch-a", "batch-b"])

    def test_identity_hash_ignores_the_freeze_timestamp_and_notes(self):
        other = build_manifest(
            self.spec,
            project_root=PROJECT,
            checkpoint_hashes=self.hashes,
            training_seeds=self.seeds,
            planned_batches=["batch-a", "batch-b"],
            notes="different note",
        )
        self.assertEqual(other.identity_hash(), self.manifest.identity_hash())

    def test_an_unplanned_batch_is_refused(self):
        with self.assertRaises(FreezeMismatch) as caught:
            self._check(batch_id="batch-improvised")
        self.assertIn("was not in the frozen plan", str(caught.exception))

    def test_different_checkpoints_are_refused(self):
        with self.assertRaises(FreezeMismatch) as caught:
            self._check(checkpoint_hashes=["c" * 64, "d" * 64])
        self.assertIn("checkpoint hashes differ", str(caught.exception))

    def test_different_training_streams_are_refused(self):
        with self.assertRaises(FreezeMismatch) as caught:
            self._check(training_seeds={0: [9, 9, 9], 1: [4, 5, 6]})
        self.assertIn("training stream identities differ", str(caught.exception))

    def test_a_changed_specification_is_refused(self):
        value = json.loads(json.dumps(self.spec.to_dict()))
        value["confirmation"]["episodes_per_family"] += 1
        from aaa.benchmark.spec import BenchmarkSpec

        with self.assertRaises(FreezeMismatch) as caught:
            check_manifest(
                self.manifest,
                BenchmarkSpec.parse(value),
                project_root=PROJECT,
                checkpoint_hashes=self.hashes,
                batch_id="batch-a",
                training_seeds=self.seeds,
            )
        self.assertIn("specification hash", str(caught.exception))

    def test_every_mismatch_is_reported_together(self):
        with self.assertRaises(FreezeMismatch) as caught:
            self._check(batch_id="nope", checkpoint_hashes=["z" * 64])
        message = str(caught.exception)
        self.assertIn("frozen plan", message)
        self.assertIn("checkpoint hashes differ", message)

    def test_manifest_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            save_manifest(self.manifest, path)
            reloaded = load_manifest(path)
            self.assertEqual(reloaded.identity_hash(), self.manifest.identity_hash())

    def test_an_unknown_manifest_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            path.write_text(json.dumps({"schema_version": "nope"}), encoding="utf-8")
            with self.assertRaises(FreezeMismatch):
                load_manifest(path)

    def test_a_missing_manifest_is_refused(self):
        with self.assertRaises(FreezeMismatch):
            load_manifest(Path("/nonexistent/freeze.json"))


if __name__ == "__main__":
    unittest.main()
