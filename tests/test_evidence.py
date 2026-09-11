"""Durable evidence: registry state machine, checksums, and structure checks."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from aaa.benchmark.evidence import (
    ExperimentRegistry,
    RegistryError,
    TrialRecord,
    dependency_lock,
    git_metadata,
    hardware_metadata,
    iter_raw_records,
    json_dump,
    sha256_file,
    sha256_text,
    verify_checksums,
    verify_records,
    write_jsonl_gz,
)

from .helpers import record


def trial(trial_id: str = "fam:main:r00:e0000", **overrides) -> TrialRecord:
    values = dict(
        trial_id=trial_id,
        family="bouncing",
        branch="main",
        replica=0,
        episode=0,
        environment_seed=99,
    )
    values.update(overrides)
    return TrialRecord(**values)


class RegistryStateMachineTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "registry.json"
        self.registry = ExperimentRegistry.load(self.path)

    def tearDown(self):
        self._directory.cleanup()

    def test_a_planned_trial_is_not_complete(self):
        self.registry.plan(trial())
        self.assertFalse(self.registry.is_complete("fam:main:r00:e0000"))
        self.assertEqual(self.registry.counts()["PLANNED"], 1)

    def test_full_lifecycle_reaches_complete(self):
        self.registry.plan(trial())
        self.registry.start("fam:main:r00:e0000")
        self.assertEqual(self.registry.counts()["RUNNING"], 1)
        done = self.registry.complete(
            "fam:main:r00:e0000", outputs=["raw/a.jsonl.gz"], checksums={"raw/a.jsonl.gz": "ab"}
        )
        self.assertEqual(done.state, "COMPLETE")
        self.assertIsNotNone(done.started_at)
        self.assertIsNotNone(done.finished_at)
        self.assertTrue(self.registry.is_complete("fam:main:r00:e0000"))

    def test_a_failure_records_its_stage_and_message(self):
        self.registry.plan(trial())
        self.registry.start("fam:main:r00:e0000")
        failed = self.registry.fail("fam:main:r00:e0000", stage="simulation", error="boom")
        self.assertEqual(failed.state, "FAILED")
        self.assertEqual(failed.failure_stage, "simulation")
        self.assertEqual(failed.error, "boom")
        self.assertFalse(self.registry.is_complete("fam:main:r00:e0000"))

    def test_failure_evidence_survives_a_reload(self):
        self.registry.plan(trial())
        self.registry.start("fam:main:r00:e0000")
        self.registry.fail("fam:main:r00:e0000", stage="simulation", error="boom")
        self.registry.save()
        reloaded = ExperimentRegistry.load(self.path)
        self.assertEqual(reloaded.trials["fam:main:r00:e0000"].error, "boom")

    def test_running_trials_are_marked_interrupted_on_reload(self):
        self.registry.plan(trial())
        self.registry.start("fam:main:r00:e0000")
        self.registry.save()
        reloaded = ExperimentRegistry.load(self.path)
        self.assertEqual(reloaded.mark_interrupted(), 1)
        self.assertEqual(reloaded.trials["fam:main:r00:e0000"].state, "INTERRUPTED")

    def test_planning_an_existing_completed_trial_preserves_it(self):
        self.registry.plan(trial())
        self.registry.start("fam:main:r00:e0000")
        self.registry.complete("fam:main:r00:e0000", outputs=["x"], checksums={"x": "y"})
        again = self.registry.plan(trial())
        self.assertEqual(again.state, "COMPLETE")
        self.assertEqual(again.outputs, ["x"])

    def test_replanning_must_not_change_a_recorded_seed(self):
        self.registry.plan(trial())
        self.registry.start("fam:main:r00:e0000")
        self.registry.complete("fam:main:r00:e0000", outputs=["x"], checksums={"x": "y"})
        with self.assertRaises(RegistryError):
            self.registry.plan(trial(environment_seed=12345))

    def test_incomplete_lists_everything_not_complete(self):
        self.registry.plan(trial("a"))
        self.registry.plan(trial("b"))
        self.registry.start("b")
        self.registry.complete("b", outputs=[], checksums={})
        self.assertEqual([item.trial_id for item in self.registry.incomplete()], ["a"])

    def test_an_unknown_trial_id_is_a_loud_error(self):
        with self.assertRaises(RegistryError):
            self.registry.start("never-planned")

    def test_an_unknown_registry_schema_is_rejected(self):
        self.path.write_text(json.dumps({"schema_version": "nope", "trials": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            ExperimentRegistry.load(self.path)

    def test_an_unknown_trial_state_is_rejected(self):
        with self.assertRaises(ValueError):
            TrialRecord.from_dict({**trial().to_dict(), "state": "VIBING"})

    def test_directory_existence_is_not_the_notion_of_state(self):
        # The registry, not the filesystem, is what says a trial is done.
        self.registry.plan(trial())
        (self.path.parent / "raw").mkdir()
        self.assertFalse(self.registry.is_complete("fam:main:r00:e0000"))


class ChecksumTests(unittest.TestCase):
    def test_matching_checksums_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "raw").mkdir()
            target = run / "raw" / "a.txt"
            target.write_text("hello", encoding="utf-8")
            json_dump(run / "checksums.json", {"raw/a.txt": sha256_file(target)})
            self.assertTrue(verify_checksums(run)["ok"])

    def test_a_corrupt_file_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "raw").mkdir()
            target = run / "raw" / "a.txt"
            target.write_text("hello", encoding="utf-8")
            json_dump(run / "checksums.json", {"raw/a.txt": sha256_file(target)})
            target.write_text("tampered", encoding="utf-8")
            result = verify_checksums(run)
            self.assertFalse(result["ok"])
            self.assertEqual(result["mismatched"], ["raw/a.txt"])

    def test_a_missing_file_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            json_dump(run / "checksums.json", {"raw/gone.txt": "0" * 64})
            result = verify_checksums(run)
            self.assertFalse(result["ok"])
            self.assertEqual(result["missing"], ["raw/gone.txt"])

    def test_missing_checksum_manifest_is_not_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(verify_checksums(Path(directory))["ok"])

    def test_text_and_file_digests_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.txt"
            path.write_text("payload", encoding="utf-8")
            self.assertEqual(sha256_file(path), sha256_text("payload"))


class RawRecordTests(unittest.TestCase):
    def test_written_records_reload_through_the_iterator(self):
        records = [record(step, 0.01) for step in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            digest = write_jsonl_gz(run / "raw" / "bouncing" / "main" / "e.jsonl.gz", records)
            self.assertEqual(len(digest), 64)
            loaded = list(iter_raw_records(run))
            self.assertEqual(len(loaded), 1)
            self.assertEqual([item.to_dict() for item in loaded[0][1]], [item.to_dict() for item in records])

    def test_a_truncated_raw_file_fails_loudly(self):
        records = [record(step, 0.01) for step in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            path = run / "raw" / "bouncing" / "main" / "e.jsonl.gz"
            write_jsonl_gz(path, records)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("\n".join(lines[:2]) + "\n" + lines[3][:20])
            with self.assertRaises((json.JSONDecodeError, EOFError, OSError, ValueError)):
                list(iter_raw_records(run))


class StructuralVerificationTests(unittest.TestCase):
    def test_well_formed_records_verify(self):
        records = [record(step, 0.01) for step in range(4)]
        self.assertTrue(verify_records(records, expected_predictors=["p"])["ok"])

    def test_no_records_is_not_ok(self):
        self.assertFalse(verify_records([], expected_predictors=["p"])["ok"])

    def test_a_missing_predictor_is_detected(self):
        records = [record(step, 0.01) for step in range(4)]
        result = verify_records(records, expected_predictors=["p", "q"])
        self.assertFalse(result["ok"])
        self.assertIn("missing predictors ['q']", result["problems"][0])

    def test_out_of_order_steps_are_detected(self):
        records = [record(step, 0.01) for step in (0, 2, 1, 3)]
        self.assertFalse(verify_records(records, expected_predictors=["p"])["ok"])

    def test_duplicate_steps_are_detected(self):
        records = [record(step, 0.01) for step in (0, 1, 1, 2)]
        self.assertFalse(verify_records(records, expected_predictors=["p"])["ok"])

    def test_a_non_finite_error_is_detected(self):
        records = [record(step, 0.01) for step in range(4)]
        records[2].predictions["p"]["absolute_error"] = float("nan")
        self.assertFalse(verify_records(records, expected_predictors=["p"])["ok"])

    def test_a_bounce_flag_without_walls_is_detected(self):
        records = [record(step, 0.01) for step in range(4)]
        bad = record(4, 0.01)
        object.__setattr__(bad, "bounced", True)
        object.__setattr__(bad, "bounce_walls", ())
        self.assertFalse(verify_records([*records, bad], expected_predictors=["p"])["ok"])

    def test_a_non_positive_interval_width_is_detected(self):
        records = [record(step, 0.01, width=1.0) for step in range(3)]
        object.__setattr__(records[1], "interval_width", 0.0)
        self.assertFalse(verify_records(records, expected_predictors=["p"])["ok"])


class ProvenanceTests(unittest.TestCase):
    def test_git_metadata_reports_commit_and_dirty_state(self):
        value = git_metadata(Path())
        self.assertIn("commit", value)
        self.assertIn("dirty", value)
        self.assertIn("tree_hash", value)

    def test_dependency_lock_is_hashed(self):
        lock = dependency_lock(Path())
        self.assertEqual(len(lock["hash"]), 64)
        self.assertTrue(lock["path"].endswith("requirements-lock.txt"))

    def test_hardware_metadata_records_the_machine(self):
        value = hardware_metadata()
        for key in ("os", "machine", "cpu_count", "python", "numpy"):
            self.assertIn(key, value)

    def test_hardware_metadata_never_leaks_a_home_directory(self):
        self.assertNotIn("/home/", json.dumps(hardware_metadata()))


if __name__ == "__main__":
    unittest.main()
