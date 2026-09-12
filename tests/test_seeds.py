"""Stream allocation and confirmation batch discipline.

The original protocol derived "freshness" from a user-editable ``--attempt-id``
label that never entered the seed, so two confirmations could differ only by
name and re-run identical numbers. These tests pin the repaired contract.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aaa.benchmark.seeds import (
    CONFIRMATION_ROLES,
    DEFAULT_GOLDEN_CASES,
    BatchRegistryError,
    ConfirmationBatchRegistry,
    golden_seed_fixture,
    lineage_for,
    probe_seed,
    purpose_for,
    stable_label,
    training_seed,
    trial_seed,
)
from aaa.benchmark.spec import canonical_spec_hash, load_spec

ROOT = 20260909


class StreamIdentityTests(unittest.TestCase):
    def test_seed_depends_only_on_trial_identity(self):
        self.assertEqual(
            trial_seed(ROOT, "development", "bouncing", 1, 2),
            trial_seed(ROOT, "development", "bouncing", 1, 2),
        )

    def test_every_identity_component_changes_the_stream(self):
        base = trial_seed(ROOT, "development", "bouncing", 1, 2)
        self.assertNotEqual(base, trial_seed(ROOT + 1, "development", "bouncing", 1, 2))
        self.assertNotEqual(base, trial_seed(ROOT, "high_replication", "bouncing", 1, 2))
        self.assertNotEqual(base, trial_seed(ROOT, "development", "speed_change", 1, 2))
        self.assertNotEqual(base, trial_seed(ROOT, "development", "bouncing", 2, 2))
        self.assertNotEqual(base, trial_seed(ROOT, "development", "bouncing", 1, 3))

    def test_labels_are_not_process_salted(self):
        # A plain hash() would change every interpreter run.
        self.assertEqual(stable_label("bouncing"), stable_label("bouncing"))
        self.assertNotEqual(stable_label("bouncing"), stable_label("straight"))

    def test_confirmation_batches_get_genuinely_different_streams(self):
        first = purpose_for("confirmation_a", "batch-0001")
        second = purpose_for("confirmation_a", "batch-0002")
        self.assertNotEqual(first, second)
        self.assertNotEqual(
            trial_seed(ROOT, first, "bouncing", 0, 0), trial_seed(ROOT, second, "bouncing", 0, 0)
        )

    def test_a_confirmation_role_without_a_batch_is_rejected(self):
        for role in CONFIRMATION_ROLES:
            with self.assertRaises(ValueError):
                purpose_for(role, None)
            with self.assertRaises(ValueError):
                purpose_for(role, "")

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(ValueError):
            purpose_for("whatever", None)

    def test_negative_or_boolean_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            trial_seed(ROOT, "development", "bouncing", -1, 0)
        with self.assertRaises(ValueError):
            trial_seed(ROOT, "development", "bouncing", True, 0)

    def test_shared_checkpoint_relationship_gives_a_and_b_one_lineage(self):
        a = lineage_for("confirmation_a", "shared_frozen_checkpoints")
        b = lineage_for("confirmation_b", "shared_frozen_checkpoints")
        self.assertEqual(a, b)
        self.assertEqual(training_seed(ROOT, a, 0, 0), training_seed(ROOT, b, 0, 0))

    def test_independent_relationship_gives_a_and_b_separate_lineages(self):
        a = lineage_for("confirmation_a", "independent_training")
        b = lineage_for("confirmation_b", "independent_training")
        self.assertNotEqual(a, b)
        self.assertNotEqual(training_seed(ROOT, a, 0, 0), training_seed(ROOT, b, 0, 0))

    def test_evaluation_and_training_streams_never_collide(self):
        self.assertNotEqual(
            training_seed(ROOT, "selected", 0, 0), trial_seed(ROOT, "development", "training", 0, 0)
        )

    def test_probe_bank_is_the_same_at_every_budget(self):
        self.assertEqual(probe_seed(ROOT, 900_000, 3), probe_seed(ROOT, 900_000, 3))


class GoldenSeedTests(unittest.TestCase):
    def test_committed_fixture_matches_the_current_mapping(self):
        path = Path("benchmarks/golden_seeds.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = load_spec()
        self.assertEqual(payload["root_seed"], spec.randomness.root_seed)
        self.assertEqual(
            payload["seeds"],
            golden_seed_fixture(spec.randomness.root_seed, DEFAULT_GOLDEN_CASES),
        )

    def test_fixture_keys_are_readable_identities(self):
        fixture = golden_seed_fixture(ROOT, DEFAULT_GOLDEN_CASES)
        for key in fixture:
            self.assertEqual(len(key.split("|")), 4)


class BatchRegistryTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "batches.json"
        self.registry = ConfirmationBatchRegistry.load(self.path)
        self.spec_hash = canonical_spec_hash()

    def tearDown(self):
        self._directory.cleanup()

    def test_a_fresh_declared_batch_can_be_claimed_once(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        claimed = self.registry.claim("b1", "confirmation_a", self.spec_hash)
        self.assertEqual(claimed.status, "planned")

    def test_declaring_the_same_batch_twice_is_rejected(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        with self.assertRaises(BatchRegistryError):
            self.registry.declare("b1", "confirmation_a", self.spec_hash)

    def test_an_undeclared_batch_cannot_be_claimed(self):
        with self.assertRaises(BatchRegistryError):
            self.registry.claim("never-declared", "confirmation_a", self.spec_hash)

    def test_a_consumed_batch_cannot_be_reused_as_a_fresh_confirmation(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        self.registry.save()
        self.registry.reserve("b1", "confirmation_a", self.spec_hash, run_id="run-1")
        self.registry.record_outcome("b1", "run-1", passed=True)
        with self.assertRaises(BatchRegistryError) as caught:
            self.registry.claim("b1", "confirmation_a", self.spec_hash)
        self.assertIn("fresh confirmation requires an unused batch", str(caught.exception))

    def test_a_consumed_batch_may_be_rerun_in_explicit_reproduction_mode(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        self.registry.save()
        self.registry.reserve("b1", "confirmation_a", self.spec_hash, run_id="run-1")
        self.registry.record_outcome("b1", "run-1", passed=True)
        again = self.registry.claim("b1", "confirmation_a", self.spec_hash, reproduction=True)
        self.assertEqual(again.batch_id, "b1")

    def test_reproduction_of_a_never_run_batch_is_rejected(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        with self.assertRaises(BatchRegistryError):
            self.registry.claim("b1", "confirmation_a", self.spec_hash, reproduction=True)

    def test_a_failed_batch_is_retired_and_never_reusable(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        self.registry.save()
        self.registry.reserve("b1", "confirmation_a", self.spec_hash, run_id="run-1")
        batch = self.registry.record_outcome("b1", "run-1", passed=False)
        self.assertEqual(batch.status, "retired")
        self.assertEqual(batch.outcome, "required_gate_failure")
        with self.assertRaises(BatchRegistryError):
            self.registry.claim("b1", "confirmation_a", self.spec_hash)

    def test_a_failed_batch_stays_recorded_rather_than_disappearing(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        self.registry.save()
        self.registry.reserve("b1", "confirmation_a", self.spec_hash, run_id="run-1")
        self.registry.record_outcome("b1", "run-1", passed=False)
        self.registry.save()
        reloaded = ConfirmationBatchRegistry.load(self.path)
        self.assertEqual(reloaded.get("b1").status, "retired")
        self.assertEqual(reloaded.get("b1").consumed_by, ["run-1"])

    def test_stale_registry_instances_cannot_both_reserve_one_batch(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        self.registry.save()
        stale = ConfirmationBatchRegistry.load(self.path)
        self.registry.reserve("b1", "confirmation_a", self.spec_hash, run_id="first")
        with self.assertRaises(BatchRegistryError):
            stale.reserve("b1", "confirmation_a", self.spec_hash, run_id="second")

    def test_an_unobserved_batch_can_be_cancelled_without_fabricated_consumption(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        self.registry.cancel_unobserved("b1", reason="paired attempt failed before this stream ran")
        self.registry.save()
        batch = ConfirmationBatchRegistry.load(self.path).get("b1")
        self.assertEqual(batch.status, "cancelled")
        self.assertEqual(batch.outcome, "superseded_before_observation")
        self.assertEqual(batch.consumed_by, [])
        with self.assertRaises(BatchRegistryError):
            self.registry.claim("b1", "confirmation_a", self.spec_hash)

    def test_matching_interrupted_claim_can_resume_but_not_restart_fresh(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        self.registry.save()
        self.registry.reserve("b1", "confirmation_a", self.spec_hash, run_id="run-1")
        resumed = self.registry.reserve("b1", "confirmation_a", self.spec_hash, run_id="run-1", resume=True)
        self.assertEqual(resumed.status, "running")
        with self.assertRaises(BatchRegistryError):
            self.registry.reserve("b1", "confirmation_a", self.spec_hash, run_id="run-1")

    def test_claiming_with_the_wrong_role_is_rejected(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        with self.assertRaises(BatchRegistryError):
            self.registry.claim("b1", "confirmation_b", self.spec_hash)

    def test_claiming_after_the_specification_changed_is_rejected(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash)
        with self.assertRaises(BatchRegistryError) as caught:
            self.registry.claim("b1", "confirmation_a", "0" * 64)
        self.assertIn("declare a new batch", str(caught.exception))

    def test_a_development_role_cannot_declare_a_batch(self):
        with self.assertRaises(BatchRegistryError):
            self.registry.declare("b1", "development", self.spec_hash)

    def test_registry_round_trips_through_disk(self):
        self.registry.declare("b1", "confirmation_a", self.spec_hash, notes="first")
        self.registry.declare("b2", "confirmation_b", self.spec_hash)
        self.registry.save()
        reloaded = ConfirmationBatchRegistry.load(self.path)
        self.assertEqual([batch.batch_id for batch in reloaded.batches()], ["b1", "b2"])
        self.assertEqual(reloaded.get("b1").notes, "first")

    def test_an_unknown_registry_schema_is_rejected(self):
        self.path.write_text(json.dumps({"schema_version": "nope", "batches": []}), encoding="utf-8")
        with self.assertRaises(BatchRegistryError):
            ConfirmationBatchRegistry.load(self.path)


if __name__ == "__main__":
    unittest.main()
