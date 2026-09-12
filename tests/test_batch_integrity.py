"""Serialized status edits must not erase confirmation consumption evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from aaa.benchmark.seeds import BatchRegistryError, ConfirmationBatchRegistry


class BatchTamperTests(unittest.TestCase):
    def test_serialized_contradictions_are_rejected(self):
        for passed in (False, True):
            for changes in (
                {"status": "planned"},
                {"consumed_by": []},
                {"outcome": None},
                {"outcome": "invented"},
                {"consumed_by": "run"},
                {"consumed_by": [""]},
            ):
                with self.subTest(passed=passed, changes=changes), tempfile.TemporaryDirectory() as d:
                    path = Path(d) / "registry.json"
                    registry = ConfirmationBatchRegistry(path)
                    registry.declare("batch", "confirmation_a", "hash")
                    registry.save()
                    registry.reserve("batch", "confirmation_a", "hash", run_id="run")
                    registry.record_outcome("batch", "run", passed=passed)
                    registry.save()
                    payload = json.loads(path.read_text())
                    payload["batches"][0].update(changes)
                    path.write_text(json.dumps(payload))
                    with self.assertRaises(BatchRegistryError):
                        ConfirmationBatchRegistry.load(path)

    def test_status_alone_does_not_establish_prior_execution(self):
        for status in ("consumed", "retired"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "registry.json"
                registry = ConfirmationBatchRegistry(path)
                registry.declare("batch", "confirmation_a", "hash")
                registry.save()
                payload = json.loads(path.read_text())
                payload["batches"][0]["status"] = status
                path.write_text(json.dumps(payload))
                with self.assertRaises(BatchRegistryError):
                    ConfirmationBatchRegistry.load(path).claim(
                        "batch", "confirmation_a", "hash", reproduction=True
                    )

    def test_in_memory_tamper_is_also_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ConfirmationBatchRegistry(Path(directory) / "registry.json")
            registry.declare("batch", "confirmation_a", "hash")
            registry.save()
            registry.reserve("batch", "confirmation_a", "hash", run_id="run")
            registry.record_outcome("batch", "run", passed=False).status = "planned"
            with self.assertRaises(BatchRegistryError):
                registry.claim("batch", "confirmation_a", "hash")

    def test_historical_batches_remain_readable(self):
        registry = ConfirmationBatchRegistry.load("benchmarks/confirmation_batches.json")
        for batch in registry.batches():
            if batch.consumed_by:
                self.assertEqual(
                    registry.claim(batch.batch_id, batch.role, batch.spec_hash, reproduction=True), batch
                )
