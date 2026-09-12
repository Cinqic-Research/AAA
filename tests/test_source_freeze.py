"""Freeze enforcement across clean source and legitimate bookkeeping commits."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from aaa.benchmark.manifest import FreezeMismatch, build_manifest, check_manifest
from aaa.benchmark.spec import load_spec


class SourceFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        for name in (
            "aaa/model.py",
            "tests/test_model.py",
            "tools/verify.py",
            "requirements-lock.txt",
            "benchmarks/golden_seeds.json",
        ):
            self.write(name, "original\n")
        self.write(
            "benchmarks/confirmation_batches.json",
            json.dumps(
                {
                    "batches": [
                        {
                            "batch_id": "a",
                            "role": "confirmation_a",
                            "spec_hash": "s",
                            "status": "planned",
                            "consumed_by": [],
                            "outcome": None,
                        }
                    ]
                }
            ),
        )
        self.commit()
        self.spec = load_spec()
        self.manifest = build_manifest(
            self.spec,
            project_root=self.root,
            checkpoint_hashes=["checkpoint"],
            training_seeds={0: [123]},
            planned_batches=["a", "b"],
        )

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")

    def check(self, **kwargs):
        args = dict(
            project_root=self.root, checkpoint_hashes=["checkpoint"], training_seeds={0: [123]}, batch_id="b"
        )
        args.update(kwargs)
        check_manifest(self.manifest, self.spec, **args)

    def test_clean_committed_source_or_verification_drift_is_rejected(self):
        for name in (
            "aaa/model.py",
            "tests/test_model.py",
            "tools/verify.py",
            "benchmarks/golden_seeds.json",
            "new_implementation.py",
        ):
            with self.subTest(name=name):
                self.write(name, "changed\n")
                self.commit()
                with self.assertRaises(FreezeMismatch):
                    self.check()
                self.git("reset", "--hard", "HEAD~1")

    def test_dependency_lock_drift_is_rejected(self):
        self.write("requirements-lock.txt", "changed\n")
        self.commit()
        with self.assertRaises(FreezeMismatch):
            self.check()

    def test_bookkeeping_commit_between_a_and_b_is_allowed(self):
        self.write("benchmarks/freeze_manifest.json", json.dumps(self.manifest.payload))
        self.write("results/new-attempt/summary.json", "{}")
        path = self.root / "benchmarks/confirmation_batches.json"
        payload = json.loads(path.read_text())
        payload["batches"][0].update(
            status="consumed", consumed_by=["run-a"], outcome="all_required_gates_pass"
        )
        path.write_text(json.dumps(payload))
        self.commit()
        self.check()

    def test_batch_identity_cannot_change_as_bookkeeping(self):
        path = self.root / "benchmarks/confirmation_batches.json"
        payload = json.loads(path.read_text())
        payload["batches"][0]["batch_id"] = "replacement"
        path.write_text(json.dumps(payload))
        self.commit()
        with self.assertRaises(FreezeMismatch):
            self.check()

    def test_missing_source_identity_is_rejected(self):
        self.manifest.payload["source"].pop("scientific_fingerprint", None)
        with self.assertRaises(FreezeMismatch):
            self.check()
