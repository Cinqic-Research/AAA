"""Retained aaa.1k.v2 evidence stays consistent with the repository.

Written after the v2 freeze, so it lives outside the v2 fingerprint (which
covers ``tests/test_aaa_1k_v2.py`` and ``tests/test_compute.py`` only). It
plays the role ``champion1 verify`` plays for ``aaa.1k.v1``: an edit to frozen
v2 source, a lock or the evidence it rests on fails here and in CI.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from research.aaa_1k_v2 import freeze, identities
from research.aaa_1k_v2.recompute import recompute

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/aaa_1k_v2"


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class FrozenV2EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
        cls.manifest = _load("freeze.json")

    def test_live_repository_matches_the_freeze(self) -> None:
        self.assertEqual(freeze.verify(self.manifest, ROOT, self.registry), [])

    def test_confirmation_is_bound_to_the_freeze_and_its_blocks_are_spent(self) -> None:
        confirmation = _load("confirmation.json")
        digest = hashlib.sha256((EVIDENCE / "freeze.json").read_bytes()).hexdigest()
        self.assertEqual(confirmation["freeze_sha256"], digest)
        self.assertEqual(confirmation["observer"], f"confirmation:{digest[:16]}")
        for block_id in self.manifest["confirmation_blocks"]:
            block = identities.find(self.registry, block_id)
            self.assertEqual(block["status"], "spent", block_id)
            self.assertEqual(block["observed_by"], confirmation["observer"], block_id)

    def test_recomputation_agrees_with_the_stored_outcome(self) -> None:
        confirmation = _load("confirmation.json")
        result = recompute(confirmation, self.manifest, self.registry)
        self.assertTrue(result["agrees"])
        self.assertEqual(result, _load("recomputation.json"))

    def test_capacity_blocks_are_used(self) -> None:
        for block in self.registry["blocks"]:
            if block["role"] == "capacity":
                self.assertEqual(block["status"], "used", block["block_id"])

    def test_phase_report_matches_the_evidence(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/write_aaa_1k_v2_report.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
