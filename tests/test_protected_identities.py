"""Historical scientific identities stay fixed while the repository moves on."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_protected_identities", ROOT / "tools/check_protected_identities.py"
)
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


class ProtectedIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = json.loads((ROOT / tool.EXPECTED).read_text(encoding="utf-8"))
        cls.live = tool.live_identities()

    def test_the_live_tree_matches_every_protected_identity(self) -> None:
        self.assertEqual(tool.compare(self.expected, self.live), [])

    def test_the_known_historical_hashes_are_the_recorded_ones(self) -> None:
        identities = self.expected["identities"]
        self.assertTrue(identities["aaa_1k_v1_phase_fingerprint"].startswith("5ce6e019"))
        self.assertTrue(identities["aaa_1k_v2_phase_fingerprint"].startswith("e5b11bdb"))
        self.assertTrue(identities["benchmark_v2_1_specification"].startswith("f8e1090b"))
        self.assertTrue(identities["observation_noise_v1_1_protocol"].startswith("546e2434"))

    def test_retained_evidence_and_historical_reports_are_covered(self) -> None:
        files = self.expected["files"]
        for name in (
            "docs/evidence/aaa_1k_v2/confirmation.json",
            "docs/evidence/aaa_1k_v2/freeze.json",
            "docs/evidence/aaa1k_loop_0006/champion_1.json",
            "docs/evidence/aaa1k_loop_0001/champion_0.json",
            "results/final/summary.json",
            "docs/aaa_1k_v2_report.md",
            "benchmarks/aaa1k_v2_identity_registry.json",
        ):
            self.assertIn(name, files)

    def test_a_changed_evidence_byte_is_reported(self) -> None:
        live = copy.deepcopy(self.live)
        live["files"]["docs/evidence/aaa_1k_v2/confirmation.json"] = "0" * 64
        problems = tool.compare(self.expected, live)
        self.assertEqual(len(problems), 1)
        self.assertIn("confirmation.json", problems[0])

    def test_a_removed_evidence_file_is_reported(self) -> None:
        live = copy.deepcopy(self.live)
        del live["files"]["results/final/summary.json"]
        self.assertTrue(any("no longer tracked" in p for p in tool.compare(self.expected, live)))

    def test_a_moved_fingerprint_is_reported(self) -> None:
        live = copy.deepcopy(self.live)
        live["identities"]["aaa_1k_v1_phase_fingerprint"] = "e9e65b0c"
        self.assertTrue(any("aaa_1k_v1" in p for p in tool.compare(self.expected, live)))

    def test_an_unknown_expectation_schema_is_refused(self) -> None:
        self.assertEqual(len(tool.compare({"schema": "other"}, self.live)), 1)


if __name__ == "__main__":
    unittest.main()
