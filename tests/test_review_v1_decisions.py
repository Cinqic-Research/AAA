"""Regression checks for the post-freeze v1 decision audit."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools.check_aaa_python_v1_decisions import DEFAULT, verify


class DecisionAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original = json.loads(Path(DEFAULT).read_text())

    def test_retained_confirmation(self) -> None:
        self.assertEqual(verify(self.original), [])

    def test_mutated_decision_fields_are_rejected(self) -> None:
        for name, change in (
            ("plan", lambda d: d["stage"]["primary"].clear()),
            (
                "adjudication",
                lambda d: d["stage"]["adjudications"]["scale_over_4k"].update(verdict="PROMOTE"),
            ),
            (
                "capacity",
                lambda d: d["stage"]["capacity_verdict"].update(
                    verdict="SCALE_JUSTIFIED_PENDING_ADAPTATION_AND_PLASTICITY"
                ),
            ),
            (
                "holm",
                lambda d: d["stage"]["summary"]["contrasts"]["syntax: 10k@e2 - 4k@e2 [frozen]"][
                    "holm"
                ].update(holm_adjusted_p=0.0),
            ),
        ):
            with self.subTest(name=name):
                changed = copy.deepcopy(self.original)
                change(changed)
                self.assertTrue(verify(changed))


if __name__ == "__main__":
    unittest.main()
