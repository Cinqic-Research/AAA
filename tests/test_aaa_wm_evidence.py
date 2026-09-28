"""Post-freeze evidence audit rejects plausible silent decision corruption."""

from __future__ import annotations

import copy
import unittest

from tools import check_aaa_wm_evidence as audit


class EvidenceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {}
        for kind, start, count, folder in audit.CASES:
            base = audit.ROOT / folder
            cls.cases[kind] = (
                start,
                count,
                folder,
                audit.strict_load(base / "freeze.json"),
                audit.strict_load(base / "confirmation.json"),
            )

    def check_mutation(self, kind: str, mutate: object, expected: str) -> None:
        start, count, folder, freeze, doc = self.cases[kind]
        modified = copy.deepcopy(doc)
        mutate(modified)
        problems = audit.verify(kind, start, count, folder, freeze=freeze, doc=modified)
        self.assertTrue(any(expected in p for p in problems), problems)

    def test_altered_ratio_with_same_verdict_fails(self) -> None:
        self.check_mutation(
            "opaque",
            lambda d: d["adjudications"]["C1_wm_over_learned_reference"].update(ratio=0.5),
            "ratio differs",
        )

    def test_altered_interval_with_same_verdict_fails(self) -> None:
        def change_lower(doc: dict) -> None:
            doc["adjudications"]["L2_language_model_over_rules"]["interval"][0] = 0.5

        self.check_mutation(
            "language",
            change_lower,
            "interval differs",
        )

    def test_missing_initialization_fails(self) -> None:
        self.check_mutation("language", lambda d: d["results"]["wms:online@lm"].pop("102"), "initialization")

    def test_invalid_bit_fails(self) -> None:
        self.check_mutation(
            "opaque",
            lambda d: d["results"]["wms:online"]["100"].update(
                bits="2" + d["results"]["wms:online"]["100"]["bits"][1:]
            ),
            "invalid correctness bits",
        )

    def test_task_order_fails(self) -> None:
        self.check_mutation("language", lambda d: d["tasks"].reverse(), "task identities")

    def test_changed_threshold_fails(self) -> None:
        self.check_mutation(
            "language",
            lambda d: d["adjudications"]["L2_language_model_over_rules"].update(threshold=0.99),
            "threshold differs",
        )

    def test_strict_json_rejects_nan(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"value": NaN}')
            with self.assertRaises(ValueError):
                audit.strict_load(path)


if __name__ == "__main__":
    unittest.main()
