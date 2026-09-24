"""Regressions from the independent review of the Python-first transition."""

from __future__ import annotations

import copy
import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aaa.promotion import adjudicate, fixtures
from research.aaa_python import experiment, generator, recompute, spec
from research.aaa_python.episode import Action, BoundaryError, Environment


class ReviewRegressions(unittest.TestCase):
    def test_promotion_extreme_finite_values_fail_closed(self) -> None:
        contract = fixtures.contract([1], draws=1000)
        records = fixtures.primitives(contract)
        for record in records:
            record["value"] = 1e300 if record["arm"] == "reference" else 1e-300
        self.assertEqual(adjudicate(contract, records)["verdict"], "INVALID_EVIDENCE")
        records[0]["value"] = 10**1000
        self.assertEqual(adjudicate(contract, records)["verdict"], "INVALID_EVIDENCE")
        for record in records:
            record["value"] = 1e308
        self.assertEqual(adjudicate(contract, records)["verdict"], "DISAGREEMENT")

    def test_forged_view_cannot_expand_label_space(self) -> None:
        task = generator.build("development", "repair", [0])[0]
        env = Environment(spec.load())
        view = env.present(task)
        with self.assertRaises(BoundaryError):
            env.commit(replace(view, labels=(99,)), Action(99, 0.5))
        env.commit(view, Action(0, 0.5))
        with self.assertRaises(BoundaryError):
            env.reveal(replace(view, labels=(99,)))

    def test_checkpoint_binds_specification_and_train_pool(self) -> None:
        current_spec = spec.load()
        plan = experiment.Plan.from_spec(current_spec, quick=True)
        tasks = {"train": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            experiment.trained_learner(current_spec, plan, tasks, 0, "lexical", path)
            changed_tasks = {"train": {"syntax": generator.build("train", "syntax", [0])}}
            with self.assertRaisesRegex(experiment.RunError, "training pool differs"):
                experiment.trained_learner(current_spec, plan, changed_tasks, 0, "lexical", path)
            changed_spec = copy.deepcopy(current_spec)
            changed_spec["learner"]["init_scale"] *= 2
            with self.assertRaisesRegex(experiment.RunError, "specification or training pool differs"):
                experiment.trained_learner(changed_spec, plan, tasks, 0, "lexical", path)

    def test_partial_recomputation_is_not_a_pass(self) -> None:
        base = Path(__file__).resolve().parents[1] / "docs/evidence/aaa_python_v0/development.json"
        evidence = recompute.load_evidence(base.read_text(encoding="utf-8"))
        result = recompute.verify(evidence, None)
        self.assertEqual(result["verdict"], "SUMMARY_ONLY_NOT_VERIFIED")

    def test_nonstandard_record_json_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write('{"confidence": NaN}\n')
            with self.assertRaisesRegex(experiment.RunError, "non-standard JSON"):
                experiment.read_records(path)

    def test_tampered_brier_primitive_is_refused_with_original_records(self) -> None:
        base = Path(__file__).resolve().parents[1] / "docs/evidence/aaa_python_v0"
        evidence = recompute.load_evidence((base / "development.json").read_text(encoding="utf-8"))
        records = experiment.read_records(base / "development_records.jsonl.gz")
        cell = next(c for c in evidence["cells"] if c["arm"] == "online" and c["family"] == "syntax")
        cell["brier_sum"] += cell["n"] * 0.25
        plan = experiment.Plan(
            **{
                **evidence["plan"],
                "representations": tuple(evidence["plan"]["representations"]),
                "adaptation_families": tuple(evidence["plan"]["adaptation_families"]),
            }
        )
        evidence["summary"] = json.loads(
            json.dumps(experiment.summarize(evidence["cells"], plan, spec.load()))
        )
        result = recompute.verify(evidence, records)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(any("cell primitives" in problem for problem in result["problems"]))


if __name__ == "__main__":
    unittest.main()
