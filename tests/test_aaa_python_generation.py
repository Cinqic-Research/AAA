"""aaa.python.v0: deterministic generation, split separation, oracle-derived answers, the causal boundary."""

from __future__ import annotations

import dataclasses
import unittest

from research.aaa_python import checks, generator, oracle
from research.aaa_python.episode import VIEW_FIELDS, Action, BoundaryError, Environment, TaskView
from research.aaa_python.subset import validate

FAMILIES = ("syntax", "outcome", "output", "localize", "repair")


class GeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        for family in FAMILIES:
            first = generator.build("development", family, range(12))
            generator._pool.cache_clear()  # rebuild earlier pools too
            second = generator.build("development", family, range(12))
            self.assertEqual(first, second, family)

    def test_order_and_batching_do_not_change_a_task(self) -> None:
        together = generator.build("development", "repair", range(8))
        alone = [generator.build("development", "repair", [i])[0] for i in reversed(range(8))]
        self.assertEqual(together, list(reversed(alone)))

    def test_splits_are_disjoint_by_identity_and_by_source(self) -> None:
        for family in FAMILIES:
            train = generator.pool_hashes("train", family)
            development = generator.pool_hashes("development", family)
            probe = {t.source_sha256 for t in generator.build("probe", family, range(40))}
            self.assertFalse(train & development, family)
            self.assertFalse((train | development) & probe, family)
            ids = {t.task_id for t in generator.pool("train", family)} | {
                t.task_id for t in generator.pool("development", family)
            }
            self.assertEqual(len(ids), 1000)

    def test_indices_outside_a_declared_pool_are_refused(self) -> None:
        for split, index in (("development", 400), ("probe", 60), ("train", -1), ("attack", 200)):
            with self.assertRaises(generator.GenerationError):
                generator.build(split, "syntax", [index])

    def test_confirmation_identities_are_refused(self) -> None:
        for family in FAMILIES:
            with self.assertRaises(generator.ConfirmationNotAdmitted):
                generator.build("confirmation", family, [0])

    def test_slices_hold_out_structure_and_literals(self) -> None:
        train_templates = {t.template for t in generator.pool("train", "output")}
        self.assertFalse(train_templates & {"nested_loop", "list_walk"})
        development = generator.pool("development", "output")
        self.assertEqual(
            {t.slice for t in development}, {"in_distribution", "novel_literals", "novel_structure"}
        )
        for task in development:
            if task.slice == "novel_structure":
                self.assertIn(task.template, {"nested_loop", "list_walk"})

    def test_answers_are_what_cpython_says(self) -> None:
        for family in ("outcome", "output", "localize"):
            for task in generator.build("development", family, range(15)):
                validate(task.source)
                outcome = oracle.execute(task.source)
                expected = {
                    "outcome": "ok" if outcome.status == "ok" else outcome.exception,
                    "output": int(outcome.stdout.strip()) if outcome.status == "ok" else None,
                    "localize": outcome.line,
                }[family]
                self.assertEqual(task.answer, expected, task.task_id)
        for task in generator.build("development", "syntax", range(15)):
            status = oracle.check_syntax(task.source).status
            self.assertEqual(task.answer, "valid" if status == "valid" else "invalid")

    def test_families_meet_their_declared_constraints(self) -> None:
        for task in generator.pool("development", "output"):
            self.assertTrue(-50 <= task.answer <= 50)
        for task in generator.pool("development", "localize"):
            self.assertEqual(task.oracle["status"], "exception")
        for task in generator.pool("development", "repair"):
            passing = [i for i, r in enumerate(task.candidate_hidden_results) if all(r)]
            self.assertEqual(passing, [task.answer], task.task_id)
            self.assertEqual(len(task.candidates), 4)
            self.assertEqual(len(task.hidden_tests), 4)
            self.assertFalse(set(task.visible_tests) & set(task.hidden_tests))

    def test_labels_are_not_degenerate(self) -> None:
        for family in FAMILIES:
            answers = [t.answer for t in generator.pool("train", family)]
            self.assertGreater(len(set(answers)), 1, family)
            top = max(answers.count(a) for a in set(answers))
            self.assertLess(top / len(answers), 0.8, family)

    def test_faulty_programs_carry_no_fixed_marker_name(self) -> None:
        names = set()
        for task in generator.pool("train", "localize"):
            line = task.source.split("\n")[task.answer - 1].strip()
            names.add(line.split(" ")[0])
        self.assertGreater(len(names), 3)


class BoundaryTests(unittest.TestCase):
    def test_every_leakage_probe_passes(self) -> None:
        self.assertEqual(checks.failed(checks.leakage_checks()), [])

    def test_the_view_has_no_answer_attribute(self) -> None:
        task = generator.build("development", "output", [0])[0]
        view = Environment(__import__("research.aaa_python.spec", fromlist=["load"]).load()).present(task)
        self.assertEqual(tuple(f.name for f in dataclasses.fields(TaskView)), VIEW_FIELDS)
        for forbidden in ("answer", "oracle", "hidden_tests", "candidate_hidden_results", "task_id", "notes"):
            self.assertFalse(hasattr(view, forbidden), forbidden)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            view.source = "x"  # type: ignore[misc]

    def test_the_order_is_enforced(self) -> None:
        from research.aaa_python import spec

        tasks = generator.build("development", "outcome", range(2))
        env = Environment(spec.load())
        view = env.present(tasks[0])
        with self.assertRaises(BoundaryError):
            env.present(tasks[1])  # the previous task was never revealed
        with self.assertRaises(BoundaryError):
            env.reveal(view)
        with self.assertRaises(BoundaryError):
            env.commit(view, Action("ok", 1.5))  # malformed confidence
        env.commit(view, Action("ok", 0.5))
        env.reveal(view)
        with self.assertRaises(BoundaryError):
            env.reveal(view)
        stale = env.present(tasks[1])
        with self.assertRaises(BoundaryError):
            env.commit(view, Action("ok", 0.5))  # refers to the previous task
        env.commit(stale, Action("ok", 0.5))


if __name__ == "__main__":
    unittest.main()
