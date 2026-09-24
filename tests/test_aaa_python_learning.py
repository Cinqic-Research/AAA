"""aaa.python.v0: learner state, controls, statistics, development runs and recomputation."""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from research.aaa_python import experiment, generator, recompute, representation
from research.aaa_python import spec as spec_module
from research.aaa_python.episode import Environment
from research.aaa_python.learners import StateError
from research.aaa_python.stats import (
    StatsError,
    balanced_accuracy,
    calibration,
    crossed_mean,
    grid,
    resolved_sign,
)

SPEC = spec_module.load()


def _learner(init: int = 0, rep: str = "lexical") -> Any:
    return experiment.make_learner(SPEC, init, rep)


class LearnerStateTests(unittest.TestCase):
    def test_clones_are_independent_and_state_round_trips_exactly(self) -> None:
        learner = _learner()
        clone = learner.clone()
        tasks = generator.build("development", "outcome", range(5))
        experiment.run_stream(clone, tasks, SPEC, feedback=True, learn=True)
        self.assertNotEqual(learner.state_hash(), clone.state_hash())
        restored = _learner()
        restored.load_state(json.loads(json.dumps(clone.state_dict())))
        self.assertEqual(restored.state_hash(), clone.state_hash())

    def test_malformed_state_is_refused_before_anything_changes(self) -> None:
        learner = _learner()
        before = learner.state_hash()
        good = learner.state_dict()
        mutations = [
            lambda s: s.update(schema="other"),
            lambda s: s.update(updates=-1),
            lambda s: s.update(updates=True),
            lambda s: s["weights"].update(syntax=[[0.0]]),
            lambda s: s["weights"]["outcome"][0].__setitem__(0, float("nan")),
            lambda s: s["weights"].update(extra=[[0.0]]),
            lambda s: s["config"].update(learning_rate=0.3),
        ]
        for mutate in mutations:
            state = copy.deepcopy(good)
            mutate(state)
            with self.assertRaises(StateError):
                learner.load_state(state)
            self.assertEqual(learner.state_hash(), before)

    def test_frozen_and_feedback_disabled_arms_never_update(self) -> None:
        tasks = generator.build("development", "localize", range(6))
        trained = _learner()
        for arm in ("frozen", "feedback_disabled"):
            agent, feedback = experiment._arm_agent(arm, trained)
            records = experiment.run_stream(agent, tasks, SPEC, feedback=feedback, learn=True)
            self.assertFalse(any(r["updated"] for r in records), arm)
            self.assertEqual(agent.state_hash(), trained.state_hash(), arm)
        online, _ = experiment._arm_agent("online", trained)
        records = experiment.run_stream(online, tasks, SPEC, feedback=True, learn=True)
        self.assertTrue(all(r["updated"] for r in records))

    def test_the_memory_disabled_control_forgets_between_tasks(self) -> None:
        trained = _learner()
        trained.remember_initial()
        tasks = generator.build("development", "syntax", range(4))
        agent, _ = experiment._arm_agent("reset_each_task", trained)
        experiment.run_stream(agent, tasks, SPEC, feedback=True, learn=True)
        agent.begin_task()
        self.assertEqual(agent.state_hash(), _reset_hash(trained))

    def test_repair_learns_only_from_the_chosen_candidate(self) -> None:
        learner = _learner()
        task = generator.build("development", "repair", [0])[0]
        env = Environment(SPEC)
        view = env.present(task)
        action = learner.act(view)
        env.commit(view, action)
        _, feedback = env.reveal(view)
        assert feedback is not None
        self.assertEqual(set(feedback.fields), {"chosen_candidate_hidden_results"})
        self.assertEqual(len(feedback.fields["chosen_candidate_hidden_results"]), 4)

    def test_parameters_and_state_are_accounted(self) -> None:
        counts = _learner().parameter_count()
        self.assertEqual(counts["trainable"], 153600)
        self.assertEqual(counts["trainable"], sum(counts["per_head"].values()))
        self.assertEqual(counts["optimizer_state"], 0)


def _reset_hash(trained: Any) -> str:
    fresh = trained.clone()
    assert fresh._initial is not None
    fresh.weights = {k: v.copy() for k, v in fresh._initial.items()}
    fresh.updates = 0
    return fresh.state_hash()


class RepresentationTests(unittest.TestCase):
    def test_representations_are_deterministic_and_normalized(self) -> None:
        source = "x = 3\nif x > 2:\n    y = x + 1\nprint(y)\n"
        for rep in representation.REPRESENTATIONS:
            a = representation.vector(source, rep, 1024)
            self.assertTrue(np.array_equal(a, representation.vector(source, rep, 1024)))
            self.assertTrue(np.all(np.isfinite(a)))

    def test_the_ast_representation_never_sees_a_syntax_verdict(self) -> None:
        self.assertEqual(representation.effective_representation("ast_nodes", "syntax"), "lexical")
        self.assertEqual(representation.effective_representation("ast_nodes", "fragment"), "lexical")
        self.assertEqual(representation.effective_representation("ast_nodes", "output"), "ast_nodes")
        with self.assertRaises(ValueError):
            representation.vector("if x = 3:\n", "ast_nodes", 64)
        learner = _learner(rep="ast_nodes")
        broken = "if x = 3:\n    pass\n"
        self.assertTrue(
            np.array_equal(learner._vector(broken, "syntax"), representation.vector(broken, "lexical", 1024))
        )

    def test_the_lexer_is_version_independent(self) -> None:
        self.assertEqual(
            representation.tokens('if x == 3:\n    y = "a"\n'),
            [
                ("KEYWORD", "if"),
                ("NAME", "x"),
                ("OP", "=="),
                ("NUMBER", "3"),
                ("OP", ":"),
                ("NEWLINE", "\n"),
                ("INDENT", "4"),
                ("NAME", "y"),
                ("OP", "="),
                ("STRING", '"a"'),
                ("NEWLINE", "\n"),
            ],
        )


class StatsTests(unittest.TestCase):
    def test_grids_refuse_gaps_duplicates_and_non_finite_values(self) -> None:
        cells = [{"init": i, "stream": s, "v": float(i + s)} for i in range(2) for s in range(2)]
        grid(cells, "v")
        for broken in (
            cells[:-1],
            [*cells, cells[0]],
            [{**cells[0], "v": float("nan")}, *cells[1:]],
            [{**cells[0], "v": True}, *cells[1:]],
        ):
            with self.assertRaises(StatsError):
                grid(broken, "v")

    def test_intervals_need_two_of_each_factor_and_signs_need_intervals(self) -> None:
        one = crossed_mean(np.ones((1, 4)), seed=1, draws=500, confidence=0.95)
        self.assertEqual(one["interval_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(resolved_sign(one), "INSUFFICIENT_EVIDENCE")
        positive = crossed_mean(
            np.full((3, 3), 0.2) + np.arange(9).reshape(3, 3) * 0.001, seed=1, draws=500, confidence=0.95
        )
        self.assertEqual(resolved_sign(positive), "POSITIVE")
        mixed = crossed_mean(np.array([[1.0, -1.0], [-1.0, 1.0]]), seed=1, draws=500, confidence=0.95)
        self.assertEqual(resolved_sign(mixed), "INCONCLUSIVE")

    def test_the_bootstrap_resamples_both_factors(self) -> None:
        # Initializations that disagree must widen the interval beyond a stream-only one.
        values = np.array([[0.0, 0.1, 0.0, 0.1], [1.0, 1.1, 1.0, 1.1], [0.5, 0.6, 0.5, 0.6]])
        crossed = crossed_mean(values, seed=3, draws=2000, confidence=0.95)
        self.assertGreater(crossed["upper"] - crossed["lower"], 0.3)

    def test_calibration_and_balanced_accuracy(self) -> None:
        self.assertAlmostEqual(calibration([1.0, 1.0], [True, False], 10)["brier"], 0.5)
        self.assertAlmostEqual(balanced_accuracy(["a", "a", "a", "b"], ["a", "a", "a", "a"]), 0.75)
        self.assertAlmostEqual(balanced_accuracy(["a", "b"], ["a", "b"]), 1.0)


class DevelopmentRunTests(unittest.TestCase):
    """A quick development run end to end, then every way its evidence can be tampered with."""

    evidence: ClassVar[dict[str, Any]]
    records: ClassVar[list[dict[str, Any]]]
    directory: ClassVar[tempfile.TemporaryDirectory[str]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoints = Path(cls.directory.name) / "ckpt"
        cls.evidence, cls.records = experiment.develop(
            quick=True, checkpoints=cls.checkpoints, log=lambda _: None
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_the_evidence_is_strict_json_and_labelled_development(self) -> None:
        text = json.dumps(self.evidence, allow_nan=False)
        loaded = recompute.load_evidence(text)
        self.assertIn("not confirmation", loaded["status"])
        self.assertTrue(loaded["plan"]["quick"])
        with self.assertRaises(recompute.RecomputeError):
            recompute.load_evidence(text.replace('"n": 6', '"n": NaN', 1))

    def test_independent_recomputation_passes(self) -> None:
        result = recompute.verify(self.evidence, self.records)
        self.assertEqual(result["verdict"], "PASS", result["problems"])
        self.assertGreater(result["tasks_reexecuted"], 50)

    def test_every_arm_and_design_is_present(self) -> None:
        arms = {c["arm"] for c in self.evidence["cells"]}
        for arm in (
            "online",
            "frozen",
            "feedback_disabled",
            "reset_each_task",
            "uniform",
            "majority",
            "lookup",
            "heuristic",
            "online@bytes",
            "frozen@ast_nodes",
            "adapt:changed:online",
            "adapt:control:frozen",
            "probe:before",
            "probe:after_changed",
        ):
            self.assertIn(arm, arms)
        summary = self.evidence["summary"]["families"]
        self.assertEqual(summary["syntax"]["arms"]["lookup"]["lookup_hit_rate"], 0.0)
        for family in SPEC["development"]["adaptation"]["families"]:
            self.assertIn("difference_of_differences", summary[family]["adaptation"])

    def test_tampered_evidence_is_refused(self) -> None:
        def tamper_record(field: str, value: Any) -> list[dict[str, Any]]:
            records = copy.deepcopy(self.records)
            records[7][field] = value
            return records

        for records in (
            tamper_record("correct", not self.records[7]["correct"]),
            tamper_record("truth", "not-the-truth"),
            tamper_record("confidence", 1.5),
            self.records[:-1],
        ):
            self.assertEqual(recompute.verify(self.evidence, records)["verdict"], "FAIL")
        cells = copy.deepcopy(self.evidence)
        cells["cells"][0]["correct"] += 1
        self.assertEqual(recompute.verify(cells, None)["verdict"], "FAIL")
        summary = copy.deepcopy(self.evidence)
        summary["summary"]["families"]["syntax"]["arms"]["online"]["brier"] = 0.0
        self.assertEqual(recompute.verify(summary, None)["verdict"], "FAIL")

    def test_summary_floats_tolerate_final_bit_rounding_but_nothing_else(self) -> None:
        # CPython 3.12 changed float sum() to compensated summation (AAA-186).
        stored = self.evidence["summary"]
        within = copy.deepcopy(stored)
        within["families"]["syntax"]["arms"]["online"]["brier"] *= 1 + 3e-16
        self.assertEqual(recompute.summary_differences(stored, within), [])
        for mutate in (
            lambda s: s["families"]["syntax"]["arms"]["online"].__setitem__(
                "brier", s["families"]["syntax"]["arms"]["online"]["brier"] * (1 + 1e-9)
            ),
            lambda s: s["families"]["syntax"]["arms"]["online"].__setitem__("brier", float("nan")),
            lambda s: s["families"]["syntax"]["contrasts"]["online - frozen"].__setitem__(
                "resolved_sign", "POSITIVE"
            ),
            lambda s: s["families"]["syntax"]["arms"]["online"].__setitem__("tasks", 1),
            lambda s: s["families"]["syntax"]["arms"].pop("online"),
        ):
            changed = copy.deepcopy(stored)
            mutate(changed)
            self.assertNotEqual(recompute.summary_differences(stored, changed), [])

    def test_pre_312_float_summation_still_recomputes(self) -> None:
        import functools
        import operator
        from unittest import mock

        naive = functools.partial(functools.reduce, operator.add)  # left-to-right, as before CPython 3.12

        def left_to_right(values: Any, start: Any = 0) -> Any:
            return naive(values, start)

        plan = experiment.Plan.from_spec(SPEC, quick=True)
        with mock.patch.object(experiment, "sum", left_to_right, create=True):
            emulated = json.loads(json.dumps(experiment.summarize(self.evidence["cells"], plan, SPEC)))
        self.assertEqual(recompute.summary_differences(self.evidence["summary"], emulated), [])

    def test_a_recorded_truth_that_cpython_disputes_is_refused_even_with_a_matching_digest(self) -> None:
        records = copy.deepcopy(self.records)
        target = next(r for r in records if r["family"] == "output")
        for record in records:
            if record["task_id"] == target["task_id"]:
                record["truth"] = 49 if record["truth"] != 49 else 48
                record["correct"] = (not record["abstain"]) and record["answer"] == record["truth"]
        evidence = copy.deepcopy(self.evidence)
        evidence["records"] = {"count": len(records), "sha256": experiment.records_sha256(records)}
        evidence["cells"] = experiment.cells_from_records(records, SPEC)
        plan = experiment.Plan.from_spec(SPEC, quick=True)
        evidence["summary"] = experiment.summarize(evidence["cells"], plan, SPEC)
        result = recompute.verify(evidence, records)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(any("fresh CPython" in p for p in result["problems"]))

    def test_resuming_from_checkpoints_reproduces_the_run_exactly(self) -> None:
        again, _ = experiment.develop(quick=True, checkpoints=self.checkpoints, log=lambda _: None)
        self.assertEqual(again["records"], self.evidence["records"])
        self.assertEqual(again["summary"], self.evidence["summary"])

    def test_a_checkpoint_from_another_plan_or_with_a_wrong_hash_is_refused(self) -> None:
        plan = experiment.Plan.from_spec(SPEC, quick=True)
        path = next(self.checkpoints.glob("init-0-lexical.json"))
        saved = json.loads(path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / path.name
            for change in ({"plan": {**saved["plan"], "draws": 1}}, {"state_hash": "0" * 64}):
                target.write_text(json.dumps({**saved, **change}), encoding="utf-8")
                with self.assertRaises(experiment.RunError):
                    experiment.trained_learner(SPEC, plan, {"train": {}}, 0, "lexical", Path(other))

    def test_the_plan_matches_the_record_schema(self) -> None:
        self.assertEqual(set(self.evidence["plan"]), {f.name for f in dataclasses.fields(experiment.Plan)})


class CommandLineTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "research.aaa_python", *args], capture_output=True, text=True, check=False
        )

    def test_refusals_exit_2(self) -> None:
        self.assertEqual(self._run("confirm").returncode, 2)
        self.assertEqual(self._run("generate", "--split", "confirmation", "--count", "1").returncode, 2)
        self.assertEqual(self._run("golden", "--write").returncode, 2)
        refused = self._run(
            "develop", "--quick", "--device", "cuda", "--output", "x.json", "--records", "x.gz"
        )
        self.assertEqual(refused.returncode, 2)
        self.assertIn("CPU only", refused.stderr)

    def test_identity_commands(self) -> None:
        spec_hash = self._run("spec-hash")
        self.assertEqual(spec_hash.returncode, 0)
        self.assertIn(spec_module.spec_hash(), spec_hash.stdout)
        self.assertEqual(self._run("fingerprint").returncode, 0)


if __name__ == "__main__":
    unittest.main()


class IdentityAndProvenanceTests(unittest.TestCase):
    def test_the_fingerprint_covers_the_phase_and_excludes_its_evidence(self) -> None:
        from research.aaa_python import identity

        record = identity.fingerprint()
        self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
        files = set(record["files"])
        for name in (
            "research/aaa_python/generator.py",
            "aaa/promotion/contract.py",
            "docs/aaa_python_protocol.md",
            "requirements-lock.txt",
        ):
            self.assertIn(name, files)
        self.assertFalse(any(name.startswith("docs/evidence/") for name in files))

    def test_provenance_is_complete_and_refuses_other_devices(self) -> None:
        from research.aaa_python import identity

        record = identity.provenance("cpu")
        self.assertTrue(record["captured_before_run"])
        self.assertEqual(record["compute"]["resolved_device"], "cpu")
        self.assertIn(record["source"]["dirty"], (True, False))
        self.assertRegex(record["source"]["commit"], r"^[0-9a-f]{40}$")
        for device in ("cuda", "cuda:0", "auto"):
            with self.assertRaises(ValueError):
                identity.provenance(device)


class InProcessCommandTests(unittest.TestCase):
    def test_commands_and_refusals(self) -> None:
        import contextlib
        import io

        from research.aaa_python import cli

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["spec-hash"]), 0)
            self.assertEqual(cli.main(["fingerprint"]), 0)
            self.assertEqual(cli.main(["subset"]), 0)
            self.assertEqual(cli.main(["audit"]), 0)
            self.assertEqual(cli.main(["golden"]), 0)
            self.assertEqual(cli.main(["generate", "--family", "repair", "--count", "3", "--show"]), 0)
            self.assertEqual(cli.main(["generate", "--count", "0"]), 2)
            self.assertEqual(cli.main(["generate", "--split", "confirmation"]), 2)
            self.assertEqual(cli.main(["golden", "--write"]), 2)
            self.assertEqual(cli.main(["confirm"]), 2)
            self.assertEqual(
                cli.main(["develop", "--quick", "--device", "cuda", "--output", "x", "--records", "y"]), 2
            )
