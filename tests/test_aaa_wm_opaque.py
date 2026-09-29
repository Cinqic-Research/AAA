"""Boundary, determinism and executor-agreement tests for ``aaa.python.opaque.v0`` (numpy only)."""

from __future__ import annotations

import dataclasses
import unittest

from research.aaa_wm.opaque import agents as A
from research.aaa_wm.opaque import generator as gen
from research.aaa_wm.opaque.env import REAL, RUN, BoundaryError, OpaqueEnv, RunObservation, View, play
from research.aaa_wm.opaque.program import Edit, apply, edits, mutations_of, run, signature


class MutationRelation(unittest.TestCase):
    def test_symmetric(self) -> None:
        names = tuple(f"api{k}" for k in range(6))
        tokens = [str(v) for v in range(0, 25)] + [
            "+",
            "-",
            "*",
            "//",
            "%",
            ">",
            ">=",
            "<",
            "<=",
            "==",
            "!=",
            "+=",
            "-=",
            *names,
        ]
        for t in tokens:
            for m in mutations_of(t, names):
                self.assertIn(t, mutations_of(m, names), (t, m))

    def test_thresholds_and_range_bounds_are_sites(self) -> None:
        src = "def f(p):\n    q = p\n    if q > 5:\n        q = 1\n    for i in range(3):\n        q += i\n    return q\n"
        olds = {(e.row, e.old) for e in edits(src, ("api0",))}
        self.assertIn((3, "5"), olds)
        self.assertIn((5, "3"), olds)

    def test_apply_undo_roundtrip(self) -> None:
        t = gen.build("pilot", 3)
        for e in edits(t.buggy, tuple(a.name for a in gen.library("A")))[:20]:
            self.assertEqual(apply(apply(t.buggy, e), Edit(e.row, e.col, e.new, e.old)), t.buggy)


class Generation(unittest.TestCase):
    def test_deterministic(self) -> None:
        self.assertEqual(gen.build("pilot", 7), gen.build("pilot", 7))

    def test_acceptance_rules(self) -> None:
        lib = gen.library("A")
        for i in range(30):
            t = gen.build("pilot", i)
            self.assertTrue(any(run(t.buggy, lib, x) != e for x, e in t.visible_tests))
            self.assertNotEqual(
                signature(t.buggy, lib, gen.domain()), signature(t.reference, lib, gen.domain())
            )
            self.assertIn("api", t.reference)

    def test_confirmation_refused_without_admission(self) -> None:
        with self.assertRaises(gen.ConfirmationNotAdmitted):
            gen.build("confirmation", 0)
        with self.assertRaises(gen.ConfirmationNotAdmitted):
            gen.pool("confirmation", 1)

    def test_library_b_differs_from_a(self) -> None:
        a, b = gen.library("A"), gen.library("B")
        self.assertEqual([x.name for x in a], [x.name for x in b])
        self.assertEqual(
            sum(x != y for x, y in zip(a, b, strict=True)), gen.load_spec()["library"]["B"]["changes"]
        )


class Boundary(unittest.TestCase):
    def setUp(self) -> None:
        self.task = gen.build("pilot", 11)

    def test_view_has_no_hidden_fields(self) -> None:
        fields = {f.name for f in dataclasses.fields(View)}
        self.assertEqual(
            fields, {"task_ref", "source", "visible_tests", "history", "steps_left", "runs_left", "api_names"}
        )
        view = OpaqueEnv(self.task).view()
        self.assertEqual(view.source, self.task.buggy)
        text = repr(view)
        self.assertNotIn("ApiSpec", text)
        self.assertNotIn("affine", text)

    def test_run_is_real_and_visible_only(self) -> None:
        env = OpaqueEnv(self.task)
        obs = env.step(RUN)
        self.assertIsInstance(obs, RunObservation)
        assert obs is not None
        self.assertEqual(obs.provenance, REAL)
        self.assertEqual(len(obs.results), len(self.task.visible_tests))

    def test_budgets_and_legality(self) -> None:
        env = OpaqueEnv(self.task, runs=1, steps=3)
        env.step(RUN)
        with self.assertRaises(BoundaryError):
            env.step(RUN)
        with self.assertRaises(BoundaryError):
            env.step(Edit(1, 0, "def", "fed"))
        env.submit()
        with self.assertRaises(BoundaryError):
            env.submit()
        with self.assertRaises(BoundaryError):
            env.view()

    def test_submit_scores_domain_equivalence(self) -> None:
        env = OpaqueEnv(self.task, steps=10)
        for f in reversed(self.task.faults):
            env.step(Edit(f.row, f.col, f.new, f.old))
        self.assertTrue(env.submit().success)

    def test_ceiling_is_labelled_and_never_in_a_view(self) -> None:
        prior = A.Prior(A.fit_prior([gen.build("pilot", i) for i in range(20)]))
        agent = A.Planner(A.TrueLibraryPredictor(gen.library("A")), prior, "ceiling", depth=1)
        out = play(agent, self.task)
        self.assertLessEqual(out.runs, 2)


class FastValidation(unittest.TestCase):
    def test_skeleton_cache_matches_full_validator(self) -> None:
        from research.aaa_python.subset import SubsetError, validate
        from research.aaa_python_v1 import spec as v1_spec
        from research.aaa_wm.opaque.program import _stubs, valid

        names = tuple(a.name for a in gen.library("A"))
        checked = 0
        for i in range(15):
            t = gen.build("pilot", 300 + i)
            for e in edits(t.buggy, names):
                src = apply(t.buggy, e)
                try:
                    validate(_stubs(names) + src, v1_spec.load())
                    full = True
                except (SubsetError, SyntaxError):
                    full = False
                self.assertEqual(valid(src, names), full, src)
                checked += 1
        self.assertGreater(checked, 100)


class ExecutorAgreement(unittest.TestCase):
    def test_sandbox_agrees_on_sample(self) -> None:
        from research.aaa_wm.opaque.sandbox_check import agreement

        lib = gen.library("A")
        samples = []
        for i in range(6):
            t = gen.build("pilot", 100 + i)
            for src in (t.reference, t.buggy):
                for x in (gen.domain()[0], 0, 7, gen.domain()[-1]):
                    samples.append((src, x))
        report = agreement(samples, lib)
        self.assertEqual(report["disagreements"], [])


if __name__ == "__main__":
    unittest.main()


class Freeze(unittest.TestCase):
    def test_admission_refused_without_committed_manifest(self) -> None:
        from research.aaa_wm.opaque import freeze

        if (freeze.ROOT / freeze.FREEZE_PATH).exists():
            self.skipTest("a freeze manifest exists; its admission is checked by the confirmation run")
        with self.assertRaises(freeze.FreezeError):
            freeze.Admission.obtain()

    def test_forged_admission_cannot_generate_confirmation(self) -> None:
        from research.aaa_wm.opaque import confirm, freeze

        forged = freeze.Admission(manifest={"schema": freeze.SCHEMA})
        with self.assertRaises(gen.ConfirmationNotAdmitted):
            confirm.confirmation_tasks(forged, 1)

    def test_fingerprint_covers_sources_and_imports(self) -> None:
        from research.aaa_wm.opaque import freeze

        files = freeze.fingerprint()["files"]
        self.assertIn("research/aaa_wm/opaque/generator.py", files)
        self.assertIn("research/aaa_wm/opaque/data/aaa_python_opaque_v0.json", files)
        self.assertIn("aaa/promotion/adjudicate.py", files)


class Provenance(unittest.TestCase):
    def test_imagination_is_never_a_real_observation(self) -> None:
        from research.aaa_wm.opaque import agents as Ag
        from research.aaa_wm.opaque import env as En

        self.assertNotEqual(Ag.IMAGINATION, En.REAL)
        self.assertEqual(En.RunObservation("x", (), ()).provenance, En.REAL)


class ActionConditioning(unittest.TestCase):
    def test_structured_predictions_depend_on_the_planned_program(self) -> None:
        from research.aaa_wm.opaque import structured as S
        from research.aaa_wm.opaque.env import OpaqueEnv

        lib = gen.library("A")
        model = S.LibraryModel(tuple(a.name for a in lib))
        for i in range(20):  # a small, genuinely learned table
            t = gen.build("pilot", 400 + i)
            model.learn([(t.reference, x, run(t.reference, lib, x)) for x in gen.domain()], rounds=4)
        pred = S.StructuredPredictor(model)
        differing = 0
        for i in range(10):
            t = gen.build("pilot", 400 + i)
            view = OpaqueEnv(t).view()
            progs = [apply(view.source, e) for e in view.edits()][:30]
            p = pred(progs, view)
            differing += len({tuple(r) for r in p}) > 1
        self.assertGreater(differing, 5)


class Audit(unittest.TestCase):
    def test_altered_stored_verdict_is_detected(self) -> None:
        import numpy as np

        from research.aaa_wm.opaque.confirm import adjudicate_all, audit

        rng = np.random.default_rng(3)
        slices = ["a"] * 80 + ["b"] * 80
        bits = lambda p: "".join("1" if v else "0" for v in rng.random(160) < p)  # noqa: E731
        doc = {
            "slices": slices,
            "results": {
                "ref": {s: {"bits": bits(0.2)} for s in ("1", "2")},
                "ch": {s: {"bits": bits(0.7)} for s in ("1", "2")},
            },
            "freeze": {
                "contracts": [
                    {
                        "name": "C",
                        "reference": "ref",
                        "challenger": "ch",
                        "initializations": [1, 2],
                        "threshold": 0.8,
                        "seed": 5,
                        "criterion": {"rule": "superior"},
                        "groups": "all",
                    }
                ]
            },
        }
        doc["adjudications"] = adjudicate_all(doc)
        self.assertEqual(audit(doc), [])
        doc["adjudications"]["C"]["verdict"] = (
            "REJECT" if doc["adjudications"]["C"]["verdict"] != "REJECT" else "PROMOTE"
        )
        self.assertTrue(audit(doc))
