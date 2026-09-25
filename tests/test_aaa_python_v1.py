"""Regression tests for ``aaa.python.v1``: exam, encoders, models, tool boundary, statistics, freeze.

Every test here uses training identities or synthetic inputs; none generates a
development, attack or confirmation identity (development-pool generation
needs the full earlier pools and is exercised by the phase CLI instead).
"""

from __future__ import annotations

import ast
import copy
import itertools
import math
import unittest
from unittest import mock

import numpy as np

from research.aaa_python import spec as v0_spec
from research.aaa_python.episode import Action, BoundaryError, view_of
from research.aaa_python_v1 import encoders, episode, freeze, generator, models, stats
from research.aaa_python_v1 import spec as spec_module
from research.aaa_python_v1.encoders import Encoder
from research.aaa_python_v1.models import CoreAgent, CoreModel, ModelConfig, StateError

SPEC = spec_module.load()
FAMILIES = ("syntax", "outcome", "output", "localize", "repair")


def _train(family: str, count: int = 6) -> list[generator.Task]:
    return generator.build("train", family, list(range(count)))


class SpecTests(unittest.TestCase):
    def test_execution_sections_equal_v0(self) -> None:
        v0 = v0_spec.load()
        for section in ("subset", "sandbox", "interpreter", "families"):
            self.assertEqual(SPEC[section], v0[section], section)

    def test_diverging_execution_environment_is_refused(self) -> None:
        bad = copy.deepcopy(SPEC)
        bad["sandbox"]["cpu_seconds"] = 99
        with self.assertRaises(spec_module.SpecError):
            spec_module.validate(bad)

    def test_name_pools_disjoint_and_valid(self) -> None:
        pools = SPEC["templates"]["names"]
        train, novel = set(pools["in_distribution"]["variables"]), set(pools["novel_names"]["variables"])
        self.assertFalse(train & novel)
        reserved = set(SPEC["subset"]["reserved_names"])
        for name in novel | {pools["novel_names"]["function"], pools["novel_names"]["parameter"]}:
            self.assertRegex(name, SPEC["subset"]["identifier_pattern"])
            self.assertNotIn(name, reserved)

    def test_development_layout_is_disjoint_and_inside_the_pool(self) -> None:
        layout = SPEC["splits"]["development_layout"]
        ranges = sorted(tuple(layout[k]) for k in ("tune", "evaluate", "adapt"))
        for (_, stop), (start, _) in itertools.pairwise(ranges):
            self.assertLessEqual(stop, start)
        self.assertLessEqual(ranges[-1][1], SPEC["splits"]["pool_per_family"]["development"])


class GeneratorTests(unittest.TestCase):
    def test_mutations_are_symmetric_and_never_negative(self) -> None:
        lines = [
            "    q = p * 2 + 3",
            "        return p - 11",
            "    if p > 4:",
            "        q += p - 0",
            "    return q - 1",
            "    if p <= 7:",
        ]
        for line in lines:
            for mutated in generator.mutations(line):
                self.assertIn(line, generator.mutations(mutated), (line, mutated))
                negative = [t for t in mutated.split(" ") if t.startswith("-") and t[1:].isdigit()]
                self.assertFalse(negative, mutated)

    def test_repair_candidates_are_symmetric_around_a_hidden_base(self) -> None:
        for task in _train("repair", 8):
            current = task.source.rstrip("\n").split("\n")[task.repair_line - 1]
            self.assertIn(current, task.candidates)
            self.assertNotEqual(task.candidates[task.answer], current)
            bases = set.intersection(*(set(generator.mutations(c)) for c in task.candidates))
            self.assertTrue(bases, "all candidates must share a one-mutation base")
            self.assertTrue(all(task.candidate_hidden_results[task.answer]))
            others = [r for i, r in enumerate(task.candidate_hidden_results) if i != task.answer]
            self.assertTrue(all(not all(r) for r in others))

    def test_generation_is_deterministic(self) -> None:
        first = [t.source_sha256 for t in _train("outcome", 5)]
        generator._pool.cache_clear()
        second = [t.source_sha256 for t in generator.build("train", "outcome", [0, 1, 2, 3, 4])]
        self.assertEqual(first, second)

    def test_localize_programs_always_have_two_faults(self) -> None:
        for index in range(10):
            d = generator.draft("train", "localize", index, 0, SPEC)
            assert d is not None
            self.assertEqual(len(d.notes["fault"]), 2)

    def test_novel_names_slice_uses_no_training_identifier(self) -> None:
        training = set(SPEC["templates"]["names"]["in_distribution"]["variables"]) | {"f", "p", "i", "j"}
        found = 0
        for index in range(60):
            if generator.slice_of("development", index, SPEC) != "novel_names":
                continue
            for family in ("output", "outcome", "repair"):
                d = generator.draft("development", family, index, 0, SPEC)
                if d is None:
                    continue
                names = {n.id for n in ast.walk(ast.parse(d.source)) if isinstance(n, ast.Name)}
                names |= {n.name for n in ast.walk(ast.parse(d.source)) if isinstance(n, ast.FunctionDef)}
                self.assertFalse(names & training, d.source)
                found += 1
        self.assertGreater(found, 5)

    def test_confirmation_is_refused_without_an_admission(self) -> None:
        with self.assertRaises(generator.ConfirmationNotAdmitted):
            generator.build("confirmation", "syntax", [0])
        with self.assertRaises(generator.ConfirmationNotAdmitted):
            generator.build("confirmation", "syntax", [0], admission=object())
        forged = freeze.Admission(manifest={"forged": True})
        with self.assertRaises(generator.ConfirmationNotAdmitted):
            generator.build("confirmation", "syntax", [0], admission=forged)

    def test_indices_outside_the_pool_or_duplicated_are_refused(self) -> None:
        with self.assertRaises(generator.GenerationError):
            generator.build("train", "syntax", [2400])
        with self.assertRaises(generator.GenerationError):
            generator.build("train", "syntax", [1, 1])


class EncoderTests(unittest.TestCase):
    def test_syntax_never_reaches_the_parser(self) -> None:
        tasks = _train("syntax", 6)
        with (
            mock.patch.object(ast, "parse", side_effect=AssertionError("parser used for syntax")),
            mock.patch("builtins.compile", side_effect=AssertionError("compile used for syntax")),
        ):
            for task in tasks:
                for name in encoders.ENCODERS:
                    encoders._encode_cached.cache_clear()
                    Encoder(name, 64).encode(view_of(task, 0, SPEC))

    def test_encodings_are_read_only_deterministic_and_normalized(self) -> None:
        for family in FAMILIES:
            task = _train(family, 1)[0]
            view = view_of(task, 0, SPEC)
            for name in encoders.ENCODERS:
                encoders._encode_cached.cache_clear()
                a = Encoder(name, 64).encode(view)
                encoders._encode_cached.cache_clear()
                b = Encoder(name, 64).encode(view)
                self.assertTrue(np.array_equal(a.program.value, b.program.value))
                self.assertFalse(a.program.value.flags.writeable)
                self.assertAlmostEqual(float(np.linalg.norm(a.program.value)), 1.0, places=12)
                self.assertTrue(np.all(a.program.index < 64))
                if family == "localize":
                    self.assertEqual(len(a.lines), task.source.rstrip("\n").count("\n") + 1)
                if family == "repair":
                    self.assertEqual(len(a.candidates), 4)

    def test_encoders_have_no_parameters(self) -> None:
        for name in encoders.ENCODERS:
            account = encoders.accounting(Encoder(name, 256))
            self.assertEqual(account["trainable_parameters"], 0)
            self.assertEqual(account["adaptive_state"], 0)
            self.assertFalse(account["evaluates_programs"])

    def test_alpha_renaming_is_invariant_to_identifier_choice(self) -> None:
        a = "x = 3\ny = x + 2\nprint(y)\n"
        b = "cnt = 3\nhold = cnt + 2\nprint(hold)\n"
        self.assertEqual(encoders.renamed_tokens(a), encoders.renamed_tokens(b))

    def test_flow_marks_undefined_use_statically(self) -> None:
        flow = encoders.flow_features("a = 1\nif a > 3:\n    b = ghost + 1\nprint(a)\n")
        assert flow is not None
        self.assertIn("f:use:undefined:0", flow[3])

    def test_matched_capacity_across_encoders(self) -> None:
        counts = {
            name: CoreModel(ModelConfig(name, 256, 16)).accounting()["trainable_parameters"]
            for name in encoders.ENCODERS
        }
        self.assertEqual(len(set(counts.values())), 1, counts)


class ModelTests(unittest.TestCase):
    def _perturbed(self, config: ModelConfig) -> CoreModel:
        model = CoreModel(config)
        rng = np.random.default_rng(5)
        for name in model.params:
            model.params[name] = model.params[name] + rng.normal(0, 0.3, model.params[name].shape)
        return model

    def test_finite_difference_gradients(self) -> None:
        tasks = {family: _train(family, 2) for family in FAMILIES}
        rng = np.random.default_rng(2)
        for hidden, output_head, localize_head, tool in itertools.product(
            (0, 4), ("onehot", "gauss"), ("onehot", "pointer"), (0, 2)
        ):
            model = self._perturbed(
                ModelConfig("e2", 24, hidden, output_head, localize_head, seed=1, tool_inputs=tool)
            )
            for family, items in tasks.items():
                if tool and family != "repair":
                    continue
                for task in items:
                    view = view_of(task, 0, SPEC)
                    encoding = model.encoder.encode(view)
                    tool_in = rng.normal(0, 1, (4, tool)) if tool else None
                    target, chosen = (1.0, 2) if family == "repair" else (task.answer, None)
                    grads = model.gradients(view, encoding, target, tool_in, chosen)
                    assert grads is not None
                    for name, array in model.params.items():
                        flat = array.reshape(-1)
                        for i in rng.choice(flat.size, size=min(3, flat.size), replace=False):
                            old = flat[i]
                            flat[i] = old + 1e-6
                            up = model.loss(view, encoding, target, tool_in, chosen)
                            flat[i] = old - 1e-6
                            down = model.loss(view, encoding, target, tool_in, chosen)
                            flat[i] = old
                            numeric = (up - down) / 2e-6
                            self.assertAlmostEqual(numeric, float(grads[name].reshape(-1)[i]), delta=1e-5)

    def test_state_round_trip_and_hash(self) -> None:
        config = ModelConfig("e1", 32, 4, "gauss", "pointer", momentum=0.9, learning_rate=0.05, seed=3)
        model = self._perturbed(config)
        model.velocity = {k: v + 0.1 for k, v in model.velocity.items()}
        model.updates = 7
        other = CoreModel(config)
        other.load_state(model.state_dict())
        self.assertEqual(other.state_hash(), model.state_hash())
        other.velocity["core.W"][0, 0] += 1e-9
        self.assertNotEqual(other.state_hash(), model.state_hash())

    def test_invalid_states_change_nothing(self) -> None:
        config = ModelConfig("e1", 32, 4)
        model = CoreModel(config)
        before = model.state_hash()
        good = model.state_dict()
        cases = []
        wrong_config = copy.deepcopy(good)
        wrong_config["config"]["init_scale"] = 0.5
        cases.append(wrong_config)
        nonfinite = copy.deepcopy(good)
        nonfinite["params"]["core.b"][0] = float("nan")
        cases.append(nonfinite)
        extra = copy.deepcopy(good)
        extra["params"]["ghost.W"] = [0.0]
        cases.append(extra)
        shape = copy.deepcopy(good)
        shape["params"]["core.W"] = [[0.0]]
        cases.append(shape)
        counter = copy.deepcopy(good)
        counter["updates"] = True
        cases.append(counter)
        for case in cases:
            with self.assertRaises(StateError):
                model.load_state(case)
            self.assertEqual(model.state_hash(), before)

    def test_clone_is_independent_and_reset_restores_everything(self) -> None:
        task = _train("syntax", 1)[0]
        agent = CoreAgent(CoreModel(ModelConfig("e1", 32, 4, momentum=0.5, seed=4)))
        agent.remember_initial()
        initial = agent.model.state_hash()
        env = episode.ToolEnvironment(SPEC)
        view = env.present(task)
        action = agent.act(view)
        env.commit(view, action)
        _, feedback = env.reveal(view)
        clone = agent.clone()
        self.assertTrue(agent.learn(view, action, feedback))
        self.assertNotEqual(agent.model.state_hash(), clone.model.state_hash())
        agent.reset_each_task = True
        agent.begin_task()
        self.assertEqual(agent.model.state_hash(), initial)
        self.assertEqual(agent.model.updates, 0)

    def test_accounting_counts_optimizer_state(self) -> None:
        plain = CoreModel(ModelConfig("e1", 256, 16)).accounting()
        momentum = CoreModel(ModelConfig("e1", 256, 16, momentum=0.9)).accounting()
        self.assertEqual(plain["optimizer_state"], 0)
        self.assertEqual(momentum["optimizer_state"], momentum["trainable_parameters"])
        self.assertEqual(plain["trainable_parameters"], sum(plain["per_block"].values()))

    def test_budget_search_is_close(self) -> None:
        for budget in (1000, 4000, 10000):
            hidden = models.hidden_for_budget(
                budget, encoder="e2", dimensions=256, output_head="gauss", localize_head="pointer"
            )
            count = CoreModel(ModelConfig("e2", 256, hidden, "gauss", "pointer")).accounting()[
                "trainable_parameters"
            ]
            self.assertLess(abs(count - budget) / budget, 0.15)

    def test_tool_model_accepts_families_without_tool_evidence(self) -> None:
        model = CoreModel(ModelConfig("e1", 32, 4, tool_inputs=2, seed=1))
        plain = CoreModel(ModelConfig("e1", 32, 4, seed=1))
        plain.params = {
            k: (v if k != "core.W" else model.params["core.W"][:, :32]) for k, v in model.params.items()
        }
        for family in ("syntax", "outcome"):
            task = _train(family, 1)[0]
            view = view_of(task, 0, SPEC)
            _, p_tool = model.distribution(view, model.encoder.encode(view))
            _, p_plain = plain.distribution(view, plain.encoder.encode(view))
            self.assertTrue(np.allclose(p_tool, p_plain))
        with self.assertRaises(ValueError):
            model.core(model.encoder.encode(view_of(_train("syntax", 1)[0], 0, SPEC)).program, np.zeros(3))

    def test_frozen_agent_never_updates(self) -> None:
        task = _train("outcome", 1)[0]
        agent = CoreAgent(CoreModel(ModelConfig("e0", 32, 0)))
        agent.update_enabled = False
        before = agent.model.state_hash()
        env = episode.ToolEnvironment(SPEC)
        view = env.present(task)
        action = agent.act(view)
        env.commit(view, action)
        _, feedback = env.reveal(view)
        self.assertFalse(agent.learn(view, action, feedback))
        self.assertEqual(agent.model.state_hash(), before)


class ToolBoundaryTests(unittest.TestCase):
    def test_tool_runs_only_visible_tests_before_commit(self) -> None:
        task = _train("repair", 1)[0]
        env = episode.ToolEnvironment(SPEC)
        view = env.present(task)
        seen: list[str] = []
        real = episode.run_many

        def capture(jobs, spec=None):  # type: ignore[no-untyped-def]
            seen.extend(source for _, source in jobs)
            return real(jobs, spec)

        episode._run.cache_clear()
        with mock.patch.object(episode, "run_many", side_effect=capture):
            results = env.run_visible_tests(view)
        self.assertEqual(len(results), 4)
        hidden_inputs = {i for i, _ in task.hidden_tests} - {i for i, _ in task.visible_tests}
        for source in seen:
            called = {int(x) for x in __import__("re").findall(r"print\(\w+\((-?\d+)\)\)", source)}
            self.assertFalse(called & hidden_inputs, source)
        self.assertEqual([e for _, e, _ in env.events], ["present", "tool"])
        env.commit(view, Action(0, 0.5))
        with self.assertRaises(BoundaryError):
            env.run_visible_tests(view)

    def test_tool_refuses_other_families(self) -> None:
        task = _train("output", 1)[0]
        env = episode.ToolEnvironment(SPEC)
        view = env.present(task)
        with self.assertRaises(BoundaryError):
            env.run_visible_tests(view)


class StatsTests(unittest.TestCase):
    def test_degenerate_and_resolution(self) -> None:
        zero = stats.crossed(np.zeros((3, 4)), seed=1, draws=1000, confidence=0.95)
        self.assertEqual(stats.resolved_sign(zero, disagreements=0), "DEGENERATE")
        positive = stats.crossed(
            np.full((3, 4), 0.2) + np.arange(12).reshape(3, 4) * 1e-3, seed=1, draws=1000, confidence=0.95
        )
        self.assertEqual(stats.resolved_sign(positive, disagreements=5), "POSITIVE")
        single = stats.crossed(np.zeros((1, 4)), seed=1, draws=1000, confidence=0.95)
        self.assertEqual(stats.resolved_sign(single), "INSUFFICIENT_EVIDENCE")
        with self.assertRaises(stats.StatsError):
            stats.crossed(np.array([[0.0, math.nan], [1.0, 2.0]]), seed=1, draws=1000, confidence=0.95)

    def test_holm_is_monotone_and_conservative(self) -> None:
        rng = np.random.default_rng(0)
        grids = {f"c{k}": rng.normal(k * 0.05, 0.1, size=(5, 8)) for k in range(5)}
        results = {name: {"mean": float(v.mean())} for name, v in grids.items()}
        out = stats.holm(results, grids, seed=3, draws=2000, alpha=0.05)
        for row in out.values():
            self.assertGreaterEqual(row["holm_adjusted_p"], row["p_value"])

    def test_variance_components_recover_structure(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(0, 1.0, size=(20, 1))[:, [0] * 30] + rng.normal(0, 0.1, size=(20, 30))
        parts = stats.variance_components(values)
        self.assertGreater(parts["initialization"], 10 * parts["stream"])


class FreezeTests(unittest.TestCase):
    def test_no_admission_without_a_committed_manifest(self) -> None:
        with mock.patch.object(freeze, "load_manifest", side_effect=freeze.FreezeError("none")):
            self.assertFalse(freeze.Admission(manifest={}).verify())


if __name__ == "__main__":
    unittest.main()


class TuningCheckpointTests(unittest.TestCase):
    def test_a_checkpoint_equals_a_model_trained_for_that_many_epochs(self) -> None:
        from research.aaa_python_v1 import experiment

        small = {family: _train(family, 3) for family in FAMILIES}
        tune = {family: _train(family, 6)[3:] for family in FAMILIES}
        arm = experiment.ArmSpec("t", "e1", 3, train_per_family=3)
        with (
            mock.patch.object(experiment, "train_tasks", return_value=small),
            mock.patch.object(experiment, "dev_range", side_effect=lambda _role, family: tune[family]),
            mock.patch.dict(experiment.DESIGN["tune"], {"epochs": [1, 2, 3]}),
        ):
            run = experiment.tune_job(arm, 0.1, 1000)
            direct = experiment.trained_agent(arm, 1000, 0.1, 2)
            accuracy = {}
            for family in FAMILIES:
                correct, _ = experiment.frozen_pass(direct.clone(), tune[family])
                accuracy[family] = sum(correct) / len(correct)
        self.assertEqual([c["epochs"] for c in run["checkpoints"]], [1, 2, 3])
        self.assertEqual(run["checkpoints"][1]["accuracy"], accuracy)


class RecomputeTests(unittest.TestCase):
    def _document(self) -> dict:
        from research.aaa_python_v1 import experiment, summarize

        design = experiment.DESIGN["evaluate"]
        n = design["streams"] * design["stream_length"]
        rng = np.random.default_rng(11)
        rows = []
        for arm, rate in (("a", 0.6), ("b", 0.5)):
            for init in range(design["initializations"]):
                fams = {}
                for family in FAMILIES:
                    fams[family] = {
                        mode: {"bits": experiment.bits(rng.random(n) < rate)} for mode in ("frozen", "online")
                    }
                rows.append({"arm": arm, "init": init, "families": fams})
        stage = {"stage": "synthetic", "evaluations": rows, "arms": {}}
        primary = [("a", "b", "frozen")]
        stage["primary"], stage["secondary"] = [list(c) for c in primary], []
        stage["summary"] = summarize.summarize(stage, primary)
        return {"stage": stage}

    def test_untouched_evidence_passes_and_tampering_fails(self) -> None:
        from research.aaa_python_v1 import experiment, recompute

        document = self._document()
        self.assertEqual(recompute.verify_stage(document)["verdict"], "PASS")
        tampered = copy.deepcopy(document)
        tampered["stage"]["summary"]["arms"]["a"]["syntax"]["frozen"]["mean"] += 0.01
        self.assertEqual(recompute.verify_stage(tampered)["verdict"], "FAIL")
        flipped = copy.deepcopy(document)
        payload = flipped["stage"]["evaluations"][0]["families"]["output"]["frozen"]
        values = experiment.unbits(payload["bits"])
        values[0] = not values[0]
        payload["bits"] = experiment.bits(values)
        self.assertEqual(recompute.verify_stage(flipped)["verdict"], "FAIL")
        dropped = copy.deepcopy(document)
        dropped["stage"]["evaluations"] = [r for r in dropped["stage"]["evaluations"] if r["init"] != 3]
        self.assertEqual(recompute.verify_stage(dropped)["verdict"], "FAIL")


class GoldenKeyTests(unittest.TestCase):
    def test_this_interpreter_reproduces_the_packaged_golden_keys(self) -> None:
        from research.aaa_python_v1 import golden

        self.assertEqual(golden.differences(), [])


class CliExitTests(unittest.TestCase):
    def test_fingerprint_outside_a_checkout_is_a_refusal(self) -> None:
        from research.aaa_python_v1 import cli, identity

        with mock.patch.object(identity, "_tracked", side_effect=identity.IdentityError("no checkout")):
            self.assertEqual(cli.main(["fingerprint"]), 2)

    def test_confirmation_without_a_freeze_is_a_refusal(self) -> None:
        from research.aaa_python_v1 import cli

        with mock.patch.object(freeze, "load_manifest", side_effect=freeze.FreezeError("none")):
            self.assertEqual(cli.main(["confirm", "--output", "/nonexistent/x.json"]), 2)
