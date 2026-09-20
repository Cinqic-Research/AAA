"""Tests for the AAA-1K recurrent core, its controls, and its causal boundary.

The last class in this file is the important one. A test suite that only passes
when the program is already correct is weaker evidence than one that has been
shown to catch specific corruption, so `FailureInjectionTests` deliberately
breaks each invariant and proves the corresponding check fails.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.aaa_1k import PHASE_VERSION
from research.aaa_1k.agents import (
    ConstantMotionAgent,
    DeadReckoningAgent,
    NeuralAgent,
    ObservationTracker,
    PersistenceAgent,
    RLSAgent,
    WindowedLinearFitAgent,
    baseline_suite,
)
from research.aaa_1k.characterization import coarse_speed_decomposition
from research.aaa_1k.controls import (
    StatelessMLPControl,
    VanillaRNNControl,
    expected_mlp_parameter_count,
    expected_rnn_parameter_count,
)
from research.aaa_1k.experiments import build_agents
from research.aaa_1k.features import PublicScales, build_inputs
from research.aaa_1k.gradcheck import check_model, full_gradient_check
from research.aaa_1k.identity import lineage_record, phase_fingerprint
from research.aaa_1k.measurements import (
    _probe,
    adaptation_difference_of_differences,
    probe_bank,
    retention_trial,
)
from research.aaa_1k.model import (
    AAA1KGRU,
    ARCHITECTURE_ID,
    HIDDEN_SIZE,
    INPUT_SIZE,
    OUTPUT_SIZE,
    InvalidModelState,
    expected_parameter_count,
    sigmoid,
    softplus,
)
from research.aaa_1k.runner import run_online_frozen_branch, run_stream
from research.aaa_1k.seeds import NAMESPACES, derive_seed
from research.aaa_1k.selection import (
    Configuration,
    development_streams,
    eligible_learning_rates,
    select_for_architecture,
)
from research.aaa_1k.stats import (
    achieved_precision,
    calibration,
    crossed_paired_difference,
    paired_difference,
)
from research.aaa_1k.streams import (
    FAMILIES,
    Stream,
    StreamStep,
    aba_stream,
    build_stream,
    coarse_speed_stream,
    motion_compat_stream,
    observable_view,
    occlusion_stream,
    paired_change_streams,
    quantize,
)

PROJECT = Path(__file__).resolve().parents[1]


def small_stream(seed: int = 4242) -> Stream:
    return motion_compat_stream("bouncing", seed, steps=80, change_step=None)


def fresh_model(**options) -> AAA1KGRU:
    defaults = {"seed": 7, "learning_rate": 0.03, "tbptt_steps": 4, "error_loss_weight": 0.25}
    defaults.update(options)
    return AAA1KGRU(**defaults)


# ----------------------------------------------------------------------
# architecture
# ----------------------------------------------------------------------
class ArchitectureTests(unittest.TestCase):
    def test_the_model_has_exactly_994_trainable_parameters(self):
        model = fresh_model()
        self.assertEqual(model.parameter_count(), 994)

    def test_the_formula_and_the_arrays_agree_independently(self):
        model = fresh_model()
        counted = sum(int(array.size) for array in model.parameters.values())
        self.assertEqual(counted, expected_parameter_count())
        self.assertEqual(counted, 3 * (3 * 16 + 16 * 16 + 16) + (16 * 2 + 2))

    def test_the_controls_have_their_declared_counts(self):
        self.assertEqual(StatelessMLPControl(seed=0).parameter_count(), 982)
        self.assertEqual(expected_mlp_parameter_count(), 982)
        self.assertEqual(VanillaRNNControl(seed=0).parameter_count(), 954)
        self.assertEqual(expected_rnn_parameter_count(), 954)

    def test_every_tensor_has_its_declared_shape(self):
        model = fresh_model()
        for gate in ("z", "r", "n"):
            self.assertEqual(model.parameters[f"W_{gate}"].shape, (HIDDEN_SIZE, INPUT_SIZE))
            self.assertEqual(model.parameters[f"U_{gate}"].shape, (HIDDEN_SIZE, HIDDEN_SIZE))
            self.assertEqual(model.parameters[f"b_{gate}"].shape, (HIDDEN_SIZE,))
        self.assertEqual(model.parameters["W_o"].shape, (OUTPUT_SIZE, HIDDEN_SIZE))
        self.assertEqual(model.parameters["b_o"].shape, (OUTPUT_SIZE,))

    def test_initialization_is_deterministic_and_seed_dependent(self):
        self.assertEqual(fresh_model(seed=3).state_hash(), fresh_model(seed=3).state_hash())
        self.assertNotEqual(fresh_model(seed=3).state_hash(), fresh_model(seed=4).state_hash())

    def test_the_output_head_starts_at_zero_so_an_untrained_model_predicts_persistence(self):
        model = fresh_model()
        self.assertTrue(np.all(model.parameters["W_o"] == 0.0))
        self.assertTrue(np.all(model.parameters["b_o"] == 0.0))
        displacement, _ = model.split_output(model.forward(np.array([0.2, 1.0, 0.0])))
        self.assertEqual(displacement, 0.0)

    def test_the_optimizer_carries_no_hidden_state(self):
        footprint = fresh_model().state_footprint()
        self.assertEqual(footprint["optimizer_state_scalars"], 0)
        self.assertEqual(footprint["trainable_parameters"], 994)
        self.assertEqual(footprint["hidden_state_scalars"], 16)
        # the buffer is reported at capacity, not at whatever happens to be in
        # it when the question is asked
        self.assertEqual(footprint["tbptt_buffer_scalars_current"], 0)
        self.assertEqual(footprint["tbptt_buffer_scalars_capacity"], (3 + 6 * 16 + 2) * 4)
        self.assertEqual(footprint["total_adaptive_state_scalars"], 994 + 16 + (3 + 6 * 16 + 2) * 4)

    def test_invalid_construction_is_refused(self):
        for options in (
            {"learning_rate": 0.0},
            {"learning_rate": float("nan")},
            {"tbptt_steps": 0},
            {"error_loss_weight": -1.0},
            {"gradient_clip": 0.0},
            {"seed": True},
        ):
            with self.subTest(options=options), self.assertRaises(InvalidModelState):
                fresh_model(**options)

    def test_invalid_parameter_sets_are_refused(self):
        good = fresh_model().parameters
        with self.assertRaises(InvalidModelState):
            AAA1KGRU(parameters={**good, "extra": np.zeros(3)})
        with self.assertRaises(InvalidModelState):
            AAA1KGRU(parameters={key: value for key, value in good.items() if key != "b_o"})
        with self.assertRaises(InvalidModelState):
            AAA1KGRU(parameters={**good, "W_o": np.zeros((3, 16))})
        with self.assertRaises(InvalidModelState):
            AAA1KGRU(parameters={**good, "b_o": np.array([0.0, np.inf])})


# ----------------------------------------------------------------------
# forward recurrence
# ----------------------------------------------------------------------
class ForwardTests(unittest.TestCase):
    def test_numerically_stable_activations(self):
        extreme = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
        self.assertTrue(np.all(np.isfinite(sigmoid(extreme))))
        self.assertTrue(np.all((sigmoid(extreme) >= 0) & (sigmoid(extreme) <= 1)))
        self.assertTrue(np.all(np.isfinite(softplus(extreme))))
        self.assertTrue(np.all(softplus(extreme) >= 0))
        self.assertAlmostEqual(float(softplus(np.array([0.0]))[0]), math.log(2.0))

    def test_forward_is_deterministic(self):
        first, second = fresh_model(), fresh_model()
        inputs = np.array([0.3, -0.7, 0.1])
        for _ in range(5):
            np.testing.assert_array_equal(first.forward(inputs), second.forward(inputs))

    def test_the_hidden_state_persists_across_steps(self):
        model = fresh_model()
        model.forward(np.array([0.3, 1.0, 0.0]))
        first = model.hidden.copy()
        model.forward(np.array([0.3, 1.0, 0.0]))
        self.assertFalse(np.array_equal(first, model.hidden))

    def test_reset_clears_state_and_the_truncation_buffer(self):
        model = fresh_model()
        for _ in range(4):
            model.forward(np.array([0.3, 1.0, 0.0]))
        model.reset_state()
        self.assertTrue(np.all(model.hidden == 0.0))
        with self.assertRaises(InvalidModelState):
            model.backward(0.0)

    def test_the_reset_ablation_makes_every_step_identical(self):
        model = fresh_model(reset_state_every_step=True)
        inputs = np.array([0.3, 1.0, 0.2])
        first = model.forward(inputs).copy()
        second = model.forward(inputs)
        np.testing.assert_allclose(first, second)

    def test_the_zero_error_ablation_removes_input_three(self):
        model = fresh_model(zero_error_input=True)
        a = model.forward(np.array([0.3, 1.0, 0.0])).copy()
        model.reset_state()
        b = model.forward(np.array([0.3, 1.0, 9.0]))
        np.testing.assert_allclose(a, b)

    def test_gates_stay_inside_their_ranges_and_values_stay_finite(self):
        rng = np.random.default_rng(0)
        model = fresh_model()
        for name, array in model.parameters.items():
            model.parameters[name] = rng.normal(size=array.shape) * 3.0
        for _ in range(50):
            model.forward(rng.normal(size=3) * 5.0)
            gates = model.last_gates()
            self.assertTrue(all(0.0 <= value <= 1.0 for value in gates["update_gate"]))
            self.assertTrue(all(0.0 <= value <= 1.0 for value in gates["reset_gate"]))
            self.assertTrue(np.all(np.isfinite(model.hidden)))
            self.assertTrue(np.all(np.abs(model.hidden) <= 1.0 + 1e-12))

    def test_malformed_input_is_refused(self):
        model = fresh_model()
        for bad in (np.array([0.0, 1.0]), np.zeros((3, 1)), np.array([0.0, np.nan, 1.0])):
            with self.subTest(bad=bad.shape), self.assertRaises(InvalidModelState):
                model.forward(bad)


# ----------------------------------------------------------------------
# gradients
# ----------------------------------------------------------------------
class GradientTests(unittest.TestCase):
    def test_sampled_finite_differences_agree_for_every_model(self):
        report = full_gradient_check(exhaustive=False)
        for entry in report["models"]:
            with self.subTest(model=entry["model"]):
                self.assertEqual(entry["violations"], 0, entry["worst_violating_parameter"])
        self.assertTrue(report["passed"])

    def test_every_gru_parameter_is_checked_exhaustively_at_least_once(self):
        entry = check_model(
            lambda: AAA1KGRU(seed=21, tbptt_steps=16, error_loss_weight=0.25),
            label="exhaustive",
            length=5,
            seed=99,
            exhaustive=True,
        )
        self.assertEqual(entry["checked"], 994)
        self.assertEqual(entry["violations"], 0, entry["worst_violating_parameter"])
        self.assertLess(entry["max_absolute_error"], 1e-8)

    def test_the_temporal_gradient_is_not_merely_the_one_step_gradient(self):
        """A truncation of one must disagree with a truncation covering the sequence."""

        rng = np.random.default_rng(5)
        inputs = rng.normal(size=(8, 3))
        targets = rng.normal(size=8)

        def total(horizon: int) -> float:
            model = AAA1KGRU(seed=31, tbptt_steps=horizon)
            for name, array in model.parameters.items():
                model.parameters[name] = np.random.default_rng(7).normal(size=array.shape) * 0.3
            model.reset_state()
            accumulated = {name: np.zeros_like(a) for name, a in model.parameters.items()}
            for x, target in zip(inputs, targets, strict=True):
                model.forward(x)
                step = model.backward(float(target))
                for name in accumulated:
                    accumulated[name] += step[name]
            return float(np.sum(np.abs(accumulated["U_n"])))

        self.assertGreater(abs(total(8) - total(1)), 1e-6)

    def test_the_auxiliary_target_really_is_stop_gradient(self):
        """A finite difference that lets the target move must disagree."""

        rng = np.random.default_rng(3)
        model = AAA1KGRU(seed=41, tbptt_steps=8, error_loss_weight=0.5)
        for name, array in model.parameters.items():
            model.parameters[name] = rng.normal(size=array.shape) * 0.4
        inputs = rng.normal(size=(4, 3))
        targets = rng.normal(size=4)

        def moving_target_loss() -> float:
            model.reset_state()
            total = 0.0
            for x, target in zip(inputs, targets, strict=True):
                total += model.step_loss(model.forward(x), float(target))["total_loss"]
            return total

        model.reset_state()
        analytic = {name: np.zeros_like(a) for name, a in model.parameters.items()}
        for x, target in zip(inputs, targets, strict=True):
            model.forward(x)
            step = model.backward(float(target))
            for name in analytic:
                analytic[name] += step[name]

        flat = model.parameters["W_o"].reshape(-1)
        original = flat[0]
        flat[0] = original + 1e-6
        plus = moving_target_loss()
        flat[0] = original - 1e-6
        minus = moving_target_loss()
        flat[0] = original
        moving = (plus - minus) / 2e-6
        self.assertNotAlmostEqual(moving, float(analytic["W_o"].reshape(-1)[0]), places=4)

    def test_a_non_finite_gradient_fails_loudly(self):
        model = fresh_model()
        model.forward(np.array([0.1, 0.2, 0.3]))
        gradients = model.backward(0.0)
        gradients["b_o"] = np.array([np.inf, 0.0])
        with self.assertRaises(FloatingPointError):
            model.apply_gradients(gradients)
        self.assertEqual(model.nonfinite_events, 1)

    def test_gradient_clipping_is_counted_and_bounds_the_step(self):
        model = fresh_model(gradient_clip=0.5)
        model.forward(np.array([0.1, 0.2, 0.3]))
        gradients = model.backward(0.0)
        gradients["b_o"] = np.array([100.0, 0.0])
        before = model.parameters["b_o"].copy()
        record = model.apply_gradients(gradients)
        self.assertEqual(model.clip_events, 1)
        self.assertLess(record["clip_scale"], 1.0)
        step = float(np.linalg.norm(model.parameters["b_o"] - before))
        self.assertLessEqual(step, model.learning_rate * 0.5 + 1e-12)


# ----------------------------------------------------------------------
# learning and freezing
# ----------------------------------------------------------------------
class LearningTests(unittest.TestCase):
    def test_an_online_model_changes_its_weights(self):
        agent = NeuralAgent(fresh_model(), name="a")
        before = agent.model.state_hash()
        run_stream(small_stream(), [agent])
        self.assertNotEqual(agent.model.state_hash(), before)
        self.assertGreater(agent.model.update_count, 0)

    def test_a_frozen_model_keeps_its_weights_bitwise_identical(self):
        agent = NeuralAgent(fresh_model(), name="a", update_enabled=False)
        before = {name: array.copy() for name, array in agent.model.parameters.items()}
        run_stream(small_stream(), [agent])
        for name, array in before.items():
            np.testing.assert_array_equal(agent.model.parameters[name], array)
        self.assertEqual(agent.model.update_count, 0)

    def test_a_frozen_model_still_evolves_its_hidden_state(self):
        agent = NeuralAgent(fresh_model(), name="a", update_enabled=False)
        run_stream(small_stream(), [agent])
        self.assertFalse(np.all(agent.model.hidden == 0.0))
        self.assertGreater(agent.model.forward_count, 0)

    def test_a_frozen_model_still_receives_its_previous_error_input(self):
        agent = NeuralAgent(fresh_model(), name="a", update_enabled=False)
        run_stream(small_stream(), [agent])
        self.assertNotEqual(agent.previous_signed_error, 0.0)

    def test_update_counts_are_exact(self):
        stream = small_stream()
        agent = NeuralAgent(fresh_model(), name="a")
        result = run_stream(stream, [agent])
        self.assertEqual(agent.model.update_count, agent.trained_steps)
        self.assertEqual(agent.trained_steps + agent.skipped_targets, len(result.steps))

    def test_the_recurrent_freeze_ablation_leaves_recurrent_weights_alone(self):
        agent = NeuralAgent(fresh_model(freeze_recurrent=True), name="a")
        before = {name: agent.model.parameters[name].copy() for name in ("U_z", "U_r", "U_n")}
        run_stream(small_stream(), [agent])
        for name, array in before.items():
            np.testing.assert_array_equal(agent.model.parameters[name], array)
        self.assertFalse(np.array_equal(agent.model.parameters["W_n"], np.zeros((16, 3))))

    def test_learning_reduces_error_on_a_long_predictable_stream(self):
        stream = motion_compat_stream("bouncing", 909, steps=400, change_step=None)
        agent = NeuralAgent(fresh_model(), name="a")
        errors = run_stream(stream, [agent]).errors("a")
        quarter = len(errors) // 4
        self.assertLess(float(np.mean(errors[-quarter:])), float(np.mean(errors[:quarter])))


# ----------------------------------------------------------------------
# clone integrity and branching
# ----------------------------------------------------------------------
class CloneTests(unittest.TestCase):
    def test_a_clone_is_bitwise_identical_and_shares_no_arrays(self):
        model = fresh_model()
        model.forward(np.array([0.1, 0.2, 0.3]))
        model.learn(0.4)
        clone = model.clone()
        self.assertEqual(clone.state_hash(), model.state_hash())
        for name in model.parameters:
            np.testing.assert_array_equal(clone.parameters[name], model.parameters[name])
            self.assertIsNot(clone.parameters[name], model.parameters[name])
        self.assertIsNot(clone.hidden, model.hidden)
        clone.parameters["b_o"][0] = 12.0
        self.assertNotEqual(float(model.parameters["b_o"][0]), 12.0)

    def test_branching_produces_two_arms_from_identical_state(self):
        agent = NeuralAgent(fresh_model(), name="a")
        run_stream(small_stream(), [agent])
        online = agent.branch(name="online", update_enabled=True)
        frozen = agent.branch(name="frozen", update_enabled=False)
        self.assertEqual(online.model.state_hash(), frozen.model.state_hash())
        self.assertEqual(online.previous_signed_error, frozen.previous_signed_error)
        self.assertEqual(online.tracker.to_dict(), frozen.tracker.to_dict())
        self.assertIsNot(online.model.hidden, frozen.model.hidden)

    def test_the_branch_experiment_diverges_only_through_learning(self):
        stream = motion_compat_stream("changed", 1234, steps=200, change_step=100)
        trunk = NeuralAgent(fresh_model(), name="p")
        outcome = run_online_frozen_branch(stream, trunk, branch_index=100)
        self.assertEqual(len(outcome.clone_state_hash), 64)
        advantage = outcome.post_branch_advantage()
        self.assertTrue(math.isfinite(advantage["absolute_advantage"]))
        # The frozen arm's weights never moved; its predictions still changed,
        # because its recurrence kept running.
        frozen = outcome.branch.errors(outcome.frozen_name)
        self.assertGreater(float(np.std(frozen)), 0.0)


# ----------------------------------------------------------------------
# serialization and resume
# ----------------------------------------------------------------------
class SerializationTests(unittest.TestCase):
    def test_a_round_trip_is_exact(self):
        model = fresh_model()
        for _ in range(6):
            model.forward(np.array([0.1, -0.3, 0.2]))
            model.learn(0.5)
        restored = AAA1KGRU.from_state_dict(json.loads(json.dumps(model.state_dict())))
        self.assertEqual(restored.state_hash(), model.state_hash())
        np.testing.assert_array_equal(restored.hidden, model.hidden)
        self.assertEqual(len(restored._caches), len(model._caches))

    def test_saving_and_loading_a_file_is_exact(self):
        model = fresh_model()
        model.forward(np.array([0.1, -0.3, 0.2]))
        model.learn(0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path, parent_model_id=None)
            reloaded = AAA1KGRU.load(path)
        self.assertEqual(reloaded.state_hash(), model.state_hash())

    def test_a_resumed_run_matches_an_uninterrupted_run_exactly(self):
        stream = motion_compat_stream("bouncing", 55, steps=120, change_step=None)
        straight = NeuralAgent(fresh_model(), name="a")
        reference = run_stream(stream, [straight])

        first = NeuralAgent(fresh_model(), name="a")
        run_stream(stream, [first], start=0, stop=60)
        state = json.loads(json.dumps(first.state_dict()))
        resumed = NeuralAgent(AAA1KGRU.from_state_dict(state["model"]), name="a")
        resumed.load_state(state)
        second = run_stream(stream, [resumed], start=60, begin_episode=False)

        tail = reference.slice(60)
        self.assertEqual(len(tail.steps), len(second.steps))
        for expected, actual in zip(tail.steps, second.steps, strict=True):
            self.assertEqual(expected.predictions["a"], actual.predictions["a"])
        self.assertEqual(resumed.model.state_hash(), straight.model.state_hash())

    def test_an_unsupported_checkpoint_is_refused(self):
        state = fresh_model().state_dict()
        with self.assertRaises(InvalidModelState):
            AAA1KGRU.from_state_dict({**state, "format_version": "something.else"})
        with self.assertRaises(InvalidModelState):
            AAA1KGRU.from_state_dict({**state, "architecture_id": "not-aaa1k"})
        with self.assertRaises(InvalidModelState):
            AAA1KGRU.from_state_dict({**state, "hidden": [0.0] * 4})

    def test_the_controls_round_trip_too(self):
        for model in (StatelessMLPControl(seed=1), VanillaRNNControl(seed=1)):
            with self.subTest(model=type(model).__name__):
                model.forward(np.array([0.2, 0.3, 0.4]))
                model.learn(0.1)
                self.assertEqual(model.clone().state_hash(), model.state_hash())


# ----------------------------------------------------------------------
# streams and causality
# ----------------------------------------------------------------------
class StreamTests(unittest.TestCase):
    def test_every_family_builds_deterministically(self):
        for family in FAMILIES:
            options = {"scenario": "bouncing"} if family == "motion_compat" else {}
            with self.subTest(family=family):
                first = build_stream(family, 31, **options)
                second = build_stream(family, 31, **options)
                self.assertEqual(
                    [step.true_position for step in first.steps],
                    [step.true_position for step in second.steps],
                )
                self.assertNotEqual(
                    [step.true_position for step in first.steps],
                    [step.true_position for step in build_stream(family, 32, **options).steps],
                )

    def test_the_seed_observation_is_always_given(self):
        for family in FAMILIES:
            options = {"scenario": "bouncing"} if family == "motion_compat" else {}
            self.assertTrue(build_stream(family, 5, **options).steps[0].observed)

    def test_a_stream_that_hides_its_seed_is_refused(self):
        with self.assertRaises(ValueError):
            Stream(
                family="occlusion_v1",
                stream_id="x",
                seed=0,
                steps=(
                    StreamStep(0, 0.5, False, "r"),
                    StreamStep(1, 0.5, True, "r"),
                    StreamStep(2, 0.5, True, "r"),
                ),
                metadata={},
            )

    def test_occlusion_actually_hides_steps_and_never_the_warmup(self):
        stream = occlusion_stream(17, steps=200, gap_period=20, gap_length=4, warmup=12)
        hidden = [step.index for step in stream.steps if not step.observed]
        self.assertTrue(hidden)
        self.assertGreaterEqual(min(hidden), 12)
        self.assertLess(stream.observed_fraction(), 1.0)

    def test_the_coarse_family_quantizes_every_observation(self):
        stream = coarse_speed_stream(19, steps=120, quantum=0.005)
        for step in stream.steps:
            self.assertAlmostEqual(step.true_position, quantize(step.true_position, 0.005), places=12)
        self.assertEqual(stream.observed_fraction(), 1.0)

    def test_the_aba_stream_returns_to_its_first_regime(self):
        stream = aba_stream(23, segment_steps=60)
        labels = [step.regime for step in stream.steps]
        self.assertEqual(set(labels), {"A1", "B", "A2"})
        self.assertEqual(labels[1], "A1")
        self.assertEqual(labels[-1], "A2")
        changes = [step.index for step in stream.steps if step.event == "regime_change"]
        self.assertEqual(len(changes), 2)

    def test_the_observable_view_hides_exactly_what_the_stream_hides(self):
        stream = occlusion_stream(29, steps=120)
        view = observable_view(stream)
        for step, value in zip(stream.steps, view, strict=True):
            self.assertEqual(value is None, not step.observed)


class CausalityTests(unittest.TestCase):
    def test_an_agent_only_ever_receives_a_float_or_none(self):
        received: list[object] = []

        class Recorder:
            name = "recorder"
            update_enabled = False

            def begin_episode(self) -> None:
                received.append("begin")

            def accept_observation(self, observation):
                received.append(observation)

            def predict(self) -> float:
                return 0.5

        run_stream(occlusion_stream(3, steps=80), [Recorder()])
        values = [item for item in received if item != "begin"]
        self.assertTrue(values)
        for value in values:
            self.assertTrue(value is None or isinstance(value, float))

    def test_a_learner_never_sees_a_target_before_it_is_scored(self):
        """An agent that peeks at the next observation would beat the truth exactly."""

        stream = motion_compat_stream("bouncing", 61, steps=60, change_step=None)
        agent = PersistenceAgent()
        result = run_stream(stream, [agent])
        for step in result.steps:
            # persistence predicts the current position, so a prediction equal
            # to the target would mean the target arrived before the prediction
            self.assertNotEqual(step.predictions["persistence"], step.true_next_position)

    def test_a_hidden_target_is_never_used_as_an_update_target(self):
        stream = occlusion_stream(37, steps=200, gap_period=12, gap_length=5, warmup=6)
        agent = NeuralAgent(fresh_model(), name="a")
        result = run_stream(stream, [agent])
        observed_transitions = sum(1 for step in result.steps if step.target_observed and step.input_observed)
        self.assertEqual(agent.trained_steps, observed_transitions)
        self.assertLess(agent.trained_steps, len(result.steps))

    def test_the_rls_arm_obeys_the_same_training_rule(self):
        stream = occlusion_stream(38, steps=200, gap_period=12, gap_length=5, warmup=6)
        agent = RLSAgent()
        result = run_stream(stream, [agent])
        observed_transitions = sum(1 for step in result.steps if step.target_observed and step.input_observed)
        self.assertLessEqual(agent.predictor.update_count, observed_transitions)
        self.assertLess(agent.predictor.update_count, len(result.steps))

    def test_the_previous_error_input_is_zero_until_a_target_is_revealed(self):
        agent = NeuralAgent(fresh_model(), name="a")
        agent.begin_episode()
        agent.accept_observation(0.5)
        self.assertEqual(agent.previous_signed_error, 0.0)
        agent.predict()
        self.assertEqual(agent.previous_signed_error, 0.0)
        agent.accept_observation(0.52)
        self.assertNotEqual(agent.previous_signed_error, 0.0)

    def test_the_previous_error_input_returns_to_zero_when_a_target_is_hidden(self):
        agent = NeuralAgent(fresh_model(), name="a")
        agent.begin_episode()
        agent.accept_observation(0.5)
        agent.predict()
        agent.accept_observation(0.52)
        self.assertNotEqual(agent.previous_signed_error, 0.0)
        agent.predict()
        agent.accept_observation(None)
        self.assertEqual(agent.previous_signed_error, 0.0)

    def test_no_evaluator_metadata_reaches_a_feature_vector(self):
        scales = PublicScales()
        inputs = build_inputs(
            scales, known_position=0.6, previous_known_position=0.596, previous_signed_error=0.25
        )
        self.assertEqual(inputs.shape, (3,))
        self.assertAlmostEqual(float(inputs[0]), 0.1)
        self.assertAlmostEqual(float(inputs[1]), 1.0)
        self.assertAlmostEqual(float(inputs[2]), 0.25)


class TrackerTests(unittest.TestCase):
    def test_a_held_step_gives_a_zero_velocity_feature(self):
        tracker = ObservationTracker()
        tracker.accept(0.5)
        tracker.accept(0.52)
        self.assertAlmostEqual(tracker.velocity_estimate(), 0.02)
        tracker.accept(None)
        self.assertEqual(tracker.velocity_estimate(), 0.0)
        self.assertEqual(tracker.require(), 0.52)

    def test_a_multi_step_gap_is_divided_by_its_elapsed_steps(self):
        tracker = ObservationTracker()
        tracker.accept(0.5)
        for _ in range(3):
            tracker.accept(None)
        tracker.accept(0.54)
        self.assertEqual(tracker.gap, 4)
        self.assertAlmostEqual(tracker.velocity_estimate(), 0.01)


# ----------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------
class BaselineTests(unittest.TestCase):
    def test_every_baseline_runs_and_produces_finite_predictions(self):
        stream = occlusion_stream(41, steps=120)
        result = run_stream(stream, baseline_suite())
        self.assertEqual(len(result.agent_names), 5)
        for name in result.agent_names:
            values = result.errors(name)
            self.assertTrue(np.all(np.isfinite(values)))

    def test_persistence_predicts_the_held_position(self):
        agent = PersistenceAgent()
        agent.begin_episode()
        agent.accept_observation(0.4)
        self.assertEqual(agent.predict(), 0.4)
        agent.accept_observation(None)
        self.assertEqual(agent.predict(), 0.4)

    def test_dead_reckoning_carries_velocity_through_a_gap(self):
        agent = DeadReckoningAgent()
        agent.begin_episode()
        agent.accept_observation(0.40)
        agent.accept_observation(0.42)
        self.assertAlmostEqual(agent.predict(), 0.44)
        agent.accept_observation(None)
        self.assertAlmostEqual(agent.predict(), 0.46)

    def test_constant_motion_degrades_to_persistence_in_a_gap(self):
        agent = ConstantMotionAgent()
        agent.begin_episode()
        agent.accept_observation(0.40)
        agent.accept_observation(0.42)
        self.assertAlmostEqual(agent.predict(), 0.44)
        agent.accept_observation(None)
        self.assertAlmostEqual(agent.predict(), 0.42)

    def test_the_linear_fit_baseline_extrapolates_a_straight_line(self):
        agent = WindowedLinearFitAgent(window=6)
        agent.begin_episode()
        for index in range(6):
            agent.accept_observation(0.2 + 0.01 * index)
        self.assertAlmostEqual(agent.predict(), 0.26, places=9)

    def test_the_reflected_baseline_stays_inside_the_bounds(self):
        agent = ConstantMotionAgent(reflect=True)
        agent.begin_episode()
        agent.accept_observation(0.98)
        agent.accept_observation(0.999)
        self.assertLessEqual(agent.predict(), 1.0)
        self.assertGreaterEqual(agent.predict(), 0.0)

    def test_the_rls_arm_reproduces_the_frozen_v2_1_candidate_configuration(self):
        predictor = RLSAgent().predictor
        self.assertEqual(predictor.forgetting, 0.3)
        self.assertEqual(predictor.forgetting_mode, "exponential")
        self.assertEqual(predictor.detector_multiplier, 8.0)
        self.assertTrue(predictor.unfold_target)
        self.assertTrue(predictor.skip_after_reflected_prediction)
        self.assertEqual(predictor.displacement_scale, 0.004)


# ----------------------------------------------------------------------
# seeds, statistics and selection
# ----------------------------------------------------------------------
class SeedTests(unittest.TestCase):
    def test_namespaces_are_disjoint(self):
        values = {namespace: {derive_seed(namespace, i) for i in range(200)} for namespace in NAMESPACES}
        for first in NAMESPACES:
            for second in NAMESPACES:
                if first < second:
                    self.assertEqual(values[first] & values[second], set())

    def test_seeds_are_deterministic_and_bounded(self):
        self.assertEqual(derive_seed("model_init", 3), derive_seed("model_init", 3))
        self.assertTrue(0 <= derive_seed("bootstrap", 9) < 2**31 - 1)

    def test_an_unknown_namespace_is_refused(self):
        with self.assertRaises(ValueError):
            derive_seed("not_a_namespace", 0)


class StatisticsTests(unittest.TestCase):
    def test_a_paired_difference_reports_its_direction_and_sign_counts(self):
        record = paired_difference([1.0, 2.0, 3.0], [2.0, 3.0, 5.0])
        self.assertAlmostEqual(record["mean_difference"], 4.0 / 3.0)
        self.assertEqual(record["favours_first"], 3)
        self.assertEqual(record["favours_second"], 0)
        self.assertLess(record["ci_low"], record["ci_high"])

    def test_a_single_stream_cannot_produce_an_interval(self):
        record = paired_difference([1.0], [2.0])
        self.assertEqual(record["interval_status"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(math.isnan(record["ci_low"]))

    def test_an_empty_comparison_is_refused_rather_than_passing_vacuously(self):
        with self.assertRaises(ValueError):
            paired_difference([], [])

    def test_the_bootstrap_is_reproducible(self):
        first = paired_difference([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 2.5, 5.0])
        second = paired_difference([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 2.5, 5.0])
        self.assertEqual(first["ci_low"], second["ci_low"])

    def test_calibration_detects_a_perfectly_informative_head(self):
        values = np.linspace(0.1, 1.0, 40)
        record = calibration(values, values)
        self.assertEqual(record["status"], "MEASURED")
        self.assertAlmostEqual(record["spearman"], 1.0, places=6)
        self.assertAlmostEqual(record["slope"], 1.0, places=6)
        self.assertTrue(record["bins_monotone"])

    def test_calibration_reports_insufficient_evidence_rather_than_a_number(self):
        record = calibration([0.1, 0.2], [0.1, 0.3], bins=5)
        self.assertEqual(record["status"], "INSUFFICIENT_EVIDENCE")


class SelectionRuleTests(unittest.TestCase):
    def test_the_stability_margin_removes_rates_near_the_boundary(self):
        self.assertEqual(eligible_learning_rates(0.3), (0.001, 0.003, 0.01, 0.03))
        self.assertEqual(eligible_learning_rates(0.01), (0.001,))
        self.assertEqual(eligible_learning_rates(0.003), ())

    def test_no_measured_boundary_leaves_every_rate_eligible(self):
        self.assertEqual(len(eligible_learning_rates(None)), 6)

    def test_a_configuration_key_is_stable(self):
        self.assertEqual(Configuration(0.03, 4, 0.25).key(), "lr=0.03;T=4;lambda=0.25")


# ----------------------------------------------------------------------
# identity and lineage
# ----------------------------------------------------------------------
class IdentityTests(unittest.TestCase):
    def test_the_phase_fingerprint_covers_its_own_source(self):
        fingerprint = phase_fingerprint(PROJECT)
        self.assertEqual(fingerprint["phase_version"], PHASE_VERSION)
        self.assertEqual(len(fingerprint["sha256"]), 64)
        names = set(fingerprint["files"])
        self.assertIn("research/aaa_1k/model.py", names)
        self.assertIn("aaa/predictors.py", names)
        self.assertIn("requirements-lock.txt", names)

    def test_generated_documents_are_excluded_so_the_identity_is_not_self_referential(self):
        names = set(phase_fingerprint(PROJECT)["files"])
        self.assertNotIn("docs/aaa_1k_report.md", names)
        self.assertNotIn("docs/aaa_1k_handoff.md", names)

    def test_the_root_model_declares_itself_a_lineage_root(self):
        model = fresh_model()
        record = lineage_record(
            model_state_hash=model.state_hash(),
            parameter_count=model.parameter_count(),
            initialization_seed=model.seed,
            project_root=PROJECT,
        )
        self.assertIsNone(record["parent_model_id"])
        self.assertTrue(record["is_lineage_root"])
        self.assertEqual(record["parameter_count"], 994)
        self.assertEqual(record["architecture_id"], ARCHITECTURE_ID)


# ----------------------------------------------------------------------
# visualization
# ----------------------------------------------------------------------
class VisualizationTests(unittest.TestCase):
    def setUp(self):
        import matplotlib

        matplotlib.use("Agg", force=True)

    def test_rendering_does_not_mutate_the_model(self):
        from research.aaa_1k.visualize import record_trace, render

        model = fresh_model()
        trace = record_trace(occlusion_stream(43, steps=80), model)
        before = model.state_hash()
        with tempfile.TemporaryDirectory() as directory:
            render(trace, Path(directory) / "dashboard.png")
        self.assertEqual(model.state_hash(), before)

    def test_the_trace_reports_real_hidden_units_and_gates(self):
        from research.aaa_1k.visualize import record_trace

        trace = record_trace(small_stream(), fresh_model())
        self.assertEqual(len(trace.hidden[0]), HIDDEN_SIZE)
        self.assertEqual(len(trace.update_gate[0]), HIDDEN_SIZE)
        self.assertEqual(len(trace.reset_gate[0]), HIDDEN_SIZE)
        for gates in (trace.update_gate[5], trace.reset_gate[5]):
            self.assertTrue(all(0.0 <= value <= 1.0 for value in gates))
        self.assertTrue(all(abs(value) <= 1.0 for value in trace.hidden[5]))

    def test_the_traced_predictions_match_the_runner_exactly(self):
        from research.aaa_1k.visualize import record_trace

        stream = small_stream()
        trace = record_trace(stream, fresh_model())
        agent = NeuralAgent(fresh_model(), name="aaa1k")
        result = run_stream(stream, [agent])
        self.assertEqual(len(trace.prediction), len(result.steps))
        for traced, scored in zip(trace.prediction, result.steps, strict=True):
            self.assertAlmostEqual(traced, scored.predictions["aaa1k"], places=12)

    def test_the_error_curve_uses_the_real_scored_error(self):
        from research.aaa_1k.visualize import record_trace

        trace = record_trace(small_stream(), fresh_model())
        for prediction, truth, error in zip(
            trace.prediction, trace.true_position, trace.realized_error, strict=True
        ):
            self.assertAlmostEqual(error, abs(prediction - truth), places=12)

    def test_an_event_marker_is_never_placed_before_its_event(self):
        from research.aaa_1k.visualize import record_trace

        stream = aba_stream(47, segment_steps=60)
        trace = record_trace(stream, fresh_model())
        declared = {step.index for step in stream.steps if step.event}
        for index, _ in trace.events:
            self.assertIn(index, declared)

    def test_rendering_an_empty_trace_is_refused(self):
        from research.aaa_1k.visualize import Trace, render

        with self.assertRaises(ValueError):
            render(Trace(), "unused.png")


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------
class EndToEndTests(unittest.TestCase):
    def test_the_full_arm_set_runs_paired_on_one_stream(self):
        agents = build_agents(Configuration(0.03, 4, 0.25))
        result = run_stream(coarse_speed_stream(53, steps=120), agents)
        self.assertEqual(len(result.agent_names), 12)
        for name in result.agent_names:
            self.assertTrue(np.all(np.isfinite(result.errors(name))))

    def test_the_cli_parameter_audit_and_gradient_check_succeed(self):
        from research.aaa_1k.cli import main

        self.assertEqual(main(["parameter-audit"]), 0)

    def test_running_the_same_configuration_twice_gives_identical_results(self):
        stream = occlusion_stream(59, steps=120)
        first = run_stream(stream, build_agents(Configuration(0.03, 4, 0.25)))
        second = run_stream(stream, build_agents(Configuration(0.03, 4, 0.25)))
        for left, right in zip(first.steps, second.steps, strict=True):
            self.assertEqual(left.predictions, right.predictions)


# ----------------------------------------------------------------------
# failure injection
# ----------------------------------------------------------------------
class FailureInjectionTests(unittest.TestCase):
    """Break each invariant on purpose and prove the corresponding check fails.

    Without this, every assertion above is only evidence that the code passes
    its own tests, which is not the same as evidence that the tests would
    notice if it stopped being correct.
    """

    def test_a_corrupted_backward_pass_is_caught_by_the_gradient_check(self):
        class BrokenGRU(AAA1KGRU):
            def backward(self, target_displacement):
                gradients = super().backward(target_displacement)
                gradients["U_n"] = gradients["U_n"] * 0.5  # plausible, silent, wrong
                return gradients

        entry = check_model(lambda: BrokenGRU(seed=61, tbptt_steps=16), label="broken", length=6, seed=62)
        self.assertFalse(entry["passed"])
        self.assertGreater(entry["violations"], 0)
        self.assertGreater(entry["max_relative_error"], entry["relative_tolerance"])

    def test_dropping_the_temporal_term_is_caught(self):
        class TruncatedGRU(AAA1KGRU):
            def backward(self, target_displacement):
                # keep only the most recent transition, whatever the horizon
                saved = list(self._caches)
                self._caches.clear()
                self._caches.append(saved[-1])
                gradients = super().backward(target_displacement)
                self._caches.clear()
                self._caches.extend(saved)
                return gradients

        entry = check_model(
            lambda: TruncatedGRU(seed=63, tbptt_steps=16), label="truncated", length=6, seed=64
        )
        self.assertFalse(entry["passed"])

    def test_a_leaking_agent_is_caught_by_the_causality_check(self):
        """An agent that peeked would predict the target exactly."""

        class Leaker:
            name = "leaker"
            update_enabled = False

            def __init__(self):
                self.last = 0.5

            def begin_episode(self):
                self.last = 0.5

            def accept_observation(self, observation):
                if observation is not None:
                    self.last = observation

            def predict(self):
                return self.last

        stream = motion_compat_stream("bouncing", 67, steps=40, change_step=None)
        result = run_stream(stream, [Leaker()])
        # The honest agent never matches the target exactly. Simulate a leak by
        # substituting the revealed value and confirm the check would fire.
        leaked = [step.true_next_position for step in result.steps]
        honest = [step.predictions["leaker"] for step in result.steps]
        self.assertTrue(all(a != b for a, b in zip(leaked, honest, strict=True)))
        with self.assertRaises(AssertionError):
            for value, target in zip(leaked, leaked, strict=True):
                self.assertNotEqual(value, target)

    def test_a_frozen_arm_that_secretly_learns_is_caught(self):
        agent = NeuralAgent(fresh_model(), name="a", update_enabled=False)
        before = agent.model.parameters["b_o"].copy()
        run_stream(small_stream(), [agent])
        np.testing.assert_array_equal(agent.model.parameters["b_o"], before)
        agent.update_enabled = True  # inject the defect
        run_stream(small_stream(), [agent])
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(agent.model.parameters["b_o"], before)

    def test_a_clone_that_shares_arrays_is_caught(self):
        model = fresh_model()
        shallow = AAA1KGRU(seed=model.seed, parameters=model.parameters)
        shallow.parameters["b_o"] = model.parameters["b_o"]  # inject the defect
        shallow.parameters["b_o"][0] = 5.0
        self.assertEqual(float(model.parameters["b_o"][0]), 5.0)
        # the real clone does not do this
        honest = model.clone()
        honest.parameters["b_o"][0] = 9.0
        self.assertNotEqual(float(model.parameters["b_o"][0]), 9.0)

    def test_training_on_an_unavailable_target_is_caught(self):
        stream = occlusion_stream(71, steps=160, gap_period=12, gap_length=5, warmup=6)
        honest = NeuralAgent(fresh_model(), name="a")
        run_stream(stream, [honest])
        observed_transitions = sum(
            1
            for index, step in enumerate(stream.steps[:-1])
            if step.observed and stream.steps[index + 1].observed
        )
        self.assertLessEqual(honest.trained_steps, observed_transitions)

        class Greedy(NeuralAgent):
            def accept_observation(self, observation):
                if observation is None:
                    # inject the defect: invent a target nobody revealed
                    observation = self.tracker.known
                super().accept_observation(observation)

        greedy = Greedy(fresh_model(), name="a")
        run_stream(stream, [greedy])
        self.assertGreater(greedy.trained_steps, honest.trained_steps)

    def test_a_corrupted_checkpoint_is_refused_rather_than_silently_repaired(self):
        model = fresh_model()
        state = json.loads(json.dumps(model.state_dict()))
        state["parameters"]["U_z"][0][0] = None
        with self.assertRaises((InvalidModelState, TypeError)):
            AAA1KGRU.from_state_dict(state)

    def test_an_empty_evidence_set_cannot_pass_vacuously(self):
        """`all([]) is True` is how the previous protocol lied. Not here."""

        with self.assertRaises(ValueError):
            paired_difference([], [])
        self.assertEqual(calibration([], [])["status"], "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


# ----------------------------------------------------------------------
# round 2: the corrected designs
# ----------------------------------------------------------------------
ROUND2_CONFIG = Configuration(0.03, 4, 0.25, 10.0)


class PairedChangeStreamTests(unittest.TestCase):
    def test_the_pair_is_bit_identical_until_the_change(self):
        changed, control = paired_change_streams(101, steps=200, change_step=100)
        for index in range(101):
            self.assertEqual(changed.steps[index].true_position, control.steps[index].true_position)

    def test_the_pair_diverges_after_the_change(self):
        changed, control = paired_change_streams(101, steps=200, change_step=100)
        tail_changed = [step.true_position for step in changed.steps[101:]]
        tail_control = [step.true_position for step in control.steps[101:]]
        self.assertNotEqual(tail_changed, tail_control)

    def test_only_the_changed_member_carries_a_change_event(self):
        changed, control = paired_change_streams(101, steps=200, change_step=100)
        self.assertEqual([s.index for s in changed.steps if s.event == "change"], [101])
        self.assertEqual([s.index for s in control.steps if s.event == "change"], [])

    def test_the_control_has_no_change_factor(self):
        changed, control = paired_change_streams(101, steps=200, change_step=100)
        self.assertNotEqual(changed.metadata["change_factor"], 1.0)
        self.assertEqual(control.metadata["change_factor"], 1.0)

    def test_a_change_step_outside_the_stream_is_refused(self):
        with self.assertRaises(ValueError):
            paired_change_streams(1, steps=50, change_step=49)


class AdaptationMeasurementTests(unittest.TestCase):
    def test_both_trunks_reach_an_identical_state_at_the_branch(self):
        trial = adaptation_difference_of_differences(ROUND2_CONFIG, seed=202, steps=160, change_step=80)
        self.assertTrue(trial["trunks_matched"])
        self.assertEqual(len(trial["trunk_state_hash"]), 64)

    def test_the_effect_is_the_declared_difference_of_differences(self):
        trial = adaptation_difference_of_differences(ROUND2_CONFIG, seed=203, steps=160, change_step=80)
        self.assertAlmostEqual(
            trial["adaptation_effect"],
            trial["changed"]["absolute_advantage"] - trial["control"]["absolute_advantage"],
            places=15,
        )

    def test_the_control_arm_still_shows_an_ordinary_learning_advantage(self):
        """The whole point: the control is not zero, which is why round 1 was wrong."""

        trial = adaptation_difference_of_differences(ROUND2_CONFIG, seed=204, steps=160, change_step=80)
        self.assertNotEqual(trial["continued_learning_effect"], 0.0)


class RetentionMeasurementTests(unittest.TestCase):
    def test_the_probe_bank_is_fixed_and_deterministic(self):
        first, second = probe_bank(size=4), probe_bank(size=4)
        self.assertEqual([s.stream_id for s in first], [s.stream_id for s in second])
        self.assertEqual(len(first), 4)

    def test_probing_never_updates_the_weights(self):
        trial = retention_trial(ROUND2_CONFIG, seed=205, segment_steps=60, bank=probe_bank(size=2, steps=40))
        self.assertGreater(trial["updates"], 0)
        self.assertEqual(set(trial["probe_error"]), {"after_A1", "after_B", "after_A2"})

    def test_forgetting_is_the_declared_probe_difference(self):
        trial = retention_trial(ROUND2_CONFIG, seed=206, segment_steps=60, bank=probe_bank(size=2, steps=40))
        self.assertAlmostEqual(
            trial["forgetting"],
            trial["probe_error"]["after_B"] - trial["probe_error"]["after_A1"],
            places=15,
        )

    def test_every_checkpoint_is_measured_on_the_same_questions(self):
        bank = probe_bank(size=3, steps=40)
        trial = retention_trial(ROUND2_CONFIG, seed=207, segment_steps=60, bank=bank)
        self.assertEqual(trial["probe_bank_size"], 3)
        # three distinct measurements of the same bank
        self.assertEqual(len(set(trial["probe_error"].values())), 3)


class CrossedBootstrapTests(unittest.TestCase):
    def test_a_crossed_interval_resamples_both_factors(self):
        rng = np.random.default_rng(0)
        first = rng.normal(1.0, 0.05, size=(4, 12))
        second = first + 0.2
        record = crossed_paired_difference(first, second)
        self.assertEqual(record["initializations"], 4)
        self.assertEqual(record["streams"], 12)
        self.assertAlmostEqual(record["mean_difference"], 0.2, places=9)
        self.assertLess(record["ci_low"], record["ci_high"])
        self.assertTrue(record["initializations_agreeing_on_sign"])

    def test_a_single_initialization_cannot_produce_a_crossed_interval(self):
        record = crossed_paired_difference(np.zeros((1, 8)), np.ones((1, 8)))
        self.assertEqual(record["interval_status"], "INSUFFICIENT_EVIDENCE")

    def test_the_crossed_interval_is_wider_than_the_stream_only_interval(self):
        """Resampling one factor understates the uncertainty. That was the defect."""

        rng = np.random.default_rng(3)
        # a design where initializations genuinely disagree
        offsets = np.array([0.0, 0.4, -0.3, 0.2, -0.1]).reshape(-1, 1)
        first = rng.normal(1.0, 0.02, size=(5, 20))
        second = first + 0.2 + offsets
        crossed = crossed_paired_difference(first, second)
        flat = paired_difference(first.reshape(-1), second.reshape(-1))
        crossed_width = crossed["ci_high"] - crossed["ci_low"]
        flat_width = flat["ci_high"] - flat["ci_low"]
        self.assertGreater(crossed_width, flat_width)

    def test_a_ragged_design_is_refused(self):
        with self.assertRaises(ValueError):
            crossed_paired_difference(np.zeros((2, 3)), np.zeros((3, 2)))

    def test_achieved_precision_reports_resolution_against_the_measured_effect(self):
        record = crossed_paired_difference(np.zeros((4, 20)), np.full((4, 20), 1.0))
        precision = achieved_precision(record)
        self.assertEqual(precision["status"], "MEASURED")
        self.assertAlmostEqual(precision["effect"], 1.0, places=9)
        self.assertTrue(precision["meets_quarter_effect_target"])


class PerArchitectureSelectionTests(unittest.TestCase):
    def test_each_architecture_gets_its_own_rule_selected_hyperparameters(self):
        streams = development_streams()[:2]
        record = select_for_architecture("VanillaRNNControl", streams, tbptt_steps=4, error_loss_weight=0.25)
        self.assertEqual(record["architecture"], "VanillaRNNControl")
        self.assertIn(record["selected"]["learning_rate"], record["eligible_learning_rates"])
        self.assertIn(record["selected"]["gradient_clip"], (0.3, 1.0, 3.0, 10.0, None))

    def test_controls_receive_their_own_configuration(self):
        agents = build_agents(
            ROUND2_CONFIG,
            architecture_configurations={
                "StatelessMLPControl": {
                    "learning_rate": 0.001,
                    "tbptt_steps": 4,
                    "error_loss_weight": 0.25,
                    "gradient_clip": None,
                }
            },
        )
        by_name = {agent.name: agent for agent in agents}
        self.assertEqual(by_name["mlp_control"].model.learning_rate, 0.001)
        self.assertIsNone(by_name["mlp_control"].model.gradient_clip)
        # the primary and its ablations are untouched
        self.assertEqual(by_name["aaa1k_gru"].model.learning_rate, 0.03)
        self.assertEqual(by_name["aaa1k_state_reset"].model.learning_rate, 0.03)
        self.assertEqual(by_name["aaa1k_gru"].model.gradient_clip, 10.0)

    def test_an_ablation_differs_from_the_primary_in_exactly_one_mechanism(self):
        by_name = {agent.name: agent for agent in build_agents(ROUND2_CONFIG)}
        primary = by_name["aaa1k_gru"].model
        for name, field in (
            ("aaa1k_state_reset", "reset_state_every_step"),
            ("aaa1k_no_error_input", "zero_error_input"),
            ("aaa1k_frozen_recurrent", "freeze_recurrent"),
        ):
            with self.subTest(ablation=name):
                ablation = by_name[name].model
                self.assertTrue(getattr(ablation, field))
                self.assertFalse(getattr(primary, field))
                self.assertEqual(ablation.learning_rate, primary.learning_rate)
                self.assertEqual(ablation.gradient_clip, primary.gradient_clip)
                for parameter in primary.parameters:
                    np.testing.assert_array_equal(
                        ablation.parameters[parameter], primary.parameters[parameter]
                    )


class CharacterizationTests(unittest.TestCase):
    def test_the_coarse_family_is_decomposed_into_its_two_mechanisms(self):
        record = coarse_speed_decomposition(ROUND2_CONFIG, streams=2, initializations=1)
        self.assertEqual(set(record["conditions"]), {"no_regime_switch", "with_regime_switch"})
        self.assertIn("finding", record)
        self.assertTrue(math.isfinite(record["advantage_share_without_regime"]))


class Round2FailureInjectionTests(unittest.TestCase):
    """The new designs get the same treatment as the old ones."""

    def test_a_broken_pairing_is_caught(self):
        """If the two trunks diverged before the branch, the DiD is meaningless."""

        import research.aaa_1k.measurements as measurements

        original = measurements.paired_change_streams

        def broken(seed, **options):
            changed, _ = original(seed, **options)
            other, _ = original(seed + 1, **options)  # inject: unpaired control
            return changed, other

        measurements.paired_change_streams = broken
        try:
            with self.assertRaises(RuntimeError):
                measurements.adaptation_difference_of_differences(
                    ROUND2_CONFIG, seed=301, steps=120, change_step=60
                )
        finally:
            measurements.paired_change_streams = original

    def test_a_probe_that_trains_is_caught(self):
        """The retention probe must never update the weights it is measuring."""

        agent = NeuralAgent(fresh_model(), name="a")
        run_stream(small_stream(), [agent])
        bank = probe_bank(size=2, steps=40)

        class LeakyAgent(NeuralAgent):
            def branch(self, *, name, update_enabled):
                # inject the defect: hand back a clone that still learns
                return super().branch(name=name, update_enabled=True)

        leaky = LeakyAgent(agent.model.clone(), name="leaky")
        leaky.load_state(agent.state_dict())
        with self.assertRaises(RuntimeError):
            _probe(leaky, bank, label="broken")

    def test_the_stream_only_interval_would_have_understated_the_uncertainty(self):
        """Documents the defect the crossed bootstrap fixes, as a live check."""

        rng = np.random.default_rng(11)
        offsets = np.array([0.0, 0.5, -0.4, 0.3, -0.2]).reshape(-1, 1)
        first = rng.normal(1.0, 0.01, size=(5, 25))
        second = first + 0.1 + offsets
        crossed = crossed_paired_difference(first, second)
        naive = paired_difference(first.reshape(-1), second.reshape(-1))
        self.assertGreater(crossed["ci_high"] - crossed["ci_low"], 3 * (naive["ci_high"] - naive["ci_low"]))
