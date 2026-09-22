"""Tests for the online-TBPTT reference probe (audit finding R-02).

The probe is only worth anything if each rule is exactly what it claims to be.
So every rule is checked against an independent finite-difference reference
*after real interleaved SGD updates* -- the condition the phase's own gradient
checker never exercises -- and the instrumented live model is required to
follow the champion's trajectory bit for bit.
"""

from __future__ import annotations

import unittest

import numpy as np

from research.aaa_1k.agents import NeuralAgent
from research.aaa_1k.model import AAA1KGRU
from research.aaa_1k.runner import run_stream
from research.aaa_1k.streams import coarse_speed_stream
from research.aaa_1k_loop.arms import CHAMPION_CONFIGURATION
from research.aaa_1k_loop.diagnosis4 import (
    HISTOGRAM_BINS,
    HISTOGRAM_LOW,
    HISTOGRAM_WIDTH,
    histogram_quantile,
    log_histogram,
)
from research.aaa_1k_loop.tbptt import (
    ComparingGRU,
    ReplayGRU,
    SnapshotGRU,
    backward_through,
    build_rule_model,
    current_window_loss,
    finite_difference,
    flatten,
    live_gradient,
    realized_window_loss,
    replay_gradient,
    replay_window,
    snapshot_gradient,
    t1_gradient,
)

TOLERANCE = 1e-6


def _relative(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> float:
    a, b = flatten(first), flatten(second)
    return float(np.linalg.norm(a - b) / max(1e-12, np.linalg.norm(b)))


def _drifting_model(steps: int, *, learning_rate: float = 0.3, seed: int = 3) -> tuple[SnapshotGRU, float]:
    """A model whose parameters moved inside its window, plus a target for the next loss."""

    rng = np.random.default_rng(seed)
    model = SnapshotGRU(seed=seed, **{**CHAMPION_CONFIGURATION, "learning_rate": learning_rate})
    model.parameters["W_o"][:] = rng.normal(0.0, 0.5, model.parameters["W_o"].shape)
    target = 0.0
    for index in range(steps):
        model.forward(rng.normal(size=3))
        target = float(rng.normal())
        if index < steps - 1:
            model.learn(target)
    return model, target


class BackwardRoutineTests(unittest.TestCase):
    def test_live_matrices_reproduce_champion_backward_bitwise(self) -> None:
        model, target = _drifting_model(7)
        window = list(model._caches)
        shared = backward_through(
            window,
            [model.parameters] * len(window),
            model.parameters["W_o"],
            model._output_gradient(window[-1].output, target),
        )
        champion = AAA1KGRU.backward(model, target)
        for name, value in champion.items():
            self.assertTrue(np.array_equal(value, shared[name]), name)

    def test_mismatched_matrix_count_is_refused(self) -> None:
        model, _target = _drifting_model(5)
        with self.assertRaises(ValueError):
            backward_through(list(model._caches), [model.parameters], model.parameters["W_o"], np.zeros(2))


class ReferenceTests(unittest.TestCase):
    """Each rule against its own finite-difference definition, after interleaved updates."""

    def test_snapshot_is_the_realized_trajectory_gradient(self) -> None:
        model, target = _drifting_model(6)
        error_target = abs(float(model._caches[-1].output[0]) - target)
        reference = finite_difference(realized_window_loss, model, target, error_target=error_target)
        self.assertLess(_relative(snapshot_gradient(model, target), reference), TOLERANCE)

    def test_replay_is_the_current_parameter_window_gradient(self) -> None:
        model, target = _drifting_model(6)
        error_target = abs(float(replay_window(model, model.parameters)[-1].output[0]) - target)
        reference = finite_difference(current_window_loss, model, target, error_target=error_target)
        self.assertLess(_relative(replay_gradient(model, target), reference), TOLERANCE)

    def test_live_is_neither_reference_once_parameters_drift(self) -> None:
        model, target = _drifting_model(6)
        live = live_gradient(model, target)
        self.assertGreater(_relative(live, snapshot_gradient(model, target)), 1e-4)
        self.assertGreater(_relative(live, replay_gradient(model, target)), 1e-4)

    def test_all_rules_agree_without_drift(self) -> None:
        rng = np.random.default_rng(11)
        model = SnapshotGRU(seed=5, **CHAMPION_CONFIGURATION)
        model.parameters["W_o"][:] = rng.normal(0.0, 0.5, model.parameters["W_o"].shape)
        for _ in range(6):
            model.forward(rng.normal(size=3))
        live = live_gradient(model, 0.4)
        for other in (snapshot_gradient(model, 0.4), replay_gradient(model, 0.4)):
            self.assertLess(_relative(live, other), 1e-12)

    def test_oldest_cache_matrices_never_matter(self) -> None:
        # The earliest transition's matrices only feed the discarded boundary gradient,
        # so with two caches the live and snapshot rules coincide exactly.
        model, target = _drifting_model(2)
        self.assertEqual(len(model._caches), 2)
        self.assertEqual(_relative(live_gradient(model, target), snapshot_gradient(model, target)), 0.0)

    def test_t1_is_the_last_transition_only(self) -> None:
        model, target = _drifting_model(6)
        one = AAA1KGRU(seed=0, **{**CHAMPION_CONFIGURATION, "tbptt_steps": 1})
        one.parameters = {name: value.copy() for name, value in model.parameters.items()}
        one._caches.append(model._caches[-1])
        self.assertLess(_relative(t1_gradient(model, target), AAA1KGRU.backward(one, target)), 1e-15)


class RuleModelTests(unittest.TestCase):
    def test_rule_models_share_the_champion_initialization(self) -> None:
        reference = AAA1KGRU(seed=7, **CHAMPION_CONFIGURATION)
        for rule in ("live", "snapshot", "replay", "t1"):
            model = build_rule_model(rule, 7, CHAMPION_CONFIGURATION)
            for name, value in reference.parameters.items():
                self.assertTrue(np.array_equal(value, model.parameters[name]), (rule, name))
        self.assertIsInstance(build_rule_model("replay", 7, CHAMPION_CONFIGURATION), ReplayGRU)
        self.assertEqual(build_rule_model("t1", 7, CHAMPION_CONFIGURATION).tbptt_steps, 1)
        with self.assertRaises(ValueError):
            build_rule_model("exact", 7, CHAMPION_CONFIGURATION)

    def test_comparing_model_follows_the_champion_bitwise(self) -> None:
        stream = coarse_speed_stream(424242, steps=160, regime_length=10_000)
        champion = NeuralAgent(AAA1KGRU(seed=2, **CHAMPION_CONFIGURATION), name="champion")
        comparing = NeuralAgent(ComparingGRU(seed=2, **CHAMPION_CONFIGURATION), name="comparing")
        result = run_stream(stream, [champion, comparing])
        self.assertTrue(np.array_equal(result.errors("champion"), result.errors("comparing")))
        for name, value in champion.model.parameters.items():
            self.assertTrue(np.array_equal(value, comparing.model.parameters[name]), name)
        self.assertEqual(len(comparing.model.comparisons), champion.model.update_count)

    def test_snapshot_and_replay_rules_learn_differently_from_live(self) -> None:
        stream = coarse_speed_stream(424242, steps=160, regime_length=10_000)
        agents = [
            NeuralAgent(build_rule_model(rule, 2, CHAMPION_CONFIGURATION), name=rule)
            for rule in ("live", "snapshot", "replay")
        ]
        result = run_stream(stream, agents)
        self.assertFalse(np.array_equal(result.errors("live"), result.errors("snapshot")))
        self.assertFalse(np.array_equal(result.errors("live"), result.errors("replay")))

    def test_snapshot_model_refuses_caches_without_snapshots(self) -> None:
        model, _target = _drifting_model(4)
        restored = SnapshotGRU.from_state_dict(model.state_dict())
        with self.assertRaises(RuntimeError):
            restored.backward(0.1)


class HistogramTests(unittest.TestCase):
    def test_quantiles_are_upper_bin_edges(self) -> None:
        values = np.array([0.0, 1e-5, 1e-3, 1e-3, 2e-2])
        histogram = log_histogram(values)
        self.assertEqual(histogram["zeros"], 1)
        median = histogram_quantile([histogram], 0.5)
        self.assertGreaterEqual(median, 1e-3)
        self.assertLess(median, 1e-3 * 10**HISTOGRAM_WIDTH * 1.0001)
        self.assertEqual(histogram_quantile([histogram], 0.0), 0.0)

    def test_overflow_and_nonfinite_are_counted_not_dropped(self) -> None:
        histogram = log_histogram(np.array([1e5, np.nan, np.inf, 1e-20]))
        self.assertEqual(histogram["above"], 1)
        self.assertEqual(histogram["below"], 1)
        self.assertEqual(histogram["nonfinite"], 2)
        self.assertEqual(histogram_quantile([histogram], 1.0), float("inf"))

    def test_pooling_is_order_independent(self) -> None:
        rng = np.random.default_rng(0)
        a = log_histogram(10 ** rng.uniform(-8, 0, 500))
        b = log_histogram(10 ** rng.uniform(-4, 1, 300))
        for q in (0.1, 0.5, 0.99):
            self.assertEqual(histogram_quantile([a, b], q), histogram_quantile([b, a], q))
        self.assertEqual(HISTOGRAM_LOW + HISTOGRAM_BINS * HISTOGRAM_WIDTH, 2.0)


if __name__ == "__main__":
    unittest.main()


class LoopGainTests(unittest.TestCase):
    """The closed-loop measurements against finite differences of the loop they describe."""

    def _trained(self, **overrides: object) -> AAA1KGRU:
        rng = np.random.default_rng(21)
        model = AAA1KGRU(seed=4, **{**CHAMPION_CONFIGURATION, **overrides})
        model.parameters["W_o"][:] = rng.normal(0.0, 0.7, model.parameters["W_o"].shape)
        for _ in range(5):
            model.forward(rng.normal(size=3))
        return model

    @staticmethod
    def _loop(model: AAA1KGRU, h_prev: np.ndarray, x: np.ndarray) -> np.ndarray:
        from research.aaa_1k_loop.tbptt import step

        new = step(model.parameters, x, h_prev)
        return np.concatenate([new.h, [-new.output[0]]])  # e_t = d* - o_t, d* constant

    def test_closed_loop_jacobian_matches_finite_differences(self) -> None:
        from research.aaa_1k_loop.gain import closed_loop_jacobian

        model = self._trained()
        cache = model._caches[-1]
        analytic = closed_loop_jacobian(model)
        numeric = np.zeros_like(analytic)
        epsilon = 1e-6
        for j in range(17):
            plus_h, minus_h = cache.h_prev.copy(), cache.h_prev.copy()
            plus_x, minus_x = cache.x.copy(), cache.x.copy()
            if j < 16:
                plus_h[j] += epsilon
                minus_h[j] -= epsilon
            else:
                plus_x[2] += epsilon
                minus_x[2] -= epsilon
            numeric[:, j] = (self._loop(model, plus_h, plus_x) - self._loop(model, minus_h, minus_x)) / (
                2 * epsilon
            )
        self.assertLess(np.max(np.abs(analytic - numeric)), 1e-7)

    def test_direct_gain_is_the_error_to_error_derivative(self) -> None:
        from research.aaa_1k_loop.gain import loop_measurements

        model = self._trained()
        cache = model._caches[-1]
        plus, minus = cache.x.copy(), cache.x.copy()
        plus[2] += 1e-6
        minus[2] -= 1e-6
        numeric = (
            self._loop(model, cache.h_prev, plus)[-1] - self._loop(model, cache.h_prev, minus)[-1]
        ) / 2e-6
        self.assertAlmostEqual(loop_measurements(model)["direct_gain"], numeric, places=7)

    def test_open_loop_ablation_reports_zero_feedback(self) -> None:
        from research.aaa_1k_loop.gain import loop_measurements

        measured = loop_measurements(self._trained(zero_error_input=True))
        self.assertEqual(measured["direct_gain"], 0.0)
        self.assertAlmostEqual(measured["closed_loop_radius"], measured["state_radius"])

    def test_gain_agent_is_read_only(self) -> None:
        from research.aaa_1k_loop.gain import GainAgent

        stream = coarse_speed_stream(424242, steps=120, regime_length=10_000)
        plain = NeuralAgent(AAA1KGRU(seed=2, **CHAMPION_CONFIGURATION), name="plain")
        measured = GainAgent(AAA1KGRU(seed=2, **CHAMPION_CONFIGURATION), name="measured")
        result = run_stream(stream, [plain, measured])
        self.assertTrue(np.array_equal(result.errors("plain"), result.errors("measured")))
        self.assertEqual(len(measured.gain_trace), len(result.steps))

    def test_trailing_window_and_first_index(self) -> None:
        from research.aaa_1k_loop.diagnosis4b import first_index, trailing

        values = np.arange(10, dtype=float)
        rolled = trailing(values, 3, np.median)
        self.assertTrue(np.all(np.isnan(rolled[:2])))
        self.assertEqual(rolled[2], 1.0)
        self.assertEqual(first_index(rolled > 4.5), 6)
        self.assertEqual(first_index(values > 2, start=5), 5)
        self.assertIsNone(first_index(values > 100))


class OvershootTests(unittest.TestCase):
    def test_overshoot_model_follows_the_champion_bitwise(self) -> None:
        from research.aaa_1k_loop.overshoot import OvershootGRU

        stream = coarse_speed_stream(424242, steps=150, regime_length=10_000)
        champion = NeuralAgent(AAA1KGRU(seed=2, **CHAMPION_CONFIGURATION), name="champion")
        measured = NeuralAgent(OvershootGRU(seed=2, **CHAMPION_CONFIGURATION), name="measured")
        result = run_stream(stream, [champion, measured])
        self.assertTrue(np.array_equal(result.errors("champion"), result.errors("measured")))
        self.assertIn(len(result.steps) - len(measured.model.overshoot), (0, 1))

    def test_small_steps_agree_with_the_linearization(self) -> None:
        from research.aaa_1k_loop.overshoot import OvershootGRU

        rng = np.random.default_rng(5)
        model = OvershootGRU(seed=1, **{**CHAMPION_CONFIGURATION, "learning_rate": 1e-4})
        model.parameters["W_o"][:] = rng.normal(0.0, 0.5, model.parameters["W_o"].shape)
        for _ in range(6):
            model.forward(rng.normal(size=3))
            model.learn(float(rng.normal()))
        for record in model.overshoot[1:]:
            # |r'|/|r| = |1 - kappa| to first order; the aux head makes it approximate
            self.assertAlmostEqual(record["amplification"], abs(1.0 - record["kappa"]), places=3)
            self.assertLess(record["amplification"], 1.0)
            self.assertGreaterEqual(record["error_input_share"], 0.0)
            self.assertLessEqual(record["error_input_share"] + record["head_share"], 1.0 + 1e-12)

    def test_a_huge_step_is_reported_as_overshoot(self) -> None:
        from research.aaa_1k_loop.overshoot import OvershootGRU

        rng = np.random.default_rng(6)
        model = OvershootGRU(
            seed=1, **{**CHAMPION_CONFIGURATION, "learning_rate": 50.0, "gradient_clip": None}
        )
        model.parameters["W_o"][:] = rng.normal(0.0, 0.5, model.parameters["W_o"].shape)
        model.forward(np.array([0.1, 0.5, 2.0]))
        model.learn(-1.0)
        self.assertGreater(model.overshoot[-1]["amplification"], 1.0)
        self.assertGreater(model.overshoot[-1]["kappa"], 2.0)


class UnfoldingTests(unittest.TestCase):
    def test_trace_agent_follows_the_champion_bitwise(self) -> None:
        from research.aaa_1k_loop.unfolding import UnfoldTraceAgent

        stream = coarse_speed_stream(424242, steps=400, regime_length=10_000)
        champion = NeuralAgent(AAA1KGRU(seed=2, **CHAMPION_CONFIGURATION), name="champion")
        traced = UnfoldTraceAgent(AAA1KGRU(seed=2, **CHAMPION_CONFIGURATION), name="traced")
        result = run_stream(stream, [champion, traced])
        self.assertTrue(np.array_equal(result.errors("champion"), result.errors("traced")))
        self.assertEqual(len(traced.mirror_trace), champion.trained_steps)

    def test_a_raw_prediction_past_the_wall_mirrors_an_interior_observation(self) -> None:
        # The self-confirming branch: input 0.035 inside, raw prediction past the lower wall.
        from aaa.predictors import unfold_observation

        self.assertAlmostEqual(unfold_observation(0.04, -0.03, 0.0, 1.0), -0.04)
        self.assertAlmostEqual(unfold_observation(0.04, 0.03, 0.0, 1.0), 0.04)

    def test_dead_reckoning_reference_ignores_the_prediction(self) -> None:
        from research.aaa_1k_loop.unfolding import DeadReckoningUnfoldAgent, UnfoldTraceAgent

        for cls in (UnfoldTraceAgent, DeadReckoningUnfoldAgent):
            agent = cls(AAA1KGRU(seed=0, **CHAMPION_CONFIGURATION), name="a")
            agent.begin_episode()
            agent.accept_observation(0.040)
            agent.accept_observation(0.035)
            agent.predict()
            agent._raw_prediction = -0.03  # force the prediction past the wall
            target = agent._training_target(0.040)
            if cls is UnfoldTraceAgent:
                self.assertTrue(agent.mirror_trace[-1])
                self.assertAlmostEqual(target, (-0.040 - 0.035) / agent.scales.displacement_scale)
            else:
                self.assertFalse(agent.mirror_trace[-1])
                self.assertAlmostEqual(target, (0.040 - 0.035) / agent.scales.displacement_scale)

    def test_longest_runs(self) -> None:
        from research.aaa_1k_loop.unfolding import longest_runs

        self.assertEqual(longest_runs([False, True, True, False, True]), [(1, 2), (4, 1)])
        self.assertEqual(longest_runs([]), [])
