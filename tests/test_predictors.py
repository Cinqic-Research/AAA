import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaa.predictors import (
    ConstantMotionPredictor,
    InvalidLearnerState,
    OnlineLinearPredictor,
    OnlineRLSPredictor,
    PersistencePredictor,
    ReflectedConstantMotionPredictor,
    reflect_prediction,
    unfold_observation,
    validate_covariance,
)


class BaselineTests(unittest.TestCase):
    def test_baseline_formulas_use_observations_only(self):
        history = (0.1, 0.2, 0.4, 0.7)
        self.assertAlmostEqual(PersistencePredictor().predict(history), 0.7)
        self.assertAlmostEqual(ConstantMotionPredictor().predict(history), 1.0)

    def test_reflected_constant_motion_uses_the_same_public_policy(self):
        history = (0.1, 0.2, 0.4, 0.7)
        raw = ConstantMotionPredictor().predict(history)
        reflected = ReflectedConstantMotionPredictor(lower_bound=0.0, upper_bound=1.0).predict(history)
        self.assertAlmostEqual(reflected, reflect_prediction(raw, 0.0, 1.0))
        self.assertLessEqual(reflected, 1.0)

    def test_reflected_baseline_is_exact_at_a_wall(self):
        # The bouncing world reflects with exactly this map, so the fair
        # baseline is analytically exact on a single-wall transition.
        history = (0.90, 0.94, 0.98, 1.0)
        self.assertAlmostEqual(
            ReflectedConstantMotionPredictor(lower_bound=0.0, upper_bound=1.0).predict(history), 0.98
        )

    def test_baselines_reject_non_finite_history(self):
        for predictor in (PersistencePredictor(), ConstantMotionPredictor()):
            with self.assertRaises(ValueError):
                predictor.predict((0.1, 0.2, 0.3, float("nan")))

    def test_reflected_baseline_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            ReflectedConstantMotionPredictor(lower_bound=1.0, upper_bound=1.0)


class UnfoldTests(unittest.TestCase):
    def test_unfold_identity_inside_the_interval(self):
        self.assertAlmostEqual(unfold_observation(0.4, 0.41, 0.0, 1.0), 0.4)

    def test_unfold_recovers_the_pre_image_of_a_wall_reflection(self):
        # A raw prediction of 1.05 folds to 0.95; unfolding against the raw
        # prediction recovers 1.05 rather than the folded observation.
        self.assertAlmostEqual(unfold_observation(0.95, 1.05, 0.0, 1.0), 1.05)
        self.assertAlmostEqual(unfold_observation(0.05, -0.05, 0.0, 1.0), -0.05)

    def test_unfold_is_bounds_relative(self):
        self.assertAlmostEqual(unfold_observation(28.0, 32.0, 10.0, 30.0), 32.0)

    def test_unfold_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            unfold_observation(0.5, 0.5, 1.0, 1.0)
        with self.assertRaises(ValueError):
            unfold_observation(float("nan"), 0.5, 0.0, 1.0)


class LegacyLinearTests(unittest.TestCase):
    def test_linear_feature_order_and_gradient_update(self):
        history = (0.1, 0.2, 0.3, 0.4)
        target = 0.43
        weights = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
        learning_rate = 0.2
        model = OnlineLinearPredictor(learning_rate=learning_rate, weights=weights)
        features = np.array([1.0, *history])
        predicted_delta = float(np.dot(weights, features))
        target_delta = target - history[-1]
        expected = weights - learning_rate * (predicted_delta - target_delta) * features
        model.update(history, target)
        np.testing.assert_allclose(model.weights, expected)

    def test_prediction_does_not_update_parameters(self):
        model = OnlineLinearPredictor(weights=[0.1, 0.2, 0.3, 0.4, 0.5])
        before = model.weights.copy()
        model.predict((0.1, 0.2, 0.3, 0.4))
        np.testing.assert_array_equal(model.weights, before)
        self.assertEqual(model.update_count, 0)

    def test_save_load_preserves_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model = OnlineLinearPredictor(learning_rate=0.04, weights=[0.1, 0.2, 0.3, 0.4, 0.5])
            model.save(path)
            loaded = OnlineLinearPredictor.load(path, update_enabled=False)
            history = (0.1, 0.2, 0.3, 0.4)
            self.assertAlmostEqual(model.predict(history), loaded.predict(history))
            self.assertFalse(loaded.update_enabled)

    def test_legacy_learner_validates_finite_inputs(self):
        model = OnlineLinearPredictor(update_enabled=True)
        with self.assertRaises(ValueError):
            model.predict((0.1, 0.2, 0.3, float("inf")))
        with self.assertRaises(ValueError):
            model.update((0.1, 0.2, 0.3, 0.4), float("nan"))
        with self.assertRaises(ValueError):
            OnlineLinearPredictor(learning_rate=float("nan"))
        with self.assertRaises(ValueError):
            OnlineLinearPredictor(learning_rate=-1.0)
        with self.assertRaises(ValueError):
            OnlineLinearPredictor(weights=[0.0, float("nan"), 0.0, 0.0, 0.0])

    def test_legacy_checkpoint_rejects_missing_or_wrong_fields(self):
        state = OnlineLinearPredictor().state_dict()
        bad_version = {**state, "format_version": "aaa.linear_predictor.v0"}
        with self.assertRaises(ValueError):
            OnlineLinearPredictor.from_state_dict(bad_version)
        missing = {key: value for key, value in state.items() if key != "learning_rate"}
        with self.assertRaises(ValueError):
            OnlineLinearPredictor.from_state_dict(missing)


class RLSStateTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_bit_exact(self):
        model = OnlineRLSPredictor(displacement_scale=0.004, forgetting=0.9)
        for index in range(50):
            velocity = 0.002 * (1 + index % 3)
            position = 0.2 + 0.001 * index
            model.update(
                (position - 3 * velocity, position - 2 * velocity, position - velocity, position),
                position + velocity,
            )
        state = model.state_dict()
        reloaded = OnlineRLSPredictor.from_state_dict(state, update_enabled=True)
        self.assertEqual(
            json.dumps(state, sort_keys=True), json.dumps(reloaded.state_dict(), sort_keys=True)
        )

    def test_save_load_preserves_prediction_and_covariance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model = OnlineRLSPredictor(displacement_scale=0.01)
            history = (0.1, 0.2, 0.3, 0.4)
            model.update(history, 0.41)
            model.save(path)
            loaded = OnlineRLSPredictor.load(path, update_enabled=False)
            self.assertEqual(model.update_count, loaded.update_count)
            np.testing.assert_allclose(model.covariance, loaded.covariance)
            self.assertAlmostEqual(model.predict(history), loaded.predict(history))
            np.testing.assert_allclose(loaded.covariance, loaded.covariance.T)

    def test_covariance_is_always_symmetric_and_positive_semidefinite(self):
        model = OnlineRLSPredictor(displacement_scale=0.004, forgetting=0.5, forgetting_mode="directional")
        rng = np.random.default_rng(3)
        for _ in range(2000):
            position = float(rng.uniform(0.1, 0.9))
            velocity = float(rng.uniform(-0.004, 0.004))
            model.update(
                (position - 3 * velocity, position - 2 * velocity, position - velocity, position),
                position + velocity,
            )
        diagnostics = model.check_state()
        self.assertGreater(diagnostics["min_eigenvalue"], 0.0)
        self.assertLess(diagnostics["max_asymmetry"], 1e-12)

    def test_invalid_construction_parameters_are_rejected(self):
        cases = (
            {"forgetting": 0.0},
            {"forgetting": 1.5},
            {"forgetting": float("nan")},
            {"ridge": 0.0},
            {"ridge": -1.0},
            {"displacement_scale": 0.0},
            {"lower_bound": 1.0, "upper_bound": 1.0},
            {"trace_bound": 0.0},
            {"dead_zone": -1e-9},
            {"detector_multiplier": -1.0},
            {"detector_decay": 0.0},
            {"detector_decay": 1.5},
            {"feature_set": "nonsense"},
            {"forgetting_mode": "nonsense"},
            {"update_count": -1},
            {"error_ewma": -1.0},
            {"weights": [1.0, 2.0]},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                settings = {"displacement_scale": 0.01, **overrides}
                with self.assertRaises(ValueError):
                    OnlineRLSPredictor(**settings)

    def test_invalid_covariance_states_are_rejected_not_silently_repaired(self):
        base = OnlineRLSPredictor(displacement_scale=0.01).state_dict()
        identity = np.eye(3)
        cases = {
            "negative_definite": (-identity).tolist(),
            "indefinite": np.diag([1.0, -1.0, 1.0]).tolist(),
            "asymmetric": [[1.0, 5.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "nearly_singular": np.diag([1.0, 1e-18, 1.0]).tolist(),
            "zero": np.zeros((3, 3)).tolist(),
            "non_finite": [[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "badly_conditioned": np.diag([1e12, 1.0, 1e-12]).tolist(),
        }
        for label, covariance in cases.items():
            with self.subTest(label=label):
                state = {**base, "covariance": covariance}
                state.pop("sqrt_factor")
                with self.assertRaises(ValueError):
                    OnlineRLSPredictor.from_state_dict(state)

    def test_valid_covariance_is_accepted(self):
        base = OnlineRLSPredictor(displacement_scale=0.01).state_dict()
        state = {**base, "covariance": (2.0 * np.eye(3)).tolist()}
        state.pop("sqrt_factor")
        model = OnlineRLSPredictor.from_state_dict(state)
        np.testing.assert_allclose(model.covariance, 2.0 * np.eye(3))

    def test_checkpoint_rejects_wrong_version_and_missing_fields(self):
        state = OnlineRLSPredictor(displacement_scale=0.01).state_dict()
        with self.assertRaises(ValueError):
            OnlineRLSPredictor.from_state_dict({**state, "format_version": "aaa.rls_predictor.v1"})
        with self.assertRaises(ValueError):
            OnlineRLSPredictor.from_state_dict({**state, "history_length": 3})
        for field in ("weights", "covariance", "forgetting", "ridge", "feature_set"):
            with self.subTest(field=field):
                partial = {key: value for key, value in state.items() if key != field}
                with self.assertRaises(ValueError):
                    OnlineRLSPredictor.from_state_dict(partial)

    def test_update_requires_finite_target_and_history(self):
        model = OnlineRLSPredictor(displacement_scale=0.01, update_enabled=True)
        with self.assertRaises(ValueError):
            model.update((0.1, 0.2, 0.3, 0.4), float("inf"))
        with self.assertRaises(ValueError):
            model.update((0.1, 0.2, 0.3, float("nan")), 0.5)
        with self.assertRaises(ValueError):
            model.predict((0.1, 0.2))

    def test_frozen_copy_never_updates(self):
        model = OnlineRLSPredictor(displacement_scale=0.01, update_enabled=False)
        before = model.state_dict()
        model.update((0.1, 0.2, 0.3, 0.4), 0.5)
        self.assertEqual(json.dumps(before, sort_keys=True), json.dumps(model.state_dict(), sort_keys=True))


class ValidateCovarianceTests(unittest.TestCase):
    def test_diagnostics_are_reported_for_valid_state(self):
        diagnostics = validate_covariance(np.diag([1.0, 2.0, 4.0]))
        self.assertAlmostEqual(diagnostics["min_eigenvalue"], 1.0)
        self.assertAlmostEqual(diagnostics["max_eigenvalue"], 4.0)
        self.assertAlmostEqual(diagnostics["condition_number"], 4.0)
        self.assertAlmostEqual(diagnostics["trace"], 7.0)

    def test_non_square_matrix_is_rejected(self):
        with self.assertRaises(InvalidLearnerState):
            validate_covariance(np.ones((2, 3)))


class ReflectionTests(unittest.TestCase):
    def test_reflection_handles_multiple_crossings(self):
        self.assertTrue(0.0 <= reflect_prediction(2.5, 0.0, 1.0) <= 1.0)
        self.assertTrue(0.0 <= reflect_prediction(-3.25, 0.0, 1.0) <= 1.0)

    def test_reflection_rejects_invalid_bounds_and_values(self):
        with self.assertRaises(ValueError):
            reflect_prediction(0.5, 1.0, 0.0)
        with self.assertRaises(ValueError):
            reflect_prediction(float("nan"), 0.0, 1.0)

    def test_reflection_is_identity_inside_bounds(self):
        for value in (0.0, 0.25, 1.0):
            self.assertEqual(reflect_prediction(value, 0.0, 1.0), value)


if __name__ == "__main__":
    unittest.main()
