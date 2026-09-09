import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaa.config import WorldConfig
from aaa.environment import MovingDotEnvironment
from aaa.experiment import StepRecord
from aaa.metrics import complete_rolling_mean, recovery_metric
from aaa.predictors import OnlineLinearPredictor, OnlineRLSPredictor


def _record(step: int, error: float, *, changed: bool = False) -> StepRecord:
    return StepRecord(
        seed=1,
        environment_seed=2,
        episode=0,
        scenario="changed",
        step=step,
        target_step=step + 1,
        history=(0.1, 0.2, 0.3, 0.4),
        current_observation=0.4,
        actual_next_position=0.4,
        bounced=False,
        changed=changed,
        predictions={"p": {"raw": 0.4 + error, "scored": 0.4 + error, "absolute_error": abs(error), "normalized_absolute_error": abs(error)}},
        updates_enabled={"p": False},
    )


class ScientificRepairTests(unittest.TestCase):
    def test_sampling_is_relative_to_arbitrary_bounds(self):
        config = WorldConfig(lower_bound=10.0, upper_bound=30.0, dt=0.1, steps_per_episode=20, speed_min=0.1, speed_max=0.2)
        env = MovingDotEnvironment("bouncing", seed=9, config=config)
        self.assertGreaterEqual(env.observe(), 10.0)
        self.assertLessEqual(env.observe(), 30.0)
        for _ in range(config.steps_per_episode):
            transition = env.advance()
            self.assertGreaterEqual(transition.position, 10.0)
            self.assertLessEqual(transition.position, 30.0)

    def test_exact_boundary_contact_reverses_outward_velocity(self):
        config = WorldConfig(lower_bound=-2.0, upper_bound=3.0, dt=1.0, steps_per_episode=2, history_length=2)
        env = MovingDotEnvironment("bouncing", seed=1, config=config, initial_position=3.0, initial_velocity=1.0)
        transition = env.advance()
        self.assertTrue(transition.bounced)
        self.assertEqual(transition.position, 2.0)
        self.assertLess(env.velocity, 0.0)

    def test_invalid_world_values_are_rejected(self):
        with self.assertRaises(ValueError):
            WorldConfig(dt=0)
        with self.assertRaises(ValueError):
            WorldConfig(history_length=1)
        with self.assertRaises(ValueError):
            WorldConfig(lower_bound=float("nan"))
        with self.assertRaises(ValueError):
            WorldConfig(speed_min=0.4, speed_max=0.2)

    def test_recovery_requires_complete_windows_and_reports_confirmation(self):
        records = [_record(0, 0.001), _record(1, 0.001), _record(2, 0.001), _record(3, 0.1, changed=True), _record(4, 0.01), _record(5, 0.01), _record(6, 0.01), _record(7, 0.01)]
        result = recovery_metric(records, "p", post_change_window=4, rolling_window=2, sustain_windows=2, tolerance_multiplier=1.5, tolerance_floor=0.01)
        self.assertTrue(result["applicable"])
        self.assertEqual(result["recovery_onset_transition"], 2)
        self.assertEqual(result["recovery_confirmation_transition"], 4)
        partial = recovery_metric(records[:6], "p", post_change_window=4, rolling_window=2, sustain_windows=2)
        self.assertEqual(partial["status"], "censored")
        self.assertEqual(complete_rolling_mean([1, 2, 3], 2), [1.5, 2.5])

    def test_gradient_matches_finite_difference(self):
        history = (0.1, 0.2, 0.3, 0.4)
        target = 0.43
        weights = np.array([0.1, -0.2, 0.3, -0.1, 0.05], dtype=float)
        features = OnlineLinearPredictor.features(history)
        delta = float(weights @ features)
        target_delta = target - history[-1]
        analytic = (delta - target_delta) * features
        epsilon = 1e-6
        numeric = []
        for index in range(len(weights)):
            plus = weights.copy(); plus[index] += epsilon
            minus = weights.copy(); minus[index] -= epsilon
            loss_plus = 0.5 * (float(plus @ features) - target_delta) ** 2
            loss_minus = 0.5 * (float(minus @ features) - target_delta) ** 2
            numeric.append((loss_plus - loss_minus) / (2 * epsilon))
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)

    def test_rls_checkpoint_preserves_covariance_and_prediction(self):
        model = OnlineRLSPredictor(displacement_scale=0.01)
        history = (0.1, 0.2, 0.3, 0.4)
        model.update(history, 0.41)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = OnlineRLSPredictor.load(path, update_enabled=False)
            self.assertEqual(model.update_count, loaded.update_count)
            np.testing.assert_allclose(model.covariance, loaded.covariance)
            self.assertAlmostEqual(model.predict(history), loaded.predict(history))
            self.assertTrue(np.all(np.isfinite(loaded.covariance)))
            np.testing.assert_allclose(loaded.covariance, loaded.covariance.T)


if __name__ == "__main__":
    unittest.main()
