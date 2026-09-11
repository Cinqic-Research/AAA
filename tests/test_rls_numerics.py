"""Numerical stability suite for the candidate learner.

These are direct regression tests for the reproduced pre-repair failures:
covariance-form RLS with exponential forgetting lost positive semidefiniteness
after a few hundred weakly exciting updates and overflowed to infinity after a
few thousand. The observed pre-repair failure indices are recorded in
``docs/evidence/pre_repair_probes.json``.
"""

from __future__ import annotations

import json
import math
import unittest

import numpy as np

from aaa.predictors import (
    InvalidLearnerState,
    OnlineRLSPredictor,
    _rank_one_inflate,
    batch_least_squares,
    validate_covariance,
)

LONG_RUN = 30_000


def _model(**kwargs) -> OnlineRLSPredictor:
    settings = {"displacement_scale": 0.004, "update_enabled": True}
    settings.update(kwargs)
    return OnlineRLSPredictor(**settings)


def _drive(model: OnlineRLSPredictor, stream, count: int) -> None:
    for index in range(count):
        history, target = stream(index)
        model.update(history, target)


class WeakExcitationTests(unittest.TestCase):
    """Every case here diverged or lost PSD in the pre-repair implementation."""

    def _assert_survives(self, model: OnlineRLSPredictor, stream, count: int = LONG_RUN) -> None:
        _drive(model, stream, count)
        diagnostics = model.check_state()
        self.assertTrue(np.all(np.isfinite(model.weights)))
        self.assertGreater(diagnostics["min_eigenvalue"], 0.0)
        self.assertLess(diagnostics["condition_number"], 1e12)

    def test_repeated_identical_input(self):
        self._assert_survives(_model(forgetting=0.90), lambda i: ((0.5, 0.5, 0.5, 0.5), 0.5))

    def test_stationary_observation_stream(self):
        self._assert_survives(_model(forgetting=0.95), lambda i: ((0.3, 0.3, 0.3, 0.3), 0.3))

    def test_zero_displacement_with_directional_forgetting(self):
        self._assert_survives(
            _model(forgetting=0.5, forgetting_mode="directional"), lambda i: ((0.7, 0.7, 0.7, 0.7), 0.7)
        )

    def test_constant_features(self):
        def stream(index: int):
            return (0.2, 0.2, 0.2, 0.2), 0.2

        self._assert_survives(_model(forgetting=0.90, feature_set="displacement_only"), stream)

    def test_slowly_varying_motion(self):
        def stream(index: int):
            velocity = 1e-5
            position = 0.2 + velocity * (index % 20_000)
            return (
                position - 3 * velocity,
                position - 2 * velocity,
                position - velocity,
                position,
            ), position + velocity

        self._assert_survives(_model(forgetting=0.90), stream)

    def test_near_collinear_features(self):
        def stream(index: int):
            velocity = 1e-12
            position = 0.5 + 1e-12 * index
            return (
                position - 3 * velocity,
                position - 2 * velocity,
                position - velocity,
                position,
            ), position + velocity

        self._assert_survives(_model(forgetting=0.90), stream, count=10_000)

    def test_abrupt_dynamics_change(self):
        def stream(index: int):
            velocity = 0.004 if index < LONG_RUN // 2 else -0.001
            position = 0.5
            return (
                position - 3 * velocity,
                position - 2 * velocity,
                position - velocity,
                position,
            ), position + velocity

        self._assert_survives(_model(forgetting=0.90), stream)

    @staticmethod
    def _wide_world_stream(index: int):
        velocity = 0.5
        position = -400.0 + velocity * (index % 1500)
        return (
            (position - 3 * velocity, position - 2 * velocity, position - velocity, position),
            position + velocity,
        )

    def test_extreme_but_correctly_scaled_bounds_are_stable(self):
        # A 1000-wide interval is fine as long as displacement_scale is set to
        # the typical step size, which is what the benchmark spec derives it
        # from (dt * speed_max).
        model = OnlineRLSPredictor(
            lower_bound=-500.0,
            upper_bound=500.0,
            displacement_scale=0.5,
            forgetting=0.9,
            update_enabled=True,
        )
        _drive(model, self._wide_world_stream, 5_000)
        diagnostics = model.check_state()
        self.assertGreater(diagnostics["min_eigenvalue"], 0.0)
        self.assertLess(diagnostics["condition_number"], 1e12)

    def test_a_badly_scaled_displacement_is_reported_not_hidden(self):
        # displacement_scale 500x smaller than the actual step makes the
        # regressor columns differ by three orders of magnitude. The estimate
        # still runs, but the declared conditioning invariant is violated and
        # check_state() must say so out loud rather than continue quietly.
        model = OnlineRLSPredictor(
            lower_bound=-500.0,
            upper_bound=500.0,
            displacement_scale=1e-3,
            forgetting=0.9,
            update_enabled=True,
        )
        _drive(model, self._wide_world_stream, 5_000)
        self.assertTrue(np.all(np.isfinite(model.weights)))
        with self.assertRaises(InvalidLearnerState) as caught:
            model.check_state()
        self.assertIn("condition number", str(caught.exception))

    def test_well_excited_stream_never_suspends_forgetting(self):
        rng = np.random.default_rng(11)
        model = _model(forgetting=0.90)

        def stream(index: int):
            velocity = float(rng.uniform(-0.004, 0.004))
            position = float(rng.uniform(0.1, 0.9))
            return (
                position - 3 * velocity,
                position - 2 * velocity,
                position - velocity,
                position,
            ), position + velocity

        _drive(model, stream, 5_000)
        self.assertEqual(model.forgetting_suspensions, 0)
        self.assertLess(model.check_state()["trace"], 100.0)

    def test_trace_bound_suspensions_are_counted_not_hidden(self):
        model = _model(forgetting=0.90, trace_bound=1e5)
        _drive(model, lambda i: ((0.5, 0.5, 0.5, 0.5), 0.5), 5_000)
        self.assertGreater(model.forgetting_suspensions, 0)
        self.assertLessEqual(model.check_state()["trace"], 1e5 * 1.01)


class BatchReferenceTests(unittest.TestCase):
    """Cross-check against an independent stable solver, not the same formula."""

    def test_no_forgetting_matches_ridge_batch_least_squares(self):
        model = _model(forgetting=1.0, ridge=1e-4)
        rng = np.random.default_rng(17)
        features, targets = [], []
        for _ in range(600):
            velocity = float(rng.uniform(-0.004, 0.004))
            position = float(rng.uniform(0.1, 0.9))
            history = (position - 3 * velocity, position - 2 * velocity, position - velocity, position)
            target = position + velocity + float(rng.normal(0, 1e-4))
            features.append(model.features(history))
            targets.append(target - position)
            model.update(history, target)
        reference = batch_least_squares(np.asarray(features), np.asarray(targets), ridge=1e-4)
        np.testing.assert_allclose(model.weights, reference, rtol=1e-9, atol=1e-14)

    def test_batch_reference_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            batch_least_squares(np.ones((3, 2)), np.ones(4), ridge=1e-4)
        with self.assertRaises(ValueError):
            batch_least_squares(np.ones((3, 2)), np.ones(3), ridge=0.0)

    def test_float64_reference_behaviour(self):
        model = _model(forgetting=1.0)
        self.assertEqual(model.weights.dtype, np.float64)
        self.assertEqual(model.covariance.dtype, np.float64)


class SquareRootFactorTests(unittest.TestCase):
    def test_rank_one_inflation_is_exact(self):
        factor = np.array([[2.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.1, 0.2, 3.0]])
        direction = np.array([1.0, 2.0, -1.0])
        weight = 0.7
        inflated = _rank_one_inflate(factor, direction, weight)
        np.testing.assert_allclose(
            inflated @ inflated.T, factor @ factor.T + weight * np.outer(direction, direction), atol=1e-12
        )

    def test_covariance_matches_the_covariance_form_recursion(self):
        """Square-root propagation must equal the textbook recursion exactly."""

        forgetting = 0.9
        model = _model(forgetting=forgetting, ridge=1e-4)
        covariance = np.eye(3) / 1e-4
        weights = np.zeros(3)
        rng = np.random.default_rng(23)
        for _ in range(200):
            velocity = float(rng.uniform(-0.004, 0.004))
            position = float(rng.uniform(0.2, 0.8))
            history = (position - 3 * velocity, position - 2 * velocity, position - velocity, position)
            target = position + velocity
            phi = model.features(history)
            # Reference covariance-form step, computed independently here.
            denominator = forgetting + float(phi @ covariance @ phi)
            gain = (covariance @ phi) / denominator
            error = (target - position) - float(phi @ weights)
            weights = weights + gain * error
            covariance = (covariance - np.outer(gain, phi @ covariance)) / forgetting
            model.update(history, target)
        np.testing.assert_allclose(model.weights, weights, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(model.covariance, covariance, rtol=1e-6, atol=1e-9)


class SerializationAndResumeTests(unittest.TestCase):
    def test_resume_from_checkpoint_is_bit_identical(self):
        rng = np.random.default_rng(29)
        stream = []
        for _ in range(200):
            velocity = float(rng.uniform(-0.004, 0.004))
            position = float(rng.uniform(0.2, 0.8))
            stream.append(
                (
                    (position - 3 * velocity, position - 2 * velocity, position - velocity, position),
                    position + velocity,
                )
            )
        uninterrupted = _model(forgetting=0.9)
        for history, target in stream:
            uninterrupted.update(history, target)
        interrupted = _model(forgetting=0.9)
        for history, target in stream[:120]:
            interrupted.update(history, target)
        resumed = OnlineRLSPredictor.from_state_dict(interrupted.state_dict(), update_enabled=True)
        for history, target in stream[120:]:
            resumed.update(history, target)
        np.testing.assert_array_equal(uninterrupted.weights, resumed.weights)
        np.testing.assert_array_equal(uninterrupted.covariance, resumed.covariance)
        self.assertEqual(uninterrupted.update_count, resumed.update_count)


class DetectorAndDeadZoneTests(unittest.TestCase):
    def test_detector_is_quiet_once_the_law_is_identified(self):
        model = _model(forgetting=0.5, detector_multiplier=8.0, unfold_target=True)

        def stream(index: int):
            velocity = 0.002
            position = 0.2 + velocity * (index % 300)
            return (
                position - 3 * velocity,
                position - 2 * velocity,
                position - velocity,
                position,
            ), position + velocity

        _drive(model, stream, 2_000)
        self.assertLess(model.detected_surprises, 20)
        self.assertEqual(model.forgetting_suspensions, 0)

    def test_detector_fires_on_a_genuine_law_change(self):
        model = _model(forgetting=0.5, detector_multiplier=8.0, unfold_target=True)

        def early(index: int):
            velocity = 0.002
            position = 0.2 + velocity * (index % 300)
            return (
                position - 3 * velocity,
                position - 2 * velocity,
                position - velocity,
                position,
            ), position + velocity

        _drive(model, early, 1_000)
        quiet = model.detected_surprises
        # The law changes: displacement is no longer the previous displacement.
        for index in range(50):
            velocity = 0.002
            position = 0.4 + velocity * index
            model.update(
                (position - 3 * velocity, position - 2 * velocity, position - velocity, position),
                position + 4.0 * velocity,
            )
        self.assertGreater(model.detected_surprises - quiet, 5)

    def test_dead_zone_skips_are_counted_and_change_nothing_else(self):
        model = _model(forgetting=0.9, dead_zone=1e-3)
        before = model.weights.copy()
        model.update((0.5, 0.5, 0.5, 0.5), 0.5)
        self.assertEqual(model.dead_zone_skips, 1)
        self.assertEqual(model.update_count, 0)
        np.testing.assert_array_equal(model.weights, before)

    def test_unfolding_makes_a_wall_transition_an_ordinary_sample(self):
        naive = _model(forgetting=1.0, unfold_target=False)
        consistent = _model(forgetting=1.0, unfold_target=True)
        history = (0.90, 0.94, 0.98, 1.0)
        # The environment folds 1.04 back to 0.96 at the upper wall.
        for model in (naive, consistent):
            model.weights[:] = np.array([0.0, 0.004, 0.0])
        naive.update(history, 0.96)
        consistent.update(history, 0.96)
        self.assertGreater(
            float(np.linalg.norm(naive.weights - np.array([0.0, 0.004, 0.0]))),
            float(np.linalg.norm(consistent.weights - np.array([0.0, 0.004, 0.0]))),
        )


class InvalidStateTests(unittest.TestCase):
    def test_validate_covariance_rejects_each_invalid_class(self):
        identity = np.eye(3)
        for label, matrix in (
            ("negative definite", -identity),
            ("indefinite", np.diag([1.0, -1.0, 1.0])),
            ("asymmetric", np.array([[1.0, 5.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
            ("singular", np.diag([1.0, 0.0, 1.0])),
            ("non-finite", np.array([[math.inf, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
            ("ill-conditioned", np.diag([1e12, 1.0, 1e-12])),
        ):
            with self.subTest(label=label), self.assertRaises(InvalidLearnerState):
                validate_covariance(matrix)


if __name__ == "__main__":
    unittest.main()


class StraddlingWindowTests(unittest.TestCase):
    """A window whose displacement feature is a folded difference is not a
    sample of the linear law, and fitting it damages a continuously updating
    instance. The learner detects this from its own reflected raw prediction."""

    @staticmethod
    def _model_at_a_wall(**kwargs) -> OnlineRLSPredictor:
        # Weights that already encode the constant-velocity law, so the raw
        # prediction from a window running into the upper wall overshoots it.
        return _model(forgetting=0.5, weights=[0.0, 0.004, 0.0], **kwargs)

    @staticmethod
    def _at_a_wall(model: OnlineRLSPredictor):
        """Drive one wall contact and return (updates_applied, skips)."""
        before = model.update_count
        # Raw prediction 1.016 -> reflected to 0.984: this window's own output
        # had to be folded, so the *next* window straddles the wall.
        approach = (0.944, 0.962, 0.980, 0.998)
        assert model.raw_predict(approach) > 1.0
        model.update(approach, 0.986)
        model.update((0.962, 0.980, 0.998, 0.986), 0.968)
        return model.update_count - before, model.reflection_skips

    def test_the_straddling_window_is_skipped_when_enabled(self):
        model = self._model_at_a_wall(skip_after_reflected_prediction=True)
        applied, skips = self._at_a_wall(model)
        self.assertEqual(skips, 1)
        self.assertEqual(applied, 1)

    def test_without_the_policy_every_window_is_fitted(self):
        model = self._model_at_a_wall(skip_after_reflected_prediction=False)
        applied, skips = self._at_a_wall(model)
        self.assertEqual(skips, 0)
        self.assertEqual(applied, 2)

    def test_the_trigger_is_the_models_own_reflected_prediction(self):
        model = self._model_at_a_wall(skip_after_reflected_prediction=True)
        # Well inside the interval nothing is reflected and nothing is skipped.
        for _ in range(20):
            model.update((0.40, 0.42, 0.44, 0.46), 0.48)
        self.assertEqual(model.reflection_skips, 0)
        self.assertEqual(model.update_count, 20)

    def test_the_policy_is_off_when_reflection_is_off(self):
        model = self._model_at_a_wall(reflect=False, skip_after_reflected_prediction=True)
        self._at_a_wall(model)
        self.assertEqual(model.reflection_skips, 0)

    def test_the_skip_counter_and_flag_survive_a_checkpoint(self):
        model = self._model_at_a_wall(skip_after_reflected_prediction=True)
        self._at_a_wall(model)
        restored = OnlineRLSPredictor.from_state_dict(
            model.state_dict(), name="restored", update_enabled=True
        )
        self.assertEqual(restored.reflection_skips, model.reflection_skips)
        self.assertEqual(restored.previous_prediction_reflected, model.previous_prediction_reflected)
        self.assertTrue(restored.skip_after_reflected_prediction)

    def test_skipping_leaves_the_learned_state_untouched(self):
        # A skipped step must move no weight, no covariance and no counter
        # other than its own. It must still record whether *its* raw
        # prediction was reflected, so a second consecutive wall contact is
        # handled correctly.
        bookkeeping = ("reflection_skips", "previous_prediction_reflected")

        def learned(model):
            return json.dumps(
                {k: v for k, v in model.state_dict().items() if k not in bookkeeping}, sort_keys=True
            )

        model = self._model_at_a_wall(skip_after_reflected_prediction=True)
        model.update((0.944, 0.962, 0.980, 0.998), 0.986)
        self.assertTrue(model.previous_prediction_reflected)
        before = learned(model)
        model.update((0.962, 0.980, 0.998, 0.986), 0.968)
        self.assertEqual(learned(model), before)
        self.assertEqual(model.reflection_skips, 1)
        self.assertFalse(model.previous_prediction_reflected)
