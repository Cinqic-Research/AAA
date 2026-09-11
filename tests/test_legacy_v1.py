"""Historical v1 track: legacy learner and legacy evaluation integrity.

The v1 evaluation is retained so the original (unfavourable) result stays
reproducible. These tests pin its behaviour and the correctness repairs made
to it; they are not benchmark acceptance evidence.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaa.config import ExperimentConfig
from aaa.evaluation import _episode_seed, run_full_evaluation
from aaa.predictors import OnlineLinearPredictor


class LegacyGradientTests(unittest.TestCase):
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
            plus = weights.copy()
            plus[index] += epsilon
            minus = weights.copy()
            minus[index] -= epsilon
            loss_plus = 0.5 * (float(plus @ features) - target_delta) ** 2
            loss_minus = 0.5 * (float(minus @ features) - target_delta) ** 2
            numeric.append((loss_plus - loss_minus) / (2 * epsilon))
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)

    def test_zero_initialized_legacy_model_is_persistence(self):
        model = OnlineLinearPredictor()
        self.assertEqual(model.predict((0.1, 0.2, 0.3, 0.4)), 0.4)


class EpisodeSeedTests(unittest.TestCase):
    def test_every_supported_scenario_maps_to_a_seed(self):
        for scenario in ("straight", "bouncing", "changed"):
            for phase in ("dev", "train", "final"):
                self.assertIsInstance(_episode_seed(1, phase, 0, scenario), int)

    def test_the_oscillator_scenario_has_its_own_stream(self):
        # The original helper raised KeyError for this scenario.
        self.assertIsInstance(_episode_seed(1, "dev", 0, "dynamics_change"), int)

    def test_unsupported_scenario_raises_a_clear_error_not_a_key_error(self):
        with self.assertRaises(ValueError) as caught:
            _episode_seed(1, "dev", 0, "teleport")
        self.assertIn("teleport", str(caught.exception))

    def test_unsupported_phase_raises_a_clear_error(self):
        with self.assertRaises(ValueError):
            _episode_seed(1, "confirmation", 0, "straight")

    def test_seeds_are_distinct_across_phase_episode_and_scenario(self):
        seeds = {
            _episode_seed(1, phase, episode, scenario)
            for phase in ("dev", "train", "final")
            for episode in range(4)
            for scenario in ("straight", "bouncing", "changed")
        }
        self.assertEqual(len(seeds), 3 * 4 * 3)


class OutputIsolationTests(unittest.TestCase):
    def test_a_run_writes_nothing_outside_the_requested_output_root(self):
        project_root = Path(__file__).resolve().parents[1]
        before = set((project_root / "reports").glob("*")) if (project_root / "reports").exists() else set()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_full_evaluation(
                ExperimentConfig().quick(),
                output_root=Path(directory),
                label="isolation",
                project_root=project_root,
            )
            self.assertTrue(run_dir.is_relative_to(Path(directory)))
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "experiment_report.md").exists())
        after = set((project_root / "reports").glob("*")) if (project_root / "reports").exists() else set()
        self.assertEqual(before, after)

    def test_frozen_integrity_is_measured_from_actual_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_full_evaluation(
                ExperimentConfig().quick(), output_root=Path(directory), label="integrity"
            )
            import json

            integrity = json.loads((run_dir / "frozen_generalization" / "integrity.json").read_text())
        self.assertTrue(integrity["checkpoint_weights_unchanged"])
        self.assertGreater(integrity["evaluated_copies"], 0)
        self.assertEqual(integrity["weights_after_distinct"], [integrity["weights_before"]])
        self.assertEqual(integrity["update_counts_after_distinct"], [integrity["update_count_before"]])


if __name__ == "__main__":
    unittest.main()
