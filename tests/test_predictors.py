import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaa.predictors import ConstantMotionPredictor, OnlineLinearPredictor, PersistencePredictor


class PredictorTests(unittest.TestCase):
    def test_baseline_formulas_use_observations_only(self):
        history = (0.1, 0.2, 0.4, 0.7)
        self.assertAlmostEqual(PersistencePredictor().predict(history), 0.7)
        self.assertAlmostEqual(ConstantMotionPredictor().predict(history), 1.0)

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


if __name__ == "__main__":
    unittest.main()
