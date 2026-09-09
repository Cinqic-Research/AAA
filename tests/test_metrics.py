import unittest

from aaa.experiment import StepRecord
from aaa.metrics import episode_metrics, mean_absolute_error, rolling_mean


def record(step, actual, error, *, bounced=False, changed=False):
    return StepRecord(
        seed=1,
        environment_seed=2,
        episode=0,
        scenario="changed",
        step=step,
        target_step=step + 1,
        history=(0.1, 0.2, 0.3, 0.4),
        current_observation=0.4,
        actual_next_position=actual,
        bounced=bounced,
        changed=changed,
        predictions={"p": {"raw": actual + error, "scored": actual + error, "absolute_error": abs(error)}},
        updates_enabled={"p": False},
    )


class MetricsTests(unittest.TestCase):
    def test_known_mean_and_rolling_values(self):
        self.assertAlmostEqual(mean_absolute_error([0.1, 0.3]), 0.2)
        self.assertEqual(rolling_mean([1.0, 3.0, 5.0], 2), [1.0, 2.0, 4.0])

    def test_episode_event_slices_and_mae(self):
        records = [
            record(0, 0.1, 0.1),
            record(1, 0.2, 0.3, bounced=True),
            record(2, 0.3, 0.2, changed=True),
            record(3, 0.4, 0.1),
        ]
        result = episode_metrics(records, ["p"], post_change_window=2, rolling_window=2)
        predictor = result["predictors"]["p"]
        self.assertAlmostEqual(predictor["mae"], 0.175)
        self.assertAlmostEqual(predictor["bounce_mae"], 0.3)
        self.assertAlmostEqual(predictor["change_mae"], 0.2)
        self.assertAlmostEqual(predictor["post_change_window_mae"], 0.15)


if __name__ == "__main__":
    unittest.main()
