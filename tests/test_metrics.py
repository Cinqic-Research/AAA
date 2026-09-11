import unittest

from aaa.experiment import StepRecord
from aaa.metrics import (
    aggregate_metrics,
    complete_rolling_mean,
    episode_metrics,
    mean_absolute_error,
    normalized_errors,
    raw_errors,
    rolling_mean,
    signed_normalized_errors,
)

from .helpers import identity, record


class MeanAndRollingTests(unittest.TestCase):
    def test_known_mean_and_rolling_values(self):
        self.assertAlmostEqual(mean_absolute_error([0.1, 0.3]), 0.2)
        self.assertEqual(rolling_mean([1.0, 3.0, 5.0], 2), [1.0, 2.0, 4.0])
        self.assertEqual(complete_rolling_mean([1.0, 2.0, 3.0], 2), [1.5, 2.5])

    def test_empty_and_invalid_inputs(self):
        self.assertIsNone(mean_absolute_error([]))
        with self.assertRaises(ValueError):
            mean_absolute_error([0.1, float("nan")])
        with self.assertRaises(ValueError):
            rolling_mean([1.0], 0)
        with self.assertRaises(ValueError):
            complete_rolling_mean([1.0], 0)


class UnitContractTests(unittest.TestCase):
    """Regression for the mixed normalized/raw units defect."""

    def _records(self, width: float):
        return [
            record(step, 1.0, width=width, bounced=step == 1, changed=step == 2)
            for step in range(4)
        ]

    def test_normalized_and_raw_accessors_are_distinct(self):
        records = self._records(10.0)
        self.assertEqual(normalized_errors(records, "p"), [0.1] * 4)
        self.assertEqual(raw_errors(records, "p"), [1.0] * 4)

    def test_every_episode_aggregate_uses_normalized_units_on_a_wide_interval(self):
        result = episode_metrics(self._records(10.0), ["p"], post_change_window=2)["predictors"]["p"]
        for key in ("mae", "bounce_mae", "non_bounce_mae", "change_mae", "post_change_window_mae"):
            with self.subTest(key=key):
                self.assertAlmostEqual(result[key], 0.1)

    def test_every_aggregate_metric_uses_normalized_units_on_a_wide_interval(self):
        result = aggregate_metrics(self._records(10.0), ["p"], post_change_window=2)["predictors"]["p"]
        for key in ("mae", "bounce_mae", "non_bounce_mae", "change_mae"):
            with self.subTest(key=key):
                self.assertAlmostEqual(result[key], 0.1)
        self.assertAlmostEqual(result["mae_raw"], 1.0)

    def test_unit_width_hides_nothing_that_a_wide_interval_reveals(self):
        # With width 1 the normalized and raw values coincide, which is exactly
        # how the original mixed-unit defect stayed invisible. Any interval
        # whose width is not 1 must still report normalized units.
        narrow = episode_metrics(self._records(1.0), ["p"], post_change_window=2)["predictors"]["p"]
        self.assertAlmostEqual(narrow["mae"], 1.0)
        wide = episode_metrics(self._records(10.0), ["p"], post_change_window=2)["predictors"]["p"]
        self.assertAlmostEqual(wide["mae"], 0.1)

    def test_arbitrary_interval_widths_scale_the_normalized_aggregate(self):
        for width in (1.0, 10.0, 7.5):
            with self.subTest(width=width):
                result = episode_metrics(self._records(width), ["p"], post_change_window=2)
                self.assertAlmostEqual(result["predictors"]["p"]["mae"], 1.0 / width)

    def test_missing_normalized_field_is_a_loud_error(self):
        broken = record(0, 1.0)
        del broken.predictions["p"]["normalized_absolute_error"]
        with self.assertRaises(KeyError):
            normalized_errors([broken], "p")

    def test_unknown_predictor_is_a_loud_error(self):
        with self.assertRaises(KeyError):
            normalized_errors([record(0, 1.0)], "missing")

    def test_signed_errors_are_normalized(self):
        self.assertAlmostEqual(signed_normalized_errors([record(0, 0.5, width=10.0)], "p")[0], 0.05)


class EpisodeMetricTests(unittest.TestCase):
    def test_event_slices_and_mae(self):
        records = [
            record(0, 0.1),
            record(1, 0.3, bounced=True),
            record(2, 0.2, changed=True),
            record(3, 0.1),
        ]
        predictor = episode_metrics(records, ["p"], post_change_window=2)["predictors"]["p"]
        self.assertAlmostEqual(predictor["mae"], 0.175)
        self.assertAlmostEqual(predictor["bounce_mae"], 0.3)
        self.assertAlmostEqual(predictor["change_mae"], 0.2)
        self.assertAlmostEqual(predictor["post_change_window_mae"], 0.15)
        self.assertAlmostEqual(predictor["post_change_window_cumulative"], 0.3)
        self.assertEqual(predictor["post_change_window_transitions"], 2)

    def test_bounce_events_count_every_wall_contact(self):
        records = [record(0, 0.1, walls=("upper", "lower"))]
        summary = episode_metrics(records, ["p"])
        self.assertEqual(summary["bounce_events"], 2)
        self.assertEqual(summary["bounce_transitions"], 1)

    def test_empty_records_are_rejected(self):
        with self.assertRaises(ValueError):
            episode_metrics([], ["p"])

    def test_identity_fields_are_carried_into_the_summary(self):
        ident = identity(trial_id="t1", replica_id=3, episode=7, family="bouncing", stratum="positive-position1-speed2")
        summary = episode_metrics([record(0, 0.1, ident=ident)], ["p"])
        self.assertEqual(summary["trial_id"], "t1")
        self.assertEqual(summary["replica_id"], 3)
        self.assertEqual(summary["episode"], 7)
        self.assertEqual(summary["stratum"], "positive-position1-speed2")


class AggregateTests(unittest.TestCase):
    def test_aggregate_reports_replica_structure(self):
        records = [
            record(step, 0.1, ident=identity(replica_id=replica, episode=0))
            for replica in (0, 1)
            for step in range(3)
        ]
        result = aggregate_metrics(records, ["p"])
        self.assertEqual(result["replica_count"], 2)
        self.assertEqual(result["episode_count"], 2)
        self.assertIn("0", result["replicas"])

    def test_empty_records_return_an_empty_shape(self):
        result = aggregate_metrics([], ["p"])
        self.assertEqual(result["scored_steps"], 0)


if __name__ == "__main__":
    unittest.main()
