"""Recovery accounting: every episode falls into exactly one status.

Regression coverage for the pre-repair defects:

- eligibility was derived from a 50-transition post-event *average*, so a large
  one-step shock followed by fast recovery was diluted below the threshold and
  the episode silently left the denominator;
- the pre-event reference used the entire early prefix, including the training
  transient;
- one learner's pre-event error sequence was reused as the reference for
  persistence, constant motion and every frozen copy.
"""

from __future__ import annotations

import unittest

from aaa.metrics import ELIGIBLE_STATUSES, RECOVERY_STATUSES, RecoveryConfig, recovery_metric

from .helpers import record

CONFIG = RecoveryConfig(
    pre_event_reference_length=20,
    post_event_horizon=50,
    shock_window=5,
    shock_multiplier=3.0,
    shock_floor=1e-4,
    tolerance_multiplier=1.5,
    tolerance_floor=1e-5,
    rolling_window=5,
    sustain_windows=3,
)


def series(pre: list[float], event: float, post: list[float]):
    records = [record(step, value) for step, value in enumerate(pre)]
    records.append(record(len(pre), event, changed=True))
    records.extend(record(len(pre) + 1 + index, value) for index, value in enumerate(post))
    return records


class StatusPartitionTests(unittest.TestCase):
    def test_every_status_is_declared(self):
        self.assertEqual(len(set(RECOVERY_STATUSES)), len(RECOVERY_STATUSES))
        for status in ELIGIBLE_STATUSES:
            self.assertIn(status, RECOVERY_STATUSES)

    def test_no_intervention(self):
        result = recovery_metric([record(step, 0.01) for step in range(80)], "p", CONFIG)
        self.assertEqual(result["status"], "no_intervention")
        self.assertFalse(result["eligible"])

    def test_insufficient_pre_event_evidence(self):
        result = recovery_metric(series([0.01] * 5, 0.9, [0.001] * 60), "p", CONFIG)
        self.assertEqual(result["status"], "insufficient_pre_event_evidence")
        self.assertEqual(result["pre_event_observations"], 5)
        self.assertEqual(result["required_pre_event_observations"], 20)

    def test_insufficient_post_event_evidence(self):
        result = recovery_metric(series([0.01] * 30, 0.9, [0.001] * 10), "p", CONFIG)
        self.assertEqual(result["status"], "insufficient_post_event_evidence")
        self.assertEqual(result["post_event_observations"], 11)

    def test_no_measured_shock(self):
        result = recovery_metric(series([0.01] * 30, 0.011, [0.01] * 60), "p", CONFIG)
        self.assertEqual(result["status"], "no_measured_shock")
        self.assertFalse(result["eligible"])

    def test_eligible_and_recovered(self):
        result = recovery_metric(series([0.01] * 30, 0.9, [0.001] * 60), "p", CONFIG)
        self.assertEqual(result["status"], "recovered")
        self.assertTrue(result["eligible"])
        self.assertTrue(result["recovered"])
        self.assertIsNotNone(result["recovery_time_steps"])

    def test_eligible_and_never_recovered_is_counted(self):
        result = recovery_metric(series([0.01] * 30, 0.9, [0.5] * 60), "p", CONFIG)
        self.assertEqual(result["status"], "unrecovered")
        self.assertTrue(result["eligible"])
        self.assertFalse(result["recovered"])
        self.assertNotIn("recovery_time_steps", result)


class ShockDilutionRegressionTests(unittest.TestCase):
    def test_a_single_large_shock_with_fast_recovery_stays_eligible(self):
        # Pre-repair this episode was dropped: the 50-step post average of a
        # single 20x shock fell below the "meaningful increase" threshold.
        records = series([0.02] * 30, 0.20, [0.02] * 60)
        result = recovery_metric(records, "p", CONFIG)
        self.assertEqual(result["status"], "recovered")
        self.assertGreaterEqual(result["peak_shock"], 0.20)
        self.assertLess(result["post_event_window_mae"], 0.0245)

    def test_peak_shock_not_the_post_average_drives_eligibility(self):
        result = recovery_metric(series([0.02] * 30, 0.20, [0.02] * 60), "p", CONFIG)
        self.assertGreater(result["peak_shock"], result["post_event_window_mae"])
        self.assertGreaterEqual(result["peak_shock"], result["shock_threshold"])


class ReferenceWindowTests(unittest.TestCase):
    def test_reference_uses_the_recent_window_not_the_whole_prefix(self):
        # A large early training transient must not inflate the reference and
        # thereby suppress a genuine shock.
        transient = [0.5] * 100
        settled = [0.001] * 20
        result = recovery_metric(series(transient + settled, 0.05, [0.001] * 60), "p", CONFIG)
        self.assertAlmostEqual(result["pre_event_reference_mae"], 0.001, places=9)
        self.assertEqual(result["status"], "recovered")

    def test_supplied_pre_event_errors_are_used_verbatim(self):
        records = series([0.001] * 30, 0.9, [0.001] * 60)
        loose = recovery_metric(records, "p", CONFIG, pre_event_errors=[0.5] * 30)
        self.assertEqual(loose["status"], "no_measured_shock")
        tight = recovery_metric(records, "p", CONFIG)
        self.assertEqual(tight["status"], "recovered")

    def test_predictor_specific_references_give_predictor_specific_results(self):
        records = [
            record(step, 0.0, predictors=("fast", "slow"), errors={"fast": 0.001, "slow": 0.5})
            for step in range(30)
        ]
        records.append(
            record(30, 0.0, changed=True, predictors=("fast", "slow"), errors={"fast": 0.9, "slow": 0.9})
        )
        records.extend(
            record(31 + index, 0.0, predictors=("fast", "slow"), errors={"fast": 0.001, "slow": 0.5})
            for index in range(60)
        )
        fast = recovery_metric(records, "fast", CONFIG)
        slow = recovery_metric(records, "slow", CONFIG)
        self.assertEqual(fast["status"], "recovered")
        self.assertEqual(slow["status"], "no_measured_shock")
        # Reusing the fast learner's reference for the slow predictor would
        # misclassify it; that is exactly the defect being guarded against.
        borrowed = recovery_metric(records, "slow", CONFIG, pre_event_errors=[0.001] * 30)
        self.assertNotEqual(borrowed["status"], slow["status"])


class BoundaryTests(unittest.TestCase):
    def test_threshold_equality_is_treated_as_a_shock(self):
        reference = 0.01
        threshold = CONFIG.shock_multiplier * reference
        result = recovery_metric(series([reference] * 30, threshold, [reference] * 60), "p", CONFIG)
        self.assertTrue(result["eligible"])

    def test_exactly_enough_post_event_evidence_is_sufficient(self):
        result = recovery_metric(series([0.01] * 30, 0.9, [0.001] * 49), "p", CONFIG)
        self.assertIn(result["status"], ("recovered", "unrecovered"))

    def test_one_transition_short_is_censored(self):
        result = recovery_metric(series([0.01] * 30, 0.9, [0.001] * 48), "p", CONFIG)
        self.assertEqual(result["status"], "insufficient_post_event_evidence")

    def test_exactly_enough_pre_event_evidence_is_sufficient(self):
        result = recovery_metric(series([0.01] * 20, 0.9, [0.001] * 60), "p", CONFIG)
        self.assertNotEqual(result["status"], "insufficient_pre_event_evidence")

    def test_recovery_confirmation_follows_the_declared_window_arithmetic(self):
        result = recovery_metric(series([0.01] * 30, 0.9, [0.001] * 60), "p", CONFIG)
        self.assertEqual(
            result["recovery_confirmation_transition"],
            result["recovery_onset_transition"] - 1 + CONFIG.rolling_window + CONFIG.sustain_windows - 1,
        )


class ConfigValidationTests(unittest.TestCase):
    def test_invalid_configurations_are_rejected(self):
        cases = (
            {"pre_event_reference_length": 0},
            {"post_event_horizon": 0},
            {"shock_window": 0},
            {"rolling_window": 0},
            {"sustain_windows": 0},
            {"shock_window": 60},
            {"rolling_window": 40, "sustain_windows": 20},
            {"shock_multiplier": 0.0},
            {"tolerance_multiplier": -1.0},
            {"shock_floor": -1.0},
            {"pre_event_reference_length": True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    RecoveryConfig(**overrides)

    def test_to_dict_round_trips(self):
        payload = CONFIG.to_dict()
        self.assertEqual(RecoveryConfig(**payload), CONFIG)


if __name__ == "__main__":
    unittest.main()
