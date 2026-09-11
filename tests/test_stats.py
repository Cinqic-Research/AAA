"""Hierarchical bootstrap behaviour and estimand/gate agreement."""

from __future__ import annotations

import unittest

import numpy as np

from aaa.benchmark.stats import (
    INSUFFICIENT,
    Interval,
    PairedSamples,
    absolute_difference_statistic,
    episode_balanced_mean,
    hierarchical_bootstrap,
    holm_bonferroni,
    mean_statistic,
    point_relative_improvement,
    relative_improvement_statistic,
)


def paired(candidate, baseline):
    return PairedSamples(candidate=candidate, baseline=baseline)


class ResamplingStructureTests(unittest.TestCase):
    def test_a_draw_preserves_each_replica_observation_count(self):
        counts = {"0": 7, "1": 3, "2": 11}
        samples = PairedSamples(
            candidate={key: [float(index) for index in range(size)] for key, size in counts.items()}
        )
        observed: list[int] = []

        def statistic(candidate: np.ndarray, baseline):
            observed.append(candidate.size)
            return float(np.mean(candidate))

        hierarchical_bootstrap(samples, statistic, estimand="mean", seed=1, draws=200)
        # Every draw resamples three replicas with replacement and then that
        # replica's own episodes, so a draw never collapses to one episode per
        # replica and its size is a sum of three replica sizes.
        possible = {a + b + c for a in counts.values() for b in counts.values() for c in counts.values()}
        self.assertTrue(set(observed[1:]).issubset(possible))
        self.assertGreater(max(observed[1:]), 3)

    def test_pairing_is_preserved_across_predictors(self):
        # Candidate equals baseline episode by episode, so every paired
        # difference must be exactly zero on every draw.
        values = {"0": [1.0, 5.0, 9.0], "1": [2.0, 4.0]}
        samples = paired({k: list(v) for k, v in values.items()}, {k: list(v) for k, v in values.items()})
        interval = hierarchical_bootstrap(
            samples, absolute_difference_statistic, estimand="difference", seed=2, draws=500
        )
        self.assertAlmostEqual(interval.estimate, 0.0)
        self.assertAlmostEqual(interval.lower, 0.0)
        self.assertAlmostEqual(interval.upper, 0.0)

    def test_misaligned_pairs_are_rejected(self):
        samples = paired({"0": [1.0, 2.0]}, {"0": [1.0]})
        with self.assertRaises(ValueError):
            samples.validate()

    def test_deterministic_bootstrap_stream(self):
        rng = np.random.default_rng(0)
        candidate = {str(key): list(rng.uniform(0.5, 2.0, size=9)) for key in range(4)}
        baseline = {str(key): list(rng.uniform(2.0, 6.0, size=9)) for key in range(4)}
        samples = paired(candidate, baseline)
        first = hierarchical_bootstrap(
            samples, relative_improvement_statistic, estimand="ri", seed=7, draws=500
        )
        second = hierarchical_bootstrap(
            samples, relative_improvement_statistic, estimand="ri", seed=7, draws=500
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        different = hierarchical_bootstrap(
            samples, relative_improvement_statistic, estimand="ri", seed=8, draws=500
        )
        self.assertAlmostEqual(first.estimate, different.estimate)
        self.assertNotEqual((first.lower, first.upper), (different.lower, different.upper))


class EstimandTests(unittest.TestCase):
    def test_ratio_of_means_is_not_mean_of_ratios(self):
        candidate = {"0": [1.0, 1.0], "1": [1.0, 1.0]}
        baseline = {"0": [1.0, 100.0], "1": [1.0, 100.0]}
        ratio_of_means = point_relative_improvement(candidate, baseline)
        mean_of_ratios = float(
            np.mean(
                [1.0 - c / b for key in candidate for c, b in zip(candidate[key], baseline[key], strict=True)]
            )
        )
        self.assertAlmostEqual(ratio_of_means, 1.0 - 4.0 / 202.0)
        self.assertNotAlmostEqual(ratio_of_means, mean_of_ratios)

    def test_bootstrap_recomputes_the_same_statistic_as_the_point_estimate(self):
        candidate = {"0": [1.0, 2.0, 3.0], "1": [2.0, 2.0, 2.0]}
        baseline = {"0": [2.0, 4.0, 6.0], "1": [4.0, 4.0, 4.0]}
        interval = hierarchical_bootstrap(
            paired(candidate, baseline), relative_improvement_statistic, estimand="ri", seed=3, draws=800
        )
        self.assertAlmostEqual(interval.estimate, point_relative_improvement(candidate, baseline))
        self.assertAlmostEqual(interval.estimate, 0.5)

    def test_episode_balanced_mean_matches_the_mean_statistic(self):
        values = {"0": [1.0, 2.0], "1": [3.0]}
        interval = hierarchical_bootstrap(
            PairedSamples(candidate=values), mean_statistic, estimand="mean", seed=4, draws=400
        )
        self.assertAlmostEqual(interval.estimate, episode_balanced_mean(values))

    def test_statistics_reject_a_missing_baseline(self):
        for statistic in (relative_improvement_statistic, absolute_difference_statistic):
            with self.subTest(statistic=statistic.__name__), self.assertRaises(ValueError):
                statistic(np.asarray([1.0]), None)

    def test_zero_baseline_mean_is_undefined_not_silently_zero(self):
        with self.assertRaises(ZeroDivisionError):
            relative_improvement_statistic(np.asarray([1.0]), np.asarray([0.0]))


class InsufficientEvidenceTests(unittest.TestCase):
    def test_a_single_replica_cannot_produce_an_interval(self):
        interval = hierarchical_bootstrap(
            PairedSamples(candidate={"0": [1.0, 2.0]}), mean_statistic, estimand="mean", seed=5, draws=400
        )
        self.assertEqual(interval.status, INSUFFICIENT)
        self.assertFalse(interval.available)
        self.assertIsNone(interval.lower)

    def test_an_undefined_statistic_reports_insufficient_evidence(self):
        interval = hierarchical_bootstrap(
            paired({"0": [1.0], "1": [1.0]}, {"0": [0.0], "1": [0.0]}),
            relative_improvement_statistic,
            estimand="ri",
            seed=6,
            draws=400,
        )
        self.assertEqual(interval.status, INSUFFICIENT)

    def test_unavailable_interval_serializes_its_reason(self):
        interval = Interval.unavailable("mean", "not enough replicas", level=0.95, draws=10)
        payload = interval.to_dict()
        self.assertEqual(payload["status"], INSUFFICIENT)
        self.assertEqual(payload["reason"], "not enough replicas")

    def test_p_value_is_reported_only_when_a_null_is_declared(self):
        samples = paired({"0": [1.0, 1.0], "1": [1.0, 1.0]}, {"0": [2.0, 2.0], "1": [2.0, 2.0]})
        without = hierarchical_bootstrap(
            samples, relative_improvement_statistic, estimand="ri", seed=9, draws=400
        )
        self.assertIsNone(without.p_value)
        with_null = hierarchical_bootstrap(
            samples, relative_improvement_statistic, estimand="ri", seed=9, draws=400, null_value=0.2
        )
        self.assertIsNotNone(with_null.p_value)
        self.assertLess(with_null.p_value, 0.01)


class MultiplicityTests(unittest.TestCase):
    def test_holm_step_down_ordering(self):
        result = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.30}, alpha=0.05)
        adjusted = {entry["name"]: entry["adjusted_p_value"] for entry in result.entries}
        self.assertAlmostEqual(adjusted["a"], 0.003)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        self.assertAlmostEqual(adjusted["c"], 0.30)
        self.assertTrue(result.rejects("a"))
        self.assertTrue(result.rejects("b"))
        self.assertFalse(result.rejects("c"))

    def test_holm_is_step_down_not_step_up(self):
        # Once a hypothesis fails, no later hypothesis may be rejected.
        result = holm_bonferroni({"a": 0.40, "b": 0.001}, alpha=0.05)
        self.assertTrue(result.rejects("b"))
        self.assertFalse(result.rejects("a"))
        harder = holm_bonferroni({"a": 0.03, "b": 0.04}, alpha=0.05)
        self.assertFalse(harder.rejects("a"))
        self.assertFalse(harder.rejects("b"))

    def test_repeated_confirmation_attempts_enlarge_the_family(self):
        alone = holm_bonferroni({"a": 0.02}, alpha=0.05)
        repeated = holm_bonferroni({"a": 0.02}, alpha=0.05, extra_family_size=3)
        self.assertEqual(alone.family_size, 1)
        self.assertEqual(repeated.family_size, 4)
        self.assertTrue(alone.rejects("a"))
        self.assertFalse(repeated.rejects("a"))

    def test_missing_p_values_are_reported_not_assumed(self):
        result = holm_bonferroni({"a": None, "b": 0.01}, alpha=0.05)
        self.assertIsNone(result.rejects("a"))
        self.assertTrue(result.rejects("b"))

    def test_none_method_passes_through(self):
        result = holm_bonferroni({"a": 0.04}, alpha=0.05, method="none")
        self.assertEqual(result.method, "none")
        self.assertTrue(result.rejects("a"))


if __name__ == "__main__":
    unittest.main()
