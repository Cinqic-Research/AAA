"""The canonical specification is the single executable source of truth.

A JSON value that the runner never reads is a lie waiting to happen, so these
tests pin the schema, the strictness of validation, the hash, and — most
importantly — that every declared leaf is actually consumed somewhere.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aaa.benchmark import spec as spec_module
from aaa.benchmark.spec import (
    BenchmarkSpec,
    SpecError,
    canonical_spec_hash,
    canonical_spec_path,
    leaf_paths,
    load_spec,
    spec_hash,
)


def canonical_dict() -> dict:
    return json.loads(canonical_spec_path().read_text(encoding="utf-8"))


class CanonicalSpecTests(unittest.TestCase):
    def test_the_canonical_spec_loads_and_validates(self):
        spec = load_spec()
        self.assertEqual(spec.spec_version, "aaa.benchmark.v2.1")
        self.assertEqual(spec.status, "active")

    def test_the_spec_ships_inside_the_installed_package(self):
        # A repository-relative path disappears when AAA is pip-installed
        # elsewhere; the canonical spec must be package data.
        self.assertEqual(canonical_spec_path().parent.name, "data")
        self.assertEqual(canonical_spec_path().parent.parent, Path(spec_module.__file__).parent)

    def test_spec_hash_is_stable_and_content_addressed(self):
        first = canonical_spec_hash()
        self.assertEqual(first, spec_hash(load_spec()))
        mutated = canonical_dict()
        mutated["confirmation"]["episodes_per_family"] += 1
        self.assertNotEqual(spec_hash(BenchmarkSpec.parse(mutated)), first)

    def test_hash_ignores_key_order_but_not_values(self):
        value = canonical_dict()
        reordered = json.loads(json.dumps(value, sort_keys=True))
        self.assertEqual(spec_hash(BenchmarkSpec.parse(reordered)), canonical_spec_hash())

    def test_round_trip_through_to_dict_is_lossless(self):
        spec = load_spec()
        self.assertEqual(spec_hash(BenchmarkSpec.parse(spec.to_dict())), spec_hash(spec))

    def test_declared_statistics_match_the_frozen_protocol(self):
        statistics = load_spec().statistics
        self.assertEqual(statistics.method, "paired_hierarchical_bootstrap")
        self.assertEqual(statistics.draws, 4000)
        self.assertEqual(statistics.interval, 0.95)
        self.assertEqual(statistics.multiplicity, "holm_bonferroni")
        self.assertTrue(statistics.exclude_no_event_episodes)

    def test_confirmation_minimums_are_declared(self):
        confirmation = load_spec().confirmation
        self.assertGreaterEqual(confirmation.replicas, 5)
        self.assertGreaterEqual(confirmation.episodes_per_family, 100)
        self.assertTrue(confirmation.require_clean_source_tree)


class StrictValidationTests(unittest.TestCase):
    def _reject(self, mutate) -> str:
        value = canonical_dict()
        mutate(value)
        with self.assertRaises(SpecError) as caught:
            BenchmarkSpec.parse(value)
        return str(caught.exception)

    def test_unknown_top_level_key_is_rejected(self):
        self._reject(lambda value: value.update({"surprise": 1}))

    def test_unknown_nested_key_is_rejected(self):
        self._reject(lambda value: value["world"].update({"gravity": 9.81}))

    def test_missing_section_is_rejected(self):
        self._reject(lambda value: value.pop("statistics"))

    def test_wrong_spec_version_is_rejected(self):
        self._reject(lambda value: value.update({"spec_version": "aaa.benchmark.v2"}))

    def test_non_finite_number_is_rejected(self):
        self._reject(lambda value: value["world"].update({"dt": float("inf")}))

    def test_negative_tolerances_and_subunit_condition_limits_are_rejected(self):
        self._reject(lambda value: value["tolerances"].update({"recompute_absolute": -1e-12}))
        self._reject(lambda value: value["tolerances"].update({"max_condition_number": 0.5}))

    def test_zero_and_negative_counts_are_rejected(self):
        self._reject(lambda value: value["confirmation"].update({"replicas": 0}))
        self._reject(lambda value: value["confirmation"].update({"replicas": -5}))

    def test_fractional_count_is_rejected(self):
        self._reject(lambda value: value["confirmation"].update({"replicas": 5.5}))

    def test_boolean_is_not_accepted_as_a_count(self):
        self._reject(lambda value: value["confirmation"].update({"replicas": True}))

    def test_interval_outside_the_open_unit_range_is_rejected(self):
        self._reject(lambda value: value["statistics"].update({"interval": 1.0}))

    def test_unknown_statistical_method_is_rejected(self):
        self._reject(lambda value: value["statistics"].update({"method": "t_test"}))

    def test_unknown_multiplicity_method_is_rejected(self):
        self._reject(lambda value: value["statistics"].update({"multiplicity": "whichever_looks_best"}))

    def test_unknown_feature_set_is_rejected(self):
        self._reject(lambda value: value["candidate"].update({"feature_set": "everything"}))

    def test_forgetting_outside_the_valid_range_is_rejected(self):
        self._reject(lambda value: value["candidate"].update({"forgetting": 0.0}))
        self._reject(lambda value: value["candidate"].update({"forgetting": 1.5}))

    def test_inverted_bounds_are_rejected(self):
        self._reject(lambda value: value["world"].update({"lower_bound": 2.0, "upper_bound": 1.0}))

    def test_unknown_gate_evaluator_is_rejected(self):
        def mutate(value):
            value["gates"][0]["evaluator"] = "vibes"

        self._reject(mutate)

    def test_malformed_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(SpecError) as caught:
                load_spec(path)
            self.assertIn("malformed JSON", str(caught.exception))

    def test_a_missing_file_is_rejected(self):
        with self.assertRaises(OSError):
            load_spec(Path("/nonexistent/aaa-spec.json"))


class ExecutableSpecTests(unittest.TestCase):
    """A declared value must change behaviour, or it is documentation pretending."""

    IGNORED_PREFIXES = (
        # Prose fields that exist to be read by humans and hashed for identity.
        "notes",
        "status",
        "spec_version",
        "randomness.description",
        "training.distribution_relationship",
        "candidate.features",
        "candidate.boundary_policy",
        "candidate.self_triggered_forgetting",
        "candidate.straddling_window_policy",
        "candidate.selection_evidence",
        "candidate.model",
        "statistics.estimands",
        "artifacts",
        "gates",
    )

    def test_every_declared_leaf_is_either_consumed_or_explicitly_descriptive(self):
        spec = load_spec()
        source = "".join(path.read_text(encoding="utf-8") for path in sorted(Path("aaa").rglob("*.py")))
        unused = []
        for path in leaf_paths(spec.to_dict()):
            if path.startswith(self.IGNORED_PREFIXES):
                continue
            leaf = path.split(".")[-1]
            if leaf.isdigit():
                continue
            if f'"{leaf}"' in source or f"'{leaf}'" in source or f".{leaf}" in source:
                continue
            unused.append(path)
        self.assertEqual(unused, [], f"declared but never read by the implementation: {unused}")

    def test_changing_a_world_value_changes_the_resolved_world(self):
        from aaa.benchmark.families import motion_world

        spec = load_spec()
        family = spec.motion_families["bouncing"]
        base = motion_world(spec, family)
        value = canonical_dict()
        value["world"]["dt"] = value["world"]["dt"] * 2
        mutated = BenchmarkSpec.parse(value)
        self.assertNotEqual(motion_world(mutated, mutated.motion_families["bouncing"]).dt, base.dt)

    def test_changing_a_candidate_value_changes_the_constructed_model(self):
        from aaa.benchmark.families import make_candidate

        spec = load_spec()
        base = make_candidate(spec, name="x", update_enabled=False)
        value = canonical_dict()
        value["candidate"]["ridge"] = value["candidate"]["ridge"] * 10
        mutated = make_candidate(BenchmarkSpec.parse(value), name="x", update_enabled=False)
        self.assertNotEqual(mutated.ridge, base.ridge)

    def test_changing_a_gate_threshold_changes_the_gate_specification(self):
        spec = load_spec()
        value = canonical_dict()
        for gate in value["gates"]:
            if gate["name"] == "recovery":
                gate["threshold"]["min_recovery_rate"] = 0.5
        mutated = BenchmarkSpec.parse(value)
        self.assertNotEqual(
            mutated.gate("recovery").threshold["min_recovery_rate"],
            spec.gate("recovery").threshold["min_recovery_rate"],
        )

    def test_required_strata_are_the_declared_cartesian_product(self):
        spec = load_spec()
        stratification = spec.stratification
        expected = len(stratification.directions) * stratification.position_bands * stratification.speed_bands
        self.assertEqual(len(spec.required_strata()), expected)
        self.assertEqual(len(set(spec.required_strata())), expected)

    def test_every_declared_gate_name_is_unique(self):
        names = [gate.name for gate in load_spec().gates]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
