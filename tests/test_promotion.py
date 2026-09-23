"""``aaa.promotion.crossed.v1``: the prospective successor to the ``aaa.1k.v2`` K5 path (``AAA-180``)."""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import math
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from aaa.promotion import (
    CONTRACT_ID,
    Contract,
    ContractError,
    Criterion,
    GroupDeclaration,
    adjudicate,
    fixtures,
    independent,
    primary,
    require_admissible,
)
from aaa.promotion.contract import combine, criterion_status

ROOT = Path(__file__).resolve().parents[1]


def _run(contract: Contract, records: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return adjudicate(contract, records, **kwargs)


class AgreementTests(unittest.TestCase):
    def assert_full_agreement(self, result: dict[str, Any]) -> None:
        self.assertEqual(result["agreement_problems"], [])
        p, i = result["primary"], result["independent"]
        self.assertTrue(math.isclose(p["geometric_ratio"], i["geometric_ratio"], rel_tol=1e-12))
        self.assertEqual(p["interval_status"], i["interval_status"])
        width = p["upper"] - p["lower"]
        self.assertLessEqual(abs(p["lower"] - i["lower"]), 0.05 * width)
        self.assertLessEqual(abs(p["upper"] - i["upper"]), 0.05 * width)
        self.assertEqual(p["criteria"], i["criteria"])
        self.assertEqual(p["verdict"], i["verdict"])
        self.assertEqual(result["verdict"], p["verdict"])

    def test_multiple_initializations_and_multiple_series_agree(self) -> None:
        contract = fixtures.contract([5, 4, 8])
        for effect, verdict in ((0.85, "PROMOTE"), (1.0, "INCONCLUSIVE"), (1.4, "REJECT")):
            result = _run(contract, fixtures.primitives(contract, effect=effect))
            self.assert_full_agreement(result)
            self.assertEqual(result["verdict"], verdict)
            self.assertEqual(result["primary"]["scope"], "population")

    def test_multiple_initializations_and_exactly_one_series_agree(self) -> None:
        # A single-series group alone, and inside a suite, for every verdict region.
        for series in ([1], [5, 4, 1]):
            contract = fixtures.contract(series)
            # With no true effect, a non-inferiority margin of 2% may be met or left
            # open depending on the fixture's realized noise; it must never reject.
            for effect, verdicts in (
                (0.85, {"PROMOTE"}),
                (1.0, {"INCONCLUSIVE", "PROMOTE"}),
                (1.4, {"REJECT"}),
            ):
                result = _run(contract, fixtures.primitives(contract, effect=effect))
                self.assert_full_agreement(result)
                self.assertEqual(result["primary"]["interval_status"], "MEASURED")
                self.assertIn(result["verdict"], verdicts, (series, effect))
                self.assertEqual(result["primary"]["scope"], "conditional_on_observed_series")
                self.assertIn(f"g{len(series) - 1}", result["primary"]["conditional_groups"])

    def test_the_single_series_interval_resamples_initializations_only(self) -> None:
        contract = fixtures.contract([1], draws=4000)
        records = fixtures.primitives(contract, effect=1.1)
        grids = primary.build_grids(contract, records)
        ref, cha = grids["g0"]["reference"][:, 0], grids["g0"]["challenger"][:, 0]
        rng = np.random.default_rng(contract.seed)
        values = []
        for start in range(0, contract.draws, primary.CHUNK):
            size = min(primary.CHUNK, contract.draws - start)
            rows = rng.integers(0, len(ref), size=(size, len(ref)))
            values.extend(cha[rows].mean(axis=1) / ref[rows].mean(axis=1))
        result = primary.evaluate(contract, records)
        self.assertAlmostEqual(result["lower"], float(np.quantile(values, 0.025)), places=12)
        self.assertAlmostEqual(result["upper"], float(np.quantile(values, 0.975)), places=12)
        self.assertAlmostEqual(result["geometric_ratio"], float(cha.mean() / ref.mean()), places=12)

    def test_one_initialization_is_insufficient_evidence_in_both_implementations(self) -> None:
        contract = fixtures.contract([5, 1], initializations=1)
        result = _run(contract, fixtures.primitives(contract, effect=0.5))
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["primary"]["interval_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["independent"]["interval_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["agreement_problems"], [])


class FailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = fixtures.contract([5, 4, 1])
        # A clearly better challenger: anything that is not fail-closed would promote it.
        self.records = fixtures.primitives(self.contract, effect=0.7)
        self.assertEqual(_run(self.contract, self.records)["verdict"], "PROMOTE")

    def assert_invalid(self, records: list[Any]) -> dict[str, Any]:
        result = _run(self.contract, records)
        self.assertEqual(result["verdict"], "INVALID_EVIDENCE", result.get("reason"))
        # Each implementation refuses on its own, not only the one that runs first.
        with self.assertRaises(ValueError):
            independent.recompute(self.contract, records)
        with self.assertRaises(ValueError):
            primary.evaluate(self.contract, records)
        return result

    def test_omitting_a_required_group_cannot_promote(self) -> None:
        for group in ("g0", "g1", "g2"):
            records = [r for r in self.records if r["group"] != group]
            self.assert_invalid(records)

    def test_omitting_a_group_for_one_arm_only_cannot_promote(self) -> None:
        records = [r for r in self.records if not (r["group"] == "g2" and r["arm"] == "challenger")]
        self.assert_invalid(records)

    def test_an_extra_undeclared_group_is_refused(self) -> None:
        extra = copy.deepcopy(self.records[:2])
        for record in extra:
            record["group"] = "undeclared"
        self.assert_invalid(self.records + extra)

    def test_mismatched_initialization_identities_are_refused(self) -> None:
        records = copy.deepcopy(self.records)
        target = records[0]["init"]
        for record in records:
            if record["init"] == target and record["arm"] == "challenger":
                record["init"] = 999_999
        self.assert_invalid(records)

    def test_mismatched_series_identities_are_refused(self) -> None:
        records = copy.deepcopy(self.records)
        for record in records:
            if record["series"] == "g2:s0" and record["arm"] == "challenger":
                record["series"] = "g2:other"
        self.assert_invalid(records)

    def test_a_missing_cell_is_refused(self) -> None:
        self.assert_invalid(self.records[:-1])

    def test_a_duplicated_cell_is_refused(self) -> None:
        self.assert_invalid([*self.records, dict(self.records[3])])

    def test_non_finite_and_non_positive_values_are_refused(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0, None, "1.0", True):
            records = copy.deepcopy(self.records)
            records[5]["value"] = bad
            self.assert_invalid(records)

    def test_malformed_records_are_refused(self) -> None:
        missing = copy.deepcopy(self.records)
        del missing[0]["series"]
        extra = copy.deepcopy(self.records)
        extra[0]["note"] = "x"
        self.assert_invalid(missing)
        self.assert_invalid(extra)
        self.assert_invalid([*self.records[1:], ["not", "a", "mapping"]])
        wrong_arm = copy.deepcopy(self.records)
        wrong_arm[0]["arm"] = "someone_else"
        self.assert_invalid(wrong_arm)
        bool_init = copy.deepcopy(self.records)
        bool_init[0]["init"] = True
        self.assert_invalid(bool_init)


class DisagreementInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = fixtures.contract([5, 4, 1])
        self.records = fixtures.primitives(self.contract, effect=0.7)

    def _tampered(self, mutate: Any) -> dict[str, Any]:
        def evaluator(contract: Contract, records: Any) -> dict[str, Any]:
            result = independent.recompute(contract, records)
            mutate(result)
            return result

        return _run(self.contract, self.records, independent_evaluator=evaluator)

    def test_a_point_value_difference_blocks_promotion(self) -> None:
        result = self._tampered(lambda r: r.update(geometric_ratio=r["geometric_ratio"] * (1 + 1e-9)))
        self.assertEqual(result["verdict"], "DISAGREEMENT")

    def test_an_interval_bound_difference_blocks_promotion(self) -> None:
        result = self._tampered(lambda r: r.update(upper=r["upper"] + 0.2 * (r["upper"] - r["lower"])))
        self.assertEqual(result["verdict"], "DISAGREEMENT")

    def test_an_interval_status_difference_blocks_promotion(self) -> None:
        result = self._tampered(
            lambda r: r.update(interval_status="INSUFFICIENT_EVIDENCE", lower=None, upper=None)
        )
        self.assertEqual(result["verdict"], "DISAGREEMENT")

    def test_a_criterion_or_verdict_difference_blocks_promotion(self) -> None:
        def flip(result: dict[str, Any]) -> None:
            result["criteria"] = dict.fromkeys(result["criteria"], "INCONCLUSIVE")
            result["verdict"] = "INCONCLUSIVE"

        self.assertEqual(self._tampered(flip)["verdict"], "DISAGREEMENT")

    def test_a_non_finite_bound_is_a_disagreement_not_a_pass(self) -> None:
        result = self._tampered(lambda r: r.update(lower=float("nan")))
        self.assertEqual(result["verdict"], "DISAGREEMENT")


class ContractTests(unittest.TestCase):
    def test_a_crossed_group_needs_two_series(self) -> None:
        with self.assertRaises(ContractError):
            GroupDeclaration("g", "crossed", ("only",))

    def test_a_conditional_group_needs_exactly_one_series(self) -> None:
        with self.assertRaises(ContractError):
            GroupDeclaration("g", "conditional_on_single_series", ("a", "b"))

    def test_malformed_contracts_are_refused(self) -> None:
        group = (GroupDeclaration("g", "crossed", ("a", "b")),)
        criterion = (Criterion("c", "noninferior", 1.02),)
        base: dict[str, Any] = {
            "reference": "r",
            "challenger": "c",
            "initializations": (1, 2),
            "groups": group,
            "criteria": criterion,
            "seed": 1,
        }
        Contract(**base)
        for change in (
            {"challenger": "r"},
            {"initializations": (1, 1)},
            {"initializations": ()},
            {"groups": group * 2},
            {"draws": 10},
            {"confidence": 1.0},
            {"seed": -1},
            {"contract_id": "aaa.promotion.crossed.v0"},
        ):
            with self.assertRaises(ContractError, msg=str(change)):
                Contract(**{**base, **change})
        for bad in (float("nan"), 0.0, True):
            with self.assertRaises(ContractError):
                Criterion("c", "noninferior", bad)

    def test_insufficient_evidence_is_never_a_pass(self) -> None:
        for rule in ("noninferior", "superior"):
            self.assertEqual(criterion_status(rule, None, None, 1.02), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(combine(["PASS", "INSUFFICIENT_EVIDENCE"]), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(combine(["PASS", "INCONCLUSIVE"]), "INCONCLUSIVE")
        self.assertEqual(combine(["INSUFFICIENT_EVIDENCE", "FAIL"]), "REJECT")
        self.assertEqual(combine([]), "INSUFFICIENT_EVIDENCE")

    def test_the_rule_boundaries_are_the_declared_ones(self) -> None:
        self.assertEqual(criterion_status("noninferior", 0.9, 1.02, 1.02), "PASS")
        self.assertEqual(criterion_status("noninferior", 1.02, 1.1, 1.02), "INCONCLUSIVE")
        self.assertEqual(criterion_status("superior", 0.9, 1.0, 1.0), "INCONCLUSIVE")
        self.assertEqual(criterion_status("superior", 1.0, 1.1, 1.0), "FAIL")


class FrozenV2PathTests(unittest.TestCase):
    def test_the_frozen_v2_k5_path_is_forbidden_for_promotion(self) -> None:
        with self.assertRaises(ContractError) as caught:
            require_admissible("aaa.1k.v2.k1-k5")
        self.assertIn("AAA-180", str(caught.exception))
        require_admissible(CONTRACT_ID)

    def test_the_frozen_defect_is_retained_not_rewritten(self) -> None:
        # The historical defect stays in the frozen source: run_confirmation still
        # calls decide without Monash, and the frozen primary still refuses a
        # one-series group. The successor lives elsewhere.
        from research.aaa_1k_v2 import confirmation, stats

        tree = ast.parse(inspect.getsource(confirmation.run_confirmation))
        calls = [
            n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "decide"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("monash_primitives", {k.arg for k in calls[0].keywords})
        grid = np.linspace(1.0, 2.0, 8).reshape(8, 1)
        frozen = stats.geometric_relative({"single": (grid, grid * 1.1)}, seed=1, draws=1000)
        self.assertEqual(frozen["interval_status"], "INSUFFICIENT_EVIDENCE")

    def test_the_successor_measures_the_case_the_frozen_primary_refused(self) -> None:
        contract = fixtures.contract([1])
        result = _run(contract, fixtures.primitives(contract, effect=1.4))
        self.assertEqual(result["primary"]["interval_status"], "MEASURED")
        self.assertEqual(result["verdict"], "REJECT")


class IndependenceTests(unittest.TestCase):
    def test_the_recomputation_shares_no_code_with_the_primary(self) -> None:
        tree = ast.parse((ROOT / "aaa/promotion/independent.py").read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertNotIn("primary", " ".join(imported))
        # Only the declared contract values and the error type are shared.
        self.assertEqual(
            {i for i in imported if i.startswith("contract.")},
            {"contract.Contract", "contract.PrimitiveError"},
        )


class RetainedV2DescriptiveTests(unittest.TestCase):
    """The successor agrees with itself on the real retained K5 groups, including saugeen."""

    def test_both_successor_implementations_agree_on_every_retained_arm(self) -> None:
        import json

        spec = importlib.util.spec_from_file_location("cps", ROOT / "tools/check_promotion_successor.py")
        assert spec is not None and spec.loader is not None
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        evidence = ROOT / "docs/evidence/aaa_1k_v2"
        confirmation = json.loads((evidence / "confirmation.json").read_text(encoding="utf-8"))
        manifest = json.loads((evidence / "freeze.json").read_text(encoding="utf-8"))
        arms = sorted(a for a in confirmation["monash_primitives"] if a != tool.REFERENCE)
        self.assertEqual(len(arms), 6)
        for arm in arms:
            contract, records = tool.successor_inputs(confirmation, manifest, arm)
            self.assertEqual(contract.conditional_groups, ["monash:saugeen"])
            self.assertEqual(len(contract.groups), 19)
            result = adjudicate(contract, records)
            self.assertEqual(result["agreement_problems"], [], arm)
            self.assertNotIn(result["verdict"], {"DISAGREEMENT", "INVALID_EVIDENCE"}, arm)


if __name__ == "__main__":
    unittest.main()
