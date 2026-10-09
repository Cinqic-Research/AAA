"""``aaa.promotion.paired.v1``: verdict regions, fail-closed evidence handling, implementation agreement."""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np

from research.aaa_erudition import promotion as P


def contract(n: int, *criteria: P.Criterion, seed: int = 7) -> P.Contract:
    return P.Contract("test", tuple(f"s{i:02d}" for i in range(n)), criteria, seed, draws=4000)


def cells(metric: str, values: dict[str, list[float]]) -> list[dict[str, Any]]:
    return [
        {"metric": metric, "arm": arm, "stream": f"s{i:02d}", "value": float(v)}
        for arm, vs in values.items()
        for i, v in enumerate(vs)
    ]


def superior(threshold: float = 0.0) -> P.Criterion:
    return P.Criterion("target", "failure", "joint", "frozen", "superior", threshold)


def noninferior(threshold: float = 0.03) -> P.Criterion:
    return P.Criterion("regression", "retention", "joint", "frozen", "noninferior", threshold)


class PairedContractTests(unittest.TestCase):
    def test_clear_improvement_promotes(self) -> None:
        g = np.random.default_rng(1)
        ref = list(g.uniform(0.4, 0.6, 12))
        evidence = cells("failure", {"frozen": ref, "joint": [r - 0.2 for r in ref]})
        evidence += cells("retention", {"frozen": [0.1] * 12, "joint": [0.1] * 12})
        result = P.adjudicate(contract(12, superior(), noninferior()), evidence)
        self.assertEqual(result["verdict"], "PROMOTE")

    def test_regression_blocks_promotion_even_with_target_gain(self) -> None:
        ref = [0.5] * 12
        evidence = cells("failure", {"frozen": ref, "joint": [0.3] * 12})
        evidence += cells("retention", {"frozen": [0.05] * 12, "joint": [0.25] * 12})
        self.assertEqual(P.adjudicate(contract(12, superior(), noninferior()), evidence)["verdict"], "REJECT")

    def test_worse_challenger_fails_superiority(self) -> None:
        evidence = cells("failure", {"frozen": [0.3] * 10, "joint": [0.5] * 10})
        result = P.adjudicate(contract(10, superior()), evidence)
        self.assertEqual(result["verdict"], "REJECT")
        self.assertEqual(result["primary"]["criteria"][0]["status"], "FAIL")

    def test_noise_is_inconclusive(self) -> None:
        g = np.random.default_rng(3)
        ref = list(g.uniform(0.3, 0.7, 10))
        noisy = [r + d for r, d in zip(ref, g.normal(0, 0.15, 10), strict=True)]
        noisy = [min(1.0, max(0.0, v)) for v in noisy]
        evidence = cells("failure", {"frozen": ref, "joint": noisy})
        self.assertEqual(P.adjudicate(contract(10, superior()), evidence)["verdict"], "INCONCLUSIVE")

    def test_too_few_streams_is_insufficient(self) -> None:
        evidence = cells("failure", {"frozen": [0.5] * 4, "joint": [0.0] * 4})
        self.assertEqual(P.adjudicate(contract(4, superior()), evidence)["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_zero_rates_are_valid(self) -> None:
        evidence = cells("failure", {"frozen": [0.2] * 8, "joint": [0.0] * 8})
        self.assertEqual(P.adjudicate(contract(8, superior()), evidence)["verdict"], "PROMOTE")

    def test_malformed_evidence_fails_closed(self) -> None:
        good = cells("failure", {"frozen": [0.5] * 8, "joint": [0.1] * 8})
        broken = {
            "missing": good[:-1],
            "duplicate": [*good, good[0]],
            "extra": [*good, {"metric": "failure", "arm": "joint", "stream": "s99", "value": 0.1}],
            "nan": [{**good[0], "value": float("nan")}, *good[1:]],
            "range": [{**good[0], "value": 1.5}, *good[1:]],
            "bool": [{**good[0], "value": True}, *good[1:]],
            "numpy": [{**good[0], "value": np.float32(0.5)}, *good[1:]],
            "string": [{**good[0], "value": "0.1"}, *good[1:]],
        }
        for name, evidence in broken.items():
            with self.subTest(name):
                self.assertEqual(
                    P.adjudicate(contract(8, superior()), evidence)["verdict"], "INVALID_EVIDENCE"
                )

    def test_a_contract_without_criteria_is_refused(self) -> None:
        evidence = cells("failure", {"frozen": [0.5] * 8, "joint": [0.1] * 8})
        self.assertEqual(P.adjudicate(contract(8), evidence)["verdict"], "INVALID_EVIDENCE")

    def test_other_contracts_are_refused(self) -> None:
        c = P.Contract("x", ("a",) * 0, (superior(),), 1, contract="aaa.promotion.crossed.v1")
        self.assertEqual(P.adjudicate(c, [])["verdict"], "INVALID_EVIDENCE")

    def test_injected_disagreement_never_promotes(self) -> None:
        evidence = cells("failure", {"frozen": [0.5] * 8, "joint": [0.1] * 8})
        honest = P.independent(contract(8, superior()), evidence)
        for field, value in (("point", 0.3), ("status", "FAIL"), ("lower", -10.0)):
            tampered = {**honest, "criteria": [{**honest["criteria"][0], field: value}]}
            with self.subTest(field), mock.patch.object(P, "independent", return_value=tampered):
                self.assertEqual(P.adjudicate(contract(8, superior()), evidence)["verdict"], "DISAGREEMENT")

    def test_implementations_agree_on_random_fixtures(self) -> None:
        g = np.random.default_rng(11)
        for trial in range(20):
            n = int(g.integers(6, 20))
            ref = g.uniform(0, 1, n)
            ch = np.clip(ref + g.normal(g.uniform(-0.2, 0.2), 0.1, n), 0, 1)
            evidence = cells("failure", {"frozen": list(ref), "joint": list(ch)})
            with self.subTest(trial):
                self.assertNotEqual(
                    P.adjudicate(contract(n, superior(), seed=trial), evidence)["verdict"], "DISAGREEMENT"
                )


if __name__ == "__main__":
    unittest.main()
