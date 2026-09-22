"""Tests for iteration 0004's precommitted challenger cycle.

Synthetic primitives drive the screen, the attack adjudication and the
confirmation decision through every outcome, and the independent
recomputation must agree with the decision on each. No real identity is
touched.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from research.aaa_1k.identity import phase_files
from research.aaa_1k_loop import iteration4 as it
from research.aaa_1k_loop import outer4, recompute4
from research.aaa_1k_loop.identities import IdentityError, empty_ledger, reserve

ROOT = Path(__file__).resolve().parents[1]
PLAN = [
    "plan:aba_v1",
    "plan:coarse_speed_v1",
    "plan:motion_compat:bouncing",
    "plan:motion_compat:changed",
    "plan:motion_compat:dynamics_change",
    "plan:occlusion_v1",
]


def primitives(
    *,
    arm: str = "c7_unfold_dr",
    ratio: float = 1.0,
    ratio_for: str | None = None,
    champion_diverges: float = 0.4,
    challenger_diverges: float = 0.0,
    noise: float = 0.002,
    seed: int = 0,
    lock: bool = False,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records = []
    for condition in [*PLAN, *it.LONG_CONDITIONS]:
        coarse = condition in it.LONG_COARSE
        for stream in range(8):
            for init in range(5):
                base = 1.0 + 0.1 * stream
                scale = ratio if (ratio_for is None or ratio_for == condition) else 1.0
                records.append(
                    {
                        "condition": condition,
                        "stream_id": f"s{stream}",
                        "init_index": init,
                        "init_seed": 100 + init,
                        "stream_seed": 1000 + stream,
                        "failures": {},
                        "persistence_mae": 1.0,
                        "arms": {
                            "gru": {
                                "mae": base,
                                "diverged": bool(coarse and rng.random() < champion_diverges),
                                "first_lock": None,
                            },
                            arm: {
                                "mae": base * scale * (1.0 + noise * rng.standard_normal()),
                                "diverged": bool(coarse and rng.random() < challenger_diverges),
                                "first_lock": 5 if lock else None,
                            },
                        },
                    }
                )
    return records


class DecisionTests(unittest.TestCase):
    def test_a_clean_fix_is_promoted_and_recomputed_identically(self) -> None:
        records = primitives()
        decision = it.decide4(records, "c7_unfold_dr")
        self.assertEqual(decision["outcome"], "PROMOTE", decision["statuses"])
        independent = recompute4.recompute(records, "c7_unfold_dr", it.FROZEN_RULES)
        self.assertEqual(independent["statuses"], decision["statuses"])

    def test_a_regression_on_one_family_rejects(self) -> None:
        records = primitives(ratio=1.10, ratio_for="plan:aba_v1")
        decision = it.decide4(records, "c7_unfold_dr")
        self.assertEqual(decision["statuses"]["K3_plan:aba_v1"], "FAIL")
        self.assertEqual(decision["outcome"], "REJECT")
        self.assertEqual(recompute4.recompute(records, "c7_unfold_dr", it.FROZEN_RULES)["outcome"], "REJECT")

    def test_a_straddling_interval_is_inconclusive_not_pass(self) -> None:
        records = primitives(ratio=1.02, ratio_for="plan:occlusion_v1", noise=0.02)
        decision = it.decide4(records, "c7_unfold_dr")
        self.assertEqual(decision["statuses"]["K3_plan:occlusion_v1"], "INCONCLUSIVE")
        self.assertEqual(decision["outcome"], "INCONCLUSIVE")

    def test_no_stability_gain_fails_the_primary_gate(self) -> None:
        records = primitives(champion_diverges=0.3, challenger_diverges=0.3)
        self.assertEqual(it.decide4(records, "c7_unfold_dr")["statuses"]["K1_long_coarse_stability"], "FAIL")

    def test_a_failed_arm_has_no_primitive(self) -> None:
        records = primitives()
        records[-1]["failures"] = {"c7_unfold_dr": "FloatingPointError"}
        with self.assertRaises(ValueError):
            it.decide4(records, "c7_unfold_dr")


class ScreenAndAttackTests(unittest.TestCase):
    def test_screen_requires_every_interval_below_the_margin(self) -> None:
        self.assertTrue(it.screen(primitives(), "c7_unfold_dr")["passed"])
        self.assertFalse(it.screen(primitives(ratio=1.05, ratio_for="plan:aba_v1"), "c7_unfold_dr")["passed"])

    def test_selection_follows_the_precommitted_order(self) -> None:
        passed, failed = {"passed": True}, {"passed": False}
        self.assertEqual(it.select_for_attack({"c7_unfold_dr": passed, "c8_folded": passed}), "c7_unfold_dr")
        self.assertEqual(it.select_for_attack({"c7_unfold_dr": failed, "c8_folded": passed}), "c8_folded")
        self.assertIsNone(it.select_for_attack({"c7_unfold_dr": failed, "c8_folded": failed}))

    def test_a_lock_in_any_attack_cell_fails_the_attack(self) -> None:
        records = primitives(lock=True)
        for condition in it.NEARBY_CONDITIONS:
            for r in [r for r in records if r["condition"] == "long:coarse_no_switch"]:
                records.append({**r, "condition": condition})
        result = it.adjudicate_attack(records, "c7_unfold_dr")
        self.assertEqual(result["outcome"]["A5_no_frame_lock"], "FAIL")
        self.assertFalse(result["advance_to_freeze"])


class IdentityTests(unittest.TestCase):
    def test_confirmation_cells_refuse_a_non_confirmation_block(self) -> None:
        ledger = reserve(
            empty_ledger(),
            block_id=it.CONFIRMATION_ENV_BLOCK,
            role="attack",
            namespace="confirmation_env",
            count=it.CONFIRMATION_ENV_COUNT,
            purpose="test",
            iteration_id=it.ITERATION_ID,
        )
        with self.assertRaises(IdentityError):
            it.confirmation_cells(ledger)

    def test_block_sizes_match_the_designs(self) -> None:
        self.assertEqual(it.DEVELOPMENT_BLOCK_SIZE, 64)
        self.assertEqual(it.ATTACK_ENV_BLOCK_SIZE, 80)
        self.assertEqual(it.CONFIRMATION_ENV_COUNT, 112)


class ConfirmationPathTests(unittest.TestCase):
    def test_the_confirmation_path_imports_only_frozen_sources(self) -> None:
        allowed = {
            Path(n).stem for n in outer4.CONFIRMATION_SOURCES_4 if n.startswith("research/aaa_1k_loop/")
        }
        for name in outer4.CONFIRMATION_SOURCES_4:
            if not name.startswith("research/aaa_1k_loop/"):
                continue
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 1:
                        module = (node.module or "__init__").split(".")[0]
                        self.assertIn(module, allowed, f"{name} imports .{module}")
                    elif node.module and node.module.startswith("research.aaa_1k_loop"):
                        self.assertIn(node.module.split(".")[-1], allowed, f"{name} imports {node.module}")
                    elif node.module and node.module.startswith("aaa."):
                        frozen = set(outer4.CONFIRMATION_SOURCES_4) | set(phase_files(ROOT))
                        self.assertIn(f"{node.module.replace('.', '/')}.py", frozen)

    def test_the_recomputation_does_not_import_the_decision(self) -> None:
        tree = ast.parse((ROOT / "research/aaa_1k_loop/recompute4.py").read_text(encoding="utf-8"))
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertNotIn("iteration4", modules)
        self.assertNotIn("outer4", modules)

    def test_confirm_refuses_without_a_freeze(self) -> None:
        if (ROOT / outer4.FREEZE).exists():
            self.skipTest("a real freeze exists")
        from research.aaa_1k_loop.freeze import FreezeError

        with self.assertRaises(FreezeError):
            outer4.main(["confirm", "--output", "docs/evidence/aaa1k_loop_0004/_never.json"])


if __name__ == "__main__":
    unittest.main()
