"""Tests for iteration 0004's precommitted challenger cycle.

Synthetic primitives drive the screen, the attack adjudication and the
confirmation decision through every outcome, and the independent
recomputation must agree with the decision on each. No real identity is
touched.
"""

from __future__ import annotations

import ast
import json
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
            it.confirmation_cells(ledger, observer="anyone")

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


class Iteration5Tests(unittest.TestCase):
    def test_iteration5_judges_with_the_identical_functions(self) -> None:
        from research.aaa_1k_loop import iteration5 as it5

        self.assertIs(it5.screen, it.screen)
        self.assertIs(it5.adjudicate_attack, it.adjudicate_attack)
        self.assertIs(it5.decide, it.decide4)
        self.assertEqual(len(it5.CANDIDATES), 1)
        self.assertEqual({k: v for k, v in it5.FROZEN_RULES.items() if k != "shared_with"}, it.FROZEN_RULES)

    def test_iteration5_confirmation_path_is_closed(self) -> None:
        from research.aaa_1k_loop import outer5

        allowed = {
            Path(n).stem for n in outer5.CONFIRMATION_SOURCES_5 if n.startswith("research/aaa_1k_loop/")
        }
        frozen = set(outer5.CONFIRMATION_SOURCES_5) | set(phase_files(ROOT))
        for name in outer5.CONFIRMATION_SOURCES_5:
            if not name.startswith("research/aaa_1k_loop/"):
                continue
            for node in ast.walk(ast.parse((ROOT / name).read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 1:
                        self.assertIn((node.module or "__init__").split(".")[0], allowed, name)
                    elif node.module and node.module.startswith("research.aaa_1k_loop"):
                        self.assertIn(node.module.split(".")[-1], allowed, name)
                    elif node.module and node.module.startswith("aaa."):
                        self.assertIn(f"{node.module.replace('.', '/')}.py", frozen, name)


class Iteration6Tests(unittest.TestCase):
    def test_iteration6_judges_with_the_identical_functions_and_one_candidate(self) -> None:
        from research.aaa_1k_loop import iteration6 as it6

        self.assertIs(it6.screen, it.screen)
        self.assertIs(it6.adjudicate_attack, it.adjudicate_attack)
        self.assertIs(it6.decide, it.decide4)
        self.assertEqual([c["id"] for c in it6.CANDIDATES], ["c10"])

    def test_iteration6_confirmation_path_is_closed(self) -> None:
        from research.aaa_1k_loop import outer6

        allowed = {
            Path(n).stem for n in outer6.CONFIRMATION_SOURCES_6 if n.startswith("research/aaa_1k_loop/")
        }
        frozen = set(outer6.CONFIRMATION_SOURCES_6) | set(phase_files(ROOT))
        for name in outer6.CONFIRMATION_SOURCES_6:
            if not name.startswith("research/aaa_1k_loop/"):
                continue
            for node in ast.walk(ast.parse((ROOT / name).read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 1:
                        self.assertIn((node.module or "__init__").split(".")[0], allowed, name)
                    elif node.module and node.module.startswith("research.aaa_1k_loop"):
                        self.assertIn(node.module.split(".")[-1], allowed, name)
                    elif node.module and node.module.startswith("aaa."):
                        self.assertIn(f"{node.module.replace('.', '/')}.py", frozen, name)


class ConfirmationAdmissionTests(unittest.TestCase):
    """The admission-to-cells sequence whose defect aborted iteration 0006's first attempt."""

    def _ledger(self, it_module: Any) -> dict[str, Any]:
        ledger = empty_ledger()
        for block_id, namespace, count in (
            (
                it_module.CONFIRMATION_ENV_BLOCK,
                getattr(it_module, "CONFIRMATION_ENV_NAMESPACE", "confirmation_env"),
                it.CONFIRMATION_ENV_COUNT,
            ),
            (
                it_module.CONFIRMATION_INIT_BLOCK,
                getattr(it_module, "CONFIRMATION_INIT_NAMESPACE", "confirmation_init"),
                it.CONFIRMATION_INIT_COUNT,
            ),
        ):
            ledger = reserve(
                ledger,
                block_id=block_id,
                role="confirmation",
                namespace=namespace,
                count=count,
                purpose="test",
                iteration_id=it_module.ITERATION_ID,
            )
        return ledger

    def test_spend_then_build_yields_the_full_design_for_every_outer_module(self) -> None:
        from research.aaa_1k_loop import iteration5, iteration6, outer5, outer6

        for it_module, outer in ((it, outer4), (iteration5, outer5), (iteration6, outer6)):
            ledger, cells = outer.spend_and_build(self._ledger(it_module), "observer-A")
            self.assertEqual(len(cells), (it.CONFIRMATION_PER_ENTRY * 6 + it.CONFIRMATION_LONG * 4) * 5)
            for block_id in (it_module.CONFIRMATION_ENV_BLOCK, it_module.CONFIRMATION_INIT_BLOCK):
                block = next(b for b in ledger["blocks"] if b["block_id"] == block_id)
                self.assertEqual((block["status"], block["observed_by"]), ("spent", "observer-A"))

    def test_cells_refuse_unspent_or_foreign_blocks(self) -> None:
        from research.aaa_1k_loop import iteration6, outer6

        ledger = self._ledger(iteration6)
        with self.assertRaises(IdentityError):
            iteration6.confirmation_cells(ledger, observer="observer-A")
        spent, _cells = outer6.spend_and_build(ledger, "observer-A")
        with self.assertRaises(IdentityError):
            iteration6.confirmation_cells(spent, observer="observer-B")
        with self.assertRaises(IdentityError):
            outer6.spend_and_build(spent, "observer-B")

    def test_attempt_two_does_not_reuse_the_burned_blocks(self) -> None:
        from research.aaa_1k_loop import iteration6
        from research.aaa_1k_loop.identities import load_ledger

        self.assertEqual(iteration6.CONFIRMATION_ATTEMPT, 2)
        ledger = load_ledger(ROOT / "benchmarks/aaa1k_loop_identity_ledger.json")
        burned = {
            b["block_id"]: b
            for b in ledger["blocks"]
            if b["block_id"].startswith("aaa1k-loop-0006/confirmation/")
        }
        self.assertEqual({b["status"] for b in burned.values()}, {"spent"})
        self.assertNotIn(iteration6.CONFIRMATION_ENV_BLOCK, burned)


class ConfirmationEndToEndTests(unittest.TestCase):
    """Admission -> cells -> primitives -> payload -> decision -> independent recomputation, synthetically."""

    def test_the_whole_outer_path_agrees_with_its_independent_recomputation(self) -> None:
        import hashlib
        import json
        import tempfile

        from research.aaa_1k_loop import iteration6, outer6
        from research.aaa_1k_loop.freeze import seed_list_sha256

        ledger = ConfirmationAdmissionTests()._ledger(iteration6)
        ledger, cells = outer6.spend_and_build(ledger, "observer-A")
        rng = np.random.default_rng(3)
        primitives = []
        for cell in cells:
            coarse = cell.condition in it.LONG_COARSE
            base = 1.0 + rng.random()
            primitives.append(
                {
                    "condition": cell.condition,
                    "init_index": cell.init_index,
                    "init_seed": cell.init_seed,
                    "stream_id": cell.stream.stream_id,
                    "stream_seed": cell.stream.seed,
                    "family": cell.stream.family,
                    "failures": {},
                    "persistence_mae": 1.0,
                    "arms": {
                        "gru": {
                            "mae": base,
                            "diverged": bool(coarse and rng.random() < 0.4),
                            "first_lock": None,
                        },
                        "c10_reach_gated_unfold": {"mae": base, "diverged": False, "first_lock": None},
                    },
                }
            )
        manifest = {
            "confirmation_source_fingerprint": {"sha256": "f" * 64},
            "champion_phase_fingerprint": "e" * 64,
            "frozen": {"challenger": {"arm": "c10_reach_gated_unfold"}, "rules": iteration6.FROZEN_RULES},
            "confirmation_blocks": {
                block_id: {
                    "namespace": next(b for b in ledger["blocks"] if b["block_id"] == block_id)["namespace"],
                    "seed_list_sha256": seed_list_sha256(ledger, block_id),
                }
                for block_id in (iteration6.CONFIRMATION_ENV_BLOCK, iteration6.CONFIRMATION_INIT_BLOCK)
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            freeze_path = Path(directory) / "freeze.json"
            freeze_path.write_text(json.dumps(manifest), encoding="utf-8")
            payload = outer6.confirmation_payload(
                manifest,
                freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                head="0" * 40,
                primitives=primitives,
            )
            payload["decision"] = iteration6.decide(primitives, payload["challenger"])
            confirmation_path = Path(directory) / "confirmation.json"
            confirmation_path.write_text(json.dumps(payload), encoding="utf-8")
            result = recompute4.verify(confirmation_path, freeze_path, ledger)
        self.assertEqual(payload["decision"]["outcome"], "PROMOTE")
        self.assertTrue(result["agrees"], result["problems"])
        self.assertEqual(result["independent"]["outcome"], "PROMOTE")


class ChampionOneTests(unittest.TestCase):
    def test_the_committed_champion_1_record_is_derived_from_the_repository(self) -> None:
        from research.aaa_1k_loop import champion1

        if not (ROOT / champion1.RECORD).exists():
            self.skipTest("Champion 1 not yet recorded")
        self.assertEqual(champion1.verify(ROOT), [])
        record = json.loads((ROOT / champion1.RECORD).read_text(encoding="utf-8"))
        self.assertEqual(record["parameter_count"], 994)
        self.assertEqual(record["state_footprint"]["total_adaptive_state_scalars"], 1414)
        self.assertEqual(record["promoted_by"]["outcome"], "PROMOTE")
        self.assertTrue(record["round3_evidence"]["round3_identities_design_and_configuration_unchanged"])

    def test_the_champion_1_world_is_always_undone(self) -> None:
        import research.aaa_1k.agents as agents_module
        import research.aaa_1k.experiments as experiments_module
        import research.aaa_1k.measurements as measurements_module
        from research.aaa_1k.agents import NeuralAgent
        from research.aaa_1k_loop import champion1
        from research.aaa_1k_loop.unfolding import ReachGatedUnfoldAgent

        with self.assertRaises(RuntimeError), champion1.champion_1_world():
            self.assertIs(experiments_module.NeuralAgent, ReachGatedUnfoldAgent)
            self.assertIs(agents_module.NeuralAgent, ReachGatedUnfoldAgent)
            raise RuntimeError("interrupted")
        for module in (agents_module, experiments_module, measurements_module):
            self.assertIs(module.NeuralAgent, NeuralAgent)


class ReadjudicationTests(unittest.TestCase):
    """Every committed 0004-0006 verdict must re-derive exactly from its own stored records."""

    @staticmethod
    def _load(path: str) -> Any:
        return json.loads((ROOT / path).read_text(encoding="utf-8"))

    def test_diagnoses_screens_attack_and_decision_readjudicate_exactly(self) -> None:
        from research.aaa_1k_loop import (
            diagnosis4,
            diagnosis4b,
            diagnosis4c,
            diagnosis4d,
            iteration5,
            iteration6,
        )

        evidence = "docs/evidence/aaa1k_loop_000"
        for adjudicate, name in (
            (diagnosis4.adjudicate4, "4/diagnosis.json"),
            (diagnosis4b.adjudicate4b, "4/diagnosis_gain.json"),
            (diagnosis4c.adjudicate4c, "4/diagnosis_overshoot.json"),
            (diagnosis4d.adjudicate4d, "4/diagnosis_unfold.json"),
        ):
            stored = self._load(evidence + name)
            self.assertEqual(json.loads(json.dumps(adjudicate(stored["records"]))), stored["analysis"], name)
        for module, number in ((it, "4"), (iteration5, "5"), (iteration6, "6")):
            stored = self._load(f"{evidence}{number}/development.json")
            for candidate in module.CANDIDATES:
                recomputed = json.loads(json.dumps(module.screen(stored["records"], candidate["arm"])))
                self.assertEqual(recomputed, stored["screens"][candidate["arm"]], (number, candidate["id"]))
        attack = self._load(f"{evidence}6/attack.json")
        self.assertEqual(
            json.loads(json.dumps(iteration6.adjudicate_attack(attack["records"], attack["attacked"]))),
            attack["adjudication"],
        )
        confirmation = self._load(f"{evidence}6/confirmation_2.json")
        decision = iteration6.decide(confirmation["primitives"], confirmation["challenger"])
        self.assertEqual(json.loads(json.dumps(decision)), confirmation["decision"])
        self.assertEqual(decision["outcome"], "PROMOTE")


class ReproductionRegistryTests(unittest.TestCase):
    def test_every_post_audit_evidence_file_with_primitives_is_reproducible(self) -> None:
        from research.aaa_1k_loop.stages4 import REPRODUCIBLE_4

        registered = {path for _target, path in REPRODUCIBLE_4.values()}
        for path in sorted(ROOT.glob("docs/evidence/aaa1k_loop_000[456]/*.json")):
            content = json.loads(path.read_text(encoding="utf-8"))
            if "records" in content or "primitives" in content:
                relative = str(path.relative_to(ROOT))
                self.assertIn(relative, registered, f"{relative} holds primitives but has no reproduction")

    def test_the_scheduled_workflow_covers_every_registered_stage(self) -> None:
        from research.aaa_1k_loop.stages4 import REPRODUCIBLE_4

        workflow = (ROOT / ".github/workflows/loop-reproduction.yml").read_text(encoding="utf-8")
        for stage in REPRODUCIBLE_4:
            self.assertIn(stage, workflow)


class RecomputationTamperTests(unittest.TestCase):
    def test_a_tampered_primitive_that_keeps_every_status_is_still_caught(self) -> None:
        import tempfile

        from research.aaa_1k_loop.identities import load_ledger

        confirmation_path = ROOT / "docs/evidence/aaa1k_loop_0006/confirmation_2.json"
        freeze_path = ROOT / "docs/evidence/aaa1k_loop_0006/freeze_2.json"
        ledger = load_ledger(ROOT / "benchmarks/aaa1k_loop_identity_ledger.json")
        self.assertTrue(recompute4.verify(confirmation_path, freeze_path, ledger)["agrees"])
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        cell = next(
            r
            for r in confirmation["primitives"]
            if r["condition"] == "long:coarse_no_switch" and not r["arms"]["gru"]["diverged"]
        )
        cell["arms"]["gru"]["diverged"] = True
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "confirmation.json"
            tampered.write_text(json.dumps(confirmation), encoding="utf-8")
            result = recompute4.verify(tampered, freeze_path, ledger)
        self.assertEqual(result["independent"]["outcome"], "PROMOTE")
        self.assertFalse(result["agrees"])
        self.assertTrue(any(p.startswith("K1_long_coarse_stability.") for p in result["problems"]))


class TolerantComparisonTests(unittest.TestCase):
    def test_floats_tolerate_last_digit_noise_but_nothing_else_does(self) -> None:
        from research.aaa_1k_loop.stages4 import compare_tolerant

        mismatches, deviations = compare_tolerant({"x": 0.00010928177700710461}, {"x": 0.0001092817770071042})
        self.assertEqual(mismatches, [])
        self.assertEqual(len(deviations), 1)
        self.assertTrue(compare_tolerant({"x": 1.0}, {"x": 1.0 + 1e-6})[0])
        self.assertTrue(compare_tolerant({"diverged": True}, {"diverged": False})[0])
        self.assertTrue(compare_tolerant({"diverged": True}, {"diverged": 1.0})[0])
        self.assertTrue(compare_tolerant({"n": 3}, {"n": 4})[0])
        self.assertTrue(compare_tolerant({"verdict": "PASS"}, {"verdict": "FAIL"})[0])
        self.assertTrue(compare_tolerant({"a": [1.0, 2.0]}, {"a": [1.0]})[0])
        self.assertEqual(compare_tolerant({"git": "a", "x": 2.0}, {"git": "b", "x": 2.0}), ([], []))


class CrossPlatformReproductionTests(unittest.TestCase):
    """Chaotic cells may drift; stable cells may not beyond tolerance; verdicts never may."""

    def _pair(self) -> tuple[dict[str, Any], dict[str, Any]]:
        confirmation = json.loads(
            (ROOT / "docs/evidence/aaa1k_loop_0006/confirmation_2.json").read_text(encoding="utf-8")
        )
        committed = {"primitives": confirmation["primitives"]}
        return committed, json.loads(json.dumps(committed))

    def test_identical_reruns_reproduce(self) -> None:
        from research.aaa_1k_loop.stages4 import compare_reproduction

        committed, fresh = self._pair()
        result = compare_reproduction(committed, fresh, "c10_reach_gated_unfold")
        self.assertEqual(result["mismatches"], [])
        self.assertGreater(result["chaotic_cells"], 0)

    def test_a_drifting_chaotic_cell_is_reported_not_failed(self) -> None:
        from research.aaa_1k_loop.stages4 import chaotic, compare_reproduction

        committed, fresh = self._pair()
        cell = next(r for r in fresh["primitives"] if chaotic(r))
        cell["arms"]["gru"]["mae"] *= 1.3
        result = compare_reproduction(committed, fresh, "c10_reach_gated_unfold")
        self.assertEqual(result["mismatches"], [])
        self.assertGreater(result["largest_chaotic_deviation"], 0.2)

    def test_a_drifting_stable_cell_fails(self) -> None:
        from research.aaa_1k_loop.stages4 import chaotic, compare_reproduction

        committed, fresh = self._pair()
        cell = next(r for r in fresh["primitives"] if not chaotic(r))
        cell["arms"]["gru"]["mae"] *= 1.0 + 1e-6
        self.assertTrue(compare_reproduction(committed, fresh, "c10_reach_gated_unfold")["mismatches"])

    def test_a_changed_conclusion_fails_even_in_chaotic_cells(self) -> None:
        from research.aaa_1k_loop.stages4 import compare_reproduction

        committed, fresh = self._pair()
        coarse = [r for r in fresh["primitives"] if r["condition"].startswith("long:coarse")]
        for cell in coarse[:40]:
            cell["arms"]["c10_reach_gated_unfold"]["diverged"] = True
        result = compare_reproduction(committed, fresh, "c10_reach_gated_unfold")
        self.assertTrue(any(m.startswith("verdicts differ") for m in result["mismatches"]))
