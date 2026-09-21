"""Tests for the AAA-1K loop pilot, most of them failure injection.

A loop that only works when everybody behaves is documentation. Each class
below breaks one invariant on purpose -- reuse a spent identity, overlap
development with confirmation, change the challenger after the freeze, claim
PROMOTE with a failed criterion, drop a rejected attempt, write NaN -- and
requires the machinery to refuse.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from research.aaa_1k.experiments import EVALUATION_PLAN
from research.aaa_1k.identity import PhaseIdentityError, phase_files, phase_fingerprint
from research.aaa_1k.runner import run_stream
from research.aaa_1k.seeds import derive_seed
from research.aaa_1k.stats import crossed_paired_difference
from research.aaa_1k.streams import coarse_speed_stream, motion_compat_stream, occlusion_stream
from research.aaa_1k_loop import cli
from research.aaa_1k_loop.arms import (
    ARMS,
    CHAMPION_CONFIGURATION,
    STATELESS_CONFIGURATION,
    UNGATED_CONFIGURATION,
    ArmSet,
    gru,
    gru_with_gate_biases,
    gru_with_split_keep_bias,
    initial_parameters_equal_except,
)
from research.aaa_1k_loop.bounded import BoundedErrorAgent
from research.aaa_1k_loop.champion import build_champion, verify_champion
from research.aaa_1k_loop.decision import THRESHOLDS, decide, outcome_of
from research.aaa_1k_loop.develop import plan_entry_name
from research.aaa_1k_loop.dynamics import gru_jacobian, rnn_jacobian
from research.aaa_1k_loop.evidence import EvidenceError, read_strict_json, write_strict_json
from research.aaa_1k_loop.freeze import (
    CONFIRMATION_SOURCES,
    FreezeError,
    build_freeze,
    require_committed,
    verify_freeze,
)
from research.aaa_1k_loop.harness import Cell, _merge, _run_one, matrix
from research.aaa_1k_loop.identities import (
    IdentityError,
    block_seeds,
    derive_loop_seed,
    empty_ledger,
    find_block,
    load_ledger,
    mark,
    prove_fresh,
    require_usable,
    reserve,
    validate_ledger,
)
from research.aaa_1k_loop.iteration import IterationError, validate_iteration
from research.aaa_1k_loop.recompute import crossed, verify_confirmation

ROOT = Path(__file__).resolve().parents[1]
RECORD_0001 = ROOT / "docs/evidence/aaa1k_loop_0001/iteration.json"


def ledger_with(*blocks: dict[str, Any]) -> dict[str, Any]:
    ledger = empty_ledger()
    for block in blocks:
        ledger = reserve(ledger, **block)
    return ledger


def block(block_id: str, role: str, namespace: str, count: int = 4, start: int = 0) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "role": role,
        "namespace": namespace,
        "count": count,
        "start": start,
        "purpose": "test",
        "iteration_id": "test-iteration",
    }


# ----------------------------------------------------------------------
# synthetic confirmation primitives
# ----------------------------------------------------------------------
def synthetic_primitives(env: list[int], inits: list[int], *, deficit: float = 6e-4) -> dict[str, Any]:
    """Records shaped like iteration 0003's confirmation, with controllable effects."""

    rng = np.random.default_rng(3)
    plan: list[dict[str, Any]] = []
    cursor = 0
    for family, options in EVALUATION_PLAN:
        entry = plan_entry_name(family, options)
        for _ in range(4):
            seed = env[cursor]
            cursor += 1
            for i, init in enumerate(inits):
                base = 2e-3 + 1e-4 * rng.standard_normal()
                coarse = family == "coarse_speed_v1"
                mae = {
                    "gru": base + (deficit if coarse else -5e-5),
                    "rnn28": base,
                    "mlp": base + (7e-4 if coarse else 3e-5),
                    "persistence": 4e-3,
                    "gru_state_reset": base + 1e-4,
                }
                record: dict[str, Any] = {
                    "condition": entry,
                    "plan_entry": entry,
                    "init_index": i,
                    "init_seed": init,
                    "stream_id": f"{family}:{seed}",
                    "family": family,
                    "stream_seed": seed,
                    "mae": mae,
                    "failures": {},
                }
                if coarse:
                    record["regime_mae"] = {
                        "slow": {"gru": 2.5e-3, "rnn28": 1.4e-3},
                        "fast": {"gru": 2.1e-3, "rnn28": 1.9e-3},
                    }
                    record["jacobian_negative_mode_median"] = {"gru": 0.0, "rnn28": 0.75}
                plan.append(record)
    fixed = []
    for seed in env[cursor : cursor + 4]:
        for i, init in enumerate(inits):
            base = 2e-3 + 1e-4 * rng.standard_normal()
            fixed.append(
                {
                    "condition": "coarse_fixed_speed",
                    "plan_entry": "coarse_fixed_speed",
                    "init_index": i,
                    "init_seed": init,
                    "stream_id": f"coarse_speed_v1:{seed}",
                    "family": "coarse_speed_v1",
                    "stream_seed": seed,
                    "mae": {"gru": base + 9e-4, "rnn28": base, "mlp": base + 7e-4, "persistence": 4e-3},
                    "failures": {},
                }
            )
    return {"plan": plan, "fixed_speed": fixed, "long_fixed_speed": []}


def confirmation_fixture(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    """A spent confirmation ledger, a freeze manifest and a confirmation artifact."""

    ledger = ledger_with(
        block("test/confirmation/env", "confirmation", "confirmation_env", count=40),
        block("test/confirmation/init", "confirmation", "confirmation_init", count=3),
    )
    env = block_seeds(find_block(ledger, "test/confirmation/env"))
    inits = block_seeds(find_block(ledger, "test/confirmation/init"))
    freeze = {
        "frozen": {"thresholds": THRESHOLDS},
        "confirmation_source_fingerprint": {"sha256": "f" * 64},
        "confirmation_blocks": {
            block_id: {"seed_list_sha256": cli_seed_hash(ledger, block_id)}
            for block_id in ("test/confirmation/env", "test/confirmation/init")
        },
    }
    freeze_path = directory / "freeze.json"
    write_strict_json(freeze_path, freeze)
    payload: dict[str, Any] = {
        "thresholds": THRESHOLDS,
        "primitives": synthetic_primitives(env, inits),
        "confirmation_source_fingerprint": "f" * 64,
        "freeze": {"sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest()},
    }
    payload["decision"] = decide(payload)
    confirmation_path = directory / "confirmation.json"
    write_strict_json(confirmation_path, payload)
    for block_id in ("test/confirmation/env", "test/confirmation/init"):
        ledger = mark(ledger, block_id, status="spent", observed_by="test")
    return confirmation_path, freeze_path, ledger


def cli_seed_hash(ledger: dict[str, Any], block_id: str) -> str:
    from research.aaa_1k_loop.freeze import seed_list_sha256

    return seed_list_sha256(ledger, block_id)


# ======================================================================
# identities
# ======================================================================
class IdentityLedgerTests(unittest.TestCase):
    def test_loop_seeds_are_deterministic_and_salted_by_iteration(self):
        self.assertEqual(derive_loop_seed("x", 3), derive_loop_seed("x", 3))
        self.assertNotEqual(
            derive_loop_seed("x", 3, iteration_id="a"), derive_loop_seed("x", 3, iteration_id="b")
        )
        with self.assertRaises(IdentityError):
            derive_loop_seed("bad:namespace", 0)
        with self.assertRaises(IdentityError):
            derive_loop_seed("x", -1)

    def test_confirmation_identities_cannot_overlap_development(self):
        ledger = ledger_with(block("dev", "development", "shared"))
        with self.assertRaisesRegex(IdentityError, "overlaps"):
            reserve(ledger, **block("conf", "confirmation", "shared"))

    def test_a_block_id_cannot_be_reserved_twice(self):
        ledger = ledger_with(block("dev", "development", "a"))
        with self.assertRaisesRegex(IdentityError, "duplicate"):
            reserve(ledger, **block("dev", "development", "b"))

    def test_a_spent_confirmation_block_cannot_be_used_again(self):
        ledger = ledger_with(block("conf", "confirmation", "c"))
        require_usable(ledger, "conf", purpose="confirmation")
        spent = mark(ledger, "conf", status="spent", observed_by="first observation")
        with self.assertRaisesRegex(IdentityError, "not fresh"):
            require_usable(spent, "conf", purpose="confirmation")
        with self.assertRaisesRegex(IdentityError, "already spent"):
            mark(spent, "conf", status="spent", observed_by="second observation")

    def test_status_never_moves_backwards(self):
        ledger = mark(ledger_with(block("dev", "development", "d")), "dev", status="spent", observed_by="x")
        with self.assertRaises(IdentityError):
            mark(ledger, "dev", status="reserved", observed_by="undo")

    def test_confirmation_evidence_may_never_inform_selection(self):
        ledger = ledger_with(block("conf", "confirmation", "c"))
        with self.assertRaisesRegex(IdentityError, "may not inform selection"):
            require_usable(ledger, "conf", purpose="selection")
        dev = ledger_with(block("dev", "development", "d"))
        with self.assertRaisesRegex(IdentityError, "not a confirmation block"):
            require_usable(dev, "dev", purpose="confirmation")

    def test_a_spent_block_must_say_what_observed_it(self):
        ledger = ledger_with(block("conf", "confirmation", "c"))
        broken = copy.deepcopy(ledger)
        broken["blocks"][0]["status"] = "spent"
        with self.assertRaisesRegex(IdentityError, "must record what observed it"):
            validate_ledger(broken)
        broken["blocks"][0]["status"] = "used_for_development"
        with self.assertRaisesRegex(IdentityError, "cannot be used for development"):
            validate_ledger(broken)

    def test_freshness_proof_catches_a_collision_with_aaa1k_identities(self):
        ledger = ledger_with(block("dev", "development", "d"))
        colliding = {block_seeds(ledger["blocks"][0])[2]}
        with self.assertRaisesRegex(IdentityError, "collide"):
            prove_fresh(ledger, ROOT, declared=colliding)

    def test_freshness_proof_catches_a_collision_with_recorded_evidence(self):
        round3 = read_strict_json(ROOT / "docs/evidence/aaa_1k_evaluation_round3.json")
        seed = round3["cells"][0]["seed"]
        from research.aaa_1k_loop import identities

        ledger = ledger_with(block("dev", "development", "d"))
        original = identities.block_seeds
        try:
            identities.block_seeds = lambda _b: [seed]  # type: ignore[assignment]
            with self.assertRaisesRegex(IdentityError, "collide"):
                prove_fresh(ledger, ROOT, declared=set())
        finally:
            identities.block_seeds = original  # type: ignore[assignment]

    def test_the_committed_ledger_is_valid_and_fresh(self):
        ledger = load_ledger(ROOT / "benchmarks/aaa1k_loop_identity_ledger.json")
        proof = prove_fresh(ledger, ROOT)
        self.assertEqual(proof["status"], "DISJOINT")
        self.assertFalse(any(b["role"] == "confirmation" for b in ledger["blocks"]))


# ======================================================================
# strict JSON and historical evidence
# ======================================================================
class EvidenceTests(unittest.TestCase):
    def test_non_standard_numbers_are_refused_on_write_and_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.json"
            with self.assertRaises(EvidenceError):
                write_strict_json(path, {"value": float("nan")})
            self.assertFalse(path.exists())
            path.write_text('{"value": Infinity}', encoding="utf-8")
            with self.assertRaises(EvidenceError):
                read_strict_json(path)

    def test_an_observed_artifact_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.json"
            write_strict_json(path, {"a": 1}, overwrite=False)
            with self.assertRaises(EvidenceError):
                write_strict_json(path, {"a": 2}, overwrite=False)
            self.assertEqual(read_strict_json(path), {"a": 1})

    def test_historical_aaa1k_evidence_is_byte_identical_to_the_closure_audit(self):
        audit = (ROOT / "docs/final_audit.md").read_text(encoding="utf-8")
        for name in (
            "docs/evidence/aaa_1k_evaluation_round1_superseded.json",
            "docs/evidence/aaa_1k_evaluation_round2_superseded.json",
            "docs/evidence/aaa_1k_evaluation_round3.json",
            "docs/evidence/aaa_1k_development_selection.json",
            "docs/evidence/aaa_1k_characterization.json",
        ):
            match = re.search(rf"^\| `{re.escape(name)}` \| `([0-9a-f]{{64}})` \|", audit, re.MULTILINE)
            self.assertIsNotNone(match, name)
            assert match is not None
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), match.group(1), name)

    def test_every_loop_artifact_is_strict_json(self):
        paths = sorted((ROOT / "docs/evidence").glob("aaa1k_loop_*/*.json"))
        paths.append(ROOT / "benchmarks/aaa1k_loop_identity_ledger.json")
        self.assertGreater(len(paths), 10)
        for path in paths:
            read_strict_json(path)


# ======================================================================
# champion identity and scope
# ======================================================================
class ChampionTests(unittest.TestCase):
    def test_champion_zero_matches_the_repository(self):
        record = read_strict_json(ROOT / "docs/evidence/aaa1k_loop_0001/champion_0.json")
        self.assertEqual(verify_champion(record, ROOT), [])
        self.assertEqual(record["parameter_count"], 994)
        self.assertEqual(record["state_footprint"]["total_adaptive_state_scalars"], 1414)

    def test_a_stale_fingerprint_or_parameter_count_is_detected(self):
        record = read_strict_json(ROOT / "docs/evidence/aaa1k_loop_0001/champion_0.json")
        stale = {**record, "phase_fingerprint": "0" * 64}
        self.assertTrue(any("phase_fingerprint" in p for p in verify_champion(stale, ROOT)))
        miscounted = {**record, "parameter_count": 1000}
        problems = verify_champion(miscounted, ROOT)
        self.assertTrue(any("parameter" in p for p in problems))

    def test_the_loop_package_is_outside_the_champion_fingerprint(self):
        self.assertFalse(any(name.startswith("research/aaa_1k_loop") for name in phase_files(ROOT)))
        record = read_strict_json(ROOT / "docs/evidence/aaa1k_loop_0001/champion_0.json")
        self.assertEqual(phase_fingerprint(ROOT)["sha256"], record["phase_fingerprint"])

    def test_arm_configurations_are_the_rule_selected_ones(self):
        selection = read_strict_json(ROOT / "docs/evidence/aaa_1k_development_selection.json")
        configs = selection["architecture_configurations"]
        self.assertEqual(CHAMPION_CONFIGURATION, configs["AAA1KGRU"])
        self.assertEqual(UNGATED_CONFIGURATION, configs["VanillaRNNControl"])
        self.assertEqual(STATELESS_CONFIGURATION, configs["StatelessMLPControl"])

    def test_champion_identity_fails_closed_without_a_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)
            (fake / "docs/evidence").mkdir(parents=True)
            with self.assertRaises(PhaseIdentityError):
                build_champion(fake, source_commit="x")


# ======================================================================
# challenger construction, accounting and causality
# ======================================================================
class ChallengerTests(unittest.TestCase):
    def test_every_candidate_differs_from_the_champion_only_in_its_declared_initial_values(self):
        champion = gru(11)
        for model in (gru_with_gate_biases(11, keep_bias=-2.0), gru_with_split_keep_bias(11)):
            self.assertTrue(initial_parameters_equal_except(champion, model, ["b_z"]))
            self.assertFalse(np.array_equal(champion.parameters["b_z"], model.parameters["b_z"]))
            self.assertEqual(model.parameter_count(), 994)
            self.assertEqual(model.state_footprint(), champion.state_footprint())

    def test_arm_names_match_their_agents(self):
        agents = ArmSet(("gru", "cand:keep_bias_-2", "cand:error_bound_8", "rnn28", "mlp"))(5)
        self.assertEqual(
            [agent.name for agent in agents],
            ["gru", "cand:keep_bias_-2", "cand:error_bound_8", "rnn28", "mlp"],
        )
        with self.assertRaises(ValueError):
            ArmSet(("gru", "gru"))
        with self.assertRaises(ValueError):
            ArmSet(("no-such-arm",))

    def test_candidates_receive_only_floats_or_none_and_predict_before_every_reveal(self):
        stream = occlusion_stream(123, steps=60)
        for arm in ("cand:keep_bias_-2", "cand:error_bound_4"):
            agent = ARMS[arm].build(9)
            log: list[str] = []
            original_accept, original_predict = agent.accept_observation, agent.predict

            def accept(value, original=original_accept, log=log):
                self.assertTrue(value is None or isinstance(value, float))
                log.append("reveal")
                original(value)

            def predict(original=original_predict, log=log):
                log.append("predict")
                return original()

            agent.accept_observation = accept  # type: ignore[method-assign]
            agent.predict = predict  # type: ignore[method-assign]
            run_stream(stream, [agent])
            self.assertEqual(log[0], "reveal")
            self.assertEqual(log[1:], ["predict", "reveal"] * (len(stream.steps) - 1))


class BoundedErrorAgentTests(unittest.TestCase):
    def test_bitwise_identical_to_the_champion_while_the_bound_is_inactive(self):
        stream = motion_compat_stream("bouncing", 77, steps=120, change_step=None)
        champion, bounded = ARMS["gru"].build(3), ARMS["cand:error_bound_16"].build(3)
        result = run_stream(stream, [champion, bounded])
        self.assertTrue(
            all(s.predictions["gru"] == s.predictions["cand:error_bound_16"] for s in result.steps)
        )

    def test_the_bound_is_enforced(self):
        agent = BoundedErrorAgent(gru(1), name="b", error_input_bound=0.5)
        run_stream(coarse_speed_stream(5, steps=80), [agent])
        self.assertLessEqual(abs(agent.previous_signed_error), 0.5)

    def test_branching_keeps_the_bound_and_the_hash_covers_it(self):
        agent = ARMS["cand:error_bound_8"].build(4)
        run_stream(coarse_speed_stream(8, steps=40), [agent])
        frozen = agent.branch(name="frozen", update_enabled=False)
        self.assertIsInstance(frozen, BoundedErrorAgent)
        self.assertEqual(frozen.error_input_bound, 8.0)
        self.assertEqual(agent.interaction_state_hash(), frozen.interaction_state_hash())
        other = BoundedErrorAgent(gru(4), name="x", error_input_bound=4.0)
        other.load_state({**agent.state_dict(), "error_input_bound": 4.0, "previous_signed_error": 0.0})
        self.assertNotEqual(other.interaction_state_hash(), agent.interaction_state_hash())

    def test_a_checkpoint_saved_under_another_bound_is_refused(self):
        agent = ARMS["cand:error_bound_8"].build(4)
        state = agent.state_dict()
        with self.assertRaisesRegex(ValueError, "error_input_bound"):
            BoundedErrorAgent(gru(4), name="y", error_input_bound=4.0).load_state(state)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            ARMS["cand:error_bound_8"].build(4).load_state({**state, "previous_signed_error": 9.0})


class InstrumentationTests(unittest.TestCase):
    def _finite_difference(self, model: Any, x: np.ndarray, h0: np.ndarray) -> np.ndarray:
        numeric = np.zeros((h0.size, h0.size))
        for j in range(h0.size):
            for sign in (1.0, -1.0):
                clone = model.clone()
                clone.hidden = h0.copy()
                clone.hidden[j] += sign * 1e-6
                clone.forward(x, record=False)
                numeric[:, j] += sign * clone.hidden / 2e-6
        return numeric

    def test_the_jacobians_match_finite_differences(self):
        from research.aaa_1k.controls import VanillaRNNControl

        rng = np.random.default_rng(0)
        for model, jacobian in (
            (gru_with_gate_biases(3, keep_bias=-1.0, reset_bias=0.7), gru_jacobian),
            (VanillaRNNControl(seed=3), rnn_jacobian),
        ):
            for _ in range(4):
                model.forward(rng.normal(size=3))
            h0, x = model.hidden.copy(), rng.normal(size=3)
            model.forward(x)
            analytic = jacobian(model)
            assert analytic is not None
            self.assertLess(float(np.max(np.abs(self._finite_difference(model, x, h0) - analytic))), 1e-7)

    def test_instrumentation_does_not_change_predictions(self):
        stream = coarse_speed_stream(31, steps=100)
        plain = run_stream(stream, [ARMS["gru"].build(2)])
        instrumented = run_stream(stream, [ARMS["gru"].build(2)], collect_diagnostics=True)
        self.assertEqual(
            [s.predictions["gru"] for s in plain.steps], [s.predictions["gru"] for s in instrumented.steps]
        )


# ======================================================================
# harness
# ======================================================================
class _Exploding:
    name = "exploding"
    update_enabled = False

    def __init__(self) -> None:
        self.steps = 0

    def begin_episode(self) -> None:
        self.steps = 0

    def accept_observation(self, _value: float | None) -> None:
        self.steps += 1

    def predict(self) -> float:
        if self.steps > 5:
            raise FloatingPointError("injected divergence")
        return 0.5


def _factory_with_exploding(seed: int) -> list[Any]:
    return [ARMS["gru"].build(seed), _Exploding()]


def _reduce(_cell: Cell, result: Any, _agents: Any) -> dict[str, Any]:
    return {"mae": {name: float(np.mean(result.errors(name))) for name in result.agent_names}}


class HarnessTests(unittest.TestCase):
    def test_isolated_runs_merge_to_exactly_the_joint_run(self):
        stream = coarse_speed_stream(99, steps=60)
        names = ("gru", "rnn28", "persistence")
        joint = run_stream(stream, list(ArmSet(names)(7)))
        merged = _merge([run_stream(stream, [agent]) for agent in ArmSet(names)(7)])
        for a, b in zip(joint.steps, merged.steps, strict=True):
            self.assertEqual(a.normalized_absolute_errors, b.normalized_absolute_errors)

    def test_a_failing_arm_is_recorded_not_dropped_and_not_fatal(self):
        cell = Cell("c", 0, 5, coarse_speed_stream(3, steps=30))
        record = _run_one((cell, _factory_with_exploding, _reduce, False, True))
        self.assertIn("exploding", record["failures"])
        self.assertIn("gru", record["mae"])
        with self.assertRaises(FloatingPointError):
            _run_one((cell, _factory_with_exploding, _reduce, False, False))

    def test_an_incomplete_crossing_is_refused_rather_than_flattened(self):
        records = [{"init_index": i, "stream_id": s, "mae": {"a": 1.0}} for i in range(2) for s in ("x", "y")]
        self.assertEqual(matrix(records, "a").shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "complete"):
            matrix(records[:-1], "a")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            matrix([*records, records[0]], "a")


# ======================================================================
# decision and independent recomputation
# ======================================================================
class DecisionTests(unittest.TestCase):
    def test_outcome_rule(self):
        self.assertEqual(outcome_of({"a": "PASS", "b": "PASS"}), "PROMOTE")
        self.assertEqual(outcome_of({"a": "PASS", "b": "FAIL"}), "REJECT")
        self.assertEqual(outcome_of({"a": "PASS", "b": "INCONCLUSIVE"}), "INCONCLUSIVE")
        self.assertEqual(outcome_of({"a": "INCONCLUSIVE", "b": "FAIL"}), "REJECT")
        self.assertEqual(outcome_of({}), "INCONCLUSIVE", "absence of evidence is never PROMOTE")
        self.assertEqual(outcome_of({"a": "NOT_VERIFIED"}), "INCONCLUSIVE")

    def test_decide_recomputes_from_primitives(self):
        env = [derive_loop_seed("e", i, iteration_id="t") for i in range(40)]
        inits = [derive_loop_seed("i", i, iteration_id="t") for i in range(3)]
        payload = {"thresholds": THRESHOLDS, "primitives": synthetic_primitives(env, inits)}
        self.assertEqual(decide(payload)["outcome"], "PROMOTE")
        reversed_effect = {
            "thresholds": THRESHOLDS,
            "primitives": synthetic_primitives(env, inits, deficit=-6e-4),
        }
        self.assertEqual(decide(reversed_effect)["outcome"], "REJECT")

    def test_changed_thresholds_are_refused(self):
        env = [derive_loop_seed("e", i, iteration_id="t") for i in range(40)]
        inits = [derive_loop_seed("i", i, iteration_id="t") for i in range(3)]
        loosened = {**THRESHOLDS, "C3_share_supported": 0.1}
        with self.assertRaisesRegex(ValueError, "thresholds"):
            decide({"thresholds": loosened, "primitives": synthetic_primitives(env, inits)})

    def test_missing_or_failed_cells_are_refused_not_treated_as_passing(self):
        env = [derive_loop_seed("e", i, iteration_id="t") for i in range(40)]
        inits = [derive_loop_seed("i", i, iteration_id="t") for i in range(3)]
        primitives = synthetic_primitives(env, inits)
        primitives["plan"][0]["failures"] = {"gru": "FloatingPointError"}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            decide({"thresholds": THRESHOLDS, "primitives": primitives})
        primitives = synthetic_primitives(env, inits)
        del primitives["plan"][3]
        with self.assertRaisesRegex(ValueError, "crossing"):
            decide({"thresholds": THRESHOLDS, "primitives": primitives})

    def test_the_independent_bootstrap_matches_the_declared_estimator(self):
        rng = np.random.default_rng(1)
        first, second = rng.normal(size=(5, 12)), rng.normal(size=(5, 12)) + 0.3
        declared = crossed_paired_difference(first, second, bootstrap_index=77)
        independent = crossed(first, second, index=77, draws=10_000, confidence=0.95)
        self.assertTrue(math.isclose(declared["ci_low"], independent["low"], rel_tol=1e-12))
        self.assertTrue(math.isclose(declared["ci_high"], independent["high"], rel_tol=1e-12))
        self.assertEqual(declared["favours_first"], independent["positive_streams"])


class RecomputationTests(unittest.TestCase):
    def test_an_untampered_confirmation_recomputes(self):
        with tempfile.TemporaryDirectory() as directory:
            confirmation, freeze, ledger = confirmation_fixture(Path(directory))
            result = verify_confirmation(confirmation, freeze, ledger)
            self.assertTrue(result["agrees"], result["problems"])

    def test_a_stored_promote_is_never_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            confirmation, freeze, ledger = confirmation_fixture(Path(directory))
            payload = read_strict_json(confirmation)
            payload["primitives"] = synthetic_primitives(
                [r["stream_seed"] for r in payload["primitives"]["plan"][::3]]
                + [r["stream_seed"] for r in payload["primitives"]["fixed_speed"][::3]],
                sorted({r["init_seed"] for r in payload["primitives"]["plan"]}),
                deficit=-6e-4,
            )
            write_strict_json(confirmation, payload)  # the stored decision still says PROMOTE
            result = verify_confirmation(confirmation, freeze, ledger)
            self.assertFalse(result["agrees"])
            self.assertTrue(any("outcome" in p for p in result["problems"]))

    def test_a_non_confirmation_identity_or_an_unspent_block_is_caught(self):
        with tempfile.TemporaryDirectory() as directory:
            confirmation, freeze, ledger = confirmation_fixture(Path(directory))
            payload = read_strict_json(confirmation)
            payload["primitives"]["plan"][0]["stream_seed"] = derive_seed("evaluation_env", 40_000)
            write_strict_json(confirmation, payload)
            self.assertTrue(
                any(
                    "non-confirmation identity" in p
                    for p in verify_confirmation(confirmation, freeze, ledger)["problems"]
                )
            )
            unspent = copy.deepcopy(ledger)
            for entry in unspent["blocks"]:
                entry["status"] = "reserved"
                entry.pop("observed_by", None)
            self.assertTrue(
                any(
                    "not marked spent" in p
                    for p in verify_confirmation(confirmation, freeze, unspent)["problems"]
                )
            )

    def test_a_confirmation_from_a_different_freeze_is_caught(self):
        with tempfile.TemporaryDirectory() as directory:
            confirmation, freeze, ledger = confirmation_fixture(Path(directory))
            manifest = read_strict_json(freeze)
            manifest["frozen"]["thresholds"] = {**THRESHOLDS, "C1_min_stream_share": 0.1}
            write_strict_json(freeze, manifest)
            problems = verify_confirmation(confirmation, freeze, ledger)["problems"]
            self.assertTrue(any("freeze manifest" in p for p in problems))
            self.assertTrue(any("thresholds" in p for p in problems))


# ======================================================================
# freeze
# ======================================================================
class FreezeTests(unittest.TestCase):
    def _manifest(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        ledger = ledger_with(
            block("t/confirmation/env", "confirmation", "ce", count=3),
            block("t/confirmation/init", "confirmation", "ci", count=2),
        )
        frozen = {"claim": "x", "thresholds": THRESHOLDS}
        manifest = build_freeze(
            ROOT,
            ledger=ledger,
            freshness={"status": "DISJOINT"},
            champion_path="docs/evidence/aaa1k_loop_0001/champion_0.json",
            attack_path="docs/evidence/aaa1k_loop_0003/attack_2.json",
            blocks=("t/confirmation/env", "t/confirmation/init"),
            frozen_content=frozen,
        )
        return manifest, ledger, frozen

    def test_an_unchanged_repository_verifies(self):
        manifest, ledger, frozen = self._manifest()
        self.assertEqual(verify_freeze(manifest, ROOT, ledger=ledger, frozen_content=frozen), [])

    def test_a_challenger_modified_after_the_freeze_is_caught(self):
        manifest, ledger, frozen = self._manifest()
        manifest["confirmation_source_fingerprint"]["files"]["research/aaa_1k_loop/arms.py"] = "0" * 64
        manifest["confirmation_source_fingerprint"]["sha256"] = "0" * 64
        problems = verify_freeze(manifest, ROOT, ledger=ledger, frozen_content=frozen)
        self.assertTrue(any("arms.py" in p for p in problems))

    def test_changed_thresholds_claim_or_identities_are_caught(self):
        manifest, ledger, frozen = self._manifest()
        loosened = {**frozen, "thresholds": {**THRESHOLDS, "C1_min_stream_share": 0.1}}
        self.assertTrue(verify_freeze(manifest, ROOT, ledger=ledger, frozen_content=loosened))
        other = ledger_with(
            block("t/confirmation/env", "confirmation", "different", count=3),
            block("t/confirmation/init", "confirmation", "ci", count=2),
        )
        self.assertTrue(
            any(
                "seeds differ" in p
                for p in verify_freeze(manifest, ROOT, ledger=other, frozen_content=frozen)
            )
        )
        stale = {**manifest, "champion_phase_fingerprint": "0" * 64}
        self.assertTrue(verify_freeze(stale, ROOT, ledger=ledger, frozen_content=frozen))

    def test_an_uncommitted_or_modified_freeze_cannot_authorize_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
            (root / "freeze.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FreezeError, "not tracked"):
                require_committed(root, "freeze.json")
            subprocess.run(["git", "-C", str(root), "add", "freeze.json"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "freeze"], check=True)
            self.assertEqual(len(require_committed(root, "freeze.json")), 40)
            (root / "freeze.json").write_text('{"changed": 1}\n', encoding="utf-8")
            with self.assertRaisesRegex(FreezeError, "differs from HEAD"):
                require_committed(root, "freeze.json")

    def test_the_confirmation_path_imports_only_frozen_sources(self):
        allowed = {
            Path(name).stem for name in CONFIRMATION_SOURCES if name.startswith("research/aaa_1k_loop/")
        }
        for name in CONFIRMATION_SOURCES:
            if not name.startswith("research/aaa_1k_loop/"):
                continue
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level == 1:
                        module = node.module or "__init__"
                        self.assertIn(module.split(".")[0], allowed, f"{name} imports .{module}")
                    elif node.module and node.module.startswith("research.aaa_1k_loop"):
                        self.assertIn(node.module.split(".")[-1], allowed, f"{name} imports {node.module}")

    def test_the_cli_refuses_confirmation_without_a_freeze(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FreezeError):
            cli.main(["confirm3", "--output", str(Path(directory) / "c.json")])

    def test_the_cli_refuses_to_freeze_a_challenger_the_attack_rejected(self):
        ledger_before = (ROOT / "benchmarks/aaa1k_loop_identity_ledger.json").read_bytes()
        self.assertEqual(cli.main(["freeze3"]), 1)
        self.assertEqual((ROOT / "benchmarks/aaa1k_loop_identity_ledger.json").read_bytes(), ledger_before)
        self.assertFalse((ROOT / "docs/evidence/aaa1k_loop_0003/freeze.json").exists())


# ======================================================================
# iteration records and the state model
# ======================================================================
class IterationRecordTests(unittest.TestCase):
    def setUp(self):
        self.record = read_strict_json(RECORD_0001)

    def assertInvalid(self, record: dict[str, Any], pattern: str, root: Path = ROOT) -> None:
        with self.assertRaisesRegex(IterationError, pattern):
            validate_iteration(record, root, decide=decide)

    def test_the_committed_records_are_valid(self):
        for number in ("0001", "0002", "0003"):
            record = read_strict_json(ROOT / f"docs/evidence/aaa1k_loop_{number}/iteration.json")
            result = validate_iteration(record, ROOT, decide=decide)
            self.assertEqual(result["outcome"], "REJECT")

    def test_an_invalid_state_transition_is_refused(self):
        record = copy.deepcopy(self.record)
        record["history"] = [h for h in record["history"] if h["state"] != "DIAGNOSED"]
        self.assertInvalid(record, "illegal transition")
        record = copy.deepcopy(self.record)
        record["history"][0]["state"] = "CLASSIFIED"
        self.assertInvalid(record, "OBSERVED|twice")

    def test_a_claimed_state_without_its_artifacts_is_refused(self):
        record = copy.deepcopy(self.record)
        record["artifacts"] = [a for a in record["artifacts"] if a["role"] != "attack"]
        self.assertInvalid(record, "artifacts \\['attack'\\] are absent")

    def test_a_rewritten_or_missing_artifact_is_refused(self):
        record = copy.deepcopy(self.record)
        record["artifacts"][1]["sha256"] = "0" * 64
        self.assertInvalid(record, "changed")
        record = copy.deepcopy(self.record)
        record["artifacts"][1]["path"] = "docs/evidence/aaa1k_loop_0001/missing.json"
        self.assertInvalid(record, "missing")

    def test_a_rejected_attempt_cannot_be_discarded_or_left_unexplained(self):
        record = copy.deepcopy(self.record)
        record["candidates"] = [c for c in record["candidates"] if c["candidate_id"] != "aaa1k-loop-0001-c1"]
        self.assertInvalid(record, "missing from the record")
        record = copy.deepcopy(self.record)
        record["candidates"][0]["reason"] = ""
        self.assertInvalid(record, "lacks a reason")

    def test_promotion_without_a_decided_confirmation_is_refused(self):
        record = copy.deepcopy(self.record)
        record["outcome"] = "PROMOTE"
        self.assertInvalid(record, "must record outcome REJECT|PROMOTE requires")
        record = copy.deepcopy(self.record)
        record["history"] = [h for h in record["history"] if h["state"] not in ("REJECTED", "PRESERVED")]
        record["outcome"] = "PROMOTE"
        self.assertInvalid(record, "PROMOTE requires a decided confirmation")

    def test_a_frozen_candidate_in_a_rejected_iteration_is_refused(self):
        record = copy.deepcopy(self.record)
        record["candidates"][1]["status"] = "FROZEN"
        self.assertInvalid(record, "cannot be FROZEN")

    def test_a_non_standard_json_artifact_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.json").write_text('{"x": NaN}', encoding="utf-8")
            record = copy.deepcopy(self.record)
            record["artifacts"].append(
                {
                    "path": "bad.json",
                    "role": "note",
                    "sha256": hashlib.sha256((root / "bad.json").read_bytes()).hexdigest(),
                }
            )
            for artifact in record["artifacts"][:-1]:
                target = root / artifact["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / artifact["path"]).read_bytes())
            self.assertInvalid(record, "non-standard", root=root)

    def _decided_fixture(self, root: Path, *, outcome: str) -> dict[str, Any]:
        confirmation, _freeze, _ledger = confirmation_fixture(root)
        for name, content in {
            "champion.json": {"a": 1},
            "observation.json": {"a": 1},
            "diagnosis.json": {"a": 1},
            "development.json": {"candidates_declared": [{"candidate_id": "c"}]},
            "attack.json": {"challenger_id": "c"},
        }.items():
            write_strict_json(root / name, content)
        artifacts = [
            {"path": name, "role": role, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
            for name, role in (
                ("champion.json", "champion"),
                ("observation.json", "observation"),
                ("diagnosis.json", "diagnosis"),
                ("development.json", "development"),
                ("attack.json", "attack"),
                ("freeze.json", "freeze"),
                ("confirmation.json", "confirmation"),
            )
        ]
        states = [
            "OBSERVED",
            "CLASSIFIED",
            "DIAGNOSED",
            "TEST_DEFINED",
            "CHALLENGER_CREATED",
            "CHALLENGER_ATTACKED",
            "FROZEN",
            "CONFIRMED",
            "DECIDED",
            "PRESERVED",
        ]
        self.assertTrue(confirmation.exists())
        return {
            "schema": "aaa.loop.iteration.v1",
            "iteration_id": "test",
            "parent_champion": {},
            "problem": "p",
            "classification": {},
            "hypotheses": {},
            "candidates": [{"candidate_id": "c", "status": "FROZEN"}],
            "history": [{"state": s, "at": "2026-01-01T00:00:00Z"} for s in states],
            "artifacts": artifacts,
            "outcome": outcome,
            "accounting": {"parameters_after": 994, "parameters_counted_from_arrays": 994},
        }

    def test_a_decided_outcome_must_equal_the_recomputed_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._decided_fixture(root, outcome="PROMOTE")
            self.assertEqual(validate_iteration(record, root, decide=decide)["recomputed_outcome"], "PROMOTE")
            self.assertInvalid({**record, "outcome": "INCONCLUSIVE"}, "recompute to PROMOTE", root=root)
            with self.assertRaisesRegex(IterationError, "decision recomputation"):
                validate_iteration(record, root, decide=None)

    def test_promotion_with_a_changed_parameter_count_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._decided_fixture(root, outcome="PROMOTE")
            record["accounting"] = {"parameters_after": 994, "parameters_counted_from_arrays": 1042}
            self.assertInvalid(record, "parameter accounting", root=root)

    def test_promotion_with_a_failed_or_insufficient_criterion_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._decided_fixture(root, outcome="PROMOTE")
            payload = read_strict_json(root / "confirmation.json")
            for entry in payload["primitives"]["fixed_speed"]:
                entry["mae"]["rnn28"] = entry["mae"]["mlp"] + 1e-4  # memory no longer helps: C3 fails
            write_strict_json(root / "confirmation.json", payload)
            for artifact in record["artifacts"]:
                artifact["sha256"] = hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest()
            self.assertInvalid(record, "recompute to REJECT", root=root)


if __name__ == "__main__":
    unittest.main()


# ======================================================================
# reproduction check
# ======================================================================
class ReproductionTests(unittest.TestCase):
    def test_compare_evidence_requires_every_committed_value(self):
        committed = {"a": 1.0, "git": {"commit": "x"}, "compute_seconds": 3.0, "rows": [{"b": 2}]}
        self.assertEqual(
            cli.compare_evidence(committed, {**committed, "git": {}, "compute_seconds": 9.0}), ([], [])
        )
        mismatches, added = cli.compare_evidence(committed, {**committed, "a": 1.0000001, "new": 5})
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(added, ["$.new"])
        self.assertTrue(cli.compare_evidence(committed, {"a": 1.0, "rows": []})[0])

    def test_the_first_claim_attack_is_labelled_with_the_claim_it_judged(self):
        """Regression for AAA-164: the command must not stamp the current claim id."""

        from research.aaa_1k_loop import iteration3

        committed = read_strict_json(ROOT / "docs/evidence/aaa1k_loop_0003/attack.json")
        original = iteration3.run_attack
        try:
            iteration3.run_attack = lambda _ledger, workers=None: committed["records"]  # type: ignore[assignment]
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "rerun.json"
                self.assertEqual(cli.main(["attack3", "--output", str(output)]), 0)
                mismatches, _ = cli.compare_evidence(committed, read_strict_json(output))
        finally:
            iteration3.run_attack = original  # type: ignore[assignment]
        self.assertEqual(mismatches, [])
        self.assertEqual(committed["claim_id"], "aaa1k-claim-q4-coarse-v2")
