"""Deterministic synthetic PASS / FAIL / NOT_VERIFIED / INSUFFICIENT cases.

Every required gate is exercised in all of its reachable outcomes with
hand-built evidence, so a gate cannot quietly become unfalsifiable.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

from aaa.benchmark.gates import (
    FAIL,
    INSUFFICIENT_EVIDENCE,
    NOT_VERIFIED,
    PASS,
    GateContext,
    evaluate_gates,
)
from aaa.benchmark.families import (
    CHANGED_LAW_PREDICTOR_NAMES,
    FamilyCollector,
    ONLINE_PREDICTOR_NAMES,
)
from aaa.benchmark.spec import load_spec
from aaa.metrics import RecoveryConfig

from .helpers import identity, record

# Gate *logic* is what these cases exercise, so the bootstrap runs with a
# small draw count. The canonical draw count is asserted in test_spec.py.
SPEC = replace(load_spec(), statistics=replace(load_spec().statistics, draws=300))
RECOVERY = RecoveryConfig(
    pre_event_reference_length=5,
    post_event_horizon=10,
    shock_window=3,
    shock_multiplier=3.0,
    shock_floor=1e-4,
    tolerance_multiplier=1.5,
    tolerance_floor=1e-5,
    rolling_window=3,
    sustain_windows=2,
)


def _episode(
    family: str,
    branch: str,
    replica: int,
    episode: int,
    stratum: str,
    errors: dict[str, float],
    *,
    steps: int = 12,
    bounce_steps: tuple[int, ...] = (),
    change_step: int | None = None,
):
    ident = identity(
        trial_id=f"{family}:{branch}:r{replica}:e{episode}",
        family=family,
        branch=branch,
        replica_id=replica,
        episode=episode,
        stratum=stratum,
    )
    return [
        record(
            step,
            0.0,
            ident=ident,
            predictors=tuple(errors),
            errors=errors,
            bounced=step in bounce_steps,
            walls=(("upper",) if step % 2 == 0 else ("lower",)) if step in bounce_steps else (),
            changed=change_step is not None and step == change_step,
        )
        for step in range(steps)
    ]


def motion_collector(
    family: str,
    errors: dict[str, float],
    *,
    strata: list[str] | None = None,
    replicas: int = 5,
    episodes: int = 4,
    bounce_steps: tuple[int, ...] = (),
    change_step: int | None = None,
) -> FamilyCollector:
    collector = FamilyCollector(family, list(errors), recovery=RECOVERY)
    all_strata = strata if strata is not None else list(SPEC.required_strata())
    if not all_strata:
        all_strata = ["unstratified"]
    for replica in range(replicas):
        for episode in range(episodes * max(1, len(all_strata))):
            stratum = all_strata[episode % len(all_strata)]
            collector.add(
                _episode(
                    family,
                    "main",
                    replica,
                    episode,
                    stratum,
                    errors,
                    bounce_steps=bounce_steps,
                    change_step=change_step,
                ),
                post_change_window=RECOVERY.post_event_horizon,
            )
    return collector


def changed_law_collector(
    errors: dict[str, float],
    *,
    recovered: bool,
    replicas: int = 5,
    episodes: int = 30,
    shock: float = 0.2,
):
    collector = FamilyCollector("changed_law", list(errors), recovery=RECOVERY, branch="changed-law")
    for replica in range(replicas):
        for episode in range(episodes):
            ident = identity(
                trial_id=f"changed_law:changed-law:r{replica}:e{episode}",
                family="changed_law",
                branch="changed-law",
                replica_id=replica,
                episode=episode,
                stratum="changed-law",
            )
            records = []
            for step in range(16):
                if step == 0:
                    # The unavoidable first surprise is shared by every arm.
                    step_errors = {name: shock for name in errors}
                else:
                    step_errors = (
                        dict(errors) if recovered else {name: 0.5 for name in errors}
                    )
                records.append(
                    record(step, 0.0, ident=ident, predictors=tuple(errors), errors=step_errors, changed=step == 0)
                )
            collector.add(
                records,
                pre_event_errors={name: [0.001] * 10 for name in errors},
                post_change_window=RECOVERY.post_event_horizon,
            )
    return collector


def build_context(
    *,
    collectors: dict[str, FamilyCollector],
    learning_curve: dict[str, Any] | None = None,
    correctness: dict[str, Any] | None = None,
    reproducibility: dict[str, Any] | None = None,
    latency: dict[str, Any] | None = None,
) -> GateContext:
    results = {key: collector.finish() for key, collector in collectors.items()}
    return GateContext(
        spec=SPEC,
        collectors=collectors,
        results=results,
        learning_curve=learning_curve if learning_curve is not None else good_learning_curve(),
        correctness=correctness if correctness is not None else {"executed": True, "checks": {"ok": True}},
        reproducibility=(
            reproducibility if reproducibility is not None else {"executed": True, "checks": {"ok": True}}
        ),
        latency=latency if latency is not None else {"predict_plus_update_p95_ms": 0.05, "failures": 0},
    )


def good_learning_curve() -> dict[str, Any]:
    budgets = list(SPEC.training.learning_probe_budgets)
    curve = {}
    for index, budget in enumerate(budgets):
        value = 1.0 * (10.0 ** (-3 * index))
        curve[str(budget)] = {
            "budget": budget,
            "replica_episode_values": {str(replica): [value] * 4 for replica in range(5)},
            "mean": value,
        }
    return {"probe_episodes": 4, "budgets": budgets, "curve": curve}


def flat_learning_curve() -> dict[str, Any]:
    budgets = list(SPEC.training.learning_probe_budgets)
    curve = {
        str(budget): {
            "budget": budget,
            "replica_episode_values": {str(replica): [1.0] * 4 for replica in range(5)},
            "mean": 1.0,
        }
        for budget in budgets
    }
    return {"probe_episodes": 4, "budgets": budgets, "curve": curve}


MOTION_GOOD = {name: 1e-7 for name in ONLINE_PREDICTOR_NAMES}
MOTION_GOOD["constant_motion_reflected"] = 1e-7
CHANGED_GOOD = {
    "persistence": 0.05,
    "constant_motion": 0.05,
    "constant_motion_reflected": 0.05,
    "frozen": 0.05,
    "online": 0.001,
}


def full_collectors(**overrides) -> dict[str, FamilyCollector]:
    collectors = {
        "constant_velocity": motion_collector("constant_velocity", dict(MOTION_GOOD)),
        "bouncing": motion_collector("bouncing", dict(MOTION_GOOD), bounce_steps=(1, 4, 9)),
        "speed_change": motion_collector(
            "speed_change", dict(MOTION_GOOD), strata=["unstratified"], change_step=4
        ),
        "speed_extrapolation": motion_collector(
            "speed_extrapolation", dict(MOTION_GOOD), strata=["unstratified"]
        ),
        "always_online": motion_collector("always_online", dict(MOTION_GOOD), strata=["unstratified"]),
        "changed_law:changed": changed_law_collector(dict(CHANGED_GOOD), recovered=True),
        "changed_law:unchanged": changed_law_collector(
            {**CHANGED_GOOD, "online": 0.05}, recovered=True
        ),
    }
    collectors.update(overrides)
    return collectors


def statuses(context: GateContext) -> dict[str, str]:
    return {gate["name"]: gate["status"] for gate in evaluate_gates(context)["gates"]}


class HappyPathTests(unittest.TestCase):
    def test_complete_evidence_passes_every_required_gate(self):
        result = evaluate_gates(build_context(collectors=full_collectors()))
        self.assertEqual(result["unmet_required_gates"], [])
        self.assertTrue(result["all_required_gates_pass"])

    def test_every_declared_gate_is_evaluated(self):
        result = evaluate_gates(build_context(collectors=full_collectors()))
        self.assertEqual({gate["name"] for gate in result["gates"]}, {gate.name for gate in SPEC.gates})


class CoverageTests(unittest.TestCase):
    def test_missing_one_stratum_is_insufficient_evidence_not_pass(self):
        required = list(SPEC.required_strata())
        collectors = full_collectors(
            constant_velocity=motion_collector("constant_velocity", dict(MOTION_GOOD), strata=required[:-1])
        )
        result = statuses(build_context(collectors=collectors))
        self.assertEqual(result["stratum_coverage"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(result["constant_velocity_identification"], INSUFFICIENT_EVIDENCE)

    def test_all_strata_removed_is_insufficient_evidence_not_pass(self):
        collectors = full_collectors(
            constant_velocity=motion_collector("constant_velocity", dict(MOTION_GOOD), strata=["unstratified"])
        )
        result = statuses(build_context(collectors=collectors))
        self.assertEqual(result["stratum_coverage"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(result["constant_velocity_identification"], INSUFFICIENT_EVIDENCE)

    def test_a_thin_stratum_is_insufficient_evidence(self):
        collectors = full_collectors(
            constant_velocity=motion_collector("constant_velocity", dict(MOTION_GOOD), episodes=1)
        )
        result = statuses(build_context(collectors=collectors))
        self.assertEqual(result["stratum_coverage"], INSUFFICIENT_EVIDENCE)

    def test_coverage_concentrated_in_one_replica_is_insufficient_evidence(self):
        collectors = full_collectors(
            constant_velocity=motion_collector("constant_velocity", dict(MOTION_GOOD), replicas=1, episodes=20)
        )
        result = statuses(build_context(collectors=collectors))
        self.assertEqual(result["stratum_coverage"], INSUFFICIENT_EVIDENCE)

    def test_missing_a_wall_is_insufficient_evidence(self):
        collector = motion_collector("bouncing", dict(MOTION_GOOD), bounce_steps=(1, 5))
        collector.wall_counts["lower"] = 0
        result = statuses(build_context(collectors=full_collectors(bouncing=collector)))
        self.assertEqual(result["stratum_coverage"], INSUFFICIENT_EVIDENCE)

    def test_too_few_bounce_events_is_insufficient_evidence(self):
        collectors = full_collectors(
            bouncing=motion_collector("bouncing", dict(MOTION_GOOD), bounce_steps=(1,), episodes=1)
        )
        result = statuses(build_context(collectors=collectors))
        self.assertEqual(result["stratum_coverage"], INSUFFICIENT_EVIDENCE)

    def test_missing_family_is_not_verified(self):
        collectors = full_collectors()
        del collectors["constant_velocity"]
        result = statuses(build_context(collectors=collectors))
        self.assertEqual(result["stratum_coverage"], NOT_VERIFIED)
        self.assertEqual(result["constant_velocity_identification"], NOT_VERIFIED)


class LearningGateTests(unittest.TestCase):
    def test_a_flat_curve_fails(self):
        context = build_context(collectors=full_collectors(), learning_curve=flat_learning_curve())
        self.assertEqual(statuses(context)["learning_progress"], FAIL)

    def test_a_missing_curve_is_not_verified(self):
        context = build_context(collectors=full_collectors(), learning_curve={})
        self.assertEqual(statuses(context)["learning_progress"], NOT_VERIFIED)

    def test_a_parameterless_rule_cannot_satisfy_the_learning_gate(self):
        # A rule with no parameters produces the same probe error at every
        # training budget: exactly the flat curve above.
        context = build_context(collectors=full_collectors(), learning_curve=flat_learning_curve())
        gate = next(g for g in evaluate_gates(context)["gates"] if g["name"] == "learning_progress")
        self.assertEqual(gate["status"], FAIL)
        self.assertAlmostEqual(gate["observed"], 0.0)

    def test_a_non_monotone_curve_fails_when_monotonicity_is_required(self):
        curve = good_learning_curve()
        budgets = curve["budgets"]
        curve["curve"][str(budgets[2])]["mean"] = 10.0
        context = build_context(collectors=full_collectors(), learning_curve=curve)
        self.assertEqual(statuses(context)["learning_progress"], FAIL)


class ComparisonGateTests(unittest.TestCase):
    def test_bounce_parity_fails_when_the_candidate_is_worse_than_the_fair_baseline(self):
        errors = dict(MOTION_GOOD)
        errors["candidate_frozen"] = 1e-3
        collectors = full_collectors(
            bouncing=motion_collector("bouncing", errors, bounce_steps=(1, 4, 9))
        )
        self.assertEqual(statuses(build_context(collectors=collectors))["bounce_event_accuracy"], FAIL)

    def test_bounce_gate_is_not_verified_without_event_episodes(self):
        collectors = full_collectors(
            bouncing=motion_collector("bouncing", dict(MOTION_GOOD), bounce_steps=())
        )
        result = statuses(build_context(collectors=collectors))
        self.assertEqual(result["bounce_event_accuracy"], NOT_VERIFIED)

    def test_changed_law_fails_below_the_declared_margin(self):
        # frozen cumulative over the 10-transition window is 0.2 + 9 * 0.05.
        # An online error of 0.036 lands just inside 20 percent improvement,
        # so the declared margin must reject it.
        errors = {**CHANGED_GOOD, "online": 0.036}
        collectors = full_collectors(
            **{"changed_law:changed": changed_law_collector(errors, recovered=True)}
        )
        self.assertEqual(statuses(build_context(collectors=collectors))["changed_law_adaptation"], FAIL)

    def test_changed_law_is_not_verified_without_the_family(self):
        collectors = full_collectors()
        del collectors["changed_law:changed"]
        self.assertEqual(statuses(build_context(collectors=collectors))["changed_law_adaptation"], NOT_VERIFIED)

    def test_unchanged_control_fails_when_updating_degrades_the_model(self):
        collectors = full_collectors(
            **{"changed_law:unchanged": changed_law_collector({**CHANGED_GOOD, "online": 1.0}, recovered=True)}
        )
        self.assertEqual(statuses(build_context(collectors=collectors))["unchanged_control"], FAIL)

    def test_always_online_non_regression_fails_when_updating_hurts(self):
        errors = dict(MOTION_GOOD)
        errors["candidate_online"] = 1e-2
        collectors = full_collectors(
            always_online=motion_collector("always_online", errors, strata=["unstratified"])
        )
        self.assertEqual(statuses(build_context(collectors=collectors))["always_online_stability"], FAIL)

    def test_absolute_accuracy_gate_fails_outside_the_declared_limit(self):
        errors = dict(MOTION_GOOD)
        errors["candidate_frozen"] = 1e-2
        collectors = full_collectors(bouncing=motion_collector("bouncing", errors, bounce_steps=(1, 4, 9)))
        self.assertEqual(statuses(build_context(collectors=collectors))["frozen_prediction_accuracy"], FAIL)


class RecoveryGateTests(unittest.TestCase):
    def test_too_few_eligible_events_is_insufficient_evidence(self):
        collectors = full_collectors(
            **{"changed_law:changed": changed_law_collector(dict(CHANGED_GOOD), recovered=True, episodes=2)}
        )
        self.assertEqual(statuses(build_context(collectors=collectors))["recovery"], INSUFFICIENT_EVIDENCE)

    def test_unrecovered_events_fail_rather_than_disappear(self):
        collectors = full_collectors(
            **{"changed_law:changed": changed_law_collector(dict(CHANGED_GOOD), recovered=False)}
        )
        context = build_context(collectors=collectors)
        gate = next(g for g in evaluate_gates(context)["gates"] if g["name"] == "recovery")
        self.assertEqual(gate["status"], FAIL)
        self.assertGreater(gate["details"]["unrecovered"], 0)
        self.assertEqual(
            gate["details"]["eligible"], gate["details"]["recovered"] + gate["details"]["unrecovered"]
        )

    def test_recovery_reports_per_replica_rates(self):
        context = build_context(collectors=full_collectors())
        gate = next(g for g in evaluate_gates(context)["gates"] if g["name"] == "recovery")
        self.assertEqual(len(gate["details"]["per_replica_recovery_rate"]), 5)


class VerificationGateTests(unittest.TestCase):
    def test_correctness_not_executed_is_not_verified(self):
        context = build_context(collectors=full_collectors(), correctness={"executed": False, "checks": {}})
        self.assertEqual(statuses(context)["correctness"], NOT_VERIFIED)

    def test_correctness_with_a_failed_check_fails(self):
        context = build_context(
            collectors=full_collectors(), correctness={"executed": True, "checks": {"a": True, "b": False}}
        )
        self.assertEqual(statuses(context)["correctness"], FAIL)

    def test_reproducibility_without_evidence_is_not_verified(self):
        context = build_context(collectors=full_collectors(), reproducibility={"executed": False, "checks": {}})
        self.assertEqual(statuses(context)["reproducibility"], NOT_VERIFIED)

    def test_latency_over_the_limit_fails(self):
        context = build_context(
            collectors=full_collectors(), latency={"predict_plus_update_p95_ms": 50.0, "failures": 0}
        )
        self.assertEqual(statuses(context)["cpu_usability"], FAIL)

    def test_latency_failures_fail_the_gate(self):
        context = build_context(
            collectors=full_collectors(), latency={"predict_plus_update_p95_ms": 0.01, "failures": 3}
        )
        self.assertEqual(statuses(context)["cpu_usability"], FAIL)

    def test_missing_latency_is_not_verified(self):
        context = build_context(collectors=full_collectors(), latency={})
        self.assertEqual(statuses(context)["cpu_usability"], NOT_VERIFIED)


class RequiredGateSemanticsTests(unittest.TestCase):
    def test_only_pass_satisfies_a_required_gate(self):
        for report, expected in (
            ({"executed": False, "checks": {}}, NOT_VERIFIED),
            ({"executed": True, "checks": {"a": False}}, FAIL),
        ):
            with self.subTest(expected=expected):
                context = build_context(collectors=full_collectors(), correctness=report)
                result = evaluate_gates(context)
                self.assertIn("correctness", result["unmet_required_gates"])
                self.assertFalse(result["all_required_gates_pass"])

    def test_a_failing_optional_gate_does_not_block(self):
        errors = dict(MOTION_GOOD)
        errors["candidate_frozen"] = 1.0
        collectors = full_collectors(
            speed_extrapolation=motion_collector("speed_extrapolation", errors, strata=["unstratified"])
        )
        result = evaluate_gates(build_context(collectors=collectors))
        statuses_by_name = {gate["name"]: gate["status"] for gate in result["gates"]}
        self.assertEqual(statuses_by_name["speed_extrapolation_report"], FAIL)
        self.assertTrue(result["all_required_gates_pass"])


if __name__ == "__main__":
    unittest.main()
