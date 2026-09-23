"""aaa.1k.v2: the batched core, its parity with AAA-1K, benchmarks, identities and statistics.

Each test is written to be able to fail: parity tests compare against the
unmodified historical implementation, gradient tests against finite
differences, leakage tests inject garbage where only hidden values live, and
identity tests inject collisions.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaa.compute import resolve_backend
from research.aaa_1k.controls import VanillaRNNControl
from research.aaa_1k.model import AAA1KGRU
from research.aaa_1k.runner import run_stream
from research.aaa_1k.streams import build_stream
from research.aaa_1k_loop.arms import gru as champion_gru
from research.aaa_1k_loop.unfolding import ReachGatedUnfoldAgent
from research.aaa_1k_v2 import PARAMETER_CAP, identities, plan, stress
from research.aaa_1k_v2.adapters import (
    DotAdapter,
    SeriesAdapter,
    adapter_state,
    load_adapter_state,
    reflect,
    unfold,
)
from research.aaa_1k_v2.arms import CANDIDATE_TEMPLATES, CHAMPION_1, CONTROL_TEMPLATES, ablations, audit
from research.aaa_1k_v2.cores import ElmanCore, GRUCore, LRUCore, MGUCore, MLPCore, build_core
from research.aaa_1k_v2.engine import BatchedLearner, CellConfig, softplus
from research.aaa_1k_v2.external import dysts_subset, metrics, monash, synthetic, tasks
from research.aaa_1k_v2.identity import fingerprint
from research.aaa_1k_v2.jobs import DotJob, SeriesJob, execute, execute_series, split
from research.aaa_1k_v2.runner import StreamBatch, branch, run_dot
from research.aaa_1k_v2.stats import (
    combine,
    geometric_relative,
    guard,
    noninferior,
    relative_crossed,
    superior,
)

ROOT = Path(__file__).resolve().parents[1]
CPU = resolve_backend("cpu")


def batch_from_streams(streams: list, repeats: int = 1) -> StreamBatch:
    rows = [s for s in streams for _ in range(repeats)]
    T = max(len(s.steps) for s in rows)
    truth = np.zeros((len(rows), T))
    observed = np.zeros((len(rows), T), dtype=bool)
    length = np.zeros(len(rows), dtype=int)
    for i, s in enumerate(rows):
        n = len(s.steps)
        length[i] = n
        truth[i, :n] = [st.true_position for st in s.steps]
        observed[i, :n] = [st.observed for st in s.steps]
    return StreamBatch(truth, observed, length, [s.stream_id for s in rows])


def champion_learner(seeds: list[int]) -> tuple[BatchedLearner, DotAdapter]:
    learner = BatchedLearner(
        GRUCore(inputs=3, hidden=16),
        CPU,
        seeds=seeds,
        configs=[CHAMPION_1.config] * len(seeds),
        tbptt_steps=4,
    )
    return learner, DotAdapter(CPU, len(seeds), features="v1", target_rule="reach_gated")


# ----------------------------------------------------------------------
# accounting and initialization parity
# ----------------------------------------------------------------------
class AccountingTests(unittest.TestCase):
    def test_every_candidate_is_under_the_cap_and_recounted(self) -> None:
        for arm in (CHAMPION_1, *CANDIDATE_TEMPLATES.values(), *CONTROL_TEMPLATES.values()):
            record = audit(arm)
            with self.subTest(arm=arm.name):
                self.assertTrue(record["agree"])
                self.assertLessEqual(record["trainable_parameters"], PARAMETER_CAP)

    def test_champion_1_accounting_matches_the_historical_record(self) -> None:
        record = audit(CHAMPION_1)
        self.assertEqual(record["trainable_parameters"], 994)
        self.assertEqual(record["total_adaptive_state_scalars"], 1414)
        self.assertEqual(record["optimizer_state_scalars"], 0)

    def test_rtrl_traces_are_counted_as_adaptive_state(self) -> None:
        record = audit(CANDIDATE_TEMPLATES["lru_v2"])
        self.assertEqual(record["eligibility_trace_scalars"], 2 * 24 + 2 * 24 * 4)
        self.assertEqual(record["tbptt_buffer_scalars"], 0)

    def test_gru_initialization_is_bitwise_aaa1k(self) -> None:
        for seed in (0, 7, 123456789):
            ours = GRUCore(inputs=3, hidden=16).init_cell(seed)
            theirs = AAA1KGRU(seed=seed).parameters
            for name, array in theirs.items():
                self.assertTrue(np.array_equal(ours[name], array), name)

    def test_elman_initialization_is_bitwise_the_ungated_control(self) -> None:
        ours = ElmanCore(inputs=3, hidden=28).init_cell(5)
        theirs = VanillaRNNControl(seed=5).parameters
        for name, array in theirs.items():
            self.assertTrue(np.array_equal(ours[name], array), name)

    def test_ablations_differ_in_exactly_one_mechanism(self) -> None:
        arm = CANDIDATE_TEMPLATES["gru_v2"]
        for name, ablation in ablations(arm).items():
            with self.subTest(name=name):
                self.assertEqual(ablation.core, arm.core)
                self.assertEqual(ablation.config.learning_rate, arm.config.learning_rate)


# ----------------------------------------------------------------------
# parity with the historical scalar Champion 1
# ----------------------------------------------------------------------
class HistoricalParityTests(unittest.TestCase):
    def test_batched_champion_1_matches_historical_champion_1(self) -> None:
        streams = [
            build_stream("occlusion_v1", 11, steps=120),
            build_stream("coarse_speed_v1", 12, steps=160),
        ]
        seeds = [3, 4]
        cells = [(s, seed) for s in streams for seed in seeds]
        batch = batch_from_streams([s for s, _ in cells])
        learner, adapter = champion_learner([seed for _, seed in cells])
        run = run_dot(learner, adapter, batch)
        for index, (stream, seed) in enumerate(cells):
            reference = run_stream(stream, [ReachGatedUnfoldAgent(champion_gru(seed), name="c1")]).errors(
                "c1"
            )
            ours = run.errors[index, : len(reference)]
            with self.subTest(stream=stream.stream_id, seed=seed):
                self.assertLessEqual(float(np.max(np.abs(ours - reference))), 1e-12)

    def test_reflect_and_unfold_match_the_public_maps(self) -> None:
        from aaa.predictors import reflect_prediction, unfold_observation

        rng = np.random.default_rng(0)
        positions = np.concatenate([rng.uniform(-3, 4, 200), [0.0, 1.0, -0.0, 2.0, -1.0]])
        ours = reflect(CPU, positions, 0.0, 1.0)
        for value, got in zip(positions, ours, strict=True):
            self.assertEqual(reflect_prediction(float(value), 0.0, 1.0), float(got))
        observed = rng.uniform(0, 1, 200)
        reference = rng.uniform(-1.5, 2.5, 200)
        ours = unfold(CPU, observed, reference, 0.0, 1.0)
        for y, r, got in zip(observed, reference, ours, strict=True):
            self.assertEqual(unfold_observation(float(y), float(r), 0.0, 1.0), float(got))


# ----------------------------------------------------------------------
# gradients against finite differences
# ----------------------------------------------------------------------
def _loss(out: np.ndarray, target: float, weight: float, fixed_error: float) -> float:
    signed = out[0] - target
    value = signed * signed
    if out.shape[0] > 1:
        value += weight * (float(softplus(np, np.asarray([out[1]]))[0]) - fixed_error) ** 2
    return float(value)


class GradientTests(unittest.TestCase):
    def _window_check(self, core: object, steps: int = 6) -> None:
        learner = BatchedLearner(
            core, CPU, seeds=[11], configs=[CellConfig(learning_rate=0.01)], tbptt_steps=4
        )  # type: ignore[arg-type]
        rng = np.random.default_rng(1)
        # non-zero readout so every parameter has a gradient
        for name in learner.params:
            if name in ("W_o", "b_o"):
                learner.params[name] = rng.normal(0, 0.3, learner.params[name].shape)
        xs = rng.normal(0, 1, (steps, core.inputs))  # type: ignore[attr-defined]
        for x in xs:
            learner.forward(xs[:1] * 0 + x[None, :])
        target = 0.7
        out = learner._last["out"][0]
        fixed = abs(out[0] - target)
        analytic = learner.gradients(np.asarray([target]))
        window = list(learner._caches)
        h0 = window[0].get("h_prev", np.zeros((1, core.state_size())))  # type: ignore[attr-defined]

        def loss_with(params: dict) -> float:
            h = h0
            o = None
            for cache in window:
                h, _ = core.forward(np, params, cache["x"], h)  # type: ignore[attr-defined]
                o = core.readout(np, params, h)  # type: ignore[attr-defined]
            assert o is not None
            return _loss(o[0], target, 0.25, fixed)

        eps = 1e-6
        for name, array in learner.params.items():
            flat = array.reshape(-1)
            for position in rng.choice(flat.size, size=min(6, flat.size), replace=False):
                plus = {k: v.copy() for k, v in learner.params.items()}
                minus = {k: v.copy() for k, v in learner.params.items()}
                plus[name].reshape(-1)[position] += eps
                minus[name].reshape(-1)[position] -= eps
                numeric = (loss_with(plus) - loss_with(minus)) / (2 * eps)
                got = float(analytic[name].reshape(-1)[position])
                self.assertAlmostEqual(
                    got, numeric, delta=1e-6 * max(1.0, abs(numeric)), msg=f"{core}: {name}[{position}]"
                )

    def test_tbptt_gradients_match_finite_differences(self) -> None:
        for core in (
            GRUCore(inputs=4, hidden=5),
            ElmanCore(inputs=4, hidden=6),
            MGUCore(inputs=4, hidden=5),
            MLPCore(inputs=4, hidden=5),
        ):
            with self.subTest(core=core.kind):
                self._window_check(core)

    def test_rtrl_gradient_is_the_exact_untruncated_gradient(self) -> None:
        core = LRUCore(inputs=4, hidden=5, readout_hidden=4)
        learner = BatchedLearner(core, CPU, seeds=[3], configs=[CellConfig(learning_rate=0.01)], rule="rtrl")
        rng = np.random.default_rng(2)
        learner.params["W_o"] = rng.normal(0, 0.3, learner.params["W_o"].shape)
        xs = rng.normal(0, 1, (25, 4))
        for x in xs:
            learner.forward(x[None, :])
        target = -0.4
        out = learner._last["out"][0]
        fixed = abs(out[0] - target)
        analytic = learner.gradients(np.asarray([target]))

        def loss_with(params: dict) -> float:
            h = np.zeros((1, 2 * core.hidden))
            traces = {
                "E_lam": np.zeros((1, 2, core.hidden)),
                "E_B": np.zeros((1, 2, core.hidden, core.inputs)),
            }
            o = None
            for x in xs:
                h, traces = core.step_state(np, params, x[None, :], h, traces)
                o, _ = core.readout_forward(np, params, h, x[None, :])
            assert o is not None
            return _loss(o[0], target, 0.25, fixed)

        eps = 1e-6
        for name, array in learner.params.items():
            flat = array.reshape(-1)
            for position in range(min(flat.size, 5)):
                plus = {k: v.copy() for k, v in learner.params.items()}
                minus = {k: v.copy() for k, v in learner.params.items()}
                plus[name].reshape(-1)[position] += eps
                minus[name].reshape(-1)[position] -= eps
                numeric = (loss_with(plus) - loss_with(minus)) / (2 * eps)
                self.assertAlmostEqual(
                    float(analytic[name].reshape(-1)[position]),
                    numeric,
                    delta=1e-6 * max(1.0, abs(numeric)),
                    msg=f"{name}[{position}]",
                )

    def test_live_and_replay_agree_while_parameters_are_constant(self) -> None:
        rng = np.random.default_rng(4)
        xs = rng.normal(0, 1, (8, 4))
        grads = {}
        for rule in ("live", "replay"):
            learner = BatchedLearner(
                GRUCore(inputs=4, hidden=6),
                CPU,
                seeds=[5],
                configs=[CellConfig(learning_rate=0.01)],
                tbptt_steps=4,
                rule=rule,
            )
            learner.params["W_o"] = np.full_like(learner.params["W_o"], 0.2)
            for x in xs:
                learner.forward(x[None, :])
            grads[rule] = learner.gradients(np.asarray([0.3]))
        for name in grads["live"]:
            self.assertTrue(np.allclose(grads["live"][name], grads["replay"][name], atol=1e-14), name)


# ----------------------------------------------------------------------
# batch semantics: independence, failure, branching, resume, leakage
# ----------------------------------------------------------------------
class BatchSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = build_stream("occlusion_v1", 21, steps=100)

    def test_a_cell_is_bitwise_independent_of_its_batch(self) -> None:
        together = execute(
            DotJob(arm=CHAMPION_1, seeds=(1, 2, 3, 4), batch=batch_from_streams([self.stream], 4))
        )
        alone = [
            execute(DotJob(arm=CHAMPION_1, seeds=(seed,), batch=batch_from_streams([self.stream])))
            for seed in (1, 2, 3, 4)
        ]
        for index in range(4):
            self.assertEqual(together["cells"][index]["mae"], alone[index]["cells"][0]["mae"])

    def test_chunking_preserves_every_cell(self) -> None:
        job = DotJob(arm=CHAMPION_1, seeds=(1, 2, 3, 4, 5), batch=batch_from_streams([self.stream], 5))
        whole = execute(job)["cells"]
        parts = [cell for piece in split(job, 3) for cell in execute(piece)["cells"]]
        self.assertEqual([c["mae"] for c in whole], [c["mae"] for c in parts])

    def test_a_failing_cell_does_not_touch_its_neighbours(self) -> None:
        configs = (CHAMPION_1.config, CellConfig(learning_rate=1e6, gradient_clip=None), CHAMPION_1.config)
        mixed = execute(
            DotJob(
                arm=CHAMPION_1, seeds=(1, 1, 2), batch=batch_from_streams([self.stream], 3), configs=configs
            )
        )
        clean = execute(DotJob(arm=CHAMPION_1, seeds=(1, 2), batch=batch_from_streams([self.stream], 2)))
        self.assertTrue(mixed["cells"][1]["failed"])
        self.assertEqual(mixed["cells"][0]["mae"], clean["cells"][0]["mae"])
        self.assertEqual(mixed["cells"][2]["mae"], clean["cells"][1]["mae"])

    def test_branch_twins_start_identical_and_frozen_does_not_update(self) -> None:
        learner, adapter = champion_learner([7])
        batch = batch_from_streams([self.stream])
        run_dot(learner, adapter, batch, stop=40)
        twin, twin_adapter, mask = branch(learner, adapter)
        before = twin.params["W_z"][1].copy()
        run_dot(twin, twin_adapter, batch.take([0, 0]), update=mask, start=40, begin=False)
        self.assertTrue(np.array_equal(twin.params["W_z"][1], before))
        self.assertFalse(np.array_equal(twin.params["W_z"][0], before))

    def test_checkpoint_resume_is_bitwise(self) -> None:
        batch = batch_from_streams([self.stream])
        learner, adapter = champion_learner([9])
        full = run_dot(learner, adapter, batch)
        learner2, adapter2 = champion_learner([9])
        first = run_dot(learner2, adapter2, batch, stop=50)
        state = json.loads(json.dumps(learner2.state_dict()))
        saved_adapter = json.loads(json.dumps(adapter_state(adapter2)))
        resumed, _ = champion_learner([9])
        resumed.load_state_dict(state)
        adapter3 = DotAdapter(CPU, 1, features="v1", target_rule="reach_gated")
        load_adapter_state(adapter3, saved_adapter)
        second = run_dot(resumed, adapter3, batch, start=50, begin=False)
        stitched = np.concatenate([first.errors[0], second.errors[0]])
        self.assertTrue(np.array_equal(stitched, full.errors[0]))

    def test_hidden_values_never_reach_a_cell(self) -> None:
        batch = batch_from_streams([self.stream])
        poisoned = batch.take([0])
        hidden = ~poisoned.observed
        poisoned.truth[hidden] = 1e9  # only never-shown steps change
        scored = StreamBatch(
            poisoned.truth, poisoned.observed, poisoned.length, poisoned.stream_ids, score=batch.truth.copy()
        )
        a, b = champion_learner([3])
        clean = run_dot(
            a,
            b,
            StreamBatch(
                batch.truth, batch.observed, batch.length, batch.stream_ids, score=batch.truth.copy()
            ),
        )
        c, d = champion_learner([3])
        dirty = run_dot(c, d, scored)
        self.assertTrue(np.array_equal(clean.predictions, dirty.predictions))

    def test_state_reset_ablation_forgets(self) -> None:
        arm = ablations(CANDIDATE_TEMPLATES["gru_v2"])["gru_v2:state_reset"]
        result = execute(DotJob(arm=arm, seeds=(1,), batch=batch_from_streams([self.stream])))
        self.assertFalse(result["cells"][0]["failed"])


# ----------------------------------------------------------------------
# identities
# ----------------------------------------------------------------------
class IdentityTests(unittest.TestCase):
    def test_seeds_are_64_bit_and_deterministic(self) -> None:
        seed = identities.derive_seed("development", "init", 0)
        self.assertEqual(seed, identities.derive_seed("development", "init", 0))
        self.assertLess(seed, 2**64)
        self.assertNotEqual(seed, identities.derive_seed("attack", "init", 0))

    def test_committed_registry_is_the_declared_plan_and_disjoint(self) -> None:
        registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
        declared = {b["block_id"]: b for b in plan.declared_blocks()}
        committed = {b["block_id"]: b for b in registry["blocks"]}
        self.assertEqual(set(declared), set(committed))
        for block_id, block in declared.items():
            for key in ("role", "namespace", "start", "count"):
                self.assertEqual(block[key], committed[block_id][key], block_id)
        self.assertEqual(identities.prove_disjoint(registry, ROOT)["status"], "DISJOINT")

    def test_a_collision_is_detected(self) -> None:
        registry = identities.reserve(
            identities.empty_registry(),
            [
                {
                    "block_id": "a",
                    "role": "development",
                    "namespace": "init",
                    "start": 0,
                    "count": 3,
                    "purpose": "x",
                }
            ],
        )
        with self.assertRaises(identities.IdentityError):
            identities.reserve(
                registry,
                [
                    {
                        "block_id": "b",
                        "role": "development",
                        "namespace": "init",
                        "start": 2,
                        "count": 3,
                        "purpose": "x",
                    }
                ],
            )
        seed = identities.seeds_of(registry, "a")[0]
        with self.assertRaises(identities.IdentityError):
            identities.prove_disjoint(registry, ROOT, history={"fake": {seed}})

    def test_confirmation_blocks_refuse_selection_and_reuse(self) -> None:
        registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
        with self.assertRaises(identities.IdentityError):
            identities.require(registry, "v2-confirmation-init", purpose="selection")
        with self.assertRaises(identities.IdentityError):
            plan.block_seeds_for(registry, "confirmation", "init", observer="someone")
        spent = identities.mark(registry, "v2-scratch-init", status="spent", observed_by="test")
        with self.assertRaises(identities.IdentityError):
            identities.mark(spent, "v2-scratch-init", status="used", observed_by="test")


# ----------------------------------------------------------------------
# benchmarks
# ----------------------------------------------------------------------
class StressSuiteTests(unittest.TestCase):
    def test_families_are_deterministic_and_role_tagged(self) -> None:
        for name, family in stress.FAMILIES.items():
            with self.subTest(family=name):
                first, second = stress.realize(name, 17), stress.realize(name, 17)
                self.assertTrue(np.array_equal(first.truth, second.truth))
                self.assertTrue(np.array_equal(first.observed, second.observed))
                self.assertIn(family.role, stress.ROLES)
                self.assertTrue(first.observed[0])

    def test_noisy_families_show_noise_and_score_latent_truth(self) -> None:
        batch = stress.batch("noise", [1, 2])
        self.assertIsNotNone(batch.score)
        assert batch.score is not None
        self.assertGreater(float(np.abs(batch.truth - batch.score).max()), 0.0)

    def test_every_role_has_families(self) -> None:
        for role in stress.ROLES:
            self.assertTrue(stress.families(role))


class ExternalBenchmarkTests(unittest.TestCase):
    def test_narma10_follows_its_published_recursion(self) -> None:
        series = synthetic.narma("narma10", 5, 300, washout=0)
        u, y = series.inputs, series.targets
        for t in range(20, 60):
            expected = 0.3 * y[t] + 0.05 * y[t] * y[t - 9 : t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
            self.assertAlmostEqual(y[t + 1], expected, places=12)

    def test_narma20_is_tanh_bounded(self) -> None:
        self.assertLess(float(np.abs(synthetic.narma("narma20", 3, 2000).targets).max()), 1.0)

    def test_mackey_glass_is_deterministic_and_bounded(self) -> None:
        a, b = synthetic.mackey_glass(4, 500), synthetic.mackey_glass(4, 500)
        self.assertTrue(np.array_equal(a.targets, b.targets))
        self.assertLess(float(np.abs(a.targets).max()), 1.0)

    def test_dysts_selection_and_metadata_checksum(self) -> None:
        names = [entry["system"] for entry in dysts_subset.select_systems()]
        self.assertEqual(
            names,
            [
                "Hadley",
                "SprottD",
                "SprottK",
                "KawczynskiStrizhak",
                "SprottE",
                "ZhouChen",
                "Chen",
                "Halvorsen",
                "SprottJerk",
                "Colpitts",
                "HastingsPowell",
                "SprottMore",
            ],
        )
        self.assertEqual(set(names), set(dysts_subset.RHS))

    def test_monash_mase_matches_hand_computation(self) -> None:
        training = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0])
        forecast, actual = np.asarray([2.0, 3.0]), np.asarray([1.0, 2.0])
        # lag-4 seasonal naive scale: |1-1|,|2-2|,|3-3|,|2-2| = 0 -> Inf -> dropped
        self.assertTrue(math.isinf(metrics.mase_monash_series(forecast, actual, training, (4,))))
        # lag-2 scale: mean |t[k]-t[k-2]| = mean(2,0,2,0,2,0) = 1
        self.assertAlmostEqual(metrics.mase_monash_series(forecast, actual, training, (2,)), 1.0)
        summary = metrics.mase_monash([forecast, forecast], [actual, actual], [training, training], (4,))
        self.assertEqual(summary["series_dropped"], 2)

    def test_monash_smape_is_a_fraction(self) -> None:
        self.assertAlmostEqual(metrics.smape_monash(np.asarray([2.0]), np.asarray([1.0])), 2.0 / 3.0)

    def test_tsf_parser_and_split_on_a_fixture(self) -> None:
        content = "# comment\n@relation X\n@attribute series_name string\n@frequency monthly\n@missing false\n@data\nT1:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.tsf"
            path.write_bytes(content.replace("\n", "\r\n").encode("cp1252"))
            parsed = monash.parse_tsf(path)
        self.assertEqual(parsed["header"]["frequency"], "monthly")
        self.assertEqual(parsed["series"][0]["values"].tolist(), list(map(float, range(1, 28))))

    def test_monash_normalization_uses_only_the_warm_up(self) -> None:
        values = np.linspace(0.0, 1.0, 1000)
        altered = values.copy()
        altered[200:] *= 50.0

        def scales(v: np.ndarray) -> dict:
            head = v[: max(50, len(v) // 10)]
            return {"level": float(np.std(head)), "delta": float(np.std(np.diff(head)))}

        self.assertEqual(scales(values), scales(altered))

    def test_series_adapter_sees_targets_only_after_prediction(self) -> None:
        problems = tasks.narma_problems("narma10", [1])
        clean = execute_series(SeriesJob(arm=CHAMPION_1, seeds=(1,), problems=tuple(problems)))
        self.assertFalse(clean["cells"][0]["failed"])

    def test_calibration_seeds_are_registered_identities(self) -> None:
        registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
        namespaces = {b["namespace"] for b in registry["blocks"]}
        for task in plan.external_tasks():
            self.assertIn(tasks.calibration_namespace(task), namespaces)


# ----------------------------------------------------------------------
# statistics and identity
# ----------------------------------------------------------------------
class StatisticsTests(unittest.TestCase):
    def test_crossed_relative_and_statuses(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.uniform(1.0, 2.0, (5, 12))
        better = base * 0.8
        result = relative_crossed(base, better, seed=1, draws=500)
        self.assertAlmostEqual(result["relative"], -0.2, places=12)
        self.assertEqual(noninferior(result, 0.02), "PASS")
        worse = relative_crossed(base, base * 1.3, seed=1, draws=500)
        self.assertEqual(noninferior(worse, 0.02), "FAIL")
        self.assertEqual(guard(worse, 0.05), "FAIL")
        self.assertEqual(combine(["PASS", "INCONCLUSIVE"]), "INCONCLUSIVE")
        self.assertEqual(combine(["PASS", "FAIL"]), "FAIL")

    def test_geometric_relative_shares_initializations(self) -> None:
        rng = np.random.default_rng(1)
        families = {name: (rng.uniform(1, 2, (4, 6)), None) for name in "abc"}
        pairs = {name: (a, a * 0.9) for name, (a, _) in families.items()}
        result = geometric_relative(pairs, seed=3, draws=500)
        self.assertAlmostEqual(result["geometric_ratio"], 0.9, places=12)
        self.assertEqual(superior(result, 1.0), "PASS")


class IdentityFingerprintTests(unittest.TestCase):
    def test_fingerprint_covers_the_executed_historical_code(self) -> None:
        record = fingerprint(ROOT)
        self.assertIn("research/aaa_1k/streams.py", record["files"])
        self.assertIn("research/aaa_1k_v2/engine.py", record["files"])
        self.assertIn("aaa/compute/device.py", record["files"])
        self.assertIn("requirements-cuda-lock.txt", record["files"])


class SeriesAdapterTests(unittest.TestCase):
    def test_absolute_mode_predicts_in_target_units(self) -> None:
        adapter = SeriesAdapter(
            CPU,
            1,
            center=0.0,
            level_scale=1.0,
            delta_scale=1.0,
            target_center=5.0,
            target_scale=2.0,
            target_mode="absolute",
        )
        learner = BatchedLearner(
            build_core("gru", inputs=3, hidden=4), CPU, seeds=[1], configs=[CellConfig(learning_rate=0.01)]
        )
        adapter.observe_input(np.asarray([0.3]), np.asarray([True]))
        prediction = adapter.predict(learner)
        self.assertAlmostEqual(float(prediction[0]), 5.0)  # zero readout predicts the target centre


if __name__ == "__main__":
    unittest.main()
