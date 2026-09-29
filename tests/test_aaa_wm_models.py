"""Checkpoint, accounting and inference-shape tests for the opaque.v0 torch arms (skipped without torch)."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is an optional dependency (requirements-torch-lock.txt)")
class Checkpoints(unittest.TestCase):
    def setUp(self) -> None:
        from research.aaa_wm.opaque import models as M

        self.M = M
        self.model = M.Arm(M.ArmConfig("wm", d_model=32, layers=1, heads=2, ff=64, seed=3))
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "m.pt"

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_roundtrip_fingerprint(self) -> None:
        fp = self.M.save(self.model, self.path)
        loaded = self.M.load(self.path)
        self.assertEqual(fp, self.M.fingerprint(loaded))

    def test_refuses_corrupt_file(self) -> None:
        self.M.save(self.model, self.path)
        data = self.path.read_bytes()
        self.path.write_bytes(data[: len(data) // 2])
        with self.assertRaises(self.M.CheckpointError):
            self.M.load(self.path)

    def test_refuses_tampered_weights(self) -> None:
        import torch

        self.M.save(self.model, self.path)
        payload = torch.load(self.path, weights_only=False)
        name = next(iter(payload["state"]))
        payload["state"][name] = payload["state"][name] + 1.0
        torch.save(payload, self.path)
        with self.assertRaises(self.M.CheckpointError):
            self.M.load(self.path)

    def test_refuses_missing_block(self) -> None:
        import torch

        self.M.save(self.model, self.path)
        payload = torch.load(self.path, weights_only=False)
        payload["state"].pop(next(iter(payload["state"])))
        torch.save(payload, self.path)
        with self.assertRaises(self.M.CheckpointError):
            self.M.load(self.path)

    def test_accounting_counts_every_parameter(self) -> None:
        acc = self.M.accounting(self.model)
        self.assertEqual(acc["trainable_parameters"], sum(p.numel() for p in self.model.parameters()))
        self.assertEqual(sum(acc["per_block"].values()), acc["trainable_parameters"])

    def test_predict_pass_shape(self) -> None:
        from research.aaa_wm.opaque import generator as gen

        t = gen.build("pilot", 0)
        p = self.M.predict_pass(self.model.eval(), [t.buggy, t.reference], t.visible_tests, "cpu")
        self.assertEqual(p.shape, (2, 3))
        self.assertTrue(((p >= 0) & (p <= 1)).all())


class StructuredModel(unittest.TestCase):
    def test_abduction_learns_only_consistent_entries(self) -> None:
        from research.aaa_wm.opaque import generator as gen
        from research.aaa_wm.opaque import structured as S
        from research.aaa_wm.opaque.program import run

        lib = gen.library("A")
        truth = {a.name: a.fn() for a in lib}
        model = S.LibraryModel(tuple(a.name for a in lib))
        obs = []
        for i in range(12):
            t = gen.build("pilot", 200 + i)
            for x in gen.domain()[::3]:
                obs.append((t.reference, x, run(t.reference, lib, x)))
        model.learn(obs, rounds=6)
        self.assertGreater(len(model.table), 0)
        for (name, arg), value in model.table.items():
            self.assertEqual(truth[name](arg), value)

    def test_unknown_entries_are_not_predictions(self) -> None:
        from research.aaa_wm.opaque import generator as gen
        from research.aaa_wm.opaque import structured as S

        model = S.LibraryModel(tuple(a.name for a in gen.library("A")))
        t = gen.build("pilot", 1)
        self.assertIsNone(model.predict(t.reference, 0))


if __name__ == "__main__":
    unittest.main()
