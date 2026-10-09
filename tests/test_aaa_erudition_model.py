"""The Erudition Model network, its artifact identity and its controller contract (needs PyTorch)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.aaa_erudition import lifecycle

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is installed only in the neural environment")
class EruditionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from research.aaa_erudition import erudition

        self.torch = torch
        self.e = erudition
        torch.manual_seed(0)

    def test_parameter_count_is_exact_and_all_trainable(self) -> None:
        net = self.e.EruditionNet(self.e.Architecture())
        self.assertEqual(self.e.count_parameters(net), 1_259_700)
        self.assertTrue(all(p.requires_grad for p in net.parameters()))

    def test_every_parameter_receives_gradient(self) -> None:
        net = self.e.EruditionNet(self.e.Architecture())
        x = self.torch.randn(4, self.e.WINDOW, len(lifecycle.FEATURES))
        pad = self.torch.zeros(4, self.e.WINDOW, dtype=self.torch.bool)
        menu = self.torch.ones(4, len(lifecycle.ACTION_NAMES))
        q, d = net(x, pad, menu)
        (q.sum() + d.sum()).backward()
        dead = [name for name, p in net.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
        self.assertEqual(dead, [])

    def test_padding_does_not_change_the_reading(self) -> None:
        net = self.e.EruditionNet(self.e.Architecture()).eval()
        features = np.random.default_rng(0).normal(size=(10, len(lifecycle.FEATURES))).astype(np.float32)
        x, pad = self.e.window(features)
        x2 = x.copy()
        x2[: self.e.WINDOW - 10] = 99.0
        menu = self.torch.ones(1, len(lifecycle.ACTION_NAMES))
        with self.torch.no_grad():
            a = net(self.torch.from_numpy(x)[None], self.torch.from_numpy(pad)[None], menu)[0]
            b = net(self.torch.from_numpy(x2)[None], self.torch.from_numpy(pad)[None], menu)[0]
        self.assertTrue(self.torch.allclose(a, b, atol=1e-5))

    def controller(self, margin: float = 0.0) -> object:
        return self.e.EruditionController(self.e.EruditionNet(self.e.Architecture()), margin, {"test": True})

    def test_save_load_round_trip_and_tamper_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            controller = self.controller(0.01)
            digest = controller.save(path)  # type: ignore[attr-defined]
            loaded = self.e.EruditionController.load(path)
            self.assertEqual(loaded.meta["weights_sha256"], digest)
            self.assertEqual(loaded.meta["trainable_parameters"], 1_259_700)
            data = bytearray(path.read_bytes())
            data[-10] ^= 0xFF
            path.write_bytes(bytes(data))
            with self.assertRaises(ValueError):
                self.e.EruditionController.load(path)

    def test_feature_contract_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            self.controller().save(path)  # type: ignore[attr-defined]
            meta_path = path.with_suffix(".json")
            meta = json.loads(meta_path.read_text())
            meta["features"] = meta["features"][:-1]
            meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                self.e.EruditionController.load(path)

    def test_controller_stays_inside_the_condition_menu(self) -> None:
        controller = self.controller(-1.0)
        g = np.random.default_rng(1)
        for condition, allowed in lifecycle.CONDITIONS.items():
            for _ in range(20):
                view = lifecycle.ControllerView(
                    step=int(g.integers(1, 100)),
                    features=g.normal(size=(int(g.integers(1, 80)), len(lifecycle.FEATURES))).astype(
                        np.float32
                    ),
                    allowed=allowed,
                )
                with self.subTest(condition):
                    self.assertIn(controller.decide(view).action, allowed)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
