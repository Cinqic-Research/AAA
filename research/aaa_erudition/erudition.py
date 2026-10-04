"""The Erudition Model: ``aaa.erudition.model.v0``.

A transformer encoder over the last ``WINDOW`` steps of evidence features,
with the condition's allowed-action menu as one extra token. It has two
heads:

* ``q`` -- for every adaptation action, the expected quality of the next
  ``HORIZON`` steps (task success minus adaptation cost) if that action is
  requested now. It is trained on counterfactual returns measured by
  branching the simulator, so it learns *from consequences* when to wait,
  which component to adapt, by which mechanism, and when to roll back;
* ``diagnosis`` -- a distribution over ``contracts.Diagnosis`` (stable,
  transient, insufficient evidence, unreliable feedback, tool failure, LM,
  WM or joint deficiency), trained on the simulator's hidden regime. It is
  reported with every decision and used to measure misattribution; the
  decision itself comes from ``q``.

The parameters are initialized from scratch (seeded) and trained only on
training-split simulation. No weight comes from any earlier AAA model,
Juniper LM, or Juniper Reference.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .contracts import Diagnosis
from .lifecycle import ACTION_NAMES, FEATURES, ControllerView, Decision

MODEL_VERSION = "aaa.erudition.model.v0"
WINDOW = 64
DIAGNOSES = tuple(d.value for d in Diagnosis)


@dataclasses.dataclass(frozen=True)
class Architecture:
    kind: str = "transformer"
    d_model: int = 160
    layers: int = 4
    heads: int = 4
    ff: int = 640
    window: int = WINDOW


class EruditionNet(nn.Module):
    """The selected transformer, plus the GRU and flat-MLP alternatives compared in development."""

    def __init__(self, arch: Architecture) -> None:
        super().__init__()
        self.arch = arch
        d = arch.d_model
        self.menu = nn.Linear(len(ACTION_NAMES), d)
        if arch.kind == "transformer":
            self.features = nn.Linear(len(FEATURES), d)
            self.position = nn.Parameter(torch.zeros(arch.window + 1, d))
            layer = nn.TransformerEncoderLayer(
                d, arch.heads, arch.ff, dropout=0.0, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, arch.layers, enable_nested_tensor=False)
            nn.init.normal_(self.position, std=0.02)
        elif arch.kind == "gru":
            self.features = nn.Linear(len(FEATURES), d)
            self.gru = nn.GRU(d, d, num_layers=arch.layers, batch_first=True)
        elif arch.kind == "mlp":
            self.flat = nn.Sequential(
                nn.Linear(arch.window * len(FEATURES), arch.ff),
                nn.GELU(),
                nn.Linear(arch.ff, arch.ff),
                nn.GELU(),
                nn.Linear(arch.ff, d),
            )
        else:
            raise ValueError(f"unknown architecture kind {arch.kind!r}")
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, len(ACTION_NAMES))
        self.diagnosis = nn.Linear(d, len(DIAGNOSES))

    def forward(
        self, x: torch.Tensor, pad: torch.Tensor, menu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` [B, W, F] oldest first; ``pad`` [B, W] true where padded; ``menu`` [B, A] in {0, 1}."""

        if self.arch.kind == "transformer":
            tokens = torch.cat([self.features(x), self.menu(menu).unsqueeze(1)], dim=1) + self.position
            mask = torch.cat([pad, torch.zeros_like(pad[:, :1])], dim=1)
            h = self.encoder(tokens, src_key_padding_mask=mask)[:, -1]
        elif self.arch.kind == "gru":
            # Padding is at the front and all zeros; the GRU runs through it to the newest step.
            h = self.gru(self.features(x))[0][:, -1] + self.menu(menu)
        else:
            h = self.flat(x.flatten(1)) + self.menu(menu)
        h = self.norm(h)
        return self.q(h), self.diagnosis(h)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def window(features: np.ndarray, size: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """The last ``size`` rows, left-padded with zeros; returns (window, pad mask)."""

    rows = features[-size:]
    out = np.zeros((size, features.shape[1]), dtype=np.float32)
    pad = np.ones(size, dtype=bool)
    out[size - len(rows) :] = rows
    pad[size - len(rows) :] = False
    return out, pad


def menu_vector(allowed: frozenset[str]) -> np.ndarray:
    return np.array([float(a in allowed) for a in ACTION_NAMES], dtype=np.float32)


class EruditionController:
    """Runs the trained network as a lifecycle controller (CPU, no gradients)."""

    name = "erudition"

    def __init__(self, net: EruditionNet, margin: float, meta: dict[str, Any]) -> None:
        self.net = net.eval()
        self.margin = margin
        self.meta = meta

    def scores(self, view: ControllerView) -> tuple[np.ndarray, np.ndarray]:
        x, pad = window(view.features, self.net.arch.window)
        with torch.no_grad():
            q, diag = self.net(
                torch.from_numpy(x)[None],
                torch.from_numpy(pad)[None],
                torch.from_numpy(menu_vector(view.allowed))[None],
            )
        return q[0].numpy(), torch.softmax(diag[0], dim=0).numpy()

    def decide(self, view: ControllerView) -> Decision:
        q, diag = self.scores(view)
        allowed = [i for i, a in enumerate(ACTION_NAMES) if a in view.allowed and a != "wait"]
        wait = ACTION_NAMES.index("wait")
        diagnosis = {name: float(p) for name, p in zip(DIAGNOSES, diag, strict=True)}
        if allowed:
            best = max(allowed, key=lambda i: q[i])
            if q[best] - q[wait] > self.margin:
                return Decision(ACTION_NAMES[best], diagnosis, self.name)
        return Decision("wait", diagnosis, self.name)

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict()}, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        meta = dict(self.meta)
        meta.update(
            {
                "version": MODEL_VERSION,
                "architecture": dataclasses.asdict(self.net.arch),
                "margin": self.margin,
                "trainable_parameters": count_parameters(self.net),
                "weights_sha256": digest,
                "features": list(FEATURES),
                "actions": list(ACTION_NAMES),
                "diagnoses": list(DIAGNOSES),
            }
        )
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", "utf-8")
        return digest

    @staticmethod
    def load(path: Path) -> EruditionController:
        meta = json.loads(path.with_suffix(".json").read_text("utf-8"))
        if meta.get("version") != MODEL_VERSION:
            raise ValueError(f"{path} is not a {MODEL_VERSION} model")
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["weights_sha256"]:
            raise ValueError(f"{path} does not match its recorded digest")
        if meta["features"] != list(FEATURES) or meta["actions"] != list(ACTION_NAMES):
            raise ValueError(f"{path} was trained on a different feature or action contract")
        net = EruditionNet(Architecture(**meta["architecture"]))
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state["state_dict"])
        return EruditionController(net, float(meta["margin"]), meta)
