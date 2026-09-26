"""Learned arms for ``aaa.python.opaque.v0`` (PyTorch; optional dependency, see requirements-torch-lock.txt).

One backbone family (a small pre-LN Transformer encoder over the tokenizer's sequences) and
three heads. An *arm* is a backbone plus the heads it trains, and each arm has its own
backbone, so no arm borrows another's representation:

* ``wm``         consequence head: at every ``<Q> x`` query, a distribution over the result
                 of running the program on ``x`` (the observation a real RUN would return);
* ``value``      value head on ``[CLS]``: P(program domain-equivalent to the reference),
                 P(program passes all visible tests), given the visible tests;
* ``policy``     policy head on ``[CLS]``: P(the marked edit is part of a fix), given the tests;
* ``policy_aux`` the policy plus the consequence head as an *auxiliary* loss (same labels as
                 ``wm``); decisions use only the policy head.

Accounting (:func:`accounting`) counts every trainable parameter, the AdamW state and the
update counter. Checkpoints are strict: the schema, the configuration, the vocabulary hash and
every tensor shape must match, and they are verified before anything is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import tokens as tok
from .data import CODE_LEN, Dataset

SCHEMA = "aaa.wm.opaque.checkpoint.v1"
TAIL = 9  # visible tests (x e SEP) x 3, or queries (<Q> x) x 3 padded to 9


@dataclass(frozen=True)
class ArmConfig:
    arm: str  # wm | value | policy | policy_aux
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    ff: int = 512
    dropout: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.arm not in ("wm", "value", "policy", "policy_aux"):
            raise ValueError(self.arm)


class Backbone(nn.Module):
    def __init__(self, c: ArmConfig) -> None:
        super().__init__()
        self.tok = nn.Embedding(len(tok.VOCAB), c.d_model, padding_idx=tok.PAD)
        self.typ = nn.Embedding(5, c.d_model)
        self.pos = nn.Embedding(CODE_LEN + TAIL, c.d_model)
        layer = nn.TransformerEncoderLayer(
            c.d_model, c.heads, c.ff, c.dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.d_model)

    def forward(self, ids: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.tok(ids) + self.typ(types) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=ids == tok.PAD)
        return self.norm(h)


class Arm(nn.Module):
    def __init__(self, c: ArmConfig) -> None:
        super().__init__()
        torch.manual_seed(c.seed)
        self.config = c
        self.backbone = Backbone(c)
        if c.arm in ("wm", "policy_aux"):
            self.consequence = nn.Linear(c.d_model, tok.N_RESULTS)
        if c.arm == "value":
            self.value = nn.Linear(c.d_model, 2)
        if c.arm in ("policy", "policy_aux"):
            self.policy = nn.Linear(c.d_model, 1)

    def hidden(self, ids: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        return self.backbone(ids, types)


def accounting(model: Arm, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    blocks: dict[str, int] = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        blocks[top] = blocks.get(top, 0) + p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt_state = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            opt_state += sum(v.numel() for v in state.values() if torch.is_tensor(v) and v.dim() > 0)
    return {
        "trainable_parameters": int(trainable),
        "per_block": blocks,
        "optimizer_state": int(opt_state),
        "update_counter": 1,
        "adaptive_state_total": int(trainable + opt_state + 1),
        "persistent_recurrent_state": 0,
        "replay_memory": 0,
        "tokenizer_learned_state": 0,
        "vocab_hash": tok.vocab_hash(),
    }


# ------------------------------------------------------------------ batches
def _tail_tests(tests: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """tests [B, 3, 2] -> tail ids/types [B, 9]: x e SEP per test."""

    B = tests.shape[0]
    ids = np.zeros((B, TAIL), dtype=np.int64)
    types = np.zeros((B, TAIL), dtype=np.int64)
    num = tok.numbers
    for j in range(3):
        ids[:, 3 * j] = num(tests[:, j, 0])
        ids[:, 3 * j + 1] = num(tests[:, j, 1])
        ids[:, 3 * j + 2] = tok.SEP
        types[:, 3 * j] = tok.T_INPUT
        types[:, 3 * j + 1] = tok.T_EXPECTED
    return ids, types


def _tail_queries(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xs [B, 3] -> tail [B, 9]: (<Q> x) x 3, then 3 PAD."""

    B = xs.shape[0]
    ids = np.full((B, TAIL), tok.PAD, dtype=np.int64)
    types = np.zeros((B, TAIL), dtype=np.int64)
    num = tok.numbers
    for j in range(3):
        ids[:, 2 * j] = tok.Q
        ids[:, 2 * j + 1] = num(xs[:, j])
        types[:, 2 * j] = tok.T_QUERY
        types[:, 2 * j + 1] = tok.T_INPUT
    return ids, types


QUERY_SLOTS = [CODE_LEN + 2 * j for j in range(3)]


def assemble(code: np.ndarray, tail_ids: np.ndarray, tail_types: np.ndarray, marks: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    types = np.zeros(code.shape, dtype=np.int64)
    if marks is not None:
        rows = np.nonzero(marks > 0)[0]
        types[rows, marks[rows]] = tok.T_MARK
    ids = np.concatenate([code.astype(np.int64), tail_ids], axis=1)
    typ = np.concatenate([types, tail_types], axis=1)
    return torch.from_numpy(ids), torch.from_numpy(typ)


class Batcher:
    """Samples training batches for an arm from the shared dataset (seeded)."""

    def __init__(self, ds: Dataset, arm: str, seed: int) -> None:
        self.ds, self.arm = ds, arm
        self.rng = np.random.default_rng(seed)
        self.domain = np.arange(ds.results.shape[1])

    def wm(self, B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.rng.integers(0, len(self.ds.code), B)
        cols = self.rng.integers(0, self.ds.results.shape[1], (B, 3))
        from .generator import domain

        dom = np.array(domain())
        ids, types = assemble(self.ds.code[idx], *_tail_queries(dom[cols]))
        target = torch.from_numpy(self.ds.results[idx[:, None], cols].astype(np.int64))
        return ids, types, target

    def value(self, B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.rng.integers(0, len(self.ds.code), B)
        tests = self.ds.tests[self.ds.task_of[idx]]
        ids, types = assemble(self.ds.code[idx], *_tail_tests(tests))
        y = np.stack([self.ds.equivalent[idx], self.ds.visible_pass[idx].all(axis=1)], axis=1)
        return ids, types, torch.from_numpy(y.astype(np.float32))

    def policy(self, B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        rec = self.ds.policy[self.rng.integers(0, len(self.ds.policy), B)]
        idx = rec[:, 0]
        tests = self.ds.tests[self.ds.task_of[idx]]
        ids, types = assemble(self.ds.code[idx], *_tail_tests(tests), marks=rec[:, 1])
        return ids, types, torch.from_numpy(rec[:, 2].astype(np.float32)), idx


def loss_on_batch(model: Arm, batcher: Batcher, B: int, device: str) -> torch.Tensor:
    arm = model.config.arm
    if arm == "wm":
        ids, types, target = batcher.wm(B)
        h = model.hidden(ids.to(device), types.to(device))
        logits = model.consequence(h[:, QUERY_SLOTS]).float()
        return nn.functional.cross_entropy(logits.reshape(-1, tok.N_RESULTS), target.to(device).reshape(-1))
    if arm == "value":
        ids, types, y = batcher.value(B)
        h = model.hidden(ids.to(device), types.to(device))
        return nn.functional.binary_cross_entropy_with_logits(model.value(h[:, 0]).float(), y.to(device))
    ids, types, y, idx = batcher.policy(B)
    h = model.hidden(ids.to(device), types.to(device))
    loss = nn.functional.binary_cross_entropy_with_logits(model.policy(h[:, 0]).squeeze(-1).float(), y.to(device))
    if arm == "policy_aux":
        wids, wtypes, target = batcher.wm(B)
        wh = model.hidden(wids.to(device), wtypes.to(device))
        logits = model.consequence(wh[:, QUERY_SLOTS]).float()
        loss = loss + nn.functional.cross_entropy(logits.reshape(-1, tok.N_RESULTS), target.to(device).reshape(-1))
    return loss


def train(
    model: Arm,
    ds: Dataset,
    *,
    steps: int,
    batch: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    log_every: int = 500,
    amp: bool = False,
) -> dict[str, Any]:
    """AdamW with warmup + cosine decay. ``amp`` = fp16 autocast with a gradient scaler (validated
    against fp32 before use; see docs/wm_program/compute.md)."""

    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    batcher = Batcher(ds, model.config.arm, model.config.seed + 17)
    curve = []
    model.train()
    for step in range(steps):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = loss_on_batch(model, batcher, batch, device)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if not math.isfinite(loss.item()):
            raise FloatingPointError("non-finite training loss")
        if step % log_every == 0 or step == steps - 1:
            curve.append((step, float(loss.item())))
    model.eval()
    return {"curve": curve, "accounting": accounting(model, opt), "steps": steps, "batch": batch, "lr": lr, "amp": amp}


# ------------------------------------------------------------------ inference
def _code_rows(sources: Sequence[str]) -> np.ndarray:
    from .data import _code_row

    return np.stack([_code_row(s)[0] for s in sources])


@torch.no_grad()
def predict_pass(model: Arm, sources: Sequence[str], tests: Sequence[tuple[int, Any]], device: str, *, shuffle_queries: bool = False) -> np.ndarray:
    """[n, 3] P(the program's result on visible input j equals expected_j) (``wm``/``policy_aux``)."""

    out = []
    xs = np.array([[x for x, _ in tests]] * 1)
    exp_cls = [tok.result_class(e) for _, e in tests]
    for s in range(0, len(sources), 512):
        chunk = sources[s : s + 512]
        code = _code_rows(chunk)
        q = np.repeat(xs, len(chunk), axis=0)
        if shuffle_queries:
            q = q[:, ::-1].copy()
        ids, types = assemble(code, *_tail_queries(q))
        h = model.hidden(ids.to(device), types.to(device))
        p = torch.softmax(model.consequence(h[:, QUERY_SLOTS]), dim=-1)
        out.append(p[:, torch.arange(3), torch.tensor(exp_cls)].cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


@torch.no_grad()
def predict_value(model: Arm, sources: Sequence[str], tests: Sequence[tuple[int, Any]], device: str, which: int = 0) -> np.ndarray:
    out = []
    t = np.array([[[x, e[1]] for x, e in tests]])
    for s in range(0, len(sources), 512):
        chunk = sources[s : s + 512]
        ids, types = assemble(_code_rows(chunk), *_tail_tests(np.repeat(t, len(chunk), axis=0)))
        h = model.hidden(ids.to(device), types.to(device))
        out.append(torch.sigmoid(model.value(h[:, 0])[:, which]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,))


@torch.no_grad()
def predict_policy(model: Arm, sources: Sequence[str], marks: Sequence[int], tests: Sequence[tuple[int, Any]], device: str) -> np.ndarray:
    out = []
    t = np.array([[[x, e[1]] for x, e in tests]])
    for s in range(0, len(sources), 512):
        chunk = sources[s : s + 512]
        ids, types = assemble(
            _code_rows(chunk), *_tail_tests(np.repeat(t, len(chunk), axis=0)), marks=np.array(marks[s : s + 512])
        )
        h = model.hidden(ids.to(device), types.to(device))
        out.append(torch.sigmoid(model.policy(h[:, 0]).squeeze(-1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,))


# ------------------------------------------------------------------ checkpoints
def fingerprint(model: Arm) -> str:
    h = hashlib.sha256(json.dumps(asdict(model.config), sort_keys=True).encode())
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save(model: Arm, path: Path, extra: dict[str, Any] | None = None) -> str:
    fp = fingerprint(model)
    payload = {
        "schema": SCHEMA,
        "config": asdict(model.config),
        "vocab_hash": tok.vocab_hash(),
        "fingerprint": fp,
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return fp


class CheckpointError(ValueError):
    pass


def load(path: Path, device: str = "cpu") -> Arm:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # corrupt or truncated file
        raise CheckpointError(f"unreadable checkpoint: {error}") from None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise CheckpointError("unknown checkpoint schema")
    if payload.get("vocab_hash") != tok.vocab_hash():
        raise CheckpointError("tokenizer mismatch")
    model = Arm(ArmConfig(**payload["config"]))
    reference = model.state_dict()
    state = payload["state"]
    if set(state) != set(reference) or any(state[k].shape != reference[k].shape for k in reference):
        raise CheckpointError("tensor names or shapes differ from the configuration")
    if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise CheckpointError("non-finite values")
    model.load_state_dict(state, strict=True)
    if fingerprint(model) != payload["fingerprint"]:
        raise CheckpointError("fingerprint mismatch")
    return model.to(device).eval()
