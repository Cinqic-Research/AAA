"""A small from-scratch causal decoder for American English (``aaa.wm.lm.model.v1``).

Pre-norm Transformer: RMSNorm, rotary position embedding, SwiGLU-free GELU MLP (4x), tied input/
output embeddings, no biases. Random initialization only; no pretrained weights are ever
loaded. The tokenizer is external (bytes: vocabulary 256 + 1 document separator; or a BPE model
whose file hash is part of the checkpoint).

Accounting counts trainable parameters (embedding counted once, because it is tied), AdamW state and
the step counter. Checkpoints are strict (schema, config, tokenizer hash, shapes, finiteness,
fingerprint).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCHEMA = "aaa.wm.lm.checkpoint.v1"


@dataclass(frozen=True)
class LMConfig:
    vocab: int
    d_model: int = 256
    layers: int = 4
    heads: int = 4
    context: int = 512
    dropout: float = 0.0
    seed: int = 0
    tokenizer: str = "bytes"  # "bytes" or "bpe:<sha256 of tokenizer file>"


class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.g


def _rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    # x: [B, H, T, D]
    t, d = x.shape[-2], x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = torch.arange(t, device=x.device).float()[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: LMConfig) -> None:
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.out = nn.Linear(4 * c.d_model, c.d_model, bias=False)
        self.heads = c.heads
        self.drop = c.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = _rope(q), _rope(k)
        a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.out(nn.functional.gelu(self.fc(self.n2(x))))


class LM(nn.Module):
    def __init__(self, c: LMConfig) -> None:
        super().__init__()
        torch.manual_seed(c.seed)
        self.config = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.d_model)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * c.layers) if n.endswith(("proj.weight", "out.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.hidden(ids) @ self.emb.weight.T  # tied output head

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n: int, temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        for _ in range(n):
            logits = self(ids[:, -self.config.context :])[:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


def accounting(model: LM, opt: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.emb.weight.numel()
    opt_state = 0
    if opt is not None:
        for st in opt.state.values():
            opt_state += sum(v.numel() for v in st.values() if torch.is_tensor(v) and v.dim() > 0)
    return {
        "trainable_parameters": int(trainable),
        "embedding_parameters_tied": int(emb),
        "non_embedding_parameters": int(trainable - emb),
        "optimizer_state": int(opt_state),
        "update_counter": 1,
        "adaptive_state_total": int(trainable + opt_state + 1),
    }


def fingerprint(model: LM) -> str:
    h = hashlib.sha256(json.dumps(asdict(model.config), sort_keys=True).encode())
    for k, t in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class CheckpointError(ValueError):
    pass


def save(model: LM, path: Path, extra: dict[str, Any] | None = None) -> str:
    fp = fingerprint(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"schema": SCHEMA, "config": asdict(model.config), "fingerprint": fp, "state": {k: v.cpu() for k, v in model.state_dict().items()}, "extra": extra or {}}, tmp)
    tmp.replace(path)
    return fp


def load(path: Path, device: str = "cpu") -> LM:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointError(f"unreadable checkpoint: {e}") from None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise CheckpointError("unknown schema")
    model = LM(LMConfig(**payload["config"]))
    ref, st = model.state_dict(), payload["state"]
    if set(ref) != set(st) or any(ref[k].shape != st[k].shape for k in ref):
        raise CheckpointError("shapes or names differ")
    if not all(torch.isfinite(v).all() for v in st.values()):
        raise CheckpointError("non-finite values")
    model.load_state_dict(st, strict=True)
    if fingerprint(model) != payload["fingerprint"]:
        raise CheckpointError("fingerprint mismatch")
    return model.to(device).eval()
