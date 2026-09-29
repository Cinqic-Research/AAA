"""The language adapter: American-English statement -> structured visible tests (``aaa.wm.lang.adapter.v1``).

The adapter is the from-scratch LM fine-tuned as a conditional generator:

    <statement> "\\n=> " x1 " -> " e1 " ; " x2 " -> " e2 " ; " x3 " -> " e3 "\\n"

The loss covers only the answer after ``=>``. Training statements come from ``opaque.v0`` *train*
tasks rendered with the *train* phrasing families. Evaluation statements come from development
(or, later, confirmation) tasks rendered with the *held-out* families. The pretrained LM and an
identically shaped randomly initialized twin are fine-tuned with the same data, steps and
optimizer; their difference is the contribution of pretraining.

Output provenance: the adapter's extraction is a ``LANGUAGE_MODEL_PROPOSAL``. The agent uses it
as its only statement of the tests, and nothing marks it as observed truth.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

PROPOSAL = "LANGUAGE_MODEL_PROPOSAL"


class Codec:
    """Bytes (vocab 257) or a BPE tokenizer file; ``sep`` ends a document/answer."""

    def __init__(self, kind: str, bpe_path: str | None = None) -> None:
        self.kind = kind
        if kind == "bpe":
            from tokenizers import Tokenizer

            self.tk = Tokenizer.from_file(bpe_path)  # type: ignore[arg-type]
            self.sep = self.tk.get_vocab_size()
        else:
            self.sep = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8")) if self.kind == "bytes" else self.tk.encode(text).ids

    def decode(self, ids: Sequence[int]) -> str:
        ids = [i for i in ids if i != self.sep]
        return bytes(ids).decode("utf-8", "replace") if self.kind == "bytes" else self.tk.decode(ids)


def answer(gold: Sequence[tuple[int, int]]) -> str:
    return " ; ".join(f"{x} -> {e}" for x, e in gold) + "\n"


_PAIR = re.compile(r"(-?\d+)\s*->\s*(-?\d+)")


def parse(text: str, n: int = 3) -> list[tuple[int, int]] | None:
    pairs = [(int(a), int(b)) for a, b in _PAIR.findall(text.split("\n")[0])]
    return pairs[:n] if len(pairs) >= n else None


def examples(codec: Codec, items: Sequence[dict[str, Any]], context: int) -> list[tuple[np.ndarray, int]]:
    out = []
    for it in items:
        prompt = codec.encode(it["text"] + "\n=> ")
        target = codec.encode(answer(it["gold"])) + [codec.sep]
        ids = (prompt + target)[-(context + 1) :]
        out.append((np.array(ids, dtype=np.int64), max(1, len(ids) - len(target))))
    return out


def finetune(model: Any, codec: Codec, train_items: Sequence[dict[str, Any]], *, steps: int, batch: int = 32, lr: float = 5e-4, seed: int = 0, device: str = "cuda", checkpoints: Sequence[int] = (), on_checkpoint: Any = None) -> dict[str, Any]:
    import torch

    ex = examples(codec, train_items, model.config.context)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    curve, t0 = [], time.time()
    for step in range(steps):
        idx = rng.integers(0, len(ex), batch)
        L = max(len(ex[i][0]) for i in idx)
        x = np.full((batch, L), codec.sep, dtype=np.int64)
        mask = np.zeros((batch, L - 1), dtype=np.float32)
        for r, i in enumerate(idx):
            ids, start = ex[i]
            x[r, : len(ids)] = ids
            mask[r, start - 1 : len(ids) - 1] = 1.0
        xt = torch.from_numpy(x).to(device)
        mt = torch.from_numpy(mask).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(xt[:, :-1]).float()
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), xt[:, 1:].reshape(-1), reduction="none")
        loss = (nll * mt.reshape(-1)).sum() / mt.sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            curve.append((step, float(loss.item())))
        if on_checkpoint is not None and (step + 1) in checkpoints:
            model.eval()
            on_checkpoint(step + 1)
            model.train()
    model.eval()
    return {"curve": curve, "seconds": time.time() - t0, "examples": len(ex)}


def extract(model: Any, codec: Codec, texts: Sequence[str], device: str = "cuda", max_new: int = 64) -> list[list[tuple[int, int]] | None]:
    """Greedy decoding, batched over prompts of *identical* length (no padding is ever shown)."""

    import torch

    out: list[list[tuple[int, int]] | None] = [None] * len(texts)
    prompts = [codec.encode(t + "\n=> ")[-(model.config.context - max_new) :] for t in texts]
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    groups = [g[k : k + 64] for g in by_len.values() for k in range(0, len(g), 64)]
    for group in groups:
        L = len(prompts[group[0]])
        x = torch.tensor([prompts[i] for i in group], device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(max_new):
                nxt = model(x[:, -model.config.context :])[:, -1].argmax(-1, keepdim=True)
                x = torch.cat([x, nxt], dim=1)
        for r, i in enumerate(group):
            out[i] = parse(codec.decode(x[r, L:].tolist()))
    return out
