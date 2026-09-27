"""Tokenize the corpus and train the from-scratch LM (``python -m research.aaa_wm.lm.train``).

Subcommands:

``tokenize --kind bytes|bpe --vocab N``
    bytes: every train/dev/test shard becomes ``uint16`` ids (byte value; 256 = document
    separator). bpe: a byte-level BPE (HF ``tokenizers``) is trained on a deterministic 100 MB
    sample of the **train** split only; its JSON file hash identifies the tokenizer.
``train --kind ... --d-model --layers --steps ...``
    Random contiguous windows, sampled by source weight; AdamW, warmup + cosine decay, fp16
    autocast (validated against fp32 in docs/wm_program/compute.md); gradient clipping 1.0.
    Evaluation reports **bits per byte** on the held-out dev split, which makes the byte and BPE
    arms comparable: total nats over a fixed dev byte span / (ln 2 x bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

SOURCES = ("usgpo", "fineweb_edu", "cosmopedia", "simplewiki", "pydocs")
WEIGHTS = {"usgpo": 0.30, "fineweb_edu": 0.30, "cosmopedia": 0.20, "simplewiki": 0.15, "pydocs": 0.05}
SEP = 256


def root() -> Path:
    return Path(os.environ["AAA_DATA_ROOT"]) / "lm"


def tok_dir(kind: str, vocab: int) -> Path:
    return root() / "tokens" / ("bytes" if kind == "bytes" else f"bpe{vocab}")


def _bpe_path(vocab: int) -> Path:
    return root() / "tokenizers" / f"bpe{vocab}.json"


def bpe_hash(vocab: int) -> str:
    return hashlib.sha256(_bpe_path(vocab).read_bytes()).hexdigest()


def train_bpe(vocab: int, sample_bytes: int = 100_000_000) -> str:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    corpus = root() / "corpus"
    per = {s: int(sample_bytes * WEIGHTS[s]) for s in SOURCES}

    def docs():
        for s in SOURCES:
            data = (corpus / f"{s}.train.txt").read_bytes()[: per[s]]
            for d in data.split(b"\x00"):
                if d:
                    yield d.decode("utf-8", "replace")

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab - 1, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[]
    )
    tk.train_from_iterator(docs(), trainer)
    path = _bpe_path(vocab)
    path.parent.mkdir(parents=True, exist_ok=True)
    tk.save(str(path))
    return bpe_hash(vocab)


def tokenize(kind: str, vocab: int) -> dict[str, Any]:
    out = tok_dir(kind, vocab)
    out.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"kind": kind, "vocab": 257 if kind == "bytes" else vocab}
    if kind == "bpe":
        from tokenizers import Tokenizer

        info["tokenizer_sha256"] = bpe_hash(vocab) if _bpe_path(vocab).exists() else train_bpe(vocab)
        tk = Tokenizer.from_file(str(_bpe_path(vocab)))
        sep = vocab - 1
    for s in SOURCES:
        for split in ("train", "dev", "test"):
            path = root() / "corpus" / f"{s}.{split}.txt"
            if kind == "bytes":  # streamed in 64 MB chunks into a memory-mapped .npy (bounded RAM)
                n = path.stat().st_size
                ids = np.lib.format.open_memmap(
                    out / f"{s}.{split}.npy", mode="w+", dtype=np.uint16, shape=(n,)
                )
                with path.open("rb") as fh:
                    pos = 0
                    while chunk := fh.read(64 * 2**20):
                        a = np.frombuffer(chunk, dtype=np.uint8).astype(np.uint16)
                        a[a == 0] = SEP
                        ids[pos : pos + len(a)] = a
                        pos += len(a)
                ids.flush()
                info[f"{s}.{split}"] = {"tokens": int(n), "bytes": int(n)}
                del ids
                continue
            parts = []
            carry = b""
            with path.open("rb") as fh:  # streamed: documents are NUL-terminated
                while True:
                    chunk = fh.read(32 * 2**20)
                    data = carry + chunk
                    docs = data.split(b"\x00")
                    carry = docs.pop() if chunk else b""
                    for i in range(0, len(docs), 4096):
                        for enc in tk.encode_batch(
                            [d.decode("utf-8", "replace") for d in docs[i : i + 4096] if d]
                        ):
                            parts.append(np.array([*enc.ids, sep], dtype=np.uint16))
                    if not chunk:
                        break
            bpe_ids = np.concatenate(parts) if parts else np.zeros(0, np.uint16)
            np.save(out / f"{s}.{split}.npy", bpe_ids)
            info[f"{s}.{split}"] = {"tokens": len(bpe_ids), "bytes": path.stat().st_size}
            del parts, bpe_ids
    (out / "info.json").write_text(json.dumps(info, indent=1))
    return info


class Windows:
    """Random training windows drawn from in-RAM blocks.

    Each source keeps ``blocks`` contiguous blocks of ``block_tokens`` tokens, read from random
    offsets of its memory-mapped shard (sequential reads, which suit the HDD). Every
    ``refresh`` batches one block per source is replaced. Windows are uniform within the
    blocks, and blocks are uniform over the shard, so every window position has the same
    long-run probability. Source proportions follow ``WEIGHTS``.
    """

    def __init__(
        self,
        kind: str,
        vocab: int,
        split: str,
        context: int,
        seed: int,
        *,
        block_tokens: int = 4 * 2**20,
        blocks: int = 4,
        refresh: int = 50,
    ) -> None:
        d = tok_dir(kind, vocab)
        self.data = {s: np.load(d / f"{s}.{split}.npy", mmap_mode="r") for s in SOURCES}
        self.ctx = context
        self.rng = np.random.default_rng(seed)
        names = [s for s in SOURCES if len(self.data[s]) > context + 1]
        w = np.array([WEIGHTS[s] for s in names])
        self.names, self.p = names, w / w.sum()
        self.block_tokens, self.refresh, self.calls = block_tokens, refresh, 0
        self.buf = {s: [self._block(s) for _ in range(blocks)] for s in names}

    def _block(self, s: str) -> np.ndarray:
        arr = self.data[s]
        n = min(len(arr), self.block_tokens)
        j = int(self.rng.integers(0, len(arr) - n + 1))
        return np.array(arr[j : j + n])

    def batch(self, B: int) -> np.ndarray:
        self.calls += 1
        if self.calls % self.refresh == 0:
            for s in self.names:
                self.buf[s][int(self.rng.integers(0, len(self.buf[s])))] = self._block(s)
        src = self.rng.choice(len(self.names), B, p=self.p)
        out = np.empty((B, self.ctx + 1), dtype=np.int64)
        for i, k in enumerate(src):
            blocks = self.buf[self.names[k]]
            arr = blocks[int(self.rng.integers(0, len(blocks)))]
            j = self.rng.integers(0, len(arr) - self.ctx - 1)
            out[i] = arr[j : j + self.ctx + 1]
        return out


def bits_per_byte(
    model: Any, kind: str, vocab: int, split: str, device: str, budget_bytes: int = 2_000_000
) -> dict[str, float]:
    """Nats over the first ``budget_bytes`` bytes of each source's split, per byte, in bits."""

    import torch

    d = tok_dir(kind, vocab)
    info = json.loads((d / "info.json").read_text())
    out = {}
    ctx = model.config.context
    for s in SOURCES:
        ids = np.load(d / f"{s}.{split}.npy")
        total_bytes = info[f"{s}.{split}"]["bytes"]
        if len(ids) < ctx + 1 or total_bytes == 0:
            continue
        frac = min(1.0, budget_bytes / total_bytes)
        n_tok = int(len(ids) * frac)
        nats, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, n_tok - 1, ctx):
                chunk = torch.from_numpy(ids[start : start + ctx + 1].astype(np.int64))[None].to(device)
                if chunk.shape[1] < 2:
                    break
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(chunk[:, :-1]).float()
                nats += float(torch.nn.functional.cross_entropy(logits[0], chunk[0, 1:], reduction="sum"))
                count += chunk.shape[1] - 1
        span_bytes = total_bytes * (count / len(ids))
        out[s] = nats / (math.log(2) * span_bytes)
    out["mean"] = float(np.mean([v for k, v in out.items()]))
    return out


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from . import model as Mm

    vocab = 257 if args.kind == "bytes" else args.vocab
    tokname = "bytes" if args.kind == "bytes" else f"bpe:{bpe_hash(args.vocab)}"
    cfg = Mm.LMConfig(
        vocab=vocab,
        d_model=args.d_model,
        layers=args.layers,
        heads=max(1, args.d_model // 64),
        context=args.context,
        seed=args.seed,
        tokenizer=tokname,
    )
    model = Mm.LM(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = min(500, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    data = Windows(args.kind, args.vocab, "train", args.context, args.seed + 1)
    curve, t0 = [], time.time()
    tokens_seen = 0
    for step in range(args.steps):
        xy = torch.from_numpy(data.batch(args.batch)).cuda()
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            logits = model(xy[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, vocab), xy[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        tokens_seen += xy[:, 1:].numel()
        if not math.isfinite(loss.item()):
            raise FloatingPointError("non-finite loss")
        if step % 250 == 0 or step == args.steps - 1:
            curve.append((step, float(loss.item()), time.time() - t0))
            print(step, round(float(loss.item()), 4), round(time.time() - t0), flush=True)
    model.eval()
    info = {
        "config": cfg.__dict__,
        "args": vars(args),
        "curve": curve,
        "tokens_seen": tokens_seen,
        "seconds": time.time() - t0,
        "tokens_per_second": tokens_seen / (time.time() - t0),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "accounting": Mm.accounting(model, opt),
        "dev_bits_per_byte": bits_per_byte(model, args.kind, args.vocab, "dev", "cuda"),
    }
    name = f"{args.kind}{'' if args.kind == 'bytes' else args.vocab}_d{args.d_model}_l{args.layers}_s{args.seed}_n{args.steps}{args.tag}"
    info["fingerprint"] = Mm.save(
        model, root() / "ckpt" / f"{name}.pt", {k: v for k, v in info.items() if k != "curve"}
    )
    (root() / "logs").mkdir(parents=True, exist_ok=True)
    (root() / "logs" / f"{name}.json").write_text(json.dumps(info, indent=1, default=str))
    print(
        json.dumps(
            {k: info[k] for k in ("dev_bits_per_byte", "tokens_per_second", "peak_vram_bytes")}, indent=1
        ),
        info["accounting"],
    )
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokenize")
    t.add_argument("--kind", choices=("bytes", "bpe"), required=True)
    t.add_argument("--vocab", type=int, default=8192)
    r = sub.add_parser("train")
    r.add_argument("--kind", choices=("bytes", "bpe"), required=True)
    r.add_argument("--vocab", type=int, default=8192)
    r.add_argument("--d-model", type=int, default=256)
    r.add_argument("--layers", type=int, default=4)
    r.add_argument("--context", type=int, default=512)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--steps", type=int, default=5000)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--amp", action="store_true")
    r.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "tokenize":
        print(json.dumps(tokenize(a.kind, a.vocab), indent=1))
    else:
        train(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
