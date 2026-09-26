"""American-English evaluation of a from-scratch LM checkpoint (``aaa.wm.lm.eval.v1``).

``python -m research.aaa_wm.lm.evaluate --lm CKPT [--split test]``

* **bits per byte** per source on the held-out split (the test split is used only for the
  final report; development work uses ``dev``);
* **BLiMP** (Warstadt et al. 2020, CC BY 4.0; 67 paradigms x 1,000 minimal pairs): accuracy of
  preferring the grammatical sentence by total log-probability, per paradigm and overall;
* **American spelling preference**: sentences from the held-out split containing a US-only
  spelling (the corpus lexicon) are paired with the same sentence using the British variant. The
  score is the accuracy of preferring the American one, split by whether the word occurs in
  training text at all. The corpus dialect filter makes this partly a data property, and
  the report says so;
* **spelling-noise robustness**: the increase in bits per byte when 5% of the characters of
  held-out text are perturbed (swap, drop, duplicate, substitute adjacent key), fixed seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from .corpus import VARIANTS
from .train import SOURCES, bits_per_byte, root


def _codec(model: Any) -> Any:
    from ..lang.adapter import Codec
    from .train import _bpe_path

    if model.config.tokenizer == "bytes":
        return Codec("bytes")
    return Codec("bpe", str(_bpe_path(model.config.vocab)))


def logprob(model: Any, codec: Any, texts: list[str], device: str = "cuda") -> np.ndarray:
    """Total log-probability of each text, conditioned on a leading separator token."""

    import torch

    out = np.zeros(len(texts))
    enc = [[codec.sep] + codec.encode(t)[: model.config.context - 1] for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    for s in range(0, len(order), 32):
        group = order[s : s + 32]
        L = max(len(enc[i]) for i in group)
        x = torch.full((len(group), L), codec.sep, dtype=torch.long)
        m = torch.zeros((len(group), L - 1))
        for r, i in enumerate(group):
            x[r, : len(enc[i])] = torch.tensor(enc[i])
            m[r, : len(enc[i]) - 1] = 1
        x, m = x.to(device), m.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(x[:, :-1])
        tok = -torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none").reshape(x.shape[0], -1)
        vals = (tok * m).sum(-1).cpu().numpy()
        for r, i in enumerate(group):
            out[i] = vals[r]
    return out


def blimp(model: Any, codec: Any) -> dict[str, Any]:
    import pyarrow.parquet as pq

    d = root() / "eval" / "blimp"
    per = {}
    good_all, bad_all = [], []
    for f in sorted(d.glob("*.parquet")):
        t = pq.read_table(f).to_pydict()
        g = logprob(model, codec, t["sentence_good"])
        b = logprob(model, codec, t["sentence_bad"])
        per[f.name.split("_train")[0].split(".parquet")[0]] = float(np.mean(g > b))
        good_all.append(g)
        bad_all.append(b)
    return {"overall": float(np.mean(np.concatenate(good_all) > np.concatenate(bad_all))), "paradigms": per}


def spelling_pairs(split: str = "test", limit: int = 2000) -> list[tuple[str, str, str]]:
    us_to_uk = {}
    pairs_src = [(a, b) for a, b in VARIANTS.items()]
    by_value: dict[str, list[str]] = {}
    for w, v in pairs_src:
        by_value.setdefault(v, []).append(w)
    from .corpus import _PAIRS

    for pair in _PAIRS.split("|"):
        if pair.strip():
            us, uk = pair.split()
            if VARIANTS.get(us) == "US" and VARIANTS.get(uk) == "UK":
                us_to_uk[us] = uk
    out = []
    pat = re.compile(r"\b(" + "|".join(sorted(us_to_uk, key=len, reverse=True)) + r")\b")
    for s in SOURCES:
        data = (root() / "corpus" / f"{s}.{split}.txt").read_text(encoding="utf-8", errors="replace")
        for sent in re.split(r"(?<=[.!?])\s+", data):
            if 40 <= len(sent) <= 300 and "\x00" not in sent:
                m = pat.search(sent)
                if m:
                    w = m.group(1)
                    out.append((w, sent, sent[: m.start()] + us_to_uk[w] + sent[m.end() :]))
            if len(out) >= limit:
                return out
    return out


def noisy(text: str, rate: float, rng: np.random.Generator) -> str:
    keys = "qwertyuiopasdfghjklzxcvbnm"
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.integers(0, 4)
            if op == 0 and i + 1 < len(chars):
                out += [chars[i + 1], c]
                i += 2
                continue
            if op == 1:
                i += 1
                continue
            if op == 2:
                out += [c, c]
            else:
                k = keys.find(c.lower())
                out.append(keys[(k + 1) % len(keys)] if k >= 0 else c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def noise_robustness(model: Any, codec: Any, split: str = "dev", n_docs: int = 300, rate: float = 0.05) -> dict[str, float]:
    rng = np.random.default_rng(20260926)
    docs = []
    for s in SOURCES:
        data = (root() / "corpus" / f"{s}.{split}.txt").read_bytes().split(b"\x00")
        docs += [d.decode("utf-8", "replace")[:1500] for d in data[: n_docs // len(SOURCES)] if d]
    clean = logprob(model, codec, docs)
    dirty_docs = [noisy(d, rate, rng) for d in docs]
    dirty = logprob(model, codec, dirty_docs)
    bits_clean = -clean.sum() / math.log(2) / sum(len(d.encode()) for d in docs)
    bits_dirty = -dirty.sum() / math.log(2) / sum(len(d.encode()) for d in dirty_docs)
    return {"bpb_clean": float(bits_clean), "bpb_noisy": float(bits_dirty), "delta": float(bits_dirty - bits_clean), "rate": rate}


def main() -> int:
    from .model import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    model = load(Path(a.lm), "cuda")
    codec = _codec(model)
    kind = "bytes" if model.config.tokenizer == "bytes" else "bpe"
    res: dict[str, Any] = {"lm": a.lm, "split": a.split, "config": model.config.__dict__}
    res["bits_per_byte"] = bits_per_byte(model, kind, model.config.vocab, a.split, "cuda")
    res["blimp"] = blimp(model, codec)
    pairs = spelling_pairs("test" if a.split == "test" else "dev")
    us = logprob(model, codec, [p[1] for p in pairs])
    uk = logprob(model, codec, [p[2] for p in pairs])
    res["american_spelling"] = {"pairs": len(pairs), "prefers_american": float(np.mean(us > uk)) if pairs else None}
    res["noise"] = noise_robustness(model, codec)
    out = Path(a.output) if a.output else root() / "eval_results" / (Path(a.lm).stem + f"_{a.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if k != "blimp" else v["overall"]) for k, v in res.items() if k != "config"}, indent=1))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
