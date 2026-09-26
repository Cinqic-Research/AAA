"""Language-track experiments (development identities only).

``adapter``  fine-tune a language adapter from an LM checkpoint (or its random twin) and score
             exact extraction on development statements in the *held-out* phrasings.
``play``     play language-conditioned ``opaque.v0`` episodes: a decision arm (WM-S, policy_aux, ...)
             receives only the statement; its visible tests come from a language channel
             (``none`` | ``rules`` | an adapter checkpoint).

The language-conditioned view: :class:`LanguageAgent` replaces the numeric visible tests in
every view with the channel's ``LANGUAGE_MODEL_PROPOSAL`` *before* the wrapped agent sees it.
It never reads the true tests. With no parse, the wrapped agent gets no tests and falls back
to its no-test behavior (the planner then scores every plan equally and follows the prior).
Real ``RUN`` observations are unchanged: a real test runner still compares against the real
tests.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..opaque import generator as gen
from ..opaque.program import run
from . import rules
from .reports import statement

ROLE = {"tune": (0, 1000), "evaluate": (1000, 5000)}


def root() -> Path:
    return Path(os.environ["AAA_DATA_ROOT"])


def items_for(tasks: list[Any], family: str, variants: int = 1) -> list[dict[str, Any]]:
    out = []
    for t in tasks:
        lib = gen.library(t.library)
        br = [run(t.buggy, lib, x) for x, _ in t.visible_tests]
        for v in range(variants):
            it = statement(t, family, br, seed_label=str(v))
            it["task_id"] = t.task_id
            out.append(it)
    return out


def load_lm(path: Path, random_twin: bool, seed: int) -> Any:
    from ..lm import model as Mm

    lm = Mm.load(path, "cuda")
    if random_twin:
        cfg = dataclasses.replace(lm.config, seed=seed + 7919)
        lm = Mm.LM(cfg).cuda()
    return lm


def codec_for(lm: Any) -> Any:
    from ..lm.train import _bpe_path
    from .adapter import Codec

    if lm.config.tokenizer == "bytes":
        return Codec("bytes")
    return Codec("bpe", str(_bpe_path(lm.config.vocab)))


def adapter_cmd(a: argparse.Namespace) -> int:
    from . import adapter as AD

    train_tasks = gen.pool("train", a.train_tasks)
    train_items = items_for(train_tasks, "train", variants=2)
    from ..opaque.experiment import dev_tasks

    dev = dev_tasks(a.role)
    eval_items = items_for(dev, "heldout")
    indist_items = items_for(dev[:300], "train")
    res: dict[str, Any] = {"lm": a.lm, "random_twin": a.random_twin, "steps": a.steps, "seed": a.seed}
    lm = load_lm(Path(a.lm), a.random_twin, a.seed)
    codec = codec_for(lm)
    res["finetune"] = AD.finetune(lm, codec, train_items, steps=a.steps, seed=a.seed)
    t0 = time.time()
    ext = AD.extract(lm, codec, [it["text"] for it in eval_items])
    res["heldout_exact"] = float(np.mean([e == [tuple(g) for g in it["gold"]] for e, it in zip(ext, eval_items, strict=True)]))
    res["heldout_pair_accuracy"] = float(np.mean([sum(p == tuple(g) for p, g in zip(e or [], it["gold"], strict=False)) / 3 for e, it in zip(ext, eval_items, strict=True)]))
    ext2 = AD.extract(lm, codec, [it["text"] for it in indist_items])
    res["train_phrasing_exact"] = float(np.mean([e == [tuple(g) for g in it["gold"]] for e, it in zip(ext2, indist_items, strict=True)]))
    res["rules_heldout_exact"] = float(np.mean([rules.extract(it["text"]) == [tuple(g) for g in it["gold"]] for it in eval_items]))
    res["extract_seconds"] = time.time() - t0
    res["extractions"] = {it["task_id"]: e for it, e in zip(eval_items, ext, strict=True)}
    name = f"adapter_{Path(a.lm).stem}{'_random' if a.random_twin else ''}_s{a.seed}_n{a.steps}"
    from ..lm.model import save

    save(lm, root() / "lm" / "adapters" / f"{name}.pt", {k: v for k, v in res.items() if k != "extractions"})
    out = root() / "lm" / "adapter_eval" / f"{name}_{a.role}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, default=list))
    print(json.dumps({k: v for k, v in res.items() if k not in ("extractions", "finetune")}, indent=1), res["finetune"]["curve"][-2:])
    return 0


class LanguageAgent:
    """Wraps a decision agent; the view's visible tests are replaced by the language channel's proposal."""

    def __init__(self, inner: Any, proposal: list[tuple[int, int]] | None) -> None:
        self.inner = inner
        self.tests = tuple((x, ("ok", e)) for x, e in proposal) if proposal else ()

    def _view(self, view: Any) -> Any:
        return dataclasses.replace(view, visible_tests=self.tests)

    def begin(self, view: Any) -> None:
        if hasattr(self.inner, "begin"):
            self.inner.begin(self._view(view))

    def act(self, view: Any) -> Any:
        return self.inner.act(self._view(view))

    def observe(self, view: Any, action: Any, obs: Any) -> None:
        if hasattr(self.inner, "observe") and obs is not None and len(obs.results) == len(self.tests):
            self.inner.observe(self._view(view), action, obs)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ad = sub.add_parser("adapter")
    ad.add_argument("--lm", required=True)
    ad.add_argument("--random-twin", action="store_true")
    ad.add_argument("--steps", type=int, default=3000)
    ad.add_argument("--seed", type=int, default=0)
    ad.add_argument("--train-tasks", type=int, default=6000)
    ad.add_argument("--role", default="tune", choices=list(ROLE))
    a = ap.parse_args()
    return adapter_cmd(a)


if __name__ == "__main__":
    raise SystemExit(main())
