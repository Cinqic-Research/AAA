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
    res: dict[str, Any] = {"lm": a.lm, "random_twin": a.random_twin, "steps": a.steps, "seed": a.seed, "protocol": "v2" if a.select else "v1"}
    lm = load_lm(Path(a.lm), a.random_twin, a.seed)
    codec = codec_for(lm)
    if a.select:  # adapter v2: choose the fine-tuning length on the *selection* phrasing family only
        import copy

        select_items = items_for(dev, "select")
        gold_sel = [[tuple(g) for g in it["gold"]] for it in select_items]
        best: dict[str, Any] = {"score": -1.0}
        scores: dict[int, float] = {}

        def on_ck(step: int) -> None:
            ext_s = AD.extract(lm, codec, [it["text"] for it in select_items])
            sc = float(np.mean([e == g for e, g in zip(ext_s, gold_sel, strict=True)]))
            scores[step] = sc
            if sc > best["score"]:  # ties keep the earlier (fewer-step) checkpoint
                best.update(score=sc, step=step, state=copy.deepcopy(lm.state_dict()))

        res["finetune"] = AD.finetune(lm, codec, train_items, steps=a.steps, seed=a.seed, checkpoints=(250, 500, 1000, 2000, 3000), on_checkpoint=on_ck)
        lm.load_state_dict(best["state"])
        res["selection_scores"] = scores
        res["chosen_steps"] = best["step"]
    else:
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
    name = f"adapter_{Path(a.lm).stem}{'_random' if a.random_twin else ''}_s{a.seed}_n{a.steps}{'_v2' if a.select else ''}"
    from ..lm.model import save

    save(lm, root() / "lm" / "adapters" / f"{name}.pt", {k: v for k, v in res.items() if k != "extractions"})
    out = root() / "lm" / "adapter_eval" / f"{name}_{a.role}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, default=list))
    print(json.dumps({k: v for k, v in res.items() if k not in ("extractions", "finetune")}, indent=1), res["finetune"]["curve"][-2:])
    return 0


class LanguageAgent:
    """Wraps a decision agent; the view's visible tests are replaced by the language channel's proposal.

    ``fallback`` (an agent needing no tests) is used when the channel produced no parse and the
    wrapped agent cannot act without tests (the neural policies)."""

    def __init__(self, inner: Any, proposal: list[tuple[int, int]] | None, fallback: Any = None) -> None:
        self.tests = tuple((x, ("ok", e)) for x, e in proposal) if proposal else ()
        self.inner = fallback if (not self.tests and fallback is not None) else inner

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


def play_cmd(a: argparse.Namespace) -> int:
    from ..opaque import agents as OA
    from ..opaque.env import play
    from ..opaque.experiment import dev_tasks, make_agent, provenance

    tasks = dev_tasks(a.role)
    items = items_for(tasks, "heldout")
    if a.channel == "none":
        proposals: list[Any] = [None] * len(tasks)
    elif a.channel == "rules":
        proposals = [rules.extract(it["text"]) for it in items]
    elif a.channel == "gold":  # ceiling of the language channel (perfect understanding), not an arm
        proposals = [[tuple(g) for g in it["gold"]] for it in items]
    else:
        ext = json.loads(Path(a.channel).read_text())["extractions"]
        proposals = [ext.get(t.task_id) for t in tasks]
    prior = OA.Prior(OA.fit_prior(gen.pool("train", 6000)))
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    doc: dict[str, Any] = {"schema": "aaa.wm.lang.play.v1", "status": "development", "role": a.role, "channel": a.channel, "tasks": [t.task_id for t in tasks], "slices": [t.slice for t in tasks], "provenance": provenance(), "results": {}}
    doc["channel_exact"] = float(np.mean([p is not None and [tuple(x) for x in p] == [tuple(g) for g in it["gold"]] for p, it in zip(proposals, items, strict=True)]))
    for name in a.agents.split(","):
        per = {}
        for seed in [int(x) for x in a.seeds.split(",")]:
            outs = []
            inner = None
            for i, (t, prop) in enumerate(zip(tasks, proposals, strict=True)):
                if inner is None or ("online" in name and i % 40 == 0):
                    inner = make_agent(name, seed, 20000, prior, device)
                fb = OA.ToolSearch(lambda v, e: prior.score(e), "tool_prior") if name.split(":")[0] in ("policy", "policy_aux") else None
                outs.append(play(LanguageAgent(inner, [tuple(x) for x in prop] if prop else None, fb), t, i))
            per[str(seed)] = {"bits": "".join("1" if o.success else "0" for o in outs)}
            print(name, a.channel, seed, round(float(np.mean([o.success for o in outs])), 3), flush=True)
        doc["results"][name] = per
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(doc))
    return 0


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
    ad.add_argument("--select", action="store_true", help="adapter v2: choose fine-tuning length on the selection family")
    pl = sub.add_parser("play")
    pl.add_argument("--channel", required=True, help="none | rules | gold | <adapter eval json>")
    pl.add_argument("--agents", required=True)
    pl.add_argument("--seeds", default="0")
    pl.add_argument("--role", default="tune")
    pl.add_argument("--output", required=True)
    a = ap.parse_args()
    return adapter_cmd(a) if a.cmd == "adapter" else play_cmd(a)


if __name__ == "__main__":
    raise SystemExit(main())
