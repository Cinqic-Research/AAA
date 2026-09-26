"""Development experiments for opaque.v0: train learned arms on the shared dataset and play episodes.

Usage (from the torch environment, with ``AAA_DATA_ROOT`` on the HDD)::

    python -m research.aaa_wm.opaque.experiment train --arm wm --seed 0 --steps 20000
    python -m research.aaa_wm.opaque.experiment evaluate --role tune --agents ... --seeds 0,1

Development identities only: ``development`` indices (tune / evaluate / adapt, see ``LAYOUT``).
The ``attack`` split is used once, later, by a separate command; confirmation is not admitted.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import agents as A
from . import generator as gen
from .env import play

LAYOUT = {"tune": (0, 1000), "evaluate": (1000, 5000), "adapt": (5000, 6000)}


def data_root() -> Path:
    root = os.environ.get("AAA_DATA_ROOT")
    if not root:
        raise SystemExit("set AAA_DATA_ROOT (on the HDD)")
    return Path(root) / "opaque"


def ckpt_path(arm: str, seed: int, steps: int, tag: str = "") -> Path:
    return data_root() / "ckpt" / f"{arm}_s{seed}_n{steps}{tag}.pt"


def provenance() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    out: dict[str, Any] = {"commit": head, "dirty": dirty, "python": sys.version, "platform": platform.platform()}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    out["spec_hash"] = gen.spec_hash()
    return out


def dev_tasks(role: str) -> list[gen.OpaqueTask]:
    lo, hi = LAYOUT[role]
    cache = data_root() / f"dev_{role}.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        from .program import Edit

        return [
            gen.OpaqueTask(
                r["task_id"], r["split"], r["index"], r["slice"], r["library"], r["reference"], r["buggy"],
                tuple(Edit(**f) for f in r["faults"]), tuple((x, tuple(e)) for x, e in r["visible_tests"]), r["k"],
            )
            for r in raw
        ]
    # disjointness: no development reference or buggy program equals a pilot or train program
    earlier = gen.pool("pilot") + gen.pool("train")
    exclude = {gen.program_hash(t.reference) for t in earlier} | {gen.program_hash(t.buggy) for t in earlier}
    tasks = gen.pool("development", hi - lo, start=lo, exclude=exclude)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([t.to_json() for t in tasks]))
    return tasks


def train_cmd(args: argparse.Namespace) -> int:
    import torch

    from . import data as D
    from . import models as M

    ds = D.build("train", args.train_tasks)
    model = M.Arm(M.ArmConfig(args.arm, d_model=args.d_model, layers=args.layers, ff=4 * args.d_model, seed=args.seed))
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    t0 = time.time()
    info = M.train(model, ds, steps=args.steps, batch=args.batch, lr=args.lr, device=device, amp=args.amp)
    info["seconds"] = time.time() - t0
    info["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    info["dataset"] = D.summary(ds)
    info["provenance"] = provenance()
    fp = M.save(model, ckpt_path(args.arm, args.seed, args.steps, args.tag), info)
    info["fingerprint"] = fp
    info["checkpoint_bytes"] = ckpt_path(args.arm, args.seed, args.steps, args.tag).stat().st_size
    out = data_root() / "train_logs" / f"{args.arm}_s{args.seed}_n{args.steps}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("seconds", "peak_vram_bytes", "fingerprint")}, indent=1), info["curve"][-3:], info["accounting"]["trainable_parameters"])
    return 0


def make_agent(name: str, seed: int, steps: int, prior: A.Prior, device: str, tag: str = "") -> Any:
    from . import learned as L
    from . import models as M

    if name == "submit_asis":
        return A.SubmitAsIs()
    if name == "prior":
        return A.PriorOnly(prior)
    if name == "tool_prior":
        return A.ToolSearch(lambda v, e: prior.score(e), "tool_prior")
    if name == "tool_gap":
        return A.ToolSearch(A.gap_scorer(prior), "tool_gap", initial_run=True)
    if name.startswith("ceiling"):
        depth = 1 if name.endswith("_d1") else 2
        return A.Planner(A.TrueLibraryPredictor(gen.library("A")), prior, name, depth=depth)
    base, _, variant = name.partition(":")
    if base in ("policy", "policy_aux"):
        return L.policy_agent(M.load(ckpt_path(base, seed, steps, tag), device), device, name)
    if base == "wm":
        if variant == "random":
            model = M.Arm(M.ArmConfig("wm", seed=seed + 1000)).to(device).eval()
        else:
            model = M.load(ckpt_path("wm", seed, steps, tag), device)
        depth = 1 if variant == "d1" else 2
        pred = L.WMPredictor(model, device, shuffle_queries=variant == "shuffle", disabled=variant == "disabled")
        return A.Planner(pred, prior, name, depth=depth)
    if base == "policy_aux_wm":  # the policy_aux network's auxiliary consequence head used AS a world model
        model = M.load(ckpt_path("policy_aux", seed, steps, tag), device)
        return A.Planner(L.WMPredictor(model, device), prior, name, depth=2)
    if base == "value":
        model = M.load(ckpt_path("value", seed, steps, tag), device)
        which = 1 if variant == "visible" else 0
        return A.Planner(L.ValuePredictor(model, device, which), prior, name, depth=1 if variant == "d1" else 2)
    raise ValueError(name)


def evaluate_cmd(args: argparse.Namespace) -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tasks = dev_tasks(args.role)
    if args.limit:
        tasks = tasks[: args.limit]
    prior = A.Prior(A.fit_prior(gen.pool("train", args.train_tasks)))
    seeds = [int(s) for s in args.seeds.split(",")]
    doc: dict[str, Any] = {
        "schema": "aaa.wm.opaque.evaluation.v1",
        "status": "development",
        "role": args.role,
        "range": LAYOUT[args.role],
        "runs_budget": args.runs,
        "steps_budget": args.steps_budget,
        "tasks": [t.task_id for t in tasks],
        "slices": [t.slice for t in tasks],
        "k": [t.k for t in tasks],
        "provenance": provenance(),
        "results": {},
    }
    for name in args.agents.split(","):
        per_seed = {}
        for seed in seeds if not name.split(":")[0] in ("submit_asis", "prior", "tool_prior", "tool_gap") and not name.startswith("ceiling") else [0]:
            agent = make_agent(name, seed, args.train_steps, prior, device, args.tag)
            t0 = time.time()
            if name.startswith("ceiling"):  # the ceiling imagines with the task's *actual* library (A or B)
                ceilings = {lib: A.Planner(A.TrueLibraryPredictor(gen.library(lib)), prior, name, depth=agent.depth) for lib in ("A", "B")}
                outs = [play(ceilings[t.library], t, i, runs=args.runs, steps=args.steps_budget) for i, t in enumerate(tasks)]
            else:
                outs = [play(agent, t, i, runs=args.runs, steps=args.steps_budget) for i, t in enumerate(tasks)]
            per_seed[str(seed)] = {
                "bits": "".join("1" if o.success else "0" for o in outs),
                "runs": [o.runs for o in outs],
                "edits": [o.edits for o in outs],
                "seconds": time.time() - t0,
            }
            by = defaultdict(list)
            for t, o in zip(tasks, outs, strict=True):
                by[t.slice].append(o.success)
            print(f"{name:22s} seed={seed} all={np.mean([o.success for o in outs]):.3f} " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(by.items())) + f" runs={np.mean([o.runs for o in outs]):.2f} t={time.time() - t0:.0f}s", flush=True)
        doc["results"][name] = per_seed
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opaque-experiment")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--train-tasks", type=int, default=6000)
    t.add_argument("--tag", default="")
    t.add_argument("--cpu", action="store_true")
    t.add_argument("--amp", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--role", default="tune", choices=list(LAYOUT))
    e.add_argument("--agents", required=True)
    e.add_argument("--seeds", default="0")
    e.add_argument("--train-steps", type=int, default=20000)
    e.add_argument("--train-tasks", type=int, default=6000)
    e.add_argument("--runs", type=int, default=2)
    e.add_argument("--steps-budget", type=int, default=8)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--tag", default="")
    e.add_argument("--output", required=True)
    e.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    return train_cmd(args) if args.cmd == "train" else evaluate_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
