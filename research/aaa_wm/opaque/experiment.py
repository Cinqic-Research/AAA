"""Development experiments for opaque.v0: train learned arms on the shared dataset and play episodes.

Usage (from the torch environment, with ``AAA_DATA_ROOT`` on the HDD)::

    python -m research.aaa_wm.opaque.experiment train --arm wm --seed 0 --steps 20000
    python -m research.aaa_wm.opaque.experiment evaluate --role tune --agents ... --seeds 0,1

Development identities only: ``development`` indices (tune / evaluate / adapt, see ``LAYOUT``).
The ``attack`` split is used once, later, by a separate command; confirmation is not admitted.
"""

from __future__ import annotations

import argparse
import dataclasses
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

LAYOUT = {"tune": (0, 1000), "evaluate": (1000, 5000), "adapt": (5000, 6000), "attack": (0, 1480)}


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
    # disjointness: no development reference or buggy program equals a pilot or train program;
    # the attack split (used once) also excludes every development program
    earlier = gen.pool("pilot") + gen.pool("train") + (gen.pool("development") if role == "attack" else [])
    exclude = {gen.program_hash(t.reference) for t in earlier} | {gen.program_hash(t.buggy) for t in earlier}
    tasks = gen.pool("attack" if role == "attack" else "development", hi - lo, start=lo, exclude=exclude)
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


def jepa_cmd(args: argparse.Namespace) -> int:
    """Train J-4 (wm consequence head + action-conditioned JEPA auxiliary) at the wm step budget."""

    import torch

    from . import data as D
    from . import jepa as J
    from . import models as M

    ds = D.build("train", args.train_tasks)
    trans = J.build_transitions(ds, gen.pool("train", args.train_tasks), limit=args.transition_tasks)
    model = J.JepaArm(M.ArmConfig("wm", seed=args.seed), lam_j=args.lam_j)
    t0 = time.time()
    info = J.train_jepa(model, ds, trans, steps=args.steps, amp=True, vic=args.vic)
    info["seconds"] = time.time() - t0
    info["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    info["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["ema_target_parameters"] = sum(p.numel() for p in model.target.parameters())
    info["provenance"] = provenance()
    # the planner uses the consequence head: save the inner wm arm (plus the JEPA parts for the record)
    fp = M.save(model.inner, ckpt_path("wm", args.seed, args.steps, args.tag), {k: v for k, v in info.items() if k != "curve"})
    torch.save({"edit": model.edit.state_dict(), "pred": model.pred.state_dict(), "target": model.target.state_dict()}, ckpt_path("jepa_parts", args.seed, args.steps, args.tag))
    info["fingerprint"] = fp
    out = data_root() / "train_logs" / f"jepa_s{args.seed}_n{args.steps}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info["diagnostics"], indent=1), info["curve"][-2:])
    return 0


def make_agent(name: str, seed: int, steps: int, prior: A.Prior, device: str, tag: str = "") -> Any:
    if name == "submit_asis":
        return A.SubmitAsIs()
    if name == "prior":
        return A.PriorOnly(prior)
    if name == "tool_prior":
        return A.ToolSearch(lambda v, e: prior.score(e), "tool_prior")
    if name == "tool_gap":
        return A.ToolSearch(A.gap_scorer(prior), "tool_gap", initial_run=True)
    if name == "tells":  # red-team shortcut: grammar legality only, same verified planner (no world model)
        from .attacks import TellPredictor

        return A.Planner(TellPredictor(), prior, name, depth=2)
    if name.startswith("ceiling"):
        depth = 1 if name.endswith("_d1") else 2
        return A.Planner(A.TrueLibraryPredictor(gen.library("A")), prior, name, depth=depth)
    base, _, variant = name.partition(":")
    if base == "wms":
        return wms_agent(variant, seed, prior)
    from . import learned as L  # torch arms only below this line
    from . import models as M

    if base in ("policy", "policy_aux") and variant == "plan":  # controller-matched: same planner as WM-S
        return L.PolicyPlanner(M.load(ckpt_path(base, seed, steps, tag), device), device, prior, name)
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


def wms_table(seed: int, programs: int) -> Any:
    """Learn (or load) the WM-S library table from ``programs`` sampled training programs' real observations."""

    from . import data as D
    from . import structured as S

    names = tuple(a.name for a in gen.library("A"))
    path = data_root() / "wms" / f"table_s{seed}_p{programs}.json"
    model = S.LibraryModel(names)
    if path.exists():
        raw = json.loads(path.read_text())
        model.table = {(k.split("|")[0], int(k.split("|")[1])): v for k, v in raw["table"].items()}
        model.updates = raw["updates"]
        return model
    ds = D.build("train", 6000)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds.sources), min(programs, len(ds.sources)), replace=False)
    sub = type(ds)(ds.code[idx], [], ds.results[idx], ds.task_of[idx], ds.equivalent[idx], ds.visible_pass[idx], ds.tests, ds.policy[:0], [ds.sources[i] for i in idx])
    t0 = time.time()
    stats = model.learn(S.observations_from_dataset(sub, gen.domain()), rounds=12)
    stats["seconds"] = time.time() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stats": stats, "updates": model.updates, "table": {f"{k[0]}|{k[1]}": v for k, v in model.table.items()}}))
    return model


def wms_agent(variant: str, seed: int, prior: A.Prior) -> Any:
    import copy

    from . import structured as S

    parts = set(variant.split("+")) if variant else set()
    programs = next((int(p[1:]) for p in parts if p.startswith("p") and p[1:].isdigit()), 200000)
    model = copy.deepcopy(wms_table(seed, programs))
    if "empty" in parts:
        model.table = {}
    if "identity" in parts:  # ablation: the learned library replaced by 'every API returns its argument'
        model.table = {k: k[1] for k in model.table}
    if "random" in parts:  # ablation: learned values permuted across entries of the same API
        rng = np.random.default_rng(seed + 99)
        by: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for k in model.table:
            by[k[0]].append(k)
        new = {}
        for keys in by.values():
            vals = [model.table[k] for k in keys]
            rng.shuffle(vals)
            new.update(zip(keys, vals, strict=True))
        model.table = new
    unknown = next((float(p[1:]) for p in parts if p.startswith("u") and p[1:].replace(".", "").isdigit()), 0.3)
    pred: Any = S.StructuredPredictor(model, online="online" in parts, unknown=unknown)
    base_pred = pred
    if "shuffle" in parts:  # action control: plan i is scored with the prediction for another plan
        rng_s = np.random.default_rng(seed + 7)

        def shuffled(programs: Any, view: Any, _p: Any = base_pred) -> Any:
            out = _p(programs, view)
            return out[rng_s.permutation(len(out))]

        pred = shuffled
    if "corrupt" in parts:  # attack: every expected value shifted by +1 (a wrong specification)

        def corrupt(programs: Any, view: Any, _p: Any = pred) -> Any:
            bad = dataclasses.replace(view, visible_tests=tuple((x, (e[0], e[1] + 1) if e[0] == "ok" else e) for x, e in view.visible_tests))
            return _p(programs, bad)

        pred = corrupt
    if "stale" in parts:  # action control: every plan gets the prediction for the unedited program

        def stale(programs: Any, view: Any, _p: Any = base_pred) -> Any:
            one = _p([view.source], view)
            return np.repeat(one, len(programs), axis=0)

        pred = stale
    agent = A.Planner(pred, prior, "wms:" + variant, depth=1 if "d1" in parts else 2)
    if "online" in parts:
        agent.on_observation = lambda obs, view: base_pred.observe(obs, view)  # type: ignore[method-assign]
    return agent


def evaluate_cmd(args: argparse.Namespace) -> int:
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    except ImportError:  # non-neural agents (baselines, WM-S) need no torch
        device = "cpu"
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
        "stream_reset": args.stream_reset,
        "steps_budget": args.steps_budget,
        "tasks": [t.task_id for t in tasks],
        "slices": [t.slice for t in tasks],
        "k": [t.k for t in tasks],
        "provenance": provenance(),
        "results": {},
    }
    for name in args.agents.split(","):
        per_seed = {}
        for seed in seeds if not name.split(":")[0] in ("submit_asis", "prior", "tool_prior", "tool_gap", "tells") and not name.startswith("ceiling") else [0]:
            agent = make_agent(name, seed, args.train_steps, prior, device, args.tag)
            t0 = time.time()
            if name.startswith("ceiling"):  # the ceiling imagines with the task's *actual* library (A or B)
                ceilings = {lib: A.Planner(A.TrueLibraryPredictor(gen.library(lib)), prior, name, depth=agent.depth) for lib in ("A", "B")}
                outs = [play(ceilings[t.library], t, i, runs=args.runs, steps=args.steps_budget) for i, t in enumerate(tasks)]
            elif "online" in name and args.stream_reset:
                # every stream (block of 40 tasks = one slice) starts from the trained state
                block = gen.load_spec()["slices"]["block_size"]
                outs = []
                for i, t in enumerate(tasks):
                    if i % block == 0:
                        agent = make_agent(name, seed, args.train_steps, prior, device, args.tag)
                    outs.append(play(agent, t, i, runs=args.runs, steps=args.steps_budget))
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


def adapt_cmd(args: argparse.Namespace) -> int:
    """Adaptation, retention and plasticity on the ``adapt`` range (development identities).

    For each initialization and each library_B stream (40 tasks) paired with an in_distribution
    (library A) probe stream from the same range:

    * ``online`` / ``frozen``: WM-S starting from the trained table; per-position success on the B stream.
    * ``retention``: after the online agent has adapted through the B stream, its library model is
      frozen and scored on the A probe stream, against the unadapted frozen model on the same probe.
    * ``fresh``: an online WM-S that starts from an *empty* library (a fresh learner) on the same B stream.
      Experienced minus fresh = what prior learning contributes after the switch.
    """

    import copy

    from . import structured as S

    tasks = dev_tasks("adapt")
    prior = A.Prior(A.fit_prior(gen.pool("train", args.train_tasks)))
    block = gen.load_spec()["slices"]["block_size"]
    streams: dict[str, list[list[gen.OpaqueTask]]] = defaultdict(list)
    for k in range(0, len(tasks), block):
        streams[tasks[k].slice].append(tasks[k : k + block])
    pairs = list(zip(streams["library_B"], streams["in_distribution"], strict=False))
    doc: dict[str, Any] = {"schema": "aaa.wm.opaque.adapt.v1", "status": "development", "provenance": provenance(), "seeds": {}}
    for seed in [int(x) for x in args.seeds.split(",")]:
        base = wms_table(seed, 200000)
        rows = []
        for b_stream, a_stream in pairs:
            rec: dict[str, Any] = {"b_tasks": [t.task_id for t in b_stream], "a_tasks": [t.task_id for t in a_stream]}
            for mode in ("online", "frozen", "fresh"):
                model = copy.deepcopy(base) if mode != "fresh" else S.LibraryModel(base.names)
                pred = S.StructuredPredictor(model, online=mode != "frozen")
                agent = A.Planner(pred, prior, mode, depth=2)
                if mode != "frozen":
                    agent.on_observation = lambda obs, view, _p=pred: _p.observe(obs, view)  # type: ignore[method-assign]
                rec[mode] = "".join("1" if play(agent, t, i).success else "0" for i, t in enumerate(b_stream))
                if mode == "online":
                    adapted = model
            for label, model in (("retention_adapted", adapted), ("retention_unadapted", copy.deepcopy(base))):
                agent = A.Planner(S.StructuredPredictor(model, online=False), prior, label, depth=2)
                rec[label] = "".join("1" if play(agent, t, i).success else "0" for i, t in enumerate(a_stream))
            rows.append(rec)
            print(seed, {k: v.count("1") for k, v in rec.items() if isinstance(v, str) and set(v) <= {"0", "1"}}, flush=True)
        doc["seeds"][str(seed)] = rows
    Path(args.output).write_text(json.dumps(doc))
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
    e.add_argument("--no-stream-reset", dest="stream_reset", action="store_false", help="(tune-era behavior) online learning carries across streams")
    ad = sub.add_parser("adapt")
    ad.add_argument("--seeds", default="0")
    ad.add_argument("--train-tasks", type=int, default=6000)
    ad.add_argument("--output", required=True)
    j = sub.add_parser("jepa")
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--steps", type=int, default=20000)
    j.add_argument("--lam-j", type=float, default=1.0)
    j.add_argument("--vic", type=float, default=1.0)
    j.add_argument("--train-tasks", type=int, default=6000)
    j.add_argument("--transition-tasks", type=int, default=6000)
    j.add_argument("--tag", default="_jepa")
    args = ap.parse_args(argv)
    if args.cmd == "jepa":
        return jepa_cmd(args)
    if args.cmd == "adapt":
        return adapt_cmd(args)
    return train_cmd(args) if args.cmd == "train" else evaluate_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
