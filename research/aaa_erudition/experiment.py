"""Run streams under experimental conditions and retain their evidence.

A run writes one JSON document per (stream, condition, controller) with the
evaluator scores, every controller decision, the full adaptation lineage and
resource counts, plus the state store and the Language Model call cache it
used. ``--backend replay`` re-executes a recorded run with no model present;
the result must be byte-identical.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION
from .contracts import canonical_json
from .controllers import AlwaysAdapt, Heuristic, NeverAdapt
from .language import CachedBackend, CallCache, LlamaServer, ScriptedLM, load_profile
from .lifecycle import CONDITIONS, Controller, System
from .simulate import register_truth
from .store import StateStore
from .surrogate import SurrogateLM
from .toolshift import make_stream


def controller_for(name: str, model_path: Path | None) -> Controller:
    if name == "never":
        return NeverAdapt()
    if name == "always":
        return AlwaysAdapt()
    if name == "heuristic":
        return Heuristic()
    if name == "erudition":
        from .erudition import EruditionController

        if model_path is None:
            raise SystemExit("--model is required for the erudition controller")
        return EruditionController.load(model_path)
    raise SystemExit(f"unknown controller {name!r}")


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_one(
    split: str,
    index: int,
    condition: str,
    controller: Controller,
    backend: Any,
    out_dir: Path,
    *,
    admitted: bool = False,
) -> dict[str, Any]:
    stream = make_stream(split, index, admitted=admitted)
    if isinstance(backend, SurrogateLM):
        register_truth(backend, stream)
    name = f"{split}-{index:05d}-{condition}-{controller.name}"
    store_dir = out_dir / "stores" / name
    if store_dir.exists():
        raise SystemExit(f"{store_dir} exists; runs never overwrite retained state")
    started = time.time()
    system = System(stream, backend, controller, condition, StateStore(store_dir))
    summary = system.run()
    summary.update(
        {
            "schema": "aaa.erudition.run.v1",
            "protocol": PROTOCOL_VERSION,
            "split": split,
            "index": index,
            "backend": backend.backend_id,
            "store_problems": system.store.verify(),
        }
    )
    path = out_dir / f"{name}.json.gz"
    with gzip.GzipFile(path, "wb", mtime=0) as stream_out:
        stream_out.write(canonical_json(summary))
    timing = {"wall_seconds": time.time() - started, "lm_seconds": system.cost["seconds"]}
    (out_dir / f"{name}.timing.json").write_bytes(canonical_json(timing) + b"\n")
    summary["wall_seconds"] = timing["wall_seconds"]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_erudition.experiment")
    parser.add_argument("--split", required=True, choices=("train", "development", "attack", "confirmation"))
    parser.add_argument("--freeze", type=Path, help="committed freeze manifest; required for confirmation")
    parser.add_argument("--indices", required=True, help="comma-separated stream indices")
    parser.add_argument("--conditions", default="frozen,lm_only,wm_only,joint")
    parser.add_argument("--controller", default="heuristic")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--resume", action="store_true", help="skip runs whose result file already exists")
    parser.add_argument("--backend", choices=("llama", "replay", "scripted", "surrogate"), default="llama")
    parser.add_argument("--url", default="http://127.0.0.1:18741")
    parser.add_argument("--key", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    admitted = False
    if args.split == "confirmation":
        from .identity import admit

        if args.backend != "llama":
            raise SystemExit("confirmation runs only against the real model runtime")

        if args.freeze is None or args.model is None:
            raise SystemExit("confirmation requires --freeze and the frozen --model")
        freeze = admit(args.freeze, args.model)
        plan = freeze["confirmation"]
        requested = [int(i) for i in args.indices.split(",")]
        arms = {f"{c}/{'never' if c == 'frozen' else args.controller}" for c in args.conditions.split(",")}
        if not set(requested) <= set(plan["indices"]) or not arms <= set(plan["arms"]):
            raise SystemExit("confirmation runs must be declared in the freeze")
        admitted = True
    args.out.mkdir(parents=True, exist_ok=True)
    if args.backend == "scripted":
        backend: Any = ScriptedLM()
    elif args.backend == "surrogate":
        if args.split == "confirmation":
            raise SystemExit("the surrogate is a development instrument; it never runs confirmation streams")
        backend = SurrogateLM()
    else:
        if args.cache is None:
            raise SystemExit("--cache is required for the llama and replay backends")
        inner = None
        if args.backend == "llama":
            if args.key is None:
                raise SystemExit("--key is required for the llama backend")
            inner = LlamaServer(args.url, args.key)
        backend = CachedBackend(CallCache(args.cache), inner, backend_id=None)
    manifest = {
        "schema": "aaa.erudition.run_manifest.v1",
        "protocol": PROTOCOL_VERSION,
        "git_head": _git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "profile": load_profile()["profile"]["artifact"],
        "argv": sys.argv,
    }
    (args.out / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    for index in (int(i) for i in args.indices.split(",")):
        for condition in args.conditions.split(","):
            if condition not in CONDITIONS:
                raise SystemExit(f"unknown condition {condition!r}")
            controller = controller_for("never" if condition == "frozen" else args.controller, args.model)
            name = f"{args.split}-{index:05d}-{condition}-{controller.name}"
            if args.resume and (args.out / f"{name}.json.gz").exists():
                continue
            summary = run_one(args.split, index, condition, controller, backend, args.out, admitted=admitted)
            scores = summary["scores"]
            calls = getattr(backend, "calls", 0)
            print(
                json.dumps(
                    {
                        "stream": summary["stream"],
                        "family": summary["family"],
                        "condition": condition,
                        "success": round(sum(s["success"] for s in scores) / len(scores), 3),
                        "accepted": sum(r["kind"] == "accepted" for r in summary["lineage"]),
                        "rejected": sum(r["kind"] == "rejected" for r in summary["lineage"]),
                        "rollbacks": sum(r["kind"] == "rollback" for r in summary["lineage"]),
                        "new_lm_calls": calls,
                        "wall_seconds": round(summary["wall_seconds"], 1),
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
