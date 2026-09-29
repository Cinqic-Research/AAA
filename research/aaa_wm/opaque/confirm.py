"""Confirmation for ``aaa.python.opaque.v0``: fresh identities, observed once, under a committed freeze.

``python -m research.aaa_wm.opaque.confirm --output docs/evidence/aaa_wm_opaque_v0/confirmation.json``

1. :class:`~.freeze.Admission` is obtained and re-verified (committed manifest, clean tree,
   matching source fingerprint and specification). Nothing runs otherwise.
2. Confirmation tasks are generated for the first time here: ``confirmation`` indices
   ``[0, n)``, with every reference or buggy program that appears in pilot, train, development or
   attack replaced by a later attempt.
3. Every declared arm is played at every declared initialization. Neural arms load the checkpoints
   named in the manifest; their fingerprints must match the recorded ones.
4. Each declared ``aaa.promotion.crossed.v1`` contract is adjudicated from the primitives
   (Jeffreys-smoothed error per initialization x stream cell). The primary and independent
   implementations must agree.
5. The document records the bits for every (arm, initialization, task), the task hashes, the
   admission manifest and provenance, so the result can be recomputed without re-running anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from . import agents as A
from . import generator as gen
from .env import play
from .freeze import Admission


def confirmation_tasks(admission: Admission, n: int) -> list[gen.OpaqueTask]:
    if not admission.verify():
        raise gen.ConfirmationNotAdmitted("admission failed verification")
    spec = gen.load_spec()
    earlier = gen.pool("pilot") + gen.pool("train") + gen.pool("development") + gen.pool("attack")
    exclude = {gen.program_hash(t.reference) for t in earlier} | {gen.program_hash(t.buggy) for t in earlier}
    out = []
    for i in range(n):
        for attempt in range(500):
            t = gen.draft("confirmation", i, attempt)
            if t is None or gen.program_hash(t.reference) in exclude or gen.program_hash(t.buggy) in exclude:
                continue
            out.append(t)
            break
        else:
            raise RuntimeError(f"no confirmation task {i}")
    assert spec["slices"]["block_size"] == 40
    return out


def run(admission: Admission, output: Path) -> dict[str, Any]:
    from . import experiment as E
    from .summarize import BLOCK

    m = admission.manifest
    design = m["design"]
    verify_artifacts(m)
    tasks = confirmation_tasks(admission, design["tasks"])
    prior = A.Prior(A.fit_prior(gen.pool("train", 6000)))
    slices = [t.slice for t in tasks]
    doc: dict[str, Any] = {
        "schema": "aaa.wm.opaque.confirmation.v1",
        "status": "CONFIRMATION (fresh identities, observed once)",
        "freeze": m,
        "provenance": E.provenance(),
        "tasks": [t.task_id for t in tasks],
        "task_sha256": [hashlib.sha256(json.dumps(t.to_json(), sort_keys=True, default=str).encode()).hexdigest() for t in tasks],
        "slices": slices,
        "results": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    for arm in m["arms"]:
        per: dict[str, Any] = {}
        for seed in arm["initializations"]:
            if not admission.verify():
                raise gen.ConfirmationNotAdmitted("the source changed during confirmation")
            outs = []
            agent = None
            ceilings = {lib: A.Planner(A.TrueLibraryPredictor(gen.library(lib)), prior, "ceiling", depth=2) for lib in ("A", "B")}
            for i, t in enumerate(tasks):
                if arm["name"] == "ceiling":  # imagines with the task's actual library
                    outs.append(play(ceilings[t.library], t, i))
                    continue
                if agent is None or ("online" in arm["name"] and i % BLOCK == 0):
                    agent = E.make_agent(arm["name"], seed, arm.get("train_steps", 20000), prior, device, arm.get("tag", ""))
                outs.append(play(agent, t, i))
            per[str(seed)] = {"bits": "".join("1" if o.success else "0" for o in outs), "runs": [o.runs for o in outs]}
            print(arm["name"], seed, sum(o.success for o in outs) / len(outs), flush=True)
        doc["results"][arm["name"]] = per
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc["adjudications"] = adjudicate_all(doc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc))
    return doc


def verify_artifacts(manifest: dict[str, Any]) -> None:
    """Every learned artifact the confirmation loads must match the hash recorded in the freeze."""

    from .experiment import data_root

    for name, digest in manifest["artifacts_sha256"].items():
        path = (data_root() / "wms" / name) if name.startswith("table_") else (data_root() / "ckpt" / name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise gen.ConfirmationNotAdmitted(f"artifact {name} differs from the frozen hash")


def adjudicate_all(doc: dict[str, Any]) -> dict[str, Any]:
    """Every declared contract, recomputed from the document's own bits (usable for re-audit)."""

    import numpy as np

    from .summarize import adjudicate_ratio

    arms = {a: {s: np.array([c == "1" for c in r["bits"]], dtype=float) for s, r in per.items()} for a, per in doc["results"].items()}
    out = {}
    for c in doc["freeze"]["contracts"]:
        seeds = [str(s) for s in c["initializations"]]
        ref = {s: arms[c["reference"]][s if s in arms[c["reference"]] else "0"] for s in seeds} if c.get("reference_deterministic") else arms[c["reference"]]
        local = {c["reference"]: ref, c["challenger"]: arms[c["challenger"]]}
        only = c["groups"] if isinstance(c.get("groups"), list) else None
        out[c["name"]] = adjudicate_ratio(local, doc["slices"], seeds, c["reference"], c["challenger"], c["threshold"], seed=c["seed"], rule=c["criterion"]["rule"], only=only)
    return out


def audit(doc: dict[str, Any]) -> list[str]:
    """Recompute every adjudication from the stored bits; list every stored verdict that disagrees."""

    fresh = adjudicate_all(doc)
    problems = []
    for name, stored in doc.get("adjudications", {}).items():
        if fresh.get(name, {}).get("verdict") != stored.get("verdict"):
            problems.append(f"{name}: stored {stored.get('verdict')} != recomputed {fresh.get(name, {}).get('verdict')}")
    missing = set(fresh) - set(doc.get("adjudications", {}))
    problems += [f"{m}: not stored" for m in sorted(missing)]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run(Admission.obtain(), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
