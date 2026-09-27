"""Language-integration confirmation (``aaa.wm.lang.v0``): fresh identities, observed once, under its own freeze.

    python -m research.aaa_wm.lang.confirm freeze --reference ...   # writes the manifest (commit it next)
    python -m research.aaa_wm.lang.confirm run --output docs/evidence/aaa_wm_lang_v0/confirmation.json

**Identities.** ``opaque.v0`` ``confirmation`` indices ``[4000, 6000)``. The world-model confirmation
used ``[0, 4000)``, and those identities are spent; the indices here had never been generated. Every
program of pilot, train, development, attack and the first confirmation is excluded. Statements use
the **held-out** phrasing families only.

**Admission** mirrors ``opaque.freeze``: a committed manifest, a clean tree, a matching source
fingerprint (``research/aaa_wm/lang``, ``research/aaa_wm/opaque``, ``research/aaa_wm/lm/model.py``,
``aaa/promotion``), and matching SHA-256 hashes of every learned artifact (adapters, WM-S tables,
policy checkpoints) at the declared initializations. All of them were produced from training data only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from ..opaque import generator as gen

ROOT = Path(__file__).resolve().parents[3]
FREEZE_PATH = "docs/evidence/aaa_wm_lang_v0/freeze.json"
SCHEMA = "aaa.wm.lang.freeze.v1"
INITS = [100, 101, 102]
START, COUNT = 4000, 2000
LM_STEM = "bpe8192_d256_l4_s0_n6100"


class NotAdmitted(RuntimeError):
    pass


def data_root() -> Path:
    import os

    return Path(os.environ["AAA_DATA_ROOT"])


def adapter_path(seed: int, random_twin: bool) -> Path:
    return data_root() / "lm" / "adapters" / f"adapter_{LM_STEM}{'_random' if random_twin else ''}_s{seed}_n3000_v2.pt"


def artifacts() -> dict[str, Path]:
    out = {}
    for s in INITS:
        out[f"adapter_s{s}"] = adapter_path(s, False)
        out[f"adapter_random_s{s}"] = adapter_path(s, True)
        out[f"table_s{s}"] = data_root() / "opaque" / "wms" / f"table_s{s}_p200000.json"
        out[f"policy_aux_s{s}"] = data_root() / "opaque" / "ckpt" / f"policy_aux_s{s}_n20000.pt"
    out["bpe8192"] = data_root() / "lm" / "tokenizers" / "bpe8192.json"
    return out


def source_files() -> list[str]:
    files = []
    for d in ("research/aaa_wm/lang", "research/aaa_wm/opaque"):
        files += sorted(str(p.relative_to(ROOT)) for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return files + ["research/aaa_wm/lm/model.py", "aaa/promotion/contract.py", "aaa/promotion/primary.py", "aaa/promotion/independent.py", "aaa/promotion/adjudicate.py"]


def fingerprint() -> str:
    h = hashlib.sha256()
    for rel in source_files():
        h.update(rel.encode() + b"\0" + (ROOT / rel).read_bytes() + b"\0")
    return h.hexdigest()


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise NotAdmitted(r.stderr.strip())
    return r.stdout


def check() -> dict[str, Any]:
    path = ROOT / FREEZE_PATH
    if not path.is_file():
        raise NotAdmitted("no language freeze manifest")
    m = json.loads(path.read_text())
    if m.get("schema") != SCHEMA:
        raise NotAdmitted("schema mismatch")
    if _git("show", f"HEAD:{FREEZE_PATH}") != path.read_text():
        raise NotAdmitted("manifest not committed")
    if _git("status", "--porcelain").strip():
        raise NotAdmitted("dirty tree")
    if m["source_fingerprint"] != fingerprint():
        raise NotAdmitted("source changed since the freeze")
    for name, p in artifacts().items():
        if hashlib.sha256(p.read_bytes()).hexdigest() != m["artifacts_sha256"][name]:
            raise NotAdmitted(f"artifact {name} changed")
    return m


def contract(name: str, reference: str, challenger: str, threshold: float, rule: str = "superior") -> dict[str, Any]:
    return {"name": name, "reference": reference, "challenger": challenger, "initializations": INITS, "threshold": threshold, "seed": 20260927, "criterion": {"rule": rule}, "groups": "all"}


def write_freeze() -> Path:
    path = ROOT / FREEZE_PATH
    if path.exists():
        raise SystemExit("a freeze is never rewritten")
    m = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_fingerprint": fingerprint(),
        "source_files": source_files(),
        "artifacts_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in artifacts().items()},
        "design": {
            "identities": f"opaque.v0 confirmation indices [{START}, {START + COUNT}); earlier splits and the spent [0, 4000) excluded",
            "statements": "held-out phrasing families only (research/aaa_wm/lang/reports.py)",
            "initializations": INITS,
            "pairing": "adapter seed s, WM-S table seed s, policy_aux seed s",
            "language_model": LM_STEM + " (R1, selected on development by the pre-registered rule)",
            "runs_budget": 2,
            "steps_budget": 8,
        },
        "arms": [
            ["wms:online", "lm"], ["wms:online", "rules"], ["wms:online", "lm_random"], ["wms:online", "gold"],
            ["policy_aux:plan", "lm"], ["policy_aux:plan", "rules"],
        ],
        "contracts": [
            contract("F1_full_system_over_policy_with_LM", "policy_aux:plan@lm", "wms:online@lm", 0.90),
            contract("F2_full_system_over_policy_with_rules", "policy_aux:plan@rules", "wms:online@lm", 0.90),
            contract("L1_pretraining_end_to_end", "wms:online@lm_random", "wms:online@lm", 0.90),
            contract("L2_language_model_over_rules", "wms:online@rules", "wms:online@lm", 0.95),
        ],
        "success_rule": "LANGUAGE_INTEGRATION_SUCCESS (scoped) requires L1 and L2 PROMOTE; FULL_SYSTEM_SUCCESS (option A, full system superior to the pairwise systems measured) additionally requires F1 and F2 PROMOTE; agreeing implementations; independent review",
        "thresholds_rationale": "fixed from development effect sizes before any identity in [4000, 6000) existed; error-ratio bars: 10% fewer errors (F1, F2, L1), 5% (L2)",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1) + "\n")
    return path


def tasks(manifest: dict[str, Any]) -> list[gen.OpaqueTask]:
    check()
    earlier = gen.pool("pilot") + gen.pool("train") + gen.pool("development") + gen.pool("attack")
    exclude = {gen.program_hash(t.reference) for t in earlier} | {gen.program_hash(t.buggy) for t in earlier}
    for i in range(START):  # the spent world-model confirmation identities
        for attempt in range(500):
            t = gen.draft("confirmation", i, attempt)
            if t is not None and gen.program_hash(t.reference) not in exclude and gen.program_hash(t.buggy) not in exclude:
                exclude |= {gen.program_hash(t.reference), gen.program_hash(t.buggy)}
                break
    out = []
    for i in range(START, START + COUNT):
        for attempt in range(500):
            t = gen.draft("confirmation", i, attempt)
            if t is None or gen.program_hash(t.reference) in exclude or gen.program_hash(t.buggy) in exclude:
                continue
            out.append(t)
            break
    return out


def run(output: Path) -> dict[str, Any]:
    import numpy as np

    from ..opaque import agents as OA
    from ..opaque.env import play
    from ..opaque.experiment import make_agent, provenance
    from ..opaque.summarize import BLOCK, adjudicate_ratio
    from . import adapter as AD
    from . import rules
    from .experiment import LanguageAgent, codec_for, items_for

    m = check()
    ts = tasks(m)
    # slices: the first COUNT/BLOCK blocks cycle through the five slices exactly as in development
    items = items_for(ts, "heldout")
    prior = OA.Prior(OA.fit_prior(gen.pool("train", 6000)))
    doc: dict[str, Any] = {"schema": "aaa.wm.lang.confirmation.v1", "status": "CONFIRMATION (fresh identities, observed once)", "freeze": m, "provenance": provenance(), "tasks": [t.task_id for t in ts], "slices": [t.slice for t in ts], "results": {}, "channel_exact": {}}
    from ..lm.model import load as load_lm

    proposals: dict[str, dict[int, list[Any]]] = {"rules": {}, "gold": {}, "lm": {}, "lm_random": {}}
    for s in INITS:
        check()
        proposals["rules"][s] = [rules.extract(it["text"]) for it in items]
        proposals["gold"][s] = [[tuple(g) for g in it["gold"]] for it in items]
        for ch, rnd in (("lm", False), ("lm_random", True)):
            lm = load_lm(adapter_path(s, rnd), "cuda")
            proposals[ch][s] = AD.extract(lm, codec_for(lm), [it["text"] for it in items])
            del lm
    for ch, per in proposals.items():
        doc["channel_exact"][ch] = float(np.mean([[p is not None and [tuple(x) for x in p] == [tuple(g) for g in it["gold"]] for p, it in zip(per[s], items, strict=True)] for s in INITS]))
    for agent_name, ch in m["arms"]:
        key = f"{agent_name}@{ch}"
        doc["results"][key] = {}
        for s in INITS:
            check()
            outs = []
            inner = None
            for i, (t, prop) in enumerate(zip(ts, proposals[ch][s], strict=True)):
                if inner is None or ("online" in agent_name and i % BLOCK == 0):
                    inner = make_agent(agent_name, s, 20000, prior, "cuda")
                fb = OA.ToolSearch(lambda v, e: prior.score(e), "tool_prior")
                outs.append(play(LanguageAgent(inner, [tuple(x) for x in prop] if prop else None, fb), t, i))
            doc["results"][key][str(s)] = {"bits": "".join("1" if o.success else "0" for o in outs)}
            print(key, s, round(float(np.mean([o.success for o in outs])), 4), flush=True)
    arms = {k: {s: np.array([c == "1" for c in r["bits"]], dtype=float) for s, r in v.items()} for k, v in doc["results"].items()}
    doc["adjudications"] = {c["name"]: adjudicate_ratio(arms, doc["slices"], [str(s) for s in INITS], c["reference"], c["challenger"], c["threshold"], seed=c["seed"], rule=c["criterion"]["rule"]) for c in m["contracts"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    a = ap.parse_args()
    if a.cmd == "freeze":
        print(write_freeze())
    else:
        run(Path(a.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
