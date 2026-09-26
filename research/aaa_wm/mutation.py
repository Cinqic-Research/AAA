"""Deliberate boundary mutations: each break must make at least one test fail (fail closed).

``python -m research.aaa_wm.mutation --out docs/wm_program/mutation_results.json``

Each mutation copies the repository's ``research/``, ``tests/`` and ``aaa/`` trees to a temporary
directory, applies one textual change, runs the named tests there, and records whether they
failed ("caught") or passed ("MISSED"). The original tree is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MUTATIONS = [
    {
        "id": "M1_view_exposes_reference",
        "what": "the agent's View carries the hidden reference program",
        "file": "research/aaa_wm/opaque/env.py",
        "old": "    api_names: tuple[str, ...]\n\n    def edits",
        "new": "    api_names: tuple[str, ...]\n    reference: str = \"\"\n\n    def edits",
        "tests": ["tests.test_aaa_wm_opaque.Boundary"],
    },
    {
        "id": "M2_run_uses_whole_domain",
        "what": "RUN executes the program on the hidden domain instead of the visible tests",
        "file": "research/aaa_wm/opaque/env.py",
        "old": "results = tuple(run(self._source, self._lib, x) for x, _ in self._task.visible_tests)",
        "new": "results = tuple(run(self._source, self._lib, x) for x in domain())",
        "tests": ["tests.test_aaa_wm_opaque.Boundary"],
    },
    {
        "id": "M3_confirmation_without_admission",
        "what": "confirmation identities can be generated without an admission",
        "file": "research/aaa_wm/opaque/generator.py",
        "old": "    if split == \"confirmation\" and admission is None:\n        raise ConfirmationNotAdmitted(\"opaque.v0 confirmation",
        "new": "    if False:\n        raise ConfirmationNotAdmitted(\"opaque.v0 confirmation",
        "tests": ["tests.test_aaa_wm_opaque.Generation"],
    },
    {
        "id": "M4_forged_admission_accepted",
        "what": "confirmation tasks are generated for an admission that fails verification",
        "file": "research/aaa_wm/opaque/confirm.py",
        "old": "    if not admission.verify():\n        raise gen.ConfirmationNotAdmitted(\"admission failed verification\")",
        "new": "    pass",
        "tests": ["tests.test_aaa_wm_opaque.Freeze"],
    },
    {
        "id": "M5_fingerprint_omits_generator",
        "what": "the frozen source fingerprint omits the task generator",
        "file": "research/aaa_wm/opaque/freeze.py",
        "old": "if p.is_file() and \"__pycache__\" not in p.parts)",
        "new": "if p.is_file() and \"__pycache__\" not in p.parts and p.name != \"generator.py\")",
        "tests": ["tests.test_aaa_wm_opaque.Freeze"],
    },
    {
        "id": "M6_imagination_marked_real",
        "what": "world-model imagination carries the REAL_OBSERVATION tag",
        "file": "research/aaa_wm/opaque/agents.py",
        "old": "IMAGINATION = \"WORLD_MODEL_IMAGINATION\"",
        "new": "IMAGINATION = \"REAL_OBSERVATION\"",
        "tests": ["tests.test_aaa_wm_opaque.Provenance"],
    },
    {
        "id": "M7_world_model_ignores_action",
        "what": "WM-S predicts every plan with the unedited program (action removed)",
        "file": "research/aaa_wm/opaque/structured.py",
        "old": "                r = self.model.predict(src, x)",
        "new": "                r = self.model.predict(view.source, x)",
        "tests": ["tests.test_aaa_wm_opaque.ActionConditioning"],
    },
    {
        "id": "M8_stored_verdict_not_recomputed",
        "what": "the audit trusts stored verdicts instead of recomputing from primitives",
        "file": "research/aaa_wm/opaque/confirm.py",
        "old": "    fresh = adjudicate_all(doc)",
        "new": "    fresh = doc.get(\"adjudications\", {})",
        "tests": ["tests.test_aaa_wm_opaque.Audit"],
    },
    {
        "id": "M9_checkpoint_fingerprint_skips_tensors",
        "what": "the checkpoint fingerprint ignores parameter values (tampering undetected)",
        "file": "research/aaa_wm/opaque/models.py",
        "old": "        h.update(t.detach().cpu().contiguous().numpy().tobytes())",
        "new": "        pass",
        "tests": ["tests.test_aaa_wm_models.Checkpoints"],
        "needs_torch": True,
    },
    {
        "id": "M10_library_learner_accepts_inconsistent",
        "what": "the abductive library learner records a value that does not reproduce the observation",
        "file": "research/aaa_wm/opaque/structured.py",
        "old": "                if self.execute(source, argument, fill={key: h}) == observed:",
        "new": "                if True:",
        "tests": ["tests.test_aaa_wm_models.StructuredModel"],
    },
    {
        "id": "M11_contamination_check_blind",
        "what": "the LM corpus contamination check never flags a shared n-gram",
        "file": "research/aaa_wm/lm/corpus.py",
        "old": "                if any(\" \".join(w[i : i + n]) in grams for i in range(len(w) - n + 1)):",
        "new": "                if False:",
        "tests": ["tests.test_aaa_wm_lm.Contamination"],
    },
    {
        "id": "M12_dialect_filter_disabled",
        "what": "British-spelled documents are kept in the American-English corpus",
        "file": "research/aaa_wm/lm/corpus.py",
        "old": "    if uk >= 2 and uk > us:",
        "new": "    if False:",
        "tests": ["tests.test_aaa_wm_lm.Dialect"],
    },
]


def run_one(m: dict, python: str) -> dict:
    with tempfile.TemporaryDirectory(dir=os.environ.get("AAA_DATA_ROOT")) as tmp:
        for d in ("research", "tests", "aaa"):
            shutil.copytree(ROOT / d, Path(tmp) / d, ignore=shutil.ignore_patterns("__pycache__"))
        target = Path(tmp) / m["file"]
        text = target.read_text()
        if m["old"] not in text:
            return {**m, "result": "PATCH_DID_NOT_APPLY"}
        target.write_text(text.replace(m["old"], m["new"], 1))
        r = subprocess.run([python, "-m", "unittest", *m["tests"]], cwd=tmp, capture_output=True, text=True, timeout=900)
        tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
        return {"id": m["id"], "what": m["what"], "tests": m["tests"], "result": "caught" if r.returncode != 0 else "MISSED", "last_line": tail[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--torch-python", default=sys.executable)
    a = ap.parse_args()
    results = [run_one(m, a.torch_python if m.get("needs_torch") else sys.executable) for m in MUTATIONS]
    for r in results:
        print(r["id"], r["result"], flush=True)
    Path(a.out).write_text(json.dumps({"schema": "aaa.wm.mutation.v1", "results": results}, indent=1) + "\n")
    return 0 if all(r["result"] == "caught" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
