#!/usr/bin/env python3
"""Post-run corpus overlap audit for spent language identities; no model evaluation.

Reconstructs the exact frozen text-generation path, then checks whether any
training-corpus document shares a 13-word sequence with a confirmation report.
This is an audit of historical data, never a new confirmation or tuning input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_wm.lang import confirm  # noqa: E402
from research.aaa_wm.lang.experiment import items_for  # noqa: E402
from research.aaa_wm.lm.corpus import contamination  # noqa: E402
from research.aaa_wm.opaque import generator as gen  # noqa: E402


def task_texts() -> list[str]:
    freeze = json.loads((ROOT / confirm.FREEZE_PATH).read_text())
    if confirm.fingerprint() != freeze["source_fingerprint"]:
        raise ValueError("running source differs from the language confirmation freeze")
    earlier = gen.pool("pilot") + gen.pool("train") + gen.pool("development") + gen.pool("attack")
    excluded = {gen.program_hash(t.reference) for t in earlier} | {gen.program_hash(t.buggy) for t in earlier}
    tasks = []
    for index in range(confirm.START + confirm.COUNT):
        for attempt in range(500):
            task = gen.draft("confirmation", index, attempt)
            if task is None:
                continue
            hashes = {gen.program_hash(task.reference), gen.program_hash(task.buggy)}
            if hashes & excluded:
                continue
            if index < confirm.START:
                excluded.update(hashes)
            else:
                tasks.append(task)
            break
        else:
            raise ValueError(f"no reconstructible task for confirmation:{index}")
    if [t.task_id for t in tasks] != [
        f"aaa.python.opaque.v0:confirmation:{i}" for i in range(confirm.START, confirm.START + confirm.COUNT)
    ]:
        raise ValueError("confirmation task order differs from the freeze")
    return [item["text"] for item in items_for(tasks, "heldout")]


def audit(data_root: Path) -> dict[str, Any]:
    texts = task_texts()
    corpus = data_root / "lm" / "corpus"
    manifest = corpus / "manifest.json"
    downloads = data_root / "lm" / "raw" / "manifest_downloads.txt"
    if not manifest.is_file() or not downloads.is_file():
        raise FileNotFoundError("corpus or raw-download manifest is missing")
    result = contamination(corpus, texts, n=13)
    return {
        "status": "POST_RUN_AUDIT; not a pre-confirmation contamination gate",
        "language_freeze_commit": "a8c088dc250961ce15f49be3f9761814bff2ada6",
        "corpus_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "raw_download_manifest_sha256": hashlib.sha256(downloads.read_bytes()).hexdigest(),
        "reports_sha256": hashlib.sha256("\n\0\n".join(texts).encode()).hexdigest(),
        "overlap": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=os.environ.get("AAA_DATA_ROOT"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root or AAA_DATA_ROOT is required")
    doc = audit(args.data_root)
    encoded = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        if args.out.exists():
            if args.out.read_text() != encoded:
                raise SystemExit("the existing post-run audit differs from recomputation")
        else:
            args.out.write_text(encoded)
    print(encoded, end="")
    return int(bool(doc["overlap"]["contaminated_train_docs"]))


if __name__ == "__main__":
    raise SystemExit(main())
