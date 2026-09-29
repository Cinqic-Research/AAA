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
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_wm.lang import confirm  # noqa: E402
from research.aaa_wm.lang.experiment import items_for  # noqa: E402
from research.aaa_wm.lang.reports import CLAUSES, JOINERS, ONES, OPENERS, TENS  # noqa: E402
from research.aaa_wm.lm.corpus import contamination  # noqa: E402
from research.aaa_wm.opaque import generator as gen  # noqa: E402

WORDS = re.compile(rb"[a-z]+|[0-9]+")
NUMBER_WORDS = (
    frozenset(w.encode() for w in ONES)
    | frozenset(w.encode() for w in TENS.values())
    | {
        b"hundred",
        b"thousand",
        b"negative",
    }
)


def normalized(text: bytes) -> tuple[bytes, ...]:
    """Ignore punctuation and mask numeric spellings to test variant overlap."""

    words = WORDS.findall(text.lower())
    out: list[bytes] = []
    for word in words:
        if word.isdigit() or word in NUMBER_WORDS:
            if not out or out[-1] != b"#":
                out.append(b"#")
        else:
            out.append(word)
    return tuple(out)


def template_fragments() -> set[tuple[bytes, ...]]:
    """Static spans of held-out wording, separate from the task-specific numbers."""

    out: set[tuple[bytes, ...]] = set()
    for template in CLAUSES["heldout"] + OPENERS["heldout"] + JOINERS["heldout"]:
        for span in re.split(r"\{[xeg]\}", template):
            words = normalized(span.encode())
            if len(words) >= 5:
                out.add(words)
    return out


def extended_overlap(corpus: Path, texts: list[str]) -> dict[str, Any]:
    """Conservative 13-token normalized overlap plus held-out static phrases.

    A phrase hit alone is not contamination: generic English can occur naturally.
    Counts are retained so that a reviewer can inspect the strength of a match.
    """

    n = 13
    grams = {
        tuple(w[i : i + n])
        for text in texts
        for w in [normalized(text.encode())]
        for i in range(len(w) - n + 1)
    }
    phrases = template_fragments()
    phrase_bytes = {b" " + b" ".join(p) + b" ": p for p in phrases}
    by_source: dict[str, dict[str, int]] = {}
    phrase_hits: Counter[str] = Counter()
    for shard in sorted(corpus.glob("*.train.txt")):
        long_hits = 0
        static_hits = 0
        for doc in shard.read_bytes().split(b"\0"):
            words = normalized(doc)
            if any(tuple(words[i : i + n]) in grams for i in range(len(words) - n + 1)):
                long_hits += 1
            joined = b" " + b" ".join(words) + b" "
            found = {p for needle, p in phrase_bytes.items() if needle in joined}
            if found:
                static_hits += 1
                phrase_hits.update(" ".join(w.decode() for w in p) for p in found)
        by_source[shard.name] = {"normalized_13gram_docs": long_hits, "static_phrase_docs": static_hits}
    return {
        "normalization": "ASCII words/digits, lowercased, punctuation removed, digits and English number words masked",
        "n": n,
        "eval_texts": len(texts),
        "eval_13grams": len(grams),
        "heldout_static_fragments": len(phrases),
        "sources": by_source,
        "static_phrase_document_counts": dict(sorted(phrase_hits.items())),
    }


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


def audit(data_root: Path, *, extended: bool = False) -> dict[str, Any]:
    texts = task_texts()
    corpus = data_root / "lm" / "corpus"
    manifest = corpus / "manifest.json"
    downloads = data_root / "lm" / "raw" / "manifest_downloads.txt"
    if not manifest.is_file() or not downloads.is_file():
        raise FileNotFoundError("corpus or raw-download manifest is missing")
    result = contamination(corpus, texts, n=13)
    doc = {
        "status": "POST_RUN_AUDIT; not a pre-confirmation contamination gate",
        "language_freeze_commit": "a8c088dc250961ce15f49be3f9761814bff2ada6",
        "corpus_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "raw_download_manifest_sha256": hashlib.sha256(downloads.read_bytes()).hexdigest(),
        "reports_sha256": hashlib.sha256("\n\0\n".join(texts).encode()).hexdigest(),
        "overlap": result,
    }
    if extended:
        doc["extended_overlap"] = extended_overlap(corpus, texts)
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=os.environ.get("AAA_DATA_ROOT"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="also scan number and punctuation variants and held-out template fragments",
    )
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root or AAA_DATA_ROOT is required")
    doc = audit(args.data_root, extended=args.extended)
    encoded = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        if args.out.exists():
            if args.out.read_text() != encoded:
                raise SystemExit("the existing post-run audit differs from recomputation")
        else:
            args.out.write_text(encoded)
    print(encoded, end="")
    normalized_hits = sum(
        x["normalized_13gram_docs"] for x in doc.get("extended_overlap", {}).get("sources", {}).values()
    )
    return int(bool(doc["overlap"]["contaminated_train_docs"]) or bool(normalized_hits))


if __name__ == "__main__":
    raise SystemExit(main())
