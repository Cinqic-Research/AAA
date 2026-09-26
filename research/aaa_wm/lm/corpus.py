"""A reproducible American-English corpus for the from-scratch AAA language model (``aaa.wm.lm.corpus.v1``).

Pipeline, per source, all outputs under ``$AAA_DATA_ROOT/lm``:

1. **read** documents from the downloaded raw files. ``raw/manifest_downloads.txt`` holds the
   URL, retrieval time, SHA-256 and byte count of every file;
2. **normalize**: Unicode NFC, drop control characters, unify newlines, collapse runs of blank
   lines, truncate at ``MAX_CHARS``;
3. **filter** (each rejection is counted by reason): too short; low English-likeness
   (letter ratio, function-word ratio); **dialect**: documents whose British-only spellings
   outnumber American-only spellings, with at least two British hits, are dropped. The lexicon
   (:data:`VARIANTS`) was written for this project. It has no third-party license, and it is
   hashed into the corpus identity;
4. **deduplicate**: exact (normalized text hash), then near-duplicate by MinHash (64
   permutations over word 5-shingles, LSH 16 bands x 4 rows, estimated Jaccard >= 0.8
   against any earlier kept document of any source);
5. **split** by a hash of ``source:doc_id``: 98% train, 1% development, 1% test;
6. **write** UTF-8 shards (``NUL``-separated documents; NUL never survives normalization) and
   ``manifest.json`` (sources, licenses, counts, bytes, rejection reasons, lexicon hash and
   output hashes).

Contamination against AAA language-conditioned evaluation text is checked separately
(:func:`contamination`) before any evaluation that uses such text.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import time
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "aaa.wm.lm.corpus.v1"
MAX_CHARS = 20000
MIN_CHARS = 200

# (American, British) pairs. Only unambiguous spelling variants; words that are valid in both
# dialects with different meanings (e.g. "tire"/"tyre", "check"/"cheque") are excluded.
_PAIRS = """
color colour|colors colours|colored coloured|favorite favourite|favorites favourites|honor honour
honors honours|honored honoured|labor labour|neighbor neighbour|neighbors neighbours|neighborhood neighbourhood
behavior behaviour|behaviors behaviours|humor humour|flavor flavour|flavors flavours|harbor harbour
rumor rumour|vapor vapour|vigor vigour|endeavor endeavour|savior saviour|splendor splendour|odor odour
center centre|centers centres|centered centred|theater theatre|theaters theatres|meter metre|meters metres
liter litre|liters litres|fiber fibre|fibers fibres|caliber calibre|somber sombre|luster lustre
organize organise|organized organised|organizing organising|organization organisation|organizations organisations
realize realise|realized realised|realizing realising|recognize recognise|recognized recognised|recognizing recognising
analyze analyse|analyzed analysed|analyzing analysing|paralyze paralyse|catalyze catalyse
apologize apologise|apologized apologised|emphasize emphasise|emphasized emphasised|minimize minimise
maximize maximise|summarize summarise|summarized summarised|criticize criticise|criticized criticised
authorize authorise|authorized authorised|characterize characterise|characterized characterised
civilization civilisation|memorize memorise|modernize modernise|optimize optimise|optimized optimised
prioritize prioritise|specialize specialise|specialized specialised|standardize standardise|utilize utilise
defense defence|offense offence|license licence|licensed licenced|pretense pretence
catalog catalogue|catalogs catalogues|dialog dialogue|analog analogue|program programme|programs programmes
traveled travelled|traveling travelling|traveler traveller|travelers travellers|canceled cancelled
canceling cancelling|labeled labelled|labeling labelling|modeled modelled|modeling modelling
fueled fuelled|leveled levelled|signaled signalled|jewelry jewellery|counselor counsellor|enrollment enrolment
fulfill fulfil|skillful skilful|willful wilful|installment instalment|aluminum aluminium|gray grey
plow plough|mold mould|molded moulded|smolder smoulder|pajamas pyjamas|mustache moustache|skeptic sceptic
skeptical sceptical|maneuver manoeuvre|estrogen oestrogen|anemia anaemia|anesthesia anaesthesia
pediatric paediatric|encyclopedia encyclopaedia|fetus foetus|diarrhea diarrhoea|esophagus oesophagus
cozy cosy|draft draught|ax axe|gauge gage|aging ageing|judgment judgement|acknowledgment acknowledgement
""".replace("\n", "|")
VARIANTS: dict[str, str] = {}
for _pair in _PAIRS.split("|"):
    if _pair.strip():
        _us, _uk = _pair.split()
        VARIANTS[_us], VARIANTS[_uk] = "US", "UK"
# "program" is also standard British for software; "draft", "gauge", "ax", "judgment" are ambiguous:
for _w in ("program", "programs", "programme", "programmes", "draft", "draught", "gage", "gauge", "ax", "axe", "judgment", "judgement"):
    VARIANTS.pop(_w, None)
FUNCTION_WORDS = frozenset(
    "the of and to a in is that for it as was with be by on not he i this are or his from at which but have an they you were her she there been one all we their has would when if so no will can more who what its".split()
)
_WORD = re.compile(r"[A-Za-z]+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def lexicon_hash() -> str:
    return hashlib.sha256(json.dumps(sorted(VARIANTS.items())).encode()).hexdigest()


@dataclass(frozen=True)
class Source:
    name: str
    files: tuple[str, ...]
    license: str
    note: str
    max_train_bytes: int


SOURCES: tuple[Source, ...] = (
    Source("usgpo", ("usgpo-0000.json.gz", "usgpo-0002.json.gz"), "public domain (17 U.S.C. 105)", "common-pile/usgpo_filtered; card warns of possible metadata/license inaccuracies", 600_000_000),
    Source("fineweb_edu", ("fineweb-edu-10BT-000_00000.parquet",), "ODC-By 1.0 (+ Common Crawl terms of use)", "HuggingFaceFW/fineweb-edu sample-10BT", 600_000_000),
    Source("cosmopedia", ("cosmopedia-stories-00000.parquet", "cosmopedia-stories-00001.parquet", "cosmopedia-wikihow-00000.parquet"), "Apache-2.0", "HuggingFaceTB/cosmopedia (synthetic, Mixtral-8x7B-Instruct)", 400_000_000),
    Source("simplewiki", ("simplewiki-20231101.parquet",), "CC BY-SA 3.0 / GFDL (attribution: Wikipedia contributors)", "wikimedia/wikipedia 20231101.simple", 300_000_000),
    Source("pydocs", ("python-3.12-docs-text.tar.bz2",), "PSF License v2 (examples PSF-2.0 / 0BSD)", "docs.python.org 3.12 text archive", 50_000_000),
)


# ------------------------------------------------------------------ reading
def _read(src: Source, raw: Path) -> Iterator[tuple[str, str]]:
    for fname in src.files:
        path = raw / fname
        if fname.endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    rec = json.loads(line)
                    yield f"{fname}:{rec.get('id', i)}", rec.get("text", "")
        elif fname.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            cols = [c for c in ("id", "text") if c in pf.schema_arrow.names]
            k = 0
            for batch in pf.iter_batches(batch_size=2048, columns=cols):
                d = batch.to_pydict()
                for j, text in enumerate(d["text"]):
                    yield f"{fname}:{d['id'][j] if 'id' in d else k}", text or ""
                    k += 1
        elif fname.endswith(".tar.bz2"):
            with tarfile.open(path, "r:bz2") as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.endswith(".txt"):
                        fh = tar.extractfile(m)
                        if fh is not None:
                            yield f"{fname}:{m.name}", fh.read().decode("utf-8", "replace")
        else:
            raise ValueError(fname)


# ------------------------------------------------------------------ filtering
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_CHARS]


def dialect_counts(text: str) -> tuple[int, int]:
    us = uk = 0
    for w in _WORD.findall(text):
        v = VARIANTS.get(w.lower())
        if v == "US":
            us += 1
        elif v == "UK":
            uk += 1
    return us, uk


def verdict(text: str) -> str | None:
    """``None`` to keep, otherwise the rejection reason."""

    if len(text) < MIN_CHARS:
        return "short"
    letters = sum(c.isascii() and c.isalpha() for c in text)
    if letters / max(1, len(text)) < 0.55:
        return "not_prose"
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 30 or sum(w in FUNCTION_WORDS for w in words) / len(words) < 0.18:
        return "not_english"
    us, uk = dialect_counts(text)
    if uk >= 2 and uk > us:
        return "british_spelling"
    return None


# ------------------------------------------------------------------ dedup
_MASK = (1 << 61) - 1
_rng = np.random.default_rng(20260926)
_A = _rng.integers(1, _MASK, 64, dtype=np.uint64)
_B = _rng.integers(0, _MASK, 64, dtype=np.uint64)


def minhash(text: str) -> np.ndarray:
    words = _WORD.findall(text.lower())
    shingles = {" ".join(words[i : i + 5]) for i in range(max(1, len(words) - 4))}
    h = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") & _MASK for s in shingles], dtype=np.uint64)
    return ((h[:, None] * _A[None, :] + _B[None, :]) % np.uint64(_MASK)).min(axis=0)


class NearDup:
    def __init__(self, bands: int = 16, rows: int = 4) -> None:
        self.bands, self.rows = bands, rows
        self.buckets: list[dict[bytes, int]] = [dict() for _ in range(bands)]
        self.sigs: list[np.ndarray] = []

    def seen(self, sig: np.ndarray, threshold: float = 0.8) -> bool:
        cands = set()
        keys = []
        for b in range(self.bands):
            key = sig[b * self.rows : (b + 1) * self.rows].tobytes()
            keys.append(key)
            if key in self.buckets[b]:
                cands.add(self.buckets[b][key])
        if any(float(np.mean(self.sigs[c] == sig)) >= threshold for c in cands):
            return True
        idx = len(self.sigs)
        self.sigs.append(sig)
        for b, key in enumerate(keys):
            self.buckets[b].setdefault(key, idx)
        return False


def split_of(source: str, doc_id: str) -> str:
    v = int.from_bytes(hashlib.sha256(f"{source}:{doc_id}".encode()).digest()[:4], "big") % 100
    return "dev" if v == 0 else "test" if v == 1 else "train"


# ------------------------------------------------------------------ build
def build(root: Path, sources: tuple[Source, ...] = SOURCES) -> dict[str, Any]:
    raw, out = root / "raw", root / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    exact: set[bytes] = set()
    near = NearDup()
    manifest: dict[str, Any] = {"version": VERSION, "lexicon_sha256": lexicon_hash(), "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {}}
    downloads = (raw / "manifest_downloads.txt").read_text() if (raw / "manifest_downloads.txt").exists() else ""
    for src in sources:
        stats: Counter[str] = Counter()
        written = Counter()
        files = {s: open(out / f"{src.name}.{s}.txt", "wb") for s in ("train", "dev", "test")}
        dialect = Counter()
        for doc_id, text in _read(src, raw):
            stats["read"] += 1
            text = normalize(text)
            reason = verdict(text)
            if reason:
                stats[reason] += 1
                continue
            key = hashlib.sha1(" ".join(text.lower().split()).encode()).digest()
            if key in exact:
                stats["exact_duplicate"] += 1
                continue
            exact.add(key)
            if near.seen(minhash(text)):
                stats["near_duplicate"] += 1
                continue
            split = split_of(src.name, doc_id)
            if split == "train" and written["train"] >= src.max_train_bytes:
                stats["over_budget"] += 1
                continue
            data = text.encode("utf-8") + b"\x00"
            files[split].write(data)
            written[split] += len(data)
            stats[f"kept_{split}"] += 1
            us, uk = dialect_counts(text)
            dialect["us_hits"] += us
            dialect["uk_hits"] += uk
            if written["train"] >= src.max_train_bytes and stats["over_budget"] > 20000:
                break
        for f in files.values():
            f.close()
        manifest["sources"][src.name] = {
            "files": {
                fn: next((line for line in downloads.splitlines() if line.split(" ")[3:4] == [fn]), "missing from download manifest")
                for fn in src.files
            },
            "license": src.license,
            "note": src.note,
            "counts": dict(stats),
            "bytes": dict(written),
            "dialect_hits_kept": dict(dialect),
            "output_sha256": {s: hashlib.sha256((out / f"{src.name}.{s}.txt").read_bytes()).hexdigest() for s in ("train", "dev", "test")},
        }
        print(src.name, dict(stats), dict(written), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def contamination(corpus_dir: Path, texts: list[str], n: int = 13) -> dict[str, Any]:
    """Documents in the train shards that share a word ``n``-gram with any of ``texts``."""

    grams = set()
    for t in texts:
        w = t.lower().split()
        grams.update(" ".join(w[i : i + n]) for i in range(len(w) - n + 1))
    hits = Counter()
    for f in sorted(corpus_dir.glob("*.train.txt")):
        with open(f, "rb") as fh:
            for doc in io.BytesIO(fh.read()).read().split(b"\x00"):
                w = doc.decode("utf-8", "replace").lower().split()
                if any(" ".join(w[i : i + n]) in grams for i in range(len(w) - n + 1)):
                    hits[f.name] += 1
    return {"n": n, "eval_texts": len(texts), "contaminated_train_docs": dict(hits)}


if __name__ == "__main__":
    print(json.dumps(build(Path(os.environ["AAA_DATA_ROOT"]) / "lm")["sources"], indent=1)[:4000])
    _ = bz2  # tar.bz2 handled by tarfile
