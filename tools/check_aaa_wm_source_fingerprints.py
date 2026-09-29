#!/usr/bin/env python3
"""Verify the live world-model and language source against retained freezes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_wm.lang import confirm as language_confirm  # noqa: E402
from research.aaa_wm.opaque import freeze as opaque_freeze  # noqa: E402
from research.aaa_wm.opaque import generator as opaque_generator  # noqa: E402

OPAQUE_FREEZE = "docs/evidence/aaa_wm_opaque_v0/freeze.json"
LANGUAGE_FREEZE = "docs/evidence/aaa_wm_lang_v0/freeze.json"


def _strict_load(path: Path) -> dict[str, Any]:
    def reject(token: str) -> Any:
        raise ValueError(f"non-standard JSON constant {token}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key}")
            out[key] = value
        return out

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"freeze manifest is not an object: {path}")
    return value


def verify() -> list[str]:
    problems: list[str] = []

    try:
        opaque = _strict_load(ROOT / OPAQUE_FREEZE)
        live_opaque = opaque_freeze.fingerprint(ROOT)
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"opaque freeze/source could not be read: {exc}")
    else:
        if opaque.get("schema") != opaque_freeze.SCHEMA:
            problems.append("opaque freeze schema differs")
        if opaque.get("source_fingerprint") != live_opaque["sha256"]:
            problems.append("opaque source fingerprint differs from the committed freeze")
        if opaque.get("source_files") != live_opaque["files"]:
            problems.append("opaque source-file hashes differ from the committed freeze")
        if opaque.get("spec_sha256") != opaque_generator.spec_hash():
            problems.append("opaque specification hash differs from the committed freeze")

    try:
        language = _strict_load(ROOT / LANGUAGE_FREEZE)
        live_language_paths = language_confirm.source_files()
        live_language_fingerprint = language_confirm.fingerprint()
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"language freeze/source could not be read: {exc}")
    else:
        if language.get("schema") != language_confirm.SCHEMA:
            problems.append("language freeze schema differs")
        if language.get("source_files") != live_language_paths:
            problems.append("language source-file list differs from the committed freeze")
        if language.get("source_fingerprint") != live_language_fingerprint:
            problems.append("language source fingerprint differs from the committed freeze")

    return problems


def main() -> int:
    problems = verify()
    if problems:
        for problem in problems:
            print(f"DRIFT: {problem}", file=sys.stderr)
        return 1
    print("opaque and language frozen source fingerprints: UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
