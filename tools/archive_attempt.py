#!/usr/bin/env python3
"""Create a compact, clean-clone-verifiable archive of a benchmark attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

OMITTED_DIRECTORIES = frozenset({"raw", "plots"})


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def archive(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"archive destination already exists: {destination}")
    original = source / "checksums.json"
    if not original.is_file() or not (source / "summary.json").is_file():
        raise ValueError("source is not a completed benchmark attempt")
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or any(part in OMITTED_DIRECTORIES for part in relative.parts):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (destination / "checksums.json").replace(destination / "full_attempt_checksums.json")
    full_manifest = json.loads(original.read_text(encoding="utf-8"))
    dump(
        destination / "archive_manifest.json",
        {
            "schema_version": "aaa.compact_attempt_archive.v1",
            "full_attempt_checksum_entries": len(full_manifest),
            "omitted_directories": sorted(OMITTED_DIRECTORIES),
            "semantics": (
                "checksums.json verifies every retained archive byte; "
                "full_attempt_checksums.json describes the original complete local attempt"
            ),
        },
    )
    compact = {
        str(path.relative_to(destination)): digest(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    dump(destination / "checksums.json", compact)
    print(
        f"archived {source} -> {destination}: {len(compact)} retained files, "
        f"{len(full_manifest)} full-attempt checksum entries"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    archive(args.source, args.destination)


if __name__ == "__main__":
    main()
