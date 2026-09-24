#!/usr/bin/env python3
"""Fail closed for intentional AAA work on FLOWBOX when its HDD is unavailable."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

LABEL = "Cinqic Storage"


def find_storage(devices: list[dict], mounts: list[dict]) -> Path:
    """Match a mounted ext4 filesystem by live UUID, never by a path alone."""
    matches: list[dict] = []

    def walk(items: list[dict]) -> None:
        for item in items:
            if item.get("label") == LABEL and item.get("fstype") == "ext4":
                matches.append(item)
            walk(item.get("children") or [])

    walk(devices)
    if len(matches) != 1 or not matches[0].get("uuid"):
        raise ValueError(f"expected exactly one ext4 filesystem labelled {LABEL!r}")
    uuid = matches[0]["uuid"]
    targets: list[str] = []

    def walk_mounts(items: list[dict]) -> None:
        for item in items:
            if item.get("uuid") == uuid and item.get("fstype") == "ext4":
                targets.append(item["target"])
            walk_mounts(item.get("children") or [])

    walk_mounts(mounts)
    if len(targets) != 1:
        raise ValueError(f"{LABEL!r} ({uuid}) is not mounted exactly once")
    return Path(targets[0]).resolve()


def validate_paths(mount: Path, paths: list[Path]) -> None:
    for path in paths:
        if not path.resolve().is_relative_to(mount):
            raise ValueError(f"{path} is outside the mounted {LABEL} HDD at {mount}")


def _json_command(*command: str) -> dict:
    return json.loads(subprocess.check_output(command, text=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", action="append", type=Path, default=[], help="workspace or data path to validate"
    )
    parser.add_argument("--require-hdd", action="store_true", help="require FLOWBOX HDD even on another host")
    args = parser.parse_args(argv)
    if socket.gethostname().split(".")[0].upper() != "FLOWBOX" and not args.require_hdd:
        print("storage preflight: non-FLOWBOX host; local storage policy applies")
        return 0
    try:
        devices = _json_command("lsblk", "--json", "-o", "LABEL,UUID,FSTYPE")["blockdevices"]
        mounts = _json_command("findmnt", "--json", "-o", "TARGET,UUID,FSTYPE")["filesystems"]
        mount = find_storage(devices, mounts)
        validate_paths(mount, args.path or [Path.cwd()])
        if not os.access(mount, os.W_OK | os.X_OK):
            raise ValueError(f"{LABEL!r} is not writable at {mount}")
        with tempfile.NamedTemporaryFile(dir=mount, prefix=".aaa-preflight-", delete=True) as probe:
            probe.write(b"AAA storage preflight\n")
        print(f"storage preflight: {mount}; free bytes: {shutil.disk_usage(mount).free}")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"storage preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
