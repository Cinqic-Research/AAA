"""Verify that the installed environment matches requirements-lock.txt.

CI publishes a lock file and then has to prove it actually used it. This check
fails when an installed distribution drifts from the pinned version, so a green
tick on the locked job means the locked environment really was exercised.
"""

from __future__ import annotations

import hashlib
import sys
from importlib import metadata
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "requirements-lock.txt"


def parse(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise SystemExit(f"{path}: every requirement must be pinned with '==': {raw!r}")
        name, version = line.split("==", 1)
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def main() -> int:
    if not LOCK.exists():
        print(f"missing lock file: {LOCK}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    print(f"dependency lock hash: {digest}")

    pins = parse(LOCK)
    problems: list[str] = []
    for name, expected in sorted(pins.items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name}: pinned {expected}, not installed")
            continue
        if installed != expected:
            problems.append(f"{name}: pinned {expected}, installed {installed}")
        else:
            print(f"  ok {name}=={installed}")
    if problems:
        print("\nthe installed environment does not match the lock file:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\nall {len(pins)} pinned distributions match the installed environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
