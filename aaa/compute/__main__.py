"""``python -m aaa.compute``: probe the machine and inspect compute devices.

python -m aaa.compute probe --output benchmarks/hardware/flowbox.json
python -m aaa.compute info --device cuda
python -m aaa.compute info --device auto --cells 512
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .device import DEVICE_SPEC_GRAMMAR, DeviceUnavailableError, InvalidDeviceError, resolve_backend
from .hardware import probe, provenance_counts
from .provenance import backend_provenance


def _commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_probe(args: argparse.Namespace) -> int:
    profile = probe(commit=_commit())
    text = json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(f"wrote {destination}: {provenance_counts(profile)}")
    else:
        sys.stdout.write(text)
    return 0


def command_info(args: argparse.Namespace) -> int:
    try:
        backend = resolve_backend(args.device, cells=args.cells)
    except (InvalidDeviceError, DeviceUnavailableError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(backend_provenance(backend), indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aaa.compute", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", help="write the machine-readable hardware profile")
    p.add_argument("--output")
    p.set_defaults(func=command_probe)
    i = sub.add_parser("info", help="resolve a device and print its provenance")
    i.add_argument("--device", default="cpu", help=DEVICE_SPEC_GRAMMAR)
    i.add_argument("--cells", type=int, help="workload hint for --device auto")
    i.set_defaults(func=command_info)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
