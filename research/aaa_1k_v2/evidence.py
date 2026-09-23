"""Strict-JSON evidence for aaa.1k.v2: RFC 8259, no NaN or Infinity, never silently overwritten."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sanitize(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and non-finite floats (to ``None``)."""

    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _refuse_constants(token: str) -> Any:
    raise ValueError(f"non-standard JSON constant {token!r} in evidence")


def dumps(payload: Any, *, indent: int | None = 1) -> str:
    return json.dumps(sanitize(payload), indent=indent, sort_keys=True, allow_nan=False) + "\n"


def write(path: str | Path, payload: Any, *, overwrite: bool = False, indent: int | None = 1) -> str:
    """Write strict JSON (gzip if ``.gz``); returns the SHA-256 of the bytes written."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; evidence is never overwritten silently")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = dumps(payload, indent=indent).encode("utf-8")
    if destination.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return hashlib.sha256(data).hexdigest()


def read(path: str | Path) -> Any:
    source = Path(path)
    data = source.read_bytes()
    if source.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data.decode("utf-8"), parse_constant=_refuse_constants)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
