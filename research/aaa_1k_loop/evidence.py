"""Strict-JSON evidence I/O for the loop.

RFC 8259 JSON has no ``NaN`` or ``Infinity``. Python's encoder and decoder
accept both by default, which is how `AAA-161` happened. The loop writes with
``allow_nan=False`` and reads with a constant hook that refuses them, so a
non-standard token can neither be produced nor silently consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """Raised when an evidence artifact is not strict, finite, standards JSON."""


def _refuse_constant(token: str) -> Any:
    raise EvidenceError(f"non-standard JSON constant {token!r} is not evidence")


def read_strict_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_refuse_constant)
    except json.JSONDecodeError as error:
        raise EvidenceError(f"{path}: not valid JSON: {error}") from error


def finite_or_none(value: float) -> float | None:
    """The only sanctioned way to record a quantity that may be undefined."""

    number = float(value)
    return number if math.isfinite(number) else None


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def write_strict_json(path: Path, payload: Any, *, overwrite: bool = True) -> str:
    """Write ``payload`` atomically; refuse NaN/Infinity; return the file's sha256."""

    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as error:
        raise EvidenceError(f"refusing to write non-finite evidence to {path}: {error}") from error
    if path.exists() and not overwrite:
        raise EvidenceError(f"{path} already exists and this artifact may not be overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_keys(record: Mapping[str, Any], keys: tuple[str, ...], *, context: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise EvidenceError(f"{context} is missing {missing}")
