"""Non-self-referential scientific source identity for confirmation freezes.

All tracked and nonignored untracked files are covered, including tests, tools,
workflows, documentation and seed fixtures. Only results/, the exact freeze
manifest, and three generated review documents are excluded. The batch registry is covered
except execution-status fields; declarations and stream identities are immutable.
Git commit identity is separately retained as provenance, not used as equality.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

EXCLUDED_FILES = frozenset(
    {
        "benchmarks/freeze_manifest.json",
        "docs/final_audit.md",
        "docs/handoff_sol.md",
        "docs/sol_review.md",
    }
)
BATCH_REGISTRY = "benchmarks/confirmation_batches.json"
MUTABLE_BATCH_FIELDS = frozenset({"status", "consumed_by", "outcome", "claimed_by", "claim_started_at"})


def scientific_fingerprint(project_root: Path) -> dict[str, Any]:
    """Hash file paths, executable bits and contents; refuse unsafe identities."""
    root = project_root.resolve()
    try:
        top = (
            subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"], stderr=subprocess.PIPE
            )
            .decode()
            .strip()
        )
        if Path(top).resolve() != root:
            raise ValueError("scientific source must be a Git repository root")
        listing = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("scientific source requires a readable Git repository") from exc
    files = {}
    for name in sorted(set(listing.decode().split("\0")) - {""}):
        if name in EXCLUDED_FILES or name.startswith("results/"):
            continue
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"scientific source is missing or not a regular file: {name}")
        content = path.read_bytes()
        if name == BATCH_REGISTRY:
            registry = json.loads(content)
            registry["batches"] = [
                {key: value for key, value in batch.items() if key not in MUTABLE_BATCH_FIELDS}
                for batch in registry["batches"]
            ]
            content = json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
        files[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "executable": bool(path.stat().st_mode & 0o111),
        }
    if not files:
        raise ValueError("scientific source file set is empty")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "aaa.scientific_source.v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }
