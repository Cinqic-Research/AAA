#!/usr/bin/env python3
"""Create and restore-test a commit-addressed backup of canonical GitHub AAA."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "https://github.com/Cinqic-Research/AAA"
SCHEMA = "aaa.git-backup.v1"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify(bundle: Path, manifest: Path, scratch_root: Path) -> dict:
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA or record.get("bundle") != bundle.name:
        raise ValueError("backup manifest schema or filename differs")
    if digest(bundle) != record["sha256"]:
        raise ValueError("bundle SHA-256 differs from manifest")
    with tempfile.TemporaryDirectory(prefix="aaa-restore-", dir=scratch_root) as temporary:
        verifier = Path(temporary) / "verify.git"
        subprocess.run(["git", "init", "--bare", "--quiet", str(verifier)], check=True)
        subprocess.run(
            ["git", "-C", str(verifier), "bundle", "verify", str(bundle)],
            check=True,
            capture_output=True,
        )
        restored = Path(temporary) / "AAA"
        subprocess.run(["git", "clone", "--quiet", str(bundle), str(restored)], check=True)
        git(restored, "fsck", "--full", "--no-reflogs")
        canonical = record["canonical_main"]
        git(restored, "cat-file", "-e", f"{canonical}^{{commit}}")
        git(restored, "checkout", "--detach", canonical)
        if git(restored, "rev-parse", "HEAD") != canonical:
            raise ValueError("restored checkout differs from canonical GitHub main")
        if git(restored, "rev-parse", "HEAD^{tree}") != record["canonical_tree"]:
            raise ValueError("restored tree differs from canonical tree")
        bundle_refs = subprocess.check_output(["git", "bundle", "list-heads", str(bundle)], text=True)
        included_refs = {line.split(" ", 1)[1] for line in bundle_refs.splitlines()}
        if not set(record["refs"]).issubset(included_refs):
            raise ValueError("bundle refs differ from manifest")
        protected = restored / "tools/check_protected_identities.py"
        if protected.exists():
            subprocess.run([sys.executable, str(protected)], cwd=restored, check=True, capture_output=True)
    return record


def create(repo: Path, output_root: Path) -> tuple[Path, Path]:
    git(repo, "fetch", "--prune", "origin", "+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*")
    remote_main = git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]
    local_main = git(repo, "rev-parse", "refs/remotes/origin/main")
    if local_main != remote_main:
        raise ValueError("local and live GitHub main differ")
    git(repo, "fsck", "--full", "--no-reflogs")
    destination = output_root / local_main
    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / f"aaa-main-{local_main}.bundle"
    manifest = destination / "manifest.json"
    if bundle.exists() or manifest.exists():
        if not bundle.exists() or not manifest.exists():
            raise ValueError("an incomplete backup already occupies this commit path")
        verify(bundle, manifest, output_root)
        return bundle, manifest
    git(repo, "bundle", "create", str(bundle), "--all")
    refs = git(repo, "for-each-ref", "--format=%(refname)").splitlines()
    record = {
        "schema": SCHEMA,
        "source": SOURCE,
        "canonical": "GitHub Cinqic-Research/AAA main; backups never write back automatically",
        "canonical_main": local_main,
        "canonical_tree": git(repo, "rev-parse", "refs/remotes/origin/main^{tree}"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": bundle.name,
        "sha256": digest(bundle),
        "included_ref_policy": "all locally fetched branches, remote-tracking refs, and tags",
        "refs": refs,
        "git_version": subprocess.check_output(["git", "--version"], text=True).strip(),
    }
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    verify(bundle, manifest, output_root)
    return bundle, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="fully accessible AAA Git checkout")
    parser.add_argument("--output-root", type=Path, required=True, help="HDD backup directory")
    parser.add_argument("--verify-bundle", type=Path, help="verify a downloaded bundle with its manifest")
    parser.add_argument("--manifest", type=Path, help="manifest paired with --verify-bundle")
    args = parser.parse_args()
    from storage_preflight import main as storage_preflight

    if storage_preflight(["--path", str(args.repo), "--path", str(args.output_root)]) != 0:
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.verify_bundle:
            if args.manifest is None:
                parser.error("--manifest is required with --verify-bundle")
            record = verify(args.verify_bundle, args.manifest, args.output_root)
            print(f"RESTORE VERIFIED {record['canonical_main']} {record['sha256']}")
        else:
            bundle, manifest = create(args.repo, args.output_root)
            print(f"RESTORE VERIFIED {bundle}\nMANIFEST {manifest}\nSHA256 {digest(bundle)}")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"backup failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
