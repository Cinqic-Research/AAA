# AAA backup and recovery policy

| Role | Location | Meaning |
|---|---|---|
| Canonical | GitHub `Cinqic/AAA` `main` | Only ordinary reviewed GitHub merges advance the canonical repository. |
| Local working and backup | FLOWBOX `Cinqic Storage` ext4 HDD | Checkouts, experiment data and commit-addressed Git bundles. Two directories on one HDD are one failure domain. |
| Off-device backup | Private Hugging Face Bucket, when write access is authorized | Secondary copy of the exact Git bundle and manifest. It never changes GitHub automatically. |

The initial remote write is **not verified** until a private destination is
created, its bundle and manifest are uploaded, downloaded to a fresh HDD
directory, SHA-256 checked, and cloned and integrity-checked. A connected
read-only Hugging Face account is insufficient. Record a blocked state honestly.

## What the Git bundle covers

`tools/backup_repository.py` fetches all ordinary GitHub branches and tags,
checks that local and live `main` agree, runs `git fsck`, creates a bundle of
all fetched refs, and writes a JSON manifest. The path is keyed by the
canonical `main` commit. The manifest records source, UTC creation time,
canonical commit and tree, included refs, bundle filename and SHA-256, Git
version and schema. The tool verifies the bundle, clones it into a fresh HDD
directory, checks the restored tree, and runs the protected-identity guard
when present. A restore does not advance GitHub; changes recovered from a
backup require a separate reviewed decision.

This covers committed Git history and committed evidence. It does **not**
archive raw records, plots, checkpoints, external datasets or any other bytes
omitted from Git. In particular it does not close `AAA-077`, `AAA-134` or
`AAA-144`, and is not publication-grade archival of raw evidence. Local-only
artifacts require a separate inventory, license/privacy review, content
addressed manifest and retrieval test before any off-device claim is made.
Never upload arbitrary HDD contents, credentials, caches or third-party data.

## Local creation and restore

From a checkout on the mounted HDD, with an HDD-local Python environment:

```bash
python tools/storage_preflight.py --path "$PWD" --path "$AAA_DATA_ROOT"
python tools/backup_repository.py --repo "$PWD" \
  --output-root "$AAA_DATA_ROOT/../archives/git"
```

The output root must also be on the HDD. A commit-addressed directory holds
`aaa-main-<sha>.bundle` and `manifest.json`. Existing backups are verified
instead of overwritten. To verify a retrieved pair, use `--verify-bundle`
and `--manifest` with the same HDD output root. Do not edit a manifest to
make verification pass.

## Hugging Face copy and retrieval

Use the modern `hf` CLI after an authorized write-capable login. Prefer a
private Bucket because this is a generic Git backup, not a model or dataset
release. For each canonical SHA, copy the bundle and manifest to
`hf://buckets/Cinqic/aaa-backup/git/<sha>/`. Use `hf buckets list` to inspect
the remote objects, then `hf buckets cp` to download both to a new HDD
directory. Run the local verification command on the downloaded files and
record the exact SHA-256 and result. Do not put access tokens in commands,
repository files, shell history, CI logs or manifests. If Buckets are not
available to the authenticated account, choose the least misleading private
Hub repository type and record the limitation before uploading.

After an approved merge: fetch exact GitHub `main`; run the local tool; upload
the commit-addressed pair; retrieve and verify it; record the result in the
review or completion record. A Hugging Face write failure never rewrites
GitHub history and never turns a failed backup into a success claim.
