# AAA work, evidence and backup policy

GitHub [`Cinqic-Research/AAA` `main`](https://github.com/Cinqic-Research/AAA) is canonical. The
Cinqic HDD is the FLOWBOX working disk and a local backup location. Hugging
Face is a secondary off-device backup, never a source of automatic changes to
GitHub. Recover from a backup only to a known canonical commit or by an
explicitly reviewed recovery decision.

## Before substantive work on FLOWBOX

The 256 GB NVMe carries the operating system. The dedicated 500 GB ext4 HDD
is labeled `Cinqic Storage`. Before cloning, creating a worktree or venv,
building, running substantial tests, or writing experiment data, run:

```bash
python3 tools/storage_preflight.py --path "$PWD"
python3 tools/storage_preflight.py --path "$AAA_DATA_ROOT"
```

Run the first command from an HDD checkout. The tool matches the label and
UUID to the live mount table, tests writing, and reports the mount and free
bytes. Set `AAA_DATA_ROOT` before the second command; do not accept its default
home-directory location on FLOWBOX. A missing or unwritable HDD is a stop,
not permission to use the NVMe. Repositories, worktrees, project venvs, build
output, experiments, checkpoints, substantial caches and data roots belong on
the HDD. Small system files and unavoidable OS activity naturally remain on
the system drive. Other hosts and GitHub Actions use their own appropriate
storage roots; the tool skips the FLOWBOX requirement there unless
`--require-hdd` is passed. No scientific code assumes this mount path.

For a fresh clone, first locate and verify the HDD with `lsblk` and `findmnt`
before the script is available, then clone onto that mount and run the script.
Do not trust a directory named like the HDD without matching its mounted UUID.

## Scientific work

Reproduce a defect against the unmodified candidate before repair; keep a
deterministic regression where useful. Preserve failed, superseded, spent and
frozen evidence. Run `tools/check_protected_identities.py` before and after
changes. A frozen historical defect needs a prospective successor, not a
rewrite of old results. Use development identities for diagnosis. Python
confirmation is not admitted in `aaa.python.v0`; do not generate or inspect
it. Do not use the attack pool for ordinary development. A scientific
promotion requires a frozen contract, independent review, exact-head checks,
and its declared held-out procedure. A green check alone proves only its own
scope.

## Review roles

From 2026-10-16:

- **Implementer: Claude Opus 5.5.** Builds, tests, and documents changes, and
  prepares the evidence. It may fix defects, but it never certifies its own
  work as independently reviewed.
- **Independent technical reviewer: Claude Fable 5.1,** in a separate session
  with a fresh context. It reviews the exact head and the primary evidence
  itself, and may reject work, require changes, or withhold approval. If it
  authored, repaired, or directed the exact change, it says so and the change
  stays pending review.
- **Owner: Markus Gillyard.** Decides direction, funding, releases,
  irreversible operations, and policy exceptions. A technical approval is not
  owner authorization.

Both models are Anthropic's, so this is separate-session review, not
vendor-independent assurance. Reviews by earlier reviewers (GPT-6 Luna, GPT-6
and GPT-5.6 Sol, Codex) remain historical records. This section does not
change any review gate.

Scientific promotion still needs the frozen contract, independent review,
exact-head checks, and declared held-out procedure described above.
## Backup and recovery

The canonical [backup policy](backup_policy.md) defines the exact scope and
restore test. After GitHub `main` advances, fetch it, create a commit-addressed
HDD Git bundle with `tools/backup_repository.py`, restore-test the bundle, then
copy the bundle and manifest to private Hugging Face storage and download and
restore-test them. Record the canonical SHA, manifest, remote destination and
verification result. If remote authorization fails, report it as blocked;
never imply the off-device backup exists. Never sync Hugging Face into GitHub
automatically.
