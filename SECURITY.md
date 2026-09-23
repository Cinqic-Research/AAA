# Security policy

## Reporting

Report a vulnerability through GitHub's private security advisory form on
[Cinqic/AAA](https://github.com/Cinqic/AAA/security/advisories/new). Please do
not open a public issue for an unfixed vulnerability.

Expect an acknowledgement within a few days. AAA is a small research prototype
maintained on a best-effort basis; there is no commercial support commitment.

## Threat model

AAA runs local, seeded experiments: the dot-era simulations of a
one-dimensional world and, since `aaa.python.v0`, small generated Python
programs executed as an external oracle. It has no network client or server,
no authentication, no user accounts and no persistent service. It reads and
writes JSON under a directory the caller chooses.

Realistic concerns are correspondingly narrow:

- **Executing Python programs (`aaa.python.v0`).** AAA executes only programs
  its own deterministic generator produces, never code from the network, a
  corpus, a user or a learner. Each program must pass an allow-list AST
  validator first (no imports, attribute access, `eval`/`exec`/`open`/`getattr`
  or any unlisted builtin, `while`, recursion, `**` or `/`; bounded literals and
  loops). It then runs in a separate `python -I -S` process with an empty
  environment, a fresh temporary working directory, only the allowed builtins,
  CPU, address-space, file-size and process-count limits where the platform
  provides them, a wall-clock timeout and an output cap. Syntax checks only
  compile. `python -m research.aaa_python safety` exercises 68 escape and
  containment cases. This is defence in depth for generated code, not a
  hardened sandbox for hostile code: it relies on POSIX resource limits, has
  no network or filesystem namespace isolation, and must not be used to run
  untrusted programs. Learners run in-process and are trusted research code.
- **Untrusted checkpoint or specification files.** Everything AAA loads is JSON
  with a declared schema version and strict validation. No `pickle`, no dynamic
  import of file content, and nothing loaded from a file is executed as code.
  An unknown schema version or an out-of-range value is refused rather than
  coerced.
- **Path handling.** A run writes only under its requested output root. This is
  covered by `tests/test_legacy_v1.py::test_a_run_writes_nothing_outside_the_requested_output_root`.
- **Supply chain.** Runtime dependencies are NumPy and Matplotlib, pinned in
  `requirements-lock.txt`. CI installs the lock and verifies the installed set
  against it. The lock is part of the `aaa.1k.v1` phase fingerprint, so
  Dependabot is configured never to open a `pip` pull request, including a
  security update. Dependabot alerts still report a vulnerable pin, and the
  maintainer then regenerates the lock deliberately (`AAA-175`).
- **GitHub Actions.** Workflows declare `permissions: contents: read`, pin
  actions to commit SHAs, and pass `workflow_dispatch` inputs through the
  environment after validating them against a character-class pattern rather
  than interpolating them into shell text.

## Repository settings

These were inspected through the GitHub API again on 2026-09-19, and read back
unchanged on 2026-09-22 by the
[independent review](docs/independent_review_2026-09-22.md). See `AAA-110`
in [`docs/issue_ledger.md`](docs/issue_ledger.md).

**Applied and verified on `main`:**

| Setting | State |
|---|---|
| pull request required before merging | yes (0 approvals, so a solo maintainer can still self-merge) |
| required status checks | `Locked environment`, `Fresh install`, `Python 3.10`, `3.11`, `3.12`, `3.13` |
| branches must be up to date before merging | yes (`strict`) |
| force pushes to `main` | blocked |
| deletion of `main` | blocked |
| conversation resolution required | yes |
| delete head branches after merge | yes |
| Dependabot alerts and security updates | enabled |
| secret scanning and push protection | enabled |
| default workflow token permissions | read |
| workflows may approve pull requests | no |
| `.github/dependabot.yml` | weekly `github-actions` updates; every `pip` update ignored, because `requirements-lock.txt` is fingerprinted (`AAA-175`) |

**Deliberately not applied:**

- *Enforce for administrators.* Left off so the maintainer retains a recovery
  path if a required check is ever misconfigured or a runner is unavailable.
  Turn it on with
  `gh api -X POST repos/Cinqic/AAA/branches/main/protection/enforce_admins`
  once the check names are considered stable.
- *Required signed commits.* Not enabled. Existing scientific identity and CI
  checks verify content and behavior, but do not constitute author-signature
  verification.

Verify the current state with:

```bash
gh api repos/Cinqic/AAA/branches/main/protection
gh api repos/Cinqic/AAA --jq '.delete_branch_on_merge, .security_and_analysis'
```
