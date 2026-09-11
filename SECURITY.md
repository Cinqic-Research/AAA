# Security policy

## Reporting

Report a vulnerability through GitHub's private security advisory form on
[Cinqic/AAA](https://github.com/Cinqic/AAA/security/advisories/new). Please do
not open a public issue for an unfixed vulnerability.

Expect an acknowledgement within a few days. AAA is a small research prototype
maintained on a best-effort basis; there is no commercial support commitment.

## Threat model

AAA runs local, seeded simulations of a one-dimensional world. It has no
network client or server, no authentication, no user accounts and no persistent
service. It reads and writes JSON under a directory the caller chooses.

Realistic concerns are correspondingly narrow:

- **Untrusted checkpoint or specification files.** Everything AAA loads is JSON
  with a declared schema version and strict validation. No `pickle`, no `eval`,
  no dynamic import of file content. An unknown schema version or an
  out-of-range value is refused rather than coerced.
- **Path handling.** A run writes only under its requested output root. This is
  covered by `tests/test_legacy_v1.py::test_a_run_writes_nothing_outside_the_requested_output_root`.
- **Supply chain.** Runtime dependencies are NumPy and Matplotlib, pinned in
  `requirements-lock.txt`. CI installs the lock and verifies the installed set
  against it.
- **GitHub Actions.** Workflows declare `permissions: contents: read`, pin
  actions to commit SHAs, and pass `workflow_dispatch` inputs through the
  environment after validating them against a character-class pattern rather
  than interpolating them into shell text.

## Repository settings

These were inspected and applied through the GitHub API, and then read back to
confirm. See `AAA-110` in [`docs/issue_ledger.md`](docs/issue_ledger.md).

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
| `.github/dependabot.yml` | weekly `github-actions` and `pip` updates |

**Deliberately not applied:**

- *Enforce for administrators.* Left off so the maintainer retains a recovery
  path if a required check is ever misconfigured or a runner is unavailable.
  Turn it on with
  `gh api -X POST repos/Cinqic/AAA/branches/main/protection/enforce_admins`
  once the check names are considered stable.
- *Default workflow permissions read-only.* This is an organization- or
  account-level Actions setting rather than a repository one. Every workflow in
  this repository already declares `permissions: contents: read` explicitly,
  which is the stronger guarantee because it does not depend on an inherited
  default.

Verify the current state with:

```bash
gh api repos/Cinqic/AAA/branches/main/protection
gh api repos/Cinqic/AAA --jq '.delete_branch_on_merge, .security_and_analysis'
```
