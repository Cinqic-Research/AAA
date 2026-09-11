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

The following are **recommendations**. They are administrative settings that
the engineering environment could not verify or change, so nothing here is
claimed to be in place. See `AAA-110` in
[`docs/issue_ledger.md`](docs/issue_ledger.md).

Recommended for `main`:

- require a pull request before merging;
- require the `CPU CI` status checks to pass, including the locked-environment,
  version-matrix and fresh-install jobs;
- require branches to be up to date before merging;
- block force pushes and deletion;
- do not allow administrators to bypass the above;
- delete head branches automatically after merge;
- enable Dependabot alerts, security updates and a weekly `github-actions`
  ecosystem update so pinned action SHAs stay current;
- set default workflow permissions to read-only, and disallow Actions from
  creating or approving pull requests.
