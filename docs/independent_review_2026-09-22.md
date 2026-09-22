# Independent repository review — 2026-09-22

## 1. Scope and identity

- Review start: `2026-09-22T16:29:47-04:00`.
- Repository: `Cinqic/AAA`. Reviewed base: `main` at
  `efa1fdb925032b75b80cab4b866f80d7eba9a812`, CPU CI green (run
  `35775776610`).
- Branch: `opus/independent-review-2026-09-22`. The head whose code was
  verified is `2f034abd2989ab87cb17dada7b5dbd77d6a4b614`. Only this record, a
  README row, a `SECURITY.md` read-back date and the regenerated
  `docs/final_audit.md` were committed after it. The exact final head, CI
  and merge are recorded in the pull request.
- Platform: AMD Ryzen 7 5700G (Zen 3), Python 3.12.3, locked environment
  (dependency-lock hash `2b5a2309…`). Stage reproduction mode: `--exact`
  (bitwise), because this is the Zen 3 evidence platform (`AAA-173`).
- Reviewer: Claude Opus 5.5 (Claude Code), acting on written instructions from
  the repository owner. It is an AI review, not a human review. Commits are
  authored under the owner's Git identity, with a `Co-Authored-By` trailer.
- Method: every command wrote only to `/tmp/aaa-review/` outside the
  repository, and `git status --porcelain` was checked after each one. Nothing
  ran that reserves, claims or spends an identity or batch.

## 2. Flags

**4 open items need the owner. None blocks merge. Five Fix-now findings were
repaired in this branch. There are 0 Blockers.**

- [Recommend · low] Confirm the `pip` Dependabot policy. The config now
  suppresses bot PRs, including security PRs, against the fingerprinted lock.
- [Recommend · low] Enable the repository setting that requires actions to be
  pinned to a full commit SHA. The workflows already comply.
- [Recommend · low] Allow merge commits only. Squash and rebase merges are
  currently enabled, although evidence cites merge SHAs.
- [Defer · research] Existing research items are unchanged: M1/Q4, `AAA-162`,
  `AAA-163`, `AAA-172`, R-05–R-11, and c10's untested domains.
- [Fix now · medium, repaired] `AAA-174`: `champion1 verify` did not recompute
  the phase fingerprint.
- [Fix now · low, repaired] `AAA-175`: Dependabot could open PRs that rewrite
  the fingerprinted lock.
- [Fix now · low, repaired] `AAA-176`: active loop documentation still
  described the pre-review v0 state.
- [Fix now · low, repaired] `AAA-177`: nine observation-noise ledger statuses
  said "confirmation pending", and two more were stuck on hosted CI that has
  since run.
- [Fix now · low, repaired] `AAA-178`: the evidence policy implied a 90-day CI
  copy of confirmation evidence.

## 3. Verification performed

Every check below passed on unmodified `efa1fdb` *before* any change, and
again on branch head `2f034ab`. The only differences are the ones listed.

| Check | Base `efa1fdb` | Branch `2f034ab` |
|---|---|---|
| `ruff check .` / `ruff format --check .` | pass / pass | pass / pass |
| `mypy` | pass (108 source files) | pass (108 source files) |
| `unittest discover -s tests -t .` | 701 tests OK | 706 tests OK (701 + 5 new) |
| `tools/check_lock.py` | 24 distributions match | 24 distributions match |
| `tools/check_exit_codes.py` | contract holds | contract holds |
| `research.aaa_1k parameter-audit` | 994 / 982 / 954 | 994 / 982 / 954 |
| `research.aaa_1k gradient-check --full` | pass | pass |
| `research.aaa_1k fingerprint` | `5ce6e019…f771e`, 32 files | `5ce6e019…f771e`, 32 files |
| `research.aaa_1k recompute` (round 3) | 21 values agree | 21 values agree |
| `research.aaa_1k_loop champion --verify` | verified | verified |
| `research.aaa_1k_loop validate` | 0001–0006 VALID; ledger DISJOINT (31 blocks) | same |
| `research.aaa_1k_loop prove-fresh` | DISJOINT: 1213 loop seeds vs 499,927 declared, 569 evidence | same |
| `research.aaa_1k_loop.champion1 verify` | VALID | VALID (now also recomputes the fingerprint) |
| `research.aaa_1k_loop.recompute4` (0006) | PROMOTE; agrees with stored | PROMOTE; agrees with stored |
| `research.aaa_1k_loop reproduce observe` | REPRODUCED, 0 mismatches | REPRODUCED, 0 mismatches |
| `stages4 reproduce <stage> --exact`, all nine stages | REPRODUCED (bitwise), 0 bit-different cells | REPRODUCED (bitwise), 0 bit-different cells |
| `aaa.cli spec-hash` | `f8e1090b…a5e37` | `f8e1090b…a5e37` |
| `aaa.cli observation-noise-protocol-hash` | `546e2434…b427c` | `546e2434…b427c` |
| `aaa.cli observation-noise-fingerprint` | `ae1c265f…c04`, 240 files | `08418eb4…a733`, 241 files (expected, see §6) |
| `tools/write_audit_inventory.py` reproduces `final_audit.md` | yes (620 files) | regenerated as the last commit |
| Strict JSON (NaN/Infinity refused), every tracked `.json` | 413 files, 0 failures | 413 files, 0 failures |
| `git status --porcelain` after every command | empty | empty apart from intended edits |

The review also checked the following:

- **Frozen sets, compared by hash before and after.** All 32 `aaa.1k.v1`
  fingerprint members, both protocol files and all 430 tracked files under
  `docs/evidence/`, `results/` and `benchmarks/` are byte-identical on the
  branch.
- **Failure injection, in throwaway worktrees of the branch head** (full
  outputs outside the repository):
  1. One byte appended to fingerprinted `aaa/predictors.py`: the fingerprint
     changes, `champion --verify` fails, `champion1 verify` reports
     `STALE: phase_fingerprint` (it passed on `efa1fdb`, see `AAA-174`), and
     `validate` fails.
  2. The frozen `margin` in `freeze_2.json` set from 0.02 to −0.01:
     `champion1 verify` and `validate` fail, and `recompute4` reports
     `DISAGREES … independent REJECT`.
  3. One Champion-0 `diverged` flag flipped in a `long:coarse_no_switch` cell
     of `confirmation_2.json`: `validate` fails on the artifact hash,
     `champion1 verify` fails, and `recompute4` reports `DISAGREES` on K1's
     divergence fraction (0.2125 vs 0.20625) and interval.
  4. Iteration 0005's `outcome` changed from REJECT to PROMOTE: `validate`
     reports `a REJECTED iteration must record outcome REJECT`.
- **Claim refs.** The four `aaa-confirmation-claims/*` refs match the four
  `spent` confirmation blocks in the ledger by batch id, attempt id and source
  commit. `git ls-remote` showed every head and tag unchanged at the start and
  in the middle of the review. The final comparison is in the PR.
- **Live settings.** The branch-protection read-back matches every row of
  `SECURITY.md`: 6 required checks, strict, 0 approvals, conversation
  resolution, no force pushes or deletions, and admins not enforced. The
  default workflow token is read-only, workflows cannot approve PRs, branches
  are deleted on merge, Dependabot alerts and security updates are on, and
  secret scanning with push protection is on. Rulesets: none. Open PRs: none.
  Open Dependabot alerts: none.
- **Documentation mechanics.** A Markdown link and anchor check over 53
  tracked `.md` files found one broken anchor on `efa1fdb` (`AAA-176`) and
  none on the branch. `--help` was run for all 36 distinct documented
  command/subcommand pairs in the README, reproduction guide, handoffs and
  protocols, and every documented flag exists.
- **CI and workflows.** Every action is SHA-pinned, and every workflow
  declares `contents: read`. The `workflow_dispatch` inputs of `benchmark.yml`
  are validated through the environment. The classifiers equal the 3.10–3.13
  matrix. The `loop-reproduction.yml` path filter covers every `aaa` module
  that the loop and `aaa_1k` import (`aaa`, `aaa.config`, `aaa.environment`,
  `aaa.predictors`; `aaa.noise.reservation` only claims identities and cannot
  change a verdict). The fingerprint file list in
  `research/aaa_1k/identity.py` equals the list in `CONTRIBUTING.md`.
- **Causal boundary, spot check.** c10's reach gate reads only the tracker's
  public velocity estimate, the input position and the revealed observation
  at update time. `test_a_leaking_agent_is_caught_by_the_causality_check` and
  the predictor-boundary tests remain in place and pass.
- **Issue ledger.** All 125 entries on `efa1fdb` were tabulated with their
  status. Entries now run to `AAA-178`.

## 4. Findings

| ID | Class | Finding | Reproduction (on `efa1fdb`) | Disposition |
|---|---|---|---|---|
| `AAA-174` | Fix now (medium) | `champion1 verify` compared Champion 1's `phase_fingerprint` with a copy of Champion 0's record, never with the tree | appending one byte to `research/aaa_1k/model.py` or bumping one lock pin changed the fingerprint and failed `champion --verify`, but `champion1 verify` printed `VALID`, rc 0 | repaired; regression test |
| `AAA-175` (lead L1) | Fix now (low) | the `pip` Dependabot entry ignored only patch updates; the lock is fingerprinted | config text; `1d1ccfe` is a Dependabot minor bump of the lock; one bumped pin breaks `champion --verify`. GitHub's reference scopes `ignore` to security updates as well | repaired in configuration; regression test |
| `AAA-176` (lead L2) | Fix now (low) | `loop_protocol.md` still said "remains `v0-pilot` pending an independent review". The README row and the missing PR #20 review row were stale, and a historical anchor was broken | text, and `LOOP_PROTOCOL_VERSION == "aaa.loop.v1"` | repaired by a dated, additive note; historical sentence kept |
| `AAA-177` (leads L3, L4) | Fix now (low) | `AAA-149` and `-150` were waiting on hosted runs that have happened. Nine entries said "confirmation pending" for batches that are `retired_unobserved` | API read-back of runs `35451372899`, `35451375484`, `35451862857` and `35775776610`, and the registry status | statuses reworded, earlier wording kept in quotes with the date |
| `AAA-178` | Fix now (low) | "CI additionally retains the full attempt directory for 90 days" predates `AAA-125`. Actions never runs confirmation | `benchmark.yml` roles and batch refusal; `ci.yml` retention of 14 days | scoped; limitation restated; `AAA-077` note |
| L5 | Not a defect | no `CHANGELOG.md` entry for PR #22 | PRs #5, #10, #15 and #17 also have none; entries go to research, protocol and phase changes, and to some documentation PRs such as #21 | no change |
| — | Accepted (cosmetic) | on fingerprint drift, `champion --verify` exits 1 with a `RuntimeError` traceback instead of a `STALE:` line | injection 1 | fails closed; not changed |
| — | Accepted limitation | `AAA-077`, `AAA-124`, `AAA-134`, `AAA-144` (archival), `AAA-152`, `AAA-169` (frozen docstring), `AAA-173` | re-read; still accurate | no change |
| — | Recommend | the three settings items in §7 | live API read-back | recorded only |

## 5. Remediation, by commit

1. `bde6313` `fix(loop)`: `champion1.verify` recomputes the phase fingerprint.
   Adds a test. `AAA-174`.
2. `c63cbb1` `fix(deps)`: the Dependabot `pip` entry sets
   `open-pull-requests-limit: 0` and ignores every dependency. Adds a test and
   updates `SECURITY.md`, `dependencies.md` and an `AAA-110` note. `AAA-175`.
3. `b3619dd` `docs(loop)`: dated v1 note, README rows, restored anchor.
   `AAA-176`.
4. `6bab0bb` `docs(ledger)`: dated status updates. `AAA-177`.
5. `442afaf` and `2f034ab` `docs(evidence)`: retention scope and a wording
   correction to a claim this review could not verify. `AAA-178`.
6. `95427c9` `chore(audit)`: the inventory generator attributes the
   review-changed paths to `AAA-174`–`AAA-178`.
7. This record, the README row and the `SECURITY.md` read-back date, then the
   regenerated `docs/final_audit.md` as the last commit.

## 6. Deliberately not changed

- **The frozen `aaa.1k.v1` set.** It includes the `AAA1KGRU` docstring that
  misnames the TBPTT rule (corrected by `AAA-169` and the errata) and
  `requirements-lock.txt`. The fingerprint is `5ce6e019…` before and after.
- **Every evidence, result and benchmark file, and both protocol files.** The
  v2.1 specification hash is `f8e1090b…`, and the noise protocol hash is
  `546e2434…`.
- **The observation-noise scientific fingerprint.** By design
  (`aaa/noise/scientific_identity.py`, `AAA-152`) it covers nearly every
  tracked file, so any documentation or test edit moves it. It went from
  `ae1c265f…` (240 files) to `08418eb4…` (241 files) at `2f034ab`. This
  record and the regenerated inventory move it again at the final head. Every
  previous PR has moved it the same way. It identifies no frozen protocol and no retained freeze.
- **Historical and snapshot documents.** These include
  `loop_pilot_report.md`, the handoffs, `sol_review.md`, `handoff_sol.md` and
  the superseded reports. The broken historical anchor was repaired at its
  target, not in the report.
- **Workflow retention values, repository settings, branch protection,
  rulesets, secrets and Actions permissions.** Settings are recommendations in
  §7.
- **`research/aaa_1k_loop/champion.py`.** The cosmetic traceback above is left
  alone.
- **Research items.** M1/Q4, `AAA-162`, `AAA-163`, `AAA-172`, R-05–R-11,
  independent benchmark design, long-range memory, and wall geometry, noise,
  missingness and acceleration near walls. None of them is represented as
  solved.

## 7. Recommendations for the owner

1. **Dependabot and the lock.** The branch stops Dependabot opening `pip`
   PRs, including security PRs, because the lock is scientific identity.
   Alerts still arrive. If you would rather receive security PRs that fail CI
   by design, as a visible reminder, remove the `ignore` rule and keep
   `open-pull-requests-limit: 0`. The repository-level setting stays
   **enabled**, and this review did not change it.
2. **Require SHA-pinned actions.** `sha_pinning_required` is currently
   `false`. Every workflow already pins actions to SHAs. Enabling the setting
   makes that enforced rather than conventional: Settings → Actions → General
   → "Require actions to be pinned to a full-length commit SHA", or
   `gh api -X PUT repos/Cinqic/AAA/actions/permissions -F enabled=true -f allowed_actions=all -F sha_pinning_required=true`.
3. **Merge commits only.** `allow_squash_merge` and `allow_rebase_merge` are
   `true`. Evidence documents cite commit SHAs, and the project merges with
   merge commits. To enforce that:
   `gh api -X PATCH repos/Cinqic/AAA -F allow_squash_merge=false -F allow_rebase_merge=false`.
4. `enforce_admins` stays off, as `SECURITY.md` says. No change is recommended.

## 8. Verdict

**APPROVE WITH RECOMMENDATIONS.** This rests on the following conditions:
exact-head required CI is green; `loop-reproduction.yml`, which this branch
triggers by touching `research/aaa_1k_loop/`, passes in its cross-platform
verdict mode; `main` has not moved in a conflicting way; and the four claim
refs are unchanged at merge. No recommendation above has to be decided before
merge.

## 9. Closure

- Review end (local record): `2026-09-22T16:52:08-04:00`. Elapsed from the recorded start:
  22 minutes 21 seconds.
- Phase timings (wall clock, process timing only, not scientific evidence):
  - setup, clone, locked environment and live GitHub state: 16:29:47–16:30:20;
  - full read, lead verification and probes, with the complete baseline suite
    (including nine exact stage reproductions) running alongside:
    16:30:20–16:40:42;
  - remediation (five findings, seven commits) and adversarial self-audit,
    overlapping the end of the baseline run: about 16:33–16:42;
  - complete suite on branch head `2f034ab`: 16:42:01–16:51:54;
  - this record: 16:52 to the end time above.
- GitHub review state, the exact final head, exact-head CI, the merge SHA and
  post-merge verification happen after this file is committed. They are
  recorded in the pull request. No approval or merge is claimed by this local
  record alone.
