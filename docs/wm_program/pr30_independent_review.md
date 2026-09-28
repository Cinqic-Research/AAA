# Codex GPT-6 independent review of PR #30

Date: 2026-09-27. Reviewer: Codex GPT-6 (AI, not an independent human
reviewer). Repository: `Cinqic-Research/AAA`. Base at review start:
`dbee13c7aa624d738dc2d21dbf2cc8078274db94`. Original candidate:
`f5b3c5a584e0e73a7a01f6f9d1d385e907548a73`. The exact repaired PR head
and live CI state are recorded in the PR review comment after the final push;
recording that SHA in its own commit would be self-referential.

## Scope and method

I inventoried all 882 tracked files, mapped active and frozen packages, read
the PR's 76-file diff in related groups, reviewed the program's handoff,
designs, limitations, retained evidence and earlier independent review, and
tested the suspected failure paths against source and retained primitives.
This was a systematic repository audit, not a claim to have manually read
every generated evidence record. No subagents were used. The original review
is a source of hypotheses; its conclusions were not taken as authority.

I used the dedicated HDD worktree and existing CPU and Torch environments.
Neither confirmation was rerun. Both spent confirmation documents were parsed
and recomputed. World and language source fingerprints match their freezes;
all listed source files are unchanged since their respective provenance
commits. The four earlier scientific identity values and 493 prior protected
file hashes are unchanged.

GitHub Actions recorded a push of the world-model freeze commit `f994d91` at
2026-09-27 04:45 UTC, about four hours after the logged confirmation start.
The language freeze commit `a8c088d` was pushed before its logged run. The
world-model freeze therefore lacked a remote pre-run anchor.

## Results independently checked

- The world-model evidence contains 4,000 distinct ordered task identities,
  the declared five slice blocks, complete arm/initialization bit grids and
  11 contracts. Every retained verdict, point ratio, group ratio and interval
  agrees with a fresh computation within the contract's Monte Carlo tolerance;
  the geometric points also match a separate direct calculation from the
  correctness bits. C1 is 0.481 and all 11 stored contracts promote.
  Deterministic reconstruction matches all 4,000 retained task SHA-256 values.
  The actual cached development and attack programs have no hash overlap with
  either confirmation phase. One reference source repeats within the world
  confirmation at indices 2180 and 3663, in different slices and library
  conditions; those remain distinct tasks.
- The language evidence contains 2,000 disjoint subsequent identities and
  complete declared grids. The same checks reproduce L2 `PROMOTE` at 0.772,
  while L1, F1 and F2 are `INCONCLUSIVE`. Neither
  `LANGUAGE_INTEGRATION_SUCCESS` nor `FULL_SYSTEM_SUCCESS` is established.
  The stored channel-extraction percentages are summaries without per-task
  extraction bits, so they were not independently recomputed from retained
  primitives.
- WM-S learned a table of six opaque unary functions from program-level
  observations. Visible Python semantics and exhaustive depth-two planning
  were hand-written. The main comparison matches the number of real `RUN`
  actions, not training or inference compute. Four of five slices reuse
  training library A. The changed-library slice is weakest; the online gain
  largely fills incomplete table entries and does not establish general
  adaptation. The black-box neural WM and JEPA did not earn a decision role.
- The language task is extraction from templated American-English reports.
  The from-scratch LM beats the trained-template rule parser in L2; broad
  English understanding, general coding and full-system superiority are
  unsupported.

The local 15 world-model and 13 language learned-artifact files match their
freeze SHA-256 values. All eight local raw corpus downloads and all 15
processed corpus shards match the HDD manifests. The manifest copies added
to this review are post-run provenance snapshots; they do not retrospectively
anchor the downloads or guarantee their continued public availability. A
separate capped post-run audit deterministically reconstructed all 2,000
held-out report texts from the pinned code path
(SHA-256 `b092f0d6`) and found zero training documents sharing a 13-word
sequence with those reconstructed texts. The confirmation did not retain
report hashes for a direct byte-for-byte comparison. The audit result and
checker are retained; this was not a pre-run admission check.

## Findings, repairs, and evidence impact

The reproducible findings, severity, root cause and repair for AAA-210 through
AAA-219 are in the [issue ledger](../issue_ledger.md). AAA-210, AAA-211,
AAA-212 and AAA-215 have focused repairs outside frozen source. AAA-216 has a
new neural CI job whose live result is required before any approval. The
package wheel now contains both research specifications, and an unrelated
directory can import them after installing the wheel.

Four scientific limits remain open. `draft("confirmation")` can generate an
identity without admission (AAA-213). The language source fingerprint misses
transitive runtime helpers, including the tokenizer path resolver and RNG
(AAA-214). The benchmark's promised exclusion of visible-pass/domain-fail
nearby programs is absent and its design wording is ambiguous; 5 of 40
allowed pilot tasks have a visible-pass/domain-fail one-edit alternative
(AAA-217). The corpus contamination check was not a recorded pre-run gate,
though a post-run exact-text check now passes (AAA-218). The frozen source and spent
identities cannot be silently rewritten to repair these historical facts.
The confirmation provenance does not show that AAA-213 or AAA-214 was
exploited; the result is evidence about the recorded exact commits, with
weaker admission assurances than the design text claimed.

The opaque interpreter executes validated generated programs in-process; it
is not a hostile-code sandbox. The WM and LM checkpoint loaders and opaque
data cache use pickle-capable deserialization. Their generic entry points
must receive trusted files, while the confirmation entry points verify
declared artifact hashes before loading. The current security policy now
states those boundaries (AAA-219).

## Verification record

- `tools/check_protected_identities.py`: 500 hashes, unchanged historical
  fingerprints, pass after the new files were staged.
- `tools/check_aaa_wm_evidence.py`: world and language retained evidence pass.
  Six post-freeze audit mutations are rejected.
- Promotion self-test and AAA-180 successor reproduction: pass, including the
  historical Monash disagreement and rejected successors.
- World-model mutation harness: 12/12 deliberate breaks caught. M3 tests
  `build`, so it does not cover direct `draft` access.
- CPU research tests: 31 pass with seven optional-dependency skips. Torch
  research tests: 38 run, one GPU-only skip on this CPU test path.
- Actual hashed seed-100 policy and black-box WM checkpoints load and account
  for 985,221 and 985,092 trainable parameters; WM inference returned a
  three-test probability vector. The pretrained and random-twin LM adapters
  both load with 5,245,184 parameters, the pinned BPE tokenizer resolves,
  and both produced finite forward logits on a non-confirmation prompt.
- Full local suite: 974 tests in 539 seconds, two failures from the new
  provenance files being unstaged when that run began; all other tests passed
  or skipped. After staging, the 15 affected protected/evidence/job tests pass.
  A new full-suite CI run at the final PR head is still required.
- Ruff check, Ruff format check, mypy and the extended 64-package Torch lock
  check pass locally. The original wheel omitted both new specs; a rebuilt
  wheel installed outside the checkout now loads both.

## Verdict

**CHANGES_REQUIRED.** I cannot approve or merge this PR under the requested
gate while AAA-213, AAA-214 and AAA-217 remain open, the corpus check lacks a
pre-run record, and until the new locked neural CI and full suite complete on the
exact repaired head. The scoped numeric results above remain reproducible as
stored; this verdict concerns the stronger admission and benchmark-validity
assurances required for the merge.
