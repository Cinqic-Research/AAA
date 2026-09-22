# PR #20 independent review

## Scope and identity

- Review start: `2026-09-22T10:54:44-04:00`.
- Repository: `Cinqic/AAA`; pull request: #20.
- Reviewed base: `daea2ffe838927bc6307f1652fac6b93373b164f`.
- Original reviewed head: `3651619cd731b9babf7f132a44f2e2cf16eeded4`.
- Merge base: `daea2ffe838927bc6307f1652fac6b93373b164f`;
  original branch state: 28 commits ahead, 0 behind, 53 changed files.
- Final reviewed head: recorded in the closure section after remediation is
  committed and final-head CI completes.

The review covered the complete tracked tree by role, the 28-commit
preregistration/evidence chronology, active implementation and documentation,
strict parsing and semantic validation of retained structured evidence,
historical freeze verification, identity ownership, confirmation primitives,
Champion 1, round-3 replay, reproduction policy, CI trigger coverage and the
tracked-file inventory. Historical evidence and the frozen `research/aaa_1k/`
source were not edited.

## Independent scientific judgment

Champion 1's promotion is supported within the tested domain. The live TBPTT
rule is the documented cached-activation/current-weight approximation, but the
snapshot, replay and T=1 probes show that discrepancy does not explain M2.
The gain-above-one and one-step-overshoot explanations are contradicted before
onset. Fresh H32--H34 evidence supports a self-confirming target-unfolding
frame lock: all 54 diverged coarse cells locked before measured onset, while
both prediction-independent probes eliminated divergence.

c10 is symmetric at the upper and lower walls, uses only causal public state,
keeps equality at the reach boundary, and contains no evaluator label or future
information. Its design followed disclosed development observations, and four
candidates were screened across three iterations. That multiplicity makes the
development PASS insufficient by itself; it does not invalidate the result
because c10 then survived one independently identified fresh attack and a
fully fresh crossed confirmation under frozen rules and identities. The claim
does not extend to arbitrary geometry, wall acceleration, observation noise or
missing observations.

## Evidence findings

- The 28 original commits preserve design-before-evidence ordering. The
  exploratory observations that motivated H32--H34 and c10 remain disclosed.
- The attempt-2 freeze verifies without disagreement at historical commit
  `25ff14d648ecdf7ea0c91e9cbe3088e90acf4734`.
- Four durable remote confirmation-claim refs exist. Attempt-1 blocks are
  permanently spent by the recorded observer; its failure occurs in admission
  before cell construction. Attempt 2 uses new namespaces and seeds.
- The ledger contains 31 mutually disjoint blocks and 1,213 loop seeds. They
  are disjoint from 499,927 declared AAA-1K seeds and 569 seeds found in
  retained AAA-1K evidence.
- Direct primitive inspection found exactly 800 confirmation cells in a
  complete 5-initialization crossing: six 80-cell standard-plan conditions
  and four 80-cell long conditions. The 160 long-coarse cells contain 34
  Champion-0 divergences (`0.2125`) and no c10 divergences. Every K2/K3 MAE
  difference is exactly zero. Independent recomputation returns `PROMOTE` and
  agrees with every stored numerical criterion.
- Champion 1 retains the 994-parameter AAA-1K network, its hyperparameters and
  phase fingerprint. The changed agent identity is separately bound to the
  reach-gated source and promotion evidence; “same phase fingerprint” does not
  mean the entire agent is byte-identical to Champion 0.
- The round-3 replay keeps streams, design and configuration unchanged. It has
  720 cells; exactly two `coarse_speed_v1`, initialization-3 cells change.
  All 120 Q2 and 60 Q5 trials are unchanged. Applying the target rule to every
  neural arm, including branch clones but not analytic baselines, is the
  defensible like-for-like architecture comparison; restoration is protected
  by `finally` and tested.
- Cross-platform verdict-level reproduction is scientifically defensible only
  with its stated boundary: `--exact` is the bitwise evidence-platform check;
  default mode preserves identities, structure, divergence classifications and
  adjudicated verdicts while reporting CPU-amplified numerical drift.

## Review remediation

The review fixed active, non-scientific consistency blockers through new
commits only:

- separated the current `aaa.loop.v1` governance constant from the immutable
  `aaa.loop.v0-pilot` artifact version and added a regression test;
- reconciled the current protocol, README, changelog, contribution guidance,
  reproduction guide, evidence policy, experiment registry and loop package
  documentation while leaving historical reports and JSON unchanged;
- extended full-stage workflow triggers to retained 0004--0006 evidence and
  the workflow itself;
- taught the audit generator to classify the active loop implementation and
  findings accurately;
- regenerated and mechanically verified the final tracked-file inventory.

## Findings classification

Fixed blockers were stale and contradictory current documentation, missing
current reproduction instructions, incomplete loop-registry scope, unsafe
protocol-version transition semantics, incomplete integrity-trigger coverage,
and a stale/misclassifying final audit.

Non-blocking limitations are the intentionally frozen inaccurate AAA-1K
docstring (corrected through `AAA-169` and errata), CPU-dependent magnitudes in
chaotic Champion-0 trajectories, the tested-domain boundary of c10, and the
absence of publication-grade external archival.

Deferred research includes M1/Q4, `AAA-162`, `AAA-163`, `AAA-172`, R-05 through
R-11, independent benchmark design, long-range memory, arbitrary wall
geometry, noise and missingness near walls, and large acceleration at walls.
None is represented as solved by this review.

## Protocol decision

`aaa.loop.v1` is justified because one real fresh-confirmation outer cycle and
an independent review have now completed. It is a governance transition, not
a new scientific result and not a retroactive relabeling: all iteration
0001--0006 records and evidence remain `aaa.loop.v0-pilot`.

## Closure

The local final candidate passed:

- strict parsing of all 413 tracked JSON files;
- the 24-distribution dependency-lock check;
- Ruff lint and format checks over 178 files;
- mypy over 108 source files;
- 701 unit tests in 227.111 seconds, with 64% aggregate coverage;
- exit-code contract checks;
- AAA-1K parameter audit, exhaustive gradient checks, fingerprint and
  round-3 primitive recomputation;
- all six loop-record validations, freshness proof, Champion 1 verification
  and independent iteration-0006 recomputation;
- all nine post-audit stages bit-for-bit in Zen 3 `--exact` mode;
- isolated Champion 1 round-3 replay, with only runtime and Git provenance
  differing from the immutable artifact;
- frozen-rule, lifecycle-state, primitive and Champion-source failure
  injections, each failing in the intended way;
- historical attempt-2 freeze verification at commit `25ff14d`;
- Markdown local-link audit, diff whitespace check and tracked-file inventory
  path/hash checks.

Phase timings (wall clock, process timing rather than scientific evidence):

- live-state capture and repository/document review: about 10 minutes;
- independent evidence, freeze and adversarial checks: about 10 minutes;
- nine exact stage reproductions: 7 minutes 51 seconds;
- isolated Champion 1 round-3 replay: about 5 minutes;
- full lint/type/test/recompute suite: about 5 minutes.

Review end: `2026-09-22T11:24:01-04:00`; elapsed from the recorded start:
29 minutes 17 seconds. The scientific conclusion is **APPROVE**. GitHub review
state, final PR head, exact-head CI, merge SHA and post-merge verification are
recorded in the PR closure because they occur after this review record is
committed. No approval or merge is claimed by this local record alone.
