# Independent Codex review of PR #29

**Reviewer:** Codex GPT-6 session, with five scoped agents in the same session
(statistics, representation, provenance/packaging, literature, security/docs).
This is an independent AI review of the implementer's work, not a human
review. **Date:** 2026-09-25. **Canonical repository:**
`Cinqic-Research/AAA`. **Base:** `cb4170e86e7d15a56dbba319c695b32d01e8206b`.
**Original candidate head:** `f351d5755e82895cd4ff6661b563529bad0d6461`.
The final repaired head and merge outcome are recorded in the exact-head PR
review comment, since a committed document cannot contain its own commit SHA.

## Scope and method

The review used an independent clone on the mounted `Cinqic Storage` HDD, a
locked 24-package Python 3.12 environment, the live canonical PR/branch
metadata, the tracked-file inventory, PR diff inspection, source inspection,
mutation probes, primitive recounts, a separately salted crossed bootstrap,
freeze history and admission probes, historical AAA-180 reproduction, a fresh
wheel install from outside the checkout, and the repository's verification
commands. The inventory contains 809 tracked files in this review candidate;
the generated [inventory](final_audit.md) lists paths, hashes and categories.
This inventory and automated parsing do not imply a line-by-line human reading
of every historical file or a fresh execution of every historical experiment.

The scientific lineage was checked against original dot-era and AAA-1K
artifacts, Champion 1 and `aaa.1k.v2`, AAA-180 and its prospective promotion
successor, and `aaa.python.v0` through v1. Their evidence and claims retain
their own scope. The v1 confirmation is spent evidence, not a new run.

## Findings and repairs

1. **AAA-205, decision verification:** the original v1 `recompute` returned
   `PASS` after an in-memory change to a stored contract verdict, the capacity
   verdict, or a Holm field. A separate post-freeze audit tool now regenerates
   the summary, Holm results, capacity rule, and all four promotion
   adjudications from retained bits and the committed freeze. CI invokes it;
   mutation tests reject the three altered fields. The frozen v1 source was
   left unchanged, so the original confirmation's source identity remains
   honest. The first post-freeze implementation compared floating bounds for
   exact dictionary equality and failed cross-platform CI; the repaired gate
   requires exact decisions with the promotion contract's declared tolerance
   for interval bounds.
2. **AAA-206, protected evidence:** 22 newly added Python evidence paths were
   selected by the protected-file policy but absent from its 471-path
   manifest. Their current hashes were added without changing prior expected
   hashes. The check now covers 493 files and four unchanged historical phase
   identities.
3. **AAA-207, claim scope:** the encoder contrast matches 6,662 trainable
   parameters but compares separately tuned budgets (8 epochs for `e1`, 32
   for `e2`). The repair generator exposes one candidate equal to the visible
   buggy line, leaving three eligible choices. Current-facing text and a dated
   erratum describe these facts and the fixed Python knowledge in `e2`.
   The fingerprinted architecture and generator remain unchanged.
4. **AAA-208, literature:** a dated erratum narrows interpretations of Shaw
   relative positions, ByT5, plasticity saturation, and L2 Init. The
   pre-development literature document remains intact as chronology.
5. Reproduction, limitations, research-direction metadata, and current-facing
   spelling were updated to distinguish v1 from the historical v0 phase. The
   inventory classifier now describes v1 paths as v1 rather than v0.

## Scientific checks and interpretation

- The freeze first appears at `a73765b278ab71c1cb9d18ea43cf7014bcabc48f`;
  confirmation first appears in its child
  `7c46e4888d8c4708ba50b040a19d9074f338c7a7`. No fingerprinted source
  changed between freeze and the original PR head. The freeze and
  confirmation contain matching specification and source hashes. Dirty
  model/specification/freeze and forged/stale admission probes refuse.
- The five selected sizes count 846, 1,950, 4,158, 10,046 and 19,982
  trainable parameters directly from arrays. The output head alone uses 303,
  606, 1,212, 2,828 and 5,555 respectively; confirmation arms have zero
  momentum velocity. `e2` contains fixed structural/static knowledge and
  preprocessing cost beyond this trainable count. No direct hidden oracle or
  syntax parse-success leak was established in the reviewed paths.
- Independent per-cell Jeffreys-smoothed error recounts reproduce geometric
  ratios of 0.750454 (`e2/e1`), 0.784858 (10K/1K), 0.798257 (4K/1K), and
  0.983216 (10K/4K). A separate 50,000-draw crossed bootstrap keeps the
  first three upper bounds below 0.95 and leaves 10K/4K inconclusive against
  0.97. These are ratios of smoothed error, not accuracy differences. Simple
  rules still beat learned models on syntax and outcome; visible tests alone
  solve about 95% of generated repair tasks.
- The per-family capacity rule is `MIXED`; the overall decision
  `SCALE_NOT_JUSTIFIED` concerns the added cost of ~10K over ~4K in this
  architecture and task family. C4 (4K over 1K) was motivated by development
  evidence, then prospectively frozen before confirmation; it was not in the
  phase's initial hypothesis set. Adaptation and plasticity results remain
  development evidence. The post-hoc 4K life-stage comparison and conflicting
  mapping probe do not establish general plasticity or broad Python ability.
- Historical AAA-180 still reproduces in the frozen v2 path; its successor
  agrees on retained groups and treats single-series Saugeen conditionally.
  Historical failed, superseded and frozen evidence was not rewritten.

## Verification and limits

At the original head, the four protected identities and 471 old manifest
entries passed, as did the lock, lint, format, typing, golden keys, promotion
self-test, AAA-180 reproduction, v1 fingerprint/specification, and all
retained v1 primitive recomputations. Two independent recomputation bounds
near zero were reported `BORDERLINE`; neither changes a primary promotion
contract. The updated protected check covers 493 files; the decision audit
passes its retained evidence and mutation tests; the v1 fingerprint remains
`f1eb96177abc7b8646cacd866dbb48b4a04a066ff3903ad992d5c2aedc31ad08`.
The fresh original-head wheel installed under a path with spaces and resolved
package data, v0/v1 specification, golden keys and promotion from outside
the checkout. This is packaging smoke, not a full scientific rerun.
The local v0 safety and leakage probes passed 68/68 and 52/52 respectively;
the historical v2 fingerprint and confirmation exit-code contract also passed.

The original-head full suite passed 932 tests with one skip; three new
decision-audit tests passed separately. Coverage over the complete
suite was 64% overall. The capacity-stage source rerun was still running
when this record was drafted. Its final result and exact final-head CI belong
in the PR review comment. Full held-out re-observation would be invalid;
remaining limitations include within-pool duplicates, a single shared-core
architecture family, a generated Python subset, no confirmed adaptation
endpoint, and the absence of a 64-epoch 4K extension.

**Scientific verdict:** the retained v1 evidence supports a ~4K member as the
next experimental reference in this tested family, subject to the final
verification gates. It does not promote a model or establish broad Python
capability. ~10K has not earned its cost over ~4K; this does not prove that
all ~10K architectures lack value.

**Merge verdict at drafting:** pending source rerun,
and exact repaired-head CI. No formal GitHub approval object is claimed; the
active `MarkusGillyard` GitHub identity is the PR author.
