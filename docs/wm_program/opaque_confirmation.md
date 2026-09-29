# `opaque.v0` world-model confirmation

Status: **CONFIRMATION**, fresh identities, observed once. Freeze: `docs/evidence/aaa_wm_opaque_v0/freeze.json`,
committed at `f994d91` before any confirmation identity existed. Evidence:
`docs/evidence/aaa_wm_opaque_v0/confirmation.json`, run from that commit with a clean tree. It holds the
per-(arm, initialization, task) success bits, the task hashes, the freeze and the adjudications.

- **Design.** 4,000 fresh `confirmation` tasks (every pilot, train, development and attack program
  excluded) in 5 slices x 20 slice-pure streams of 40. Initializations 100-104 for every seeded arm. Run
  budget 2, step budget 8.
- **Artifacts.** The WM-S tables and neural checkpoints were produced from the train split before the freeze.
  Their SHA-256 hashes were verified before the run.
- **Runtime.** 7 h 59 min, one capped process, 3.3 GB peak RSS.

## Results (crossed bootstrap over initializations and streams)

| Arm | All [95%] | in-dist | two_fault | novel_literals | novel_grammar | library_B |
|---|---|---:|---:|---:|---:|---:|
| true-library planner (ceiling) | 0.779 [0.763, 0.796] | 0.819 | 0.714 | 0.845 | 0.723 | 0.797 |
| **WM-S online (candidate)** | **0.568 [0.467, 0.658]** | 0.649 | 0.553 | 0.691 | 0.539 | 0.409 |
| WM-S frozen | 0.506 [0.369, 0.631] | 0.596 | 0.481 | 0.629 | 0.480 | 0.347 |
| WM-S online, empty library | 0.197 [0.183, 0.210] | 0.218 | 0.151 | 0.214 | 0.178 | 0.223 |
| tells (grammar shortcut, no WM) | 0.171 [0.157, 0.185] | 0.179 | 0.085 | 0.206 | 0.203 | 0.182 |
| black-box neural WM, depth 1 | 0.148 [0.129, 0.168] | 0.221 | 0.085 | 0.166 | 0.078 | 0.189 |
| policy_aux inside the WM-S planner | 0.111 [0.098, 0.125] | 0.141 | 0.044 | 0.145 | 0.081 | 0.144 |
| policy_aux, tool search | 0.102 [0.088, 0.116] | 0.126 | 0.037 | 0.135 | 0.076 | 0.136 |
| WM-S, empty library (interpreter only) | 0.052 [0.046, 0.059] | 0.065 | 0.019 | 0.059 | 0.058 | 0.062 |
| tool search (gap-reading) | 0.037 [0.031, 0.045] | 0.049 | 0.005 | 0.060 | 0.031 | 0.043 |

`wms:online+empty` has no seed-dependent state, so it scores identically at every initialization, as expected.

## Contracts (`aaa.promotion.crossed.v1`; primary and independent implementations agree on every one)

The metric is the geometric mean over slices of the per-cell Jeffreys-smoothed error ratio, challenger / reference.

| Contract | Reference | Error ratio [95%] | Rule | Verdict |
|---|---|---|---|---|
| C1 WM over the strongest learned non-WM arm | policy_aux:plan | 0.481 [0.377, 0.600] | superior < 0.75 | **PROMOTE** |
| C2a the training-learned library is causal | wms:online+empty | 0.532 [0.416, 0.660] | superior < 0.80 | **PROMOTE** |
| C2b the library model is causal | wms:empty | 0.452 [0.354, 0.560] | superior < 0.75 | **PROMOTE** |
| C3 WM over the best non-learned shortcut | tells | 0.516 [0.404, 0.641] | superior < 0.75 | **PROMOTE** |
| C4 online adaptation | wms (frozen) | 0.873 [0.838, 0.932] | superior < 0.95 | **PROMOTE** |
| C5 structured over black-box WM | wm:d1 | 0.502 [0.392, 0.632] | superior < 0.75 | **PROMOTE** |
| N in_distribution not worse | policy_aux:plan | 0.417 [0.290, 0.564] | noninferior <= 1.0 | **PROMOTE** |
| N two_fault not worse | policy_aux:plan | 0.474 [0.372, 0.591] | noninferior <= 1.0 | **PROMOTE** |
| N novel_literals not worse | policy_aux:plan | 0.370 [0.257, 0.504] | noninferior <= 1.0 | **PROMOTE** |
| N novel_grammar not worse | policy_aux:plan | 0.508 [0.390, 0.634] | noninferior <= 1.0 | **PROMOTE** |
| N library_B not worse | policy_aux:plan | 0.695 [0.632, 0.764] | noninferior <= 1.0 | **PROMOTE** |

The post-run audit (`confirm.audit`) recomputed every stored verdict from the bits and found no disagreements.

## What this licenses, and what it does not (narrowed after the independent review)

The [independent adversarial review](independent_review_opaque.md) recomputed every ratio with its
own code (exact match) and found no invalidating issue and no leakage. It found five
scope-narrowing issues, all accepted. The claim therefore reads:

**Licensed, within `aaa.python.opaque.v0`, at a budget of two real test runs.** An AAA agent that plans
with WM-S makes **about half the errors** (C1 error ratio 0.481 [0.377, 0.600]; 0.32 to 0.71 depending on
the library-table seed) of:

- **the one learned model-free architecture trained here** (a ~1M-parameter transformer, 20k steps),
  run inside a planner with the same verifier and budgets (beam 8 on second edits);
- the strongest non-learned shortcut found, grammar legality (0.516). The gain persists on tasks
  the shortcut fails (0.488);
- the same agent without a library learned in training (0.532), and without any library knowledge
  (0.452);
- a black-box neural world model (0.502).

WM-S is an exact interpreter of the visible Python plus a tabular model of an opaque library,
**abduced from real observations of the same fixed six-function library that four of the five
test slices use**. The per-slice non-inferiority contracts pass, but they are **weak**: they are
checked against `policy_aux:plan` only, with a 1.0 bar that the observed ratios (at most 0.70) clear
easily. WM-S is weakest on the one slice where the library changes (`library_B`, 0.695).

**Online updating (C4) is fragile.** The pre-declared contract passes (0.873 [0.838, 0.932]), but the
benefit mostly fills gaps in an incomplete training table, mostly within the episode. It is about
zero for the most complete table (seed 100), and a seed-level t(4) interval reaches 0.975, above the
0.95 threshold. The claim is "online updating helps when the learned table is incomplete", not a
general adaptation claim.

**Not licensed:**

- that a *generic* neural world model helps (the black-box WM loses, and so does JEPA);
- that WM-S learned Python (the interpreter is hand-written knowledge);
- that the model-free reference is strong or scaled;
- transfer beyond this generated benchmark family, or beyond a library shared by training and test;
- freedom from forgetting (adapting to a changed library costs -0.067 on the old one in development);
- that compute is matched (WM-S uses about 0.3-0.4 CPU seconds per episode; the matched resource is
  real test runs).

**Design shortfalls recorded, not repaired.**

The prospective benchmark design described a nearby-repair clause, but the
frozen `generator.accept` does no neighborhood search. The clause excludes
other domain-equivalent fixes while exempting the true fix "and its
equivalents," so its intended additional constraint is ambiguous. On an
independent 40-task pilot probe, 5 tasks had a one-edit candidate that passed
every visible test while failing domain equivalence. These examples do not
violate the clause as written. The confirmation scores the full domain, and a
visible-test pass cannot be treated as a correct repair. See
[AAA-217](../issue_ledger.md) and the [PR review](pr30_independent_review.md).

- There were 5 initializations and 20 streams per slice. The design brief called for 10-20 and at least 30.
- The thresholds were fixed after part of the evaluate-stage results existed (before any
  confirmation data). They were not derived from a minimum-detectable-effect calculation.
- The freeze was pushed remotely only about 4 h into the run, and `draft("confirmation")` is not
  guarded by the admission. Every log and timestamp is consistent with a single run, but that cannot
  be proven.
- The declared CPU limit per episode was not checked during the run.

A successor phase should fix all of these prospectively. The spent confirmation identities are never reused.
