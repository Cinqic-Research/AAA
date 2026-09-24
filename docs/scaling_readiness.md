# Scaling readiness

**Current verdict (2026-09-23, Python-first transition):
`PYTHON_PHASE_IN_DEVELOPMENT`.** `aaa.python.v0` exists as a safe, causal,
recomputable development environment with a minimal learner, and its first
development result is negative. The `AAA-180` promotion precondition is
satisfied in code by a tested prospective successor
([`aaa.promotion.crossed.v1`](promotion_contract.md)). No Python confirmation
is admitted, no candidate or promotion criteria are declared, and nothing
supports a capacity increase yet. This verdict does not assert that a larger
model is needed, that 105M is feasible, or that any Python capability exists.

The previous verdict, `READY_FOR_NEXT_PHASE_DESIGN`, is recorded in
[the pre-scale review](pre_scale_review_2026-09-23.md). It described the tree
before the Python phase and the `AAA-180` successor existed.

## Evidence boundary

Champion 1 is the 994-trainable-parameter recurrent reference. On the tested
families, matched online learning beats frozen twins; hidden state and the
previous-error input matter in appropriate conditions. Champion 1 repairs the
long-horizon M2 runaway while its coarse-observation M1 weakness remains.
Analytic methods remain very strong on smooth fully observed motion. The error
head's auxiliary loss does not materially improve prediction, and the head is
only a weak uncertainty-related signal. These are task-specific findings.

The completed v2 phase had **no challenger** under its preregistered
development screen. That screen's unclipped-reference stability margin
materially capped candidate learning rates; attractive clipped configurations
outside the rule were correctly excluded. This says nothing about all possible
1K architectures or hyperparameters. Its capacity diagnosis is
`NOT_CAPACITY_LIMITED` on the declared dot/external mixture through roughly 4K
parameters. The coarse-observation deficit persists at that size. Neither
result justifies a jump to 105M for those tasks.

Round 3 measured a positive paired-change effect on its identities. Fresh v2
identities did not resolve a positive effect. The diagnostic
`python tools/check_adaptation_parity.py` replays the 5-by-24 round-3 grid:
the historical and batched designs agree on all four branch MAEs and the
difference-of-differences in every cell, including the Champion 1 replay.
Thus the tested measurement paths are equivalent on those identities; the
v2 result is a genuine **failure to replicate on fresh identities**, within
the precision of this diagnostic. Neither result is erased, and the v2 result
does not establish a negative adaptation effect. V2 retention on fresh
identities resolves favourably; this is not general immunity to forgetting.

External benchmarks are descriptive and do not demonstrate Python coding,
generalization to repositories, general agency or autonomy. Observation-noise
formal A/B remains `NOT EXECUTED`. Failed and superseded attempts remain
retained with their original identities.

## `AAA-180`: repaired prospectively; the frozen v2 defect is retained and forbidden for promotion

The frozen v2 confirmation path omitted Monash primitives when calling its K5
decision. With Monash supplied, the primary decision returns `INCONCLUSIVE`
and the independent recomputation returns `FAIL`: the single-series Saugeen
group has no series dimension to resample in the primary crossed bootstrap.
The recomputation also skipped Monash silently for a challenger without Monash
primitives (`AAA-182`). There was no v2 challenger, so no promotion result
changed. The frozen source and evidence remain unchanged, and the defect still
reproduces (`python tools/check_promotion_successor.py`).

The versioned successor `aaa.promotion.crossed.v1` meets every precondition
this section used to list:

- every declared group must be present, and an omitted group, even for one arm, is
  `INVALID_EVIDENCE` and cannot promote;
- each group's design is declared before observation. A single-series group
  is `conditional_on_single_series`: the sole series is held fixed, shared
  initializations are resampled, and the result's scope is reported as
  conditional on that series. No series dimension is fabricated;
- a primary index-resampling evaluator and a structurally independent
  count-weighted recomputation must agree on point value, interval status,
  bounds (within a derived Monte Carlo tolerance), criteria and verdict.
  Otherwise the result is `DISAGREEMENT`, which never promotes;
- regression tests cover single- and multi-series agreement, missing, extra,
  malformed, non-finite and mismatched primitives, and injected disagreement,
  all without observing held-out evidence;
- the frozen v2 contract `aaa.1k.v2.k1-k5` is in `FORBIDDEN_CONTRACTS`, and the
  v2 registry's confirmation blocks are all spent and bound to the committed
  freeze. `freeze.verify` refuses any new block set, so the frozen path cannot
  be re-armed.

A future phase that promotes anything must declare its contract before
held-out observation and use this successor or a later versioned one.

## Next phase and scaling gates

The first deliberate specialization is [Coding, beginning with
Python](research_direction.md). This expands task and representation
complexity; it is a reason to *investigate* capacity, not evidence that
capacity is already limiting.

`aaa.python.v0`'s learner has 153,600 trainable parameters, two thirds of them
in a one-hot output head, and no optimizer state. Its development result
(near the majority baseline, beaten by surface heuristics, no resolved online
benefit) points first at representation, task design and tool use, not at
capacity. The declared next steps are parameter-neutral: a tool-using baseline
that runs visible tests, representations that expose execution structure, and
a feedback curriculum. Capacity comes after those, against smaller controls.

For each material increase:

1. Define a distinct capability deficiency and reproduce it.
2. Diagnose likely causes and test parameter-neutral remedies, including
   representation, context, data, optimization and evaluation defects.
3. Show that capacity is plausibly limiting against parameter-matched and
   smaller controls, then choose the smallest meaningful increase.
4. Freeze identities, metrics, baselines, stability criteria, resource budget
   and promotion rules before held-out observation.
5. Develop, attack and confirm on separate tasks or repositories; retain all
   failed attempts and raw primitives.
6. Compare held-out gain with parameter, adaptive-state, compute, memory,
   storage and maintenance cost. Retain the smaller model if the larger one
   has not earned its parameters.

A future stability rule must evaluate the *actual clipped or otherwise
stabilized configuration* under test, with symmetric criteria across
architectures. It must be preregistered; it cannot be fitted to the v2
excluded candidates after seeing their results. A coding score must preserve
the causal and contamination boundaries in the research-direction document.

## Engineering and feasibility gates

The exact-head review must record lint, format, typing, tests, lock checks,
fingerprints, recomputation, report generation, package installation and CI.
Local CPU reproduction and prior CUDA qualification have different scopes.
The frozen v2 formal runs were CPU runs. Stable CPU/CUDA agreement and chaotic
divergence at unstable settings must be reported separately. On Zen 3, a
bitwise claim requires the documented matching platform and dependencies;
else use identity/structure and independently recomputed verdicts while
reporting numeric drift.

FLOWBOX currently has a Ryzen 7 5700G, 16 GB RAM and an RTX 2060 with 6 GB
VRAM. The machine profile and current data root are in [hardware](hardware.md).
Use `$AAA_DATA_ROOT` on the dedicated HDD for large external and experiment
outputs; preflight projected size and free space so exploratory work does not
fill the NVMe. Checkpoint/resume and atomic-evidence behavior must be proved
for the selected future workload, rather than inferred from the small current
model. At the 2026-09-23 pre-scale probe, the mounted HDD had 412,369,637,376
bytes available and the NVMe root had 15,672,995,840 bytes available. During
the Python-first transition the HDD had 412,368,965,632 bytes available.
`aaa.python.v0`'s complete development run writes about 1.4 MB of evidence
(about 42 MB more with its twelve resumable checkpoints). This supports
small next-phase diagnostics on the HDD, not an unspecified coding corpus or
105M training run. A 105M parameter-only memory estimate omits activations,
optimizer and temporary memory; feasibility requires a measured configuration-specific
probe and sustained throughput estimate. Hardware not yet installed is not
part of that probe.
