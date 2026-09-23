# Pre-scale readiness, 2026-09-23

**Verdict: `READY_FOR_NEXT_PHASE_DESIGN`.** The current repository supports
designing controlled Python coding and capacity experiments. It does not yet
support promoting a scaled candidate under the frozen `aaa.1k.v2` K5 path.
`AAA-180` is an explicit **hard precondition for any future promotion**. This
verdict does not assert that a larger model is needed, that 105M is feasible,
or that a Python coding capability exists. Exact review and validation are
recorded in [the pre-scale review](pre_scale_review_2026-09-23.md).

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

## Promotion blocker: `AAA-180`

The frozen v2 confirmation path omitted Monash primitives when calling its K5
decision. With Monash supplied, the primary decision returns `INCONCLUSIVE`
and independent recomputation returns `FAIL`: the single-series Saugeen group
has no series dimension to resample in the primary crossed bootstrap. There
was no v2 challenger, so no promotion result changed. The frozen source and
evidence remain unchanged.

Before any successor can promote a candidate, its **versioned, frozen-before-
confirmation** protocol must pass all declared Monash primitives to K5 and
state the single-series estimand. A defensible option is to keep the sole
series fixed and resample shared initializations, describing the interval as
conditional on that series; another predeclared rule may be used if justified.
The primary decision and a structurally independent recomputation must agree
on a fixture containing a single-series group, including point values,
interval status and final verdict. A mismatch or missing group must fail
closed. The rule must be tested before any held-out candidate evidence is
observed. The frozen v2 path must never be reused for promotion.

## Next phase and scaling gates

The first deliberate specialization is [Coding, beginning with
Python](research_direction.md). This expands task and representation
complexity; it is a reason to *investigate* capacity, not evidence that
capacity is already limiting. For each material increase:

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
model. At the 2026-09-23 probe, the mounted HDD had 412,369,637,376 bytes
available and the NVMe root had 15,672,995,840 bytes available. This supports
small next-phase diagnostics on the HDD, not an unspecified coding corpus or
105M training run. A 105M parameter-only memory estimate omits activations,
optimizer and temporary memory; feasibility requires a measured configuration-specific
probe and sustained throughput estimate. Hardware not yet installed is not
part of that probe.
