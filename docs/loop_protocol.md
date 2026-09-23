# The AAA improvement loop — protocol `aaa.loop.v1`

> **Status: independently reviewed v1 governance.** Iterations 0001--0003 were
> the `aaa.loop.v0-pilot` run described in
> [`loop_pilot_report.md`](loop_pilot_report.md#evaluating-the-loop); nothing
> in that historical pilot passed attack. Iteration 0006 later completed the
> first real freeze → durable claim → fresh confirmation → independent
> recomputation → promotion cycle, including a safely burned pre-observation
> attempt. Independent review of that cycle earned `aaa.loop.v1`. Historical
> records keep their recorded `aaa.loop.v0-pilot` identity; v1 is the protocol
> for future work, not a relabeling of old evidence.

## The loop

```text
Observe → Classify → Diagnose → Hypothesize → Falsify → Intervene Minimally
        → Attack → Confirm Fresh → Decide → Preserve → Repeat
```

| Stage | Meaning | What must exist before leaving it |
|---|---|---|
| Observe | a real measured weakness, from retained primitives, never invented to exercise the process | an observation artifact recomputed from primitives; observed evaluation evidence is used to state the problem only |
| Classify | say what kind of problem it is before touching the model (implementation, measurement, benchmark, estimator, optimization, representation, capacity, credit assignment, input ambiguity, instability, generalization, adaptation, baseline competitiveness, calibration, unsupported claim, insufficient evidence, expected tradeoff) | a classification that may change after diagnosis, with the change recorded |
| Diagnose | find the mechanism, with instrumentation and controlled comparisons, on identities that are not the observed evaluation cells | diagnostic evidence on `diagnostic` identities |
| Hypothesize | competing explanations with different measurable predictions, including at least one under which no model change is warranted | each hypothesis with its prediction and the observation that would contradict it |
| Falsify | tests able to prove each important hypothesis wrong, with the numeric verdict rule declared and **committed before the data exist** | computed verdicts (`SUPPORTED`, `PARTIAL`, `CONTRADICTED`, `NOT_RESOLVED`) |
| Intervene minimally | the smallest change the diagnosis supports, in the order: correctness → measurement → representation/encoding without parameters → training rule/optimization → minimal architecture → minimal parameter increase → capacity. One interpretable change at a time | the sentence *"this change should improve X because diagnosis Y showed mechanism Z"*; declared candidates and a selection rule committed before development runs |
| Attack | try to break the surviving challenger on `attack` identities: fresh initializations, nearby conditions, unaffected families, long horizons, the intervention's own neighbourhood, stability margins, ablations, implementation, causality, cost | every attack criterion `PASS`, or the challenger is rejected |
| Confirm fresh | freeze, then observe fresh `confirmation` identities once | a committed freeze manifest; identities marked spent before the first cell runs |
| Decide | `PROMOTE`, `REJECT` or `INCONCLUSIVE`, recomputed from primitives | any `FAIL` → `REJECT`; all `PASS` → `PROMOTE`; otherwise `INCONCLUSIVE` |
| Preserve | every attempt, failed or not, with its reason | a validated iteration record |
| Repeat | only from the resulting state | a new iteration with its own identity salt |

## Two nested loops, mechanically separated

**Inner loop — diagnosis and development.** May iterate. May use
`diagnostic`, `development` and `attack` identity blocks, inspect failures,
add read-only instrumentation, build probes, tune declared hyperparameters,
reject candidates. Every stage writes one strict-JSON artifact and the design
that judges it is committed first.

**Outer loop — promotion.** Runs once per frozen challenger:

1. freeze the challenger source, the evaluation protocol, the criteria, the
   estimands and the thresholds (`freeze.py`);
2. reserve fresh `confirmation` blocks and prove them disjoint;
3. **commit the freeze**; `confirm` refuses a manifest that is untracked or
   differs from `HEAD`;
4. recompute the champion fingerprint, the confirmation-source fingerprint and
   the frozen content, and refuse on any difference;
5. atomically reserve every confirmation block on immutable remote Git claim
   refs, then mark the local ledger `spent`, *before* the first cell runs;
6. run, write the primitives (never overwriting), and decide;
7. recompute independently (`recompute.py`: a different bootstrap algorithm,
   thresholds read from the freeze, identities checked against the ledger);
8. record the attempt in the iteration record whatever the outcome.

A challenger changed after confirmation is observed is a new challenger and
needs new confirmation identities. A spent confirmation identity is
contaminated for selection forever.

What enforces the separation:

| Rule | Mechanism |
|---|---|
| confirmation never informs selection | `identities.require_usable(..., purpose="selection")` refuses `confirmation` blocks; every inner-loop stage calls it |
| identities are fresh | per-iteration salt; `prove_fresh` intersects every loop seed with every AAA-1K seed below index 100,000 in every namespace and every seed recorded in AAA-1K evidence |
| no reuse | blocks never overlap; immutable remote claim refs survive crashes, resets, worktrees and fresh clones; `spent` never reverts locally |
| the observed freeze is the committed freeze | `freeze.require_committed` |
| frozen bytes exist in the recorded commit | `freeze.require_committed_confirmation_source` compares every frozen byte and executable bit with `HEAD` |
| the challenger did not change | `freeze.verify_freeze` covers the complete loop package and AAA-1K phase source, including the CLI admission path |
| stored conclusions are not evidence | `iteration.validate_iteration` recomputes a `DECIDED` outcome from the confirmation primitives |
| failures stay visible | the validator refuses a record that drops any candidate an artifact declares or attacks, or leaves a rejection unexplained |

## State model

```text
OBSERVED → CLASSIFIED → DIAGNOSED → TEST_DEFINED → CHALLENGER_CREATED
  → CHALLENGER_ATTACKED → FROZEN → CONFIRMED → DECIDED → PRESERVED
TEST_DEFINED → NO_INTERVENTION → PRESERVED
CHALLENGER_CREATED → REJECTED → PRESERVED      (every candidate fails development)
CHALLENGER_ATTACKED → REJECTED → PRESERVED     (the attack breaks the challenger)
```

Each state requires artifact roles (`champion`, `observation`, `diagnosis`,
`development`, `attack`, `freeze`, `confirmation`); the validator checks that
each exists, still has its recorded SHA-256 and parses as strict JSON. A
record cannot claim `PROMOTE` without `DECIDED`, cannot freeze a candidate in a
rejected iteration, and cannot promote with parameter accounting that
disagrees with the counted arrays.

## Evidence

* strict RFC 8259 JSON: written with `allow_nan=False`, read with a hook that
  refuses `NaN` and `Infinity`; undefined quantities are `null`;
* an observed confirmation artifact and a freeze are written once
  (`overwrite=False`);
* primitives and interpretation are kept apart: stage artifacts hold per-cell
  primitives; iteration records hold narrative and reference artifacts by
  SHA-256;
* the historical pilot command `python -m research.aaa_1k_loop reproduce
  <stage>` retains its original exact comparison; post-audit stages use
  `python -m research.aaa_1k_loop.stages4 reproduce <stage>`;
* post-audit reproduction is bitwise under `--exact` on the matching Zen 3
  evidence platform; the default cross-platform mode requires identical
  identities, structure, divergence classifications and adjudicated verdicts
  while reporting numerical drift in chaotic trajectories (`AAA-173`);
* historical evidence is never rewritten.

## Capacity is an experiment, not a reward

A parameter increase is considered only when all of the following hold:

1. an important failure persists across several fresh tests;
2. it survives benchmark and measurement scrutiny;
3. diagnosis points specifically at representation or capacity;
4. parameter-neutral remedies fail;
5. optimization and credit assignment are shown not to be the blocker;
6. a controlled capacity sweep around the current size shows a reproducible gain;
7. the gain transfers to fresh held-out evidence;
8. the larger model survives baselines, ablations and independent review.

Any increase reports old and new counts, the absolute and relative change, the
adaptive-state and compute change, and what mechanism the parameters buy.

<a id="known-gaps-to-close-before-v1"></a>

## Pilot gaps and the rules for the next iteration

These rules were chosen only in final review. They are prospective: they do
not retroactively make the pilot's deviations predeclared, and the protocol
remains `v0-pilot` until a real challenger completes fresh confirmation.

1. **Candidate budget.** Every iteration must commit a finite candidate budget
   before the first development identity is observed. The pilot's budget of
   three remains a disclosed during-execution choice, not precedent.
2. **Screen-failed tradeoff candidates.** The pilot advanced one candidate
   that failed a single non-regression screen to attack, as a labelled
   deviation. A failed screen is now an unconditional veto. A future tradeoff
   policy is a new protocol version and must be fixed before development; it
   may not be invented after a failure. L-4 remains a historical deviation.
3. **Screens without uncertainty.** Development screens used point estimates
   against a 2% margin. They happened to be right; the permanent protocol
   must use a predeclared interval or other uncertainty rule. A point estimate
   alone cannot pass a margin; an interval crossing the margin is unresolved.
4. **Attack instruments.** An attack criterion may not depend on an
   instrument already known to be fragile (iteration 0003's R4 used an
   initialization-unstable probe). Any instrument used as a gate needs a
   predeclared validation block and must pass that validation before attack.
5. **Generic freeze.** `freeze3` / `confirm3` / `recompute3` are specific to
   iteration 0003's claim. A second real use should decide what generalizes;
   the pilot deliberately did not build a framework.
6. **Claims as challengers.** They are permitted only as measurement repairs.
   Promotion freezes the exact old and proposed wording, changes active
   documentation and errata only, and never implies a model promotion. It
   requires the same attack, freeze, fresh evidence and independent decision
   recomputation as a model challenger.
7. **Unregistered observations.** Two informal looks happened (one diagnostic
   cell; one arbitrary sanity stream). Both are disclosed; the protocol should
   require a scratch identity namespace so that even sanity checks are
   ledgered. Scratch identities may never satisfy development, attack or
   confirmation criteria and every look must be recorded.

## Status after iterations 0004–0006 (2026-09-22)

The seven gaps above were made hard rules in `research/aaa_1k_loop/iteration4.py`
before any iteration-0004 development identity was observed, and iterations
0005 and 0006 imported the same judgement functions unchanged:

1. **Candidate budget:** a precommitted budget per iteration (two, one, one).
2. **Screen-failed tradeoff candidates:** a failed screen is a veto; none
   advanced.
3. **Screens without uncertainty:** non-inferiority on crossed-bootstrap
   interval bounds; a straddling interval is not a PASS (it stopped c7).
4. **Attack instruments:** only instruments validated earlier in the
   iteration.
5. **Generic freeze:** `freeze.py` takes an iteration-specific source set;
   `recompute4.py` is iteration-agnostic.
6. **Claims as challengers:** not exercised.
7. **Unregistered observations:** a scratch block for shakedowns. The two
   post-hoc looks at already-observed cells are disclosed in the records.

Stability is a primary gate of screen, attack and confirmation. Every
confirmation criterion is three-valued, and borderline evidence resolves to
INCONCLUSIVE (audit R-04).

**The outer path has run once for real** (iteration 0006: freeze, durable
remote claim, fresh confirmation, independent recomputation, PROMOTE). Its
first attempt aborted before observation on an admission defect (`AAA-171`);
the burned identities were not reused. The version remains `v0-pilot` pending
an independent review of that work. See
[`loop_report_0004_0006.md`](loop_report_0004_0006.md).

*Update (2026-09-22):* the paragraph above records the state before review.
That review has since completed
([`pr20_independent_review.md`](pr20_independent_review.md)), and the current
governance version is `aaa.loop.v1`, as the status note at the top of this
document says. The iteration 0001--0006 records and evidence remain
`aaa.loop.v0-pilot` and are not relabelled.

Post-audit iterations run from `research/aaa_1k_loop/stages4.py`
(diagnoses) and `outer4.py`/`outer5.py`/`outer6.py` (develop, attack,
freeze, confirm). Their confirmation paths import only their own frozen
source sets (tested).

## Running an iteration

The pilot's commands are in [`loop_pilot_handoff.md`](loop_pilot_handoff.md).
A new iteration adds its blocks to `INNER_LOOP_BLOCKS` in
`research/aaa_1k_loop/cli.py` under its own iteration id, runs
`ledger-sync`, commits each design before running it, and finishes with
`records` and `validate`.
