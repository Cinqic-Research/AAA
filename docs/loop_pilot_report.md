# Birth of The Loop — pilot report

> **Result: three iterations, eight candidates, eight rejections, no
> promotion, no confirmation.** Every candidate was rejected by rules committed
> before its data existed, at development screening or at attack. No freeze
> was written, no confirmation identity was reserved or spent, and
> **Champion 0 is unchanged**. The outer (promotion) loop was therefore never
> run on real evidence; it is exercised only by failure-injection tests. That
> falls short of the brief's definition of done ("fresh confirmation was
> frozen before observation"), and it is reported as unmet rather than forced.
> Reaching confirmation would have required overriding the loop's own
> rejections.

This is an implementer's report. It is not independent review and should not
be read as approval.

## Identity

| Item | Value |
|---|---|
| base commit (`main`) | `1ce716c42500003881e379f6d23dc0d807301001` |
| branch | `opus/birth-of-the-loop` |
| protocol | `aaa.loop.v0-pilot` ([`loop_protocol.md`](loop_protocol.md)) |
| implementation | `research/aaa_1k_loop/` (sibling of `research/aaa_1k/`, outside its fingerprint) |
| Champion 0 | `aaa1k-champion-0`, phase `aaa.1k.v1`, fingerprint `5ce6e019bd5389c5d8d466cc2be809ff9777e3bd6acba10917bc6a7ce98f771e` (recomputed; unchanged on this branch) |
| iterations | `aaa1k-loop-0001`, `-0002`, `-0003`, each on its own identity salt |
| evidence | `docs/evidence/aaa1k_loop_000{1,2,3}/`, ledger `benchmarks/aaa1k_loop_identity_ledger.json` |
| tests | `tests/test_aaa_1k_loop.py` (60 tests, mostly failure injection) |

## Champion 0

Derived from the repository, not typed: `docs/evidence/aaa1k_loop_0001/champion_0.json`,
re-verified by `python -m research.aaa_1k_loop champion --verify`.

| Field | Value |
|---|---|
| model | `aaa.1k.gru.v1`, `AAA1KGRU-3x16x2`, 994 parameters (formula 994) |
| adaptive state | 994 + 16 hidden + 0 optimizer + 404 TBPTT capacity = 1414 |
| hyperparameters | lr 0.03, TBPTT 4, lambda 0.25, clip 10 (controls: lr 0.01, no clip) |
| benchmark | `aaa.1k.benchmarks.v1` |
| development evidence | `docs/evidence/aaa_1k_development_selection.json` (hash recorded) |
| round-3 evidence | `docs/evidence/aaa_1k_evaluation_round3.json`, 720 cells, offsets 40,000/50,000/60,000 |
| capability vector | Q1 POSITIVE, Q2 POSITIVE, Q3 POSITIVE, Q4 INCONCLUSIVE (memory families NEGATIVE), Q5 INCONCLUSIVE, Q6 weak (Spearman 0.45), Q7 mixed and exploratory |

Round 1, 2 and 3 evidence, the Sol review and every AAA-1K report are
untouched; a test checks the round-1/2/3, selection and characterization
evidence against the SHA-256s in the closure audit.

## Observe: what Q4's "minority tail" actually is

Recomputed from round-3 primitives (`observation.json`; evaluation evidence,
used only to state the problem):

| Family | Q4 cell mean (ungated − gated) | streams favouring ungated | contribution to aggregate |
|---|---|---|---|
| motion_compat | +7.6e-05 | 0 / 72 | +3.8e-05 |
| occlusion_v1 | +1.8e-04 | 0 / 24 | +3.1e-05 |
| aba_v1 | +2.9e-06 | 7 / 24 | +4.8e-07 |
| **coarse_speed_v1** | **−7.1e-04** | **24 / 24** | **−1.17e-04** |
| all | −4.84e-05 | 31 / 144 | |

Without `coarse_speed_v1` the aggregate is **+8.3e-05**. The "minority" is one
whole family, in every initialization, not a scattered tail. On that family
the gated champion (2.33e-3) is no better than its own state-reset ablation
(2.29e-3), the stateless MLP (2.34e-3) or dead reckoning (2.34e-3); the
ungated control reaches 1.62e-3. Separately, two initialization-3 cells are
extreme outliers (online error worse than the frozen copy, clip active on ~10%
of updates).

## Diagnose: nineteen hypotheses, three fresh diagnostic blocks

All hypotheses and verdict rules were committed before the data they judge
(`git log` shows each design commit preceding its evidence commit). Streams
come from `diagnostic` ledger blocks; initializations are the champion's five.

**Round 1** (`diagnosis.json`, 16 coarse streams × 5 initializations): the
deficit reproduces off the round-3 cells — +6.5e-4 [+5.8e-4, +7.2e-4], 16/16
streams, every initialization.

| Hypothesis | Verdict | Deciding measurement |
|---|---|---|
| H2 slow recovery after switches | CONTRADICTED | deficit *larger* without switches (148%) |
| H3 gate saturation | CONTRADICTED | saturated gate fraction 0.0 |
| H4 error-input interaction | CONTRADICTED | removing input 3 makes it worse (−13% closure) |
| H5 learning rate | CONTRADICTED | lr 0.01 makes it worse (−23%) |
| H6 width | CONTRADICTED | a 16-unit, 354-parameter ungated RNN keeps 86% of the lead |
| H7 initialization | CONTRADICTED | all five initializations show it |
| H8 TBPTT | CONTRADICTED | T = 16 changes nothing (−0.8%) |
| H9 event-dominated | CONTRADICTED | 5% of the deficit within 5 steps of a switch |
| H11 statistical tail | CONTRADICTED | 100% of fresh streams show it |
| H12 keep-gate horizon (the implementer's favourite) | CONTRADICTED | keep bias +2 makes it worse (−25%) |
| H1 gating failure, H10 legitimate width tradeoff | NOT_RESOLVED in round 1 | resolved by round 2 below |

Round 1 left one clue: in the slow regime the online gated model is no better
than its persistence-like frozen copy.

**Round 2** (`diagnosis_2.json`, fresh block):

| Hypothesis | Verdict | Deciding measurement |
|---|---|---|
| H13 contractive gated dynamics cannot carry the sub-quantum phase | **SUPPORTED** (all four predeclared parts) | champion's one-step Jacobian has no sign-alternating mode (median 0.0, min real eigenvalue +0.16) while the ungated control's is 0.77; slow-regime deficit 1.07e-3 vs fast 2.4e-4; keep bias −2 closes 73% (+4.6e-4 [+4.2e-4, +5.1e-4], all initializations); keep bias +2 does not |
| H14 learning speed | CONTRADICTED — and exposed M2 | on 1120-step fixed-speed streams the deficit *grows* 25-fold |
| H15 coarse_speed_v1 rewards memory without switches | **SUPPORTED** | ungated-over-stateless advantage without switches is 99% of its switching size; the champion is *worse* than stateless at fixed speed |

H13 contradicts H1 (a parameter-neutral change to the gated model closes most
of the gap) and H10 (width is not the cause). Reset-bias probes destabilized
two of five initializations.

**Round 3** (`diagnosis_3.json`, fresh long-horizon block, four families):
the champion diverges (mean error > 2× persistence) in **60 of 160** long
coarse cells and **0 of 160** long bouncing/occlusion cells. Without input 3:
5/160. At lr 0.01: 0/160. State-reset ablation: 51/160 (recurrence not needed).
Diverged cells show |input 3| up to ~214 (stable ~4), gradient norms ~1.5e5,
the clip active on 75% of updates, output-head norm ~113. H16–H19 SUPPORTED.

Two mechanisms, therefore:

* **M1** — zero gate biases put the gated core in a contractive,
  non-alternating operating point that cannot track the slow regime's
  roughly period-2 sub-quantum phase within an episode (H13). This carries the
  Q4 sign.
* **M2** — a runaway through the previous-error feedback channel on long
  quantized streams at lr 0.03 (H16–H19). By signature it is the likely source
  of round 3's two burst cells; those evaluation cells were not re-run.

## Iteration 0001 — M1, the coarse deficit

Intervention sentence: *this change should reduce the coarse deficit because
diagnosis round 2 showed zero-bias gates confine the champion's Jacobian to
non-alternating modes (H13); a negative keep-gate bias restores one.* Only the
initial `b_z` changes: same equations, 994 parameters, 1414 state scalars,
optimizer and hyperparameters.

**Development** (`development.json`, `development_2.json`; 48 fresh streams ×
5 initializations; rule E1 stability / E2 target / E3 ≤ 2% regression on
every other plan entry):

| Candidate | coarse | other entries | stability | Verdict |
|---|---|---|---|---|
| c1, `b_z` −1 | **+140%** | −11% to −14%; occlusion +3.3% | 2 divergent cells (champion 0) | REJECTED (E1, E2, E3) |
| c2, `b_z` −2 | **−22%**, 8/8 streams | −15% to −19%; **occlusion +5.0% [+3.4%, +6.8%]** | 0 divergent | REJECTED (E3) |
| c3, −2 on 8 units / +2 on 8 | −13% | motion_compat +3.6% to +11.0%; occlusion +2.1% | 0 divergent | REJECTED (E3) |

A keep +2 probe beside c3 improves occlusion by 5.4% (occluded steps 7.07e-3
vs 8.12e-3) and costs 69–91% on smooth motion: the initial keep-gate
operating point is a genuine tradeoff axis, as the mechanism predicts.

**Deviation L-4.** c2 failed only E3. It was advanced to attack as a labelled
*tradeoff challenger*, with the occlusion criterion to be carried into any
freeze at equal or stricter strength. That rule was not predeclared; it is
recorded as a protocol gap.

**Attack** (`attack.json`; fresh streams and five fresh initializations):

| Criterion | Result |
|---|---|
| A1 target, fresh initializations | **FAIL**: 4 of 5 initializations improve by ~24%; on the fifth, c2 degrades to 2× persistence (−9.3e-4 overall, interval [−8.0e-3, +5.6e-4]) |
| A2 nearby coarse conditions | **FAIL**: quanta 0.004/0.006, regime 40, fixed speed improve; regime 90 fails on the same initialization; speeds 0.15/0.35 worse in 3 of 5 |
| A3 other families | PASS: occlusion regression replicates (+4.2% [+2.6%, +5.9%]); others −13% to −18%; Q1 POSITIVE; hidden state not negative |
| A4 long-horizon stability | PASS: c2 1 divergent cell vs champion 16 |
| A5 Q2/Q5 | PASS |
| A6 stability margin (stage 0 replayed) | PASS: unclipped boundary 0.3 for both |
| A7 intervention neighbourhood | **FAIL**: bias −1.5 fails on the same initialization; −2.5 does not |
| A8 implementation (gradients, resume, checkpoint, accounting, branch hash) | PASS |
| A9 causal boundary | PASS |
| A10 compute | PASS (ratio 1.002) |

c2 rejected. Its failure has M2's signature (not diagnosed further). Iteration
0001 ends **REJECTED** before any freeze.

## Iteration 0002 — M2, the runaway

Starts from 0001's state on a new salt. Candidates, in increasing departure,
all parameter-neutral and bitwise identical to the champion whenever
|input 3| ≤ K: c4 clip at ±8 (fixed by public arithmetic), c5 at ±4, c6 remove
input 3.

| Candidate | long coarse divergent (champion 27) | standard plan entries | Verdict |
|---|---|---|---|
| c4, K = 8 | 27 | identical to the champion | REJECTED (S1) |
| c5, K = 4 | 28 | identical (occlusion −0.2%) | REJECTED (S1) |
| c6, K = 0 | **3** | **+4% to +41%** (aba +41%, dynamics_change +39%) | REJECTED (S4) |

Bounding the input does nothing; removing the channel fixes the runaway and
costs the adaptation the channel provides. The magnitude form of H16 is
falsified. The untested refinement for a future iteration: on alternating
quantized targets `−e(t−1)` is predictive, so SGD may learn a closed-loop
gain above one. Iteration 0002 ends **REJECTED** in development.

One unregistered sanity run (arbitrary seed 4242, not a ledger identity) was
made while checking bitwise equality; it is disclosed in
`research/aaa_1k_loop/iteration2.py`, committed before development ran.

## Iteration 0003 — the Q4 interpretation (a claim, not a model)

No model challenger survived, but the diagnoses contradict two documented
claims: that Q4's negative mean is a minority tail, and that
`coarse_speed_v1` "genuinely tests hidden-regime inference" because the
recurrent model loses to the stateless control at fixed speed (that probe used
only the gated champion). The challenger was the corrected interpretation;
promotion would have changed documentation only.

| Attempt | Attack | Result |
|---|---|---|
| claim v2 | T1 nearby constructions | **FAIL**: at quantum 0.006 and speeds 0.10/0.25 only ~30% of the memory advantage survives without switches; at quantum 0.004 the champion gains slightly at fixed speed |
| | T2 16-unit RNN, T3 fresh initializations, T4 fast-dynamics gated probe | PASS |
| claim v3 (scoped to the declared construction, carrying T1's boundary) | R1 fresh initializations; R2 core claim (ungated share 1.48; champion worse than stateless at fixed speed); R3 16-unit RNN (share 1.52); R5 boundary replicates (0.69, 0.48) | PASS |
| | R4 fast-dynamics gated probe | **FAIL**: +5.1e-4 [−4.3e-5, +1.04e-3], positive in all five initializations but unresolved — the probe was c2, already known to be initialization-unstable |

Claim v3 was declared the last candidate. Iteration 0003 ends **REJECTED**. The
existing claim remains unsupported by the pilot's evidence and is flagged as
`AAA-162`; no replacement is promoted.

## What happened to Champion 0

Nothing. It remains the champion with its round-3 capability vector. No
lineage identity was created, `aaa.1k.v1` is not redefined, and parameter,
state and compute accounting are unchanged (994 / 1414 / identical).

## What the pilot measured about AAA-1K

New known limitations, measured on diagnostic, development and attack
identities (never on confirmation evidence, so they are characterizations,
not confirmed claims):

* Q4's negative sign is carried entirely by `coarse_speed_v1` (round-3
  primitives), where the champion barely uses its memory (M1, H13).
* The champion's online learning runs away on long fixed-speed quantized
  streams (M2): 44/80 and 16/80 long coarse cells in diagnosis round 3,
  27/80 in iteration 0002 development, 16/80 in iteration 0001's attack.
  Round 3 never evaluated coarse streams longer than 280 steps.
* The previous-error input carries real adaptation value (removing it costs up
  to 41%), which round 3's memory-family ablation (inconclusive) did not show.
* Capacity is not the bottleneck on the failure studied: a 354-parameter
  ungated RNN beats the 994-parameter gated champion on `coarse_speed_v1`.

## Evaluating the loop

The loop is itself the challenger here. Each answer points at evidence.

1. **Did it begin from a real measured limitation?** Yes: round 3's Q4
   mean/median disagreement, recomputed from primitives, not invented.
2. **Did classification prevent premature modification?** Yes, decisively. The
   prompt's framing ("minority tail") would have pointed at tail-robust
   losses or outlier handling. Classification from primitives showed a whole
   family and redirected the work before any model change.
3. **Did diagnosis identify a mechanism or rule hypotheses out?** Both: ten of
   twelve round-1 hypotheses were contradicted, including the implementer's
   favourite (H12); H13 was then supported by all four predeclared parts on a
   fresh block, with direct Jacobian instrumentation.
4. **Did the targeted tests distinguish explanations?** Yes. The keep +2 / keep
   −2 pair was designed so that only H13 predicted opposite signs, and the
   fixed-speed condition separated regime inference from phase integration.
5. **Was the intervention minimal?** Yes: one initial vector (`b_z`) in
   iteration 0001, one input transform in 0002, zero added parameters
   throughout.
6. **Did adversarial attack find what the initial tests missed?** Yes, twice.
   Five development initializations never showed c2's instability; five fresh
   attack initializations did. The first claim attack found the
   construction-dependence that no diagnostic had varied.
7. **Did fresh confirmation stay fresh?** Vacuously: no confirmation identity
   was ever reserved or observed. The ledger holds only diagnostic,
   development and attack blocks, all proven disjoint from every AAA-1K
   identity.
8. **Could the decision logic reject or return inconclusive?** The inner loop
   rejected eight times. The outer loop's `REJECT`, `INCONCLUSIVE` and refusal
   paths are exercised by tests on synthetic confirmations, not by real data.
9. **Were failed attempts preserved?** Yes: every candidate, probe and
   rejected claim, with evidence and reason. The validator refuses a record
   that drops one.
10. **Did it produce enough information to choose the next experiment?** Yes
    (see below).
11. **What was unnecessary?** A separate development screen *and* attack for
    a candidate whose development already showed a real regression was
    arguably redundant (deviation L-4 cost an attack for a candidate the
    screen had rejected). The 1120-step conditions in development duplicated
    diagnosis round 3.
12. **What was missing?** A predeclared candidate budget; a rule for
    screen-failed tradeoff candidates; uncertainty in screens; a scratch
    identity namespace for sanity checks; attack criteria independent of
    known-fragile instruments. The first four are process gaps the pilot hit
    in practice.
13. **What should change before the protocol becomes permanent?** The seven
    gaps in [`loop_protocol.md`](loop_protocol.md#known-gaps-to-close-before-v1),
    plus one real cycle through fresh confirmation.
14. **Did the infrastructure cost more than it produced?** About 3,000 lines
    of loop code and tests. That machinery produced a family-level
    reclassification of Q4, a supported mechanism for it, a previously
    unmeasured long-horizon failure mode, falsification of a documented claim,
    and eight honest rejections. Some of it (the iteration-3-specific
    freeze/confirm/recompute commands) has only ever run against synthetic
    confirmations and may be the wrong shape; that is the part most likely to
    be over-engineered, and it should be generalized, or deleted, on its
    first real use.

## What the pilot says about scaling

It is evidence *against* scaling AAA-1K now. The failure studied is not a
capacity failure (H6 contradicted; a 354-parameter model beats the 994-parameter
champion on it), a parameter-neutral change reaches most of the gap (H13),
and the blocker to using it is an optimization instability (M2), not
representation. Criteria 3–5 of the capacity rule are unmet.

## Recommended next iteration

Target M2 first. Its refined hypothesis, that SGD learns a closed-loop gain
above one on the previous-error channel for alternating targets, predicts a
measurable effective feedback gain in diverged cells and should be tested
before any intervention. Candidate remedies in increasing departure: bounding
the *learned* feedback gain (for example, weight decay on the input-3 column
only), then a per-group learning rate. Once M2 is closed, M1's keep-bias
remedy should be re-attacked on fresh initializations, since c2's
fresh-initialization failure had M2's signature.

## Verification

Run on the final branch state (Python 3.12, the locked environment):

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -t .` | 631 tests, all pass (571 existing, 60 new) |
| `ruff check`, `ruff format --check`, `mypy` | clean (90 source files) |
| `tools/check_lock.py`, `tools/check_exit_codes.py` | 24/24 pinned; exit-code contract holds |
| `research.aaa_1k parameter-audit` | 994 / 982 / 954, all agree |
| `research.aaa_1k gradient-check --full` | every parameter of every model, 0 violations (max abs 8.7e-10 for the GRU) |
| `research.aaa_1k fingerprint` | `5ce6e019...` — Champion 0 unchanged on this branch |
| `research.aaa_1k recompute` on round 3 | 21 of 21 stored values agree |
| `research.aaa_1k_loop validate` | three records valid; Champion 0 verified; ledger disjoint |
| `research.aaa_1k_loop reproduce <stage>` for all ten stages | all reproduce with zero mismatches, after `AAA-164` was found by this check and repaired |
| strict JSON over every tracked and new JSON file | 395 files, none non-standard |
| relative markdown links | none broken |
| wheel build and install into a clean environment | the loop package is included, imports and runs from an unrelated directory; its evidence commands fail closed outside a checkout |
| `git diff --check` | clean |

Python 3.10, 3.11 and 3.13 were not available locally; the CI matrix covers
them, and the new code uses no syntax newer than 3.10.

## Time accounting

Task timing (human/AI work, not scientific compute):

| Phase | UTC |
|---|---|
| task start (fetch, verify `main`) | 14:19:36 |
| repository review complete; Champion 0, observation and diagnosis design committed | 14:28 |
| diagnosis rounds 1–3 | 14:28–14:38 |
| iteration 0001 development and attack | 14:38–14:48 |
| iteration 0002 | 14:48–14:53 |
| iteration 0003 | 14:53–15:00 |
| records, tests, reproduction, documentation, validation, PR | 15:00–end (see the PR) |

Scientific compute is recorded per artifact (`compute_seconds`) and in each
iteration record: about 290 wall seconds in total on 16 worker processes
(diagnosis 170 s, development 56 s, attacks 65 s).
