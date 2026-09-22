# Loop iterations 0004–0006: the audit's first four steps

**Branch** `opus/audit-tbptt-semantics` · **Protocol** `aaa.loop.v0-pilot` (see
[§8](#8-what-this-means-for-the-protocol)) · **Date** 2026-09-21/22 (UTC)

This report answers the external research audit of 2026-09-21
(`AAA_Research_Audit_2026-09-21.md`, reviewing `main` at `daea2ff`). The audit's
recommended sequence was

> verify the online TBPTT update semantics → explain and stabilize M2 → make
> stability a hard promotion gate → complete one real fresh-confirmation loop →
> then expand the benchmark.

The first four steps are done. Every claim below cites a committed artifact
under `docs/evidence/aaa1k_loop_000{4,5,6}/`. Every hypothesis and threshold was
committed before the identities that judged it were observed. Every block is
in `benchmarks/aaa1k_loop_identity_ledger.json`, proven disjoint from every
AAA-1K and earlier loop identity.

## 1. Headline

| Audit item | Result |
|---|---|
| **R-02** online TBPTT semantics | The live rule is **not** the "exact realized-trajectory gradient" that `model.py` claims: it backpropagates stale activations through current matrices. It is **numerically within 7×10⁻⁵ (median)** of that gradient, and **M2 is identical under every rule**, including T = 1. It does not explain anything. (`AAA-169`) |
| **R-01** M2 | Not a learned loop gain above one (falsified), and not single-step SGD overshoot (falsified). **It is a self-confirming target-unfolding frame lock:** the unfolding branch is chosen by the learner's own raw prediction, which can lock onto the mirrored frame after a slow wall bounce. That held in 54 of 54 diverged cells. (`AAA-170`) |
| **R-01** stabilize | **c10**, a zero-parameter gate on that rule, removes M2 (fresh confirmation: 21.3% → **0%** long-coarse divergence) and is **bitwise identical** to Champion 0 everywhere else tested. |
| **R-01/R-04** promotion gates | Long-horizon stability is the primary gate of screen, attack and confirmation. Every criterion is uncertainty-aware and three-valued. An interval that crosses a boundary is INCONCLUSIVE. |
| **R-03** real outer path | **Done once.** Freeze → durable remote claim → fresh confirmation → independent recomputation → **PROMOTE**. **Champion 1** exists. |
| side result | Round 3's two unexplained "burst cells" were this frame lock. Champion 1 is 56% and 62% better there, and bitwise identical in the other 718 cells. |

Four M2 candidates were screened across three iterations (c7–c10). Three were
rejected by rules committed before their data existed, and the fourth was
promoted only by fresh attack and fresh confirmation.

## 2. R-02: which gradient does online TBPTT take?

`NeuralAgent` calls `learn()` after almost every observed step. `backward()`
walks the last `T = 4` cached transitions. Their activations were computed
under older parameters, but `backward()` multiplies through the *current*
`U_z`, `U_r`, `U_n`. The phase's gradient checker holds parameters fixed, so it
could never see this.

[`tbptt.py`](../research/aaa_1k_loop/tbptt.py) defines the alternatives exactly
and proves each one against its own finite-difference definition *after real
interleaved updates* (relative error ~10⁻⁹):

* **snapshot**: each cache backpropagated through the matrices it was computed
  with. This is the gradient the docstring describes.
* **replay**: the window recomputed under current parameters, i.e.
  constant-parameter chunked TBPTT.
* **T = 1**: the latest transition only, where all definitions coincide.

One structural fact: the earliest cache's matrices only feed the discarded
boundary gradient, so at T = 4 only two transitions carry stale matrices.

Diagnosis round 1 (`diagnosis.json`, 560 cells, H20–H23):

| | Result |
|---|---|
| H20 live ≈ snapshot where stable | **SUPPORTED**: per-step update difference median 7.1×10⁻⁵, q99 8.9×10⁻³ (516 cells, 353,545 steps) |
| H21 stale matrices drive M2 | **CONTRADICTED**: long-coarse divergences live 44 / snapshot 44 |
| H22 M2 needs no multi-step credit | **SUPPORTED**: T = 1 diverges in 46 |
| H23 chunk-consistent credit prevents M2 | **CONTRADICTED**: replay diverges in 48 |

Snapshot is EQUIVALENT (±1%) to live on all six standard plan entries.

**Disposition.** The docstring's sentence is inaccurate. The rule should be
named what it is: online TBPTT with cached activations and current weights, a
standard approximation. `model.py` is left unchanged, because editing it would
change the AAA-1K phase fingerprint that every AAA-1K result and both champions
carry. The correction is recorded in `AAA-169` and `errata.md`.

## 3. R-01: explaining M2, four rounds, two falsifications

**Round 2: the closed-loop gain (H24–H28).** The pilot's refined hypothesis
was that SGD learns a previous-error feedback gain above one.
[`gain.py`](../research/aaa_1k_loop/gain.py) measures the direct gain
`a_t = ∂e_t/∂e_{t−1}` and the spectral radius of the 17×17 closed-loop Jacobian
over (h, e), both verified against finite differences. **H24 CONTRADICTED
(0 of 41):** onset comes at steps ~150–290 while the gain is still ~0.2–0.5,
and the gain crosses one only 500–3000 steps *after* onset. **H27
CONTRADICTED:** lr 0.01 on streams three times as long diverges in 0 of 80
cells, against 36 for the champion. A learning-rate threshold, not a slower
pace.

**Round 3: single-step overshoot (H29–H31).** The LMS stability condition was
the natural next suspect. [`overshoot.py`](../research/aaa_1k_loop/overshoot.py)
measures the exact same-sample amplification of every clipped step. **H29
CONTRADICTED (0 of 39):** each update still shrank its own sample's error
(amplification ~0.6–0.8), and the clip was idle until onset. H31 passed as
declared, but the input-3 columns carry only about 1% of the curvature, so that
"support" is immaterial.

**An exploratory look (disclosed).** Round 3's already-observed cells showed
onset 13–19 steps after the first wall bounce in 34 of 39 diverged cells, 37 of
39 of them slow-regime. One traced cell showed the mechanism directly. After a
bounce at the lower wall, the champion's raw prediction stayed just past it.
`aaa.predictors.unfold_observation` picks the fold branch nearest the raw
prediction, so every later observation was mirrored. The target, taken
relative to the real-frame input position, became about −2 × distance-from-wall
/ scale and grew without bound as the dot moved away, which pulled the raw
prediction further past the wall. The scored prediction still looked
plausible, because it is reflected back before scoring.

**Round 4: the frame lock (H32–H34), on a fresh block.**

| | Result |
|---|---|
| H32 a lock (≥ 10 consecutive mirrored targets) precedes onset | **SUPPORTED** 54/54; lock leads onset by 11–20 steps; every locked cell diverged |
| H33 stable learners never lock | **SUPPORTED** (stable champion cells and all lr-0.01 cells) |
| H34 removing the self-reference removes M2 | **SUPPORTED**: folded targets 0/160, prediction-independent reference 0/160, champion 54/160 |

This re-explains the pilot's observations. The previous-error input sustains
the lock but doesn't create it (2/80 without it). A lower learning rate
doesn't chase the mirrored target fast enough to lock. The runaway was never
seen in round 3 because slow sub-quantum dots rarely reach a wall within 280
steps.

## 4. The challenger cycles, under the prospective rules

The pilot's seven declared gaps were made hard rules in
[`iteration4.py`](../research/aaa_1k_loop/iteration4.py) before any development
identity was observed:

- a finite, precommitted candidate budget;
- a failed screen is a veto;
- non-inferiority is judged on bootstrap interval bounds with a 2% margin;
- one attack, with prevalidated instruments;
- scratch identities for every look;
- long-horizon stability as a primary gate;
- a three-valued K1–K3 decision.

Iterations 0005 and 0006 import the *same* screen, attack and decision
functions (a test asserts identity), so no judgement was re-chosen after
seeing a result.

| Iter. | Candidate | Long-coarse divergence (champion → candidate) | Screen |
|---|---|---|---|
| 0004 | **c7** unfold around a prediction-independent reference | 56 → 0 of 80 | not passed: long bouncing +4.2% [+1.6, +7.4], straddles the margin |
| 0004 | **c8** train on folded observations | 56 → 0 | FAIL: `changed` +8.2%, long bouncing +17% |
| 0005 | **c9** refuse a mirror whose implied step exceeds \|v\| + \|y−p\| | 58 → 0 | FAIL: long bouncing +4.5% [+2.2, +7.1] |
| 0006 | **c10** refuse a mirror when the input is farther from the wall than \|v\| + \|y−p\| | 36 → 0 | **PASS**: every interval exactly [0, 0] |

What the rejections taught:

- c7 and c8 showed that the champion's own-prediction reference is *right*
  at genuine crossings.
- c9 showed that the short mirrored runs just after a bounce on smooth motion
  are *useful*. They keep the champion's straight-line-then-reflect scheme
  consistent.
- A disclosed look at 0005's development cells found those useful runs are
  only 1–2 steps long and next to the wall, while locks run 11+ steps and
  drift away from it.

**c10** encodes exactly that. A one-step crossing must start within one
observed step of the wall, so c10 refuses a mirrored target only when the
input is farther away than that. It was declared in advance as the last M2
candidate of this effort.

**Attack** (`0006/attack.json`, fresh streams, five fresh initializations): A1–A5
PASS. Long coarse divergences were 0 against the champion's 35. Nearby
quantized conditions (quantum 0.004/0.006, slow speed 0.08/0.15) were 0 against
43–50 each. No cell locked, and c10 was bitwise identical everywhere else.

## 5. The outer path, run for real

**Freeze** (`freeze_2.json`): the claim, K1–K3 rules and consequence; 112
streams × 5 initializations reserved and proven fresh; the confirmation source
fingerprinted (`83f363b3…`; attempt 1's freeze was `cee8ca71…`); the
Champion 0 phase fingerprint `5ce6e019…`, unchanged.

**Attempt 1 aborted before observation** (`confirmation_attempt_1.json`). Both
remote claims were created and the blocks marked spent. Then my own
`confirmation_cells()` refused them, because it re-checked for status
`reserved`. No stream was generated and no model was run. The blocks stay spent
and are never reused. The defect was fixed with the test that should have
existed (`AAA-171`): an end-to-end test from admission through cells,
primitives, payload and decision to the independent recomputation. The
unchanged challenger was then refrozen on new blocks, with the user's
approval before each remote push.

**Attempt 2** (`confirmation_2.json`, 800 cells, remote claims
`aaa-confirmation-claims/aaa1k-loop-0006-confirmation-2-{env,init}`):

| Criterion | Champion 0 | c10 | Status |
|---|---|---|---|
| K1 long-coarse divergence (primary) | 21.3% | 0% | diff 0.21 [0.08, 0.36] → **PASS** |
| K2 long bouncing, long occlusion | | bitwise identical | **PASS** |
| K3 six standard families | | bitwise identical, [0, 0] | **PASS** |

**PROMOTE.** [`recompute4.py`](../research/aaa_1k_loop/recompute4.py) agrees
(`recomputation_2.json`). It shares no decision code, uses a count-weighted
bootstrap, reads its rules from the freeze, and checks that the primitives use
exactly the frozen seeds.

## 6. Champion 1

[`champion_1.json`](evidence/aaa1k_loop_0006/champion_1.json), built and
verified by [`champion1.py`](../research/aaa_1k_loop/champion1.py):

- **Same network.** Same code, weights initialization, hyperparameters, 994
  parameters, 1414 adaptive state scalars and phase fingerprint. One
  target-construction rule differs.
- **Nothing in `research/aaa_1k/` changes.** Champion 1 is defined by the loop
  arm `ReachGatedUnfoldAgent`.
- **Its capability vector is measured, not inherited.** Round 3 was re-run on
  its own identities, design and configuration, with c10's rule for every
  neural learner (primary, ablations, controls and branch clones), so
  architecture comparisons keep one agent rule.
  - 718 of 720 cells and all 180 adaptation/retention trials equal the
    committed round 3 **exactly**, across every arm.
  - The two cells that differ (`coarse_speed_v1`, initialization 3) are round
    3's burst cells. Champion 1 improves them by 56% and 62%.
- **No capability verdict flips except one improvement:** Q7
  `coarse_speed_v1` vs dead reckoning goes INCONCLUSIVE → POSITIVE.
  - Q4 memory-families remains NEGATIVE; that is M1, still open.
  - Q2 and Q5 are identical.

## 7. What was not done, and what remains open

* **Porting c10 into AAA-1K itself.** That changes the phase fingerprint and
  is a new phase decision (`aaa.1k.v2`), not a loop promotion.
* **The same own-prediction unfolding in the v2.1 core RLS learner**
  (`aaa/predictors.py`, AAA-120) is untested for this lock (`AAA-172`, open).
  Its skip-after-reflection mitigation might prevent the chase, or might
  instead stall learning.
* **M1** (the Q4 memory-family deficit) is untouched; c2's keep-bias remedy
  should now be re-attacked on Champion 1, since its fresh-initialization
  failure had M2's signature.
* **Remaining audit items**, not started:
  - R-05: independently designed memory and long-range benchmarks.
  - R-06 / `AAA-162`: `coarse_speed_v1`'s claim.
  - R-07: wider identity digests.
  - R-08: external archival.
  - R-09: noise A/B precision.
  - R-10: missingness encoding.
  - R-11: self-error head.

  R-12 is met for this work. The outer path has an end-to-end test; every
  0004–0006 verdict re-adjudicates from its stored records in routine CI; and
  `.github/workflows/loop-reproduction.yml` re-runs all nine stages' models
  (including the confirmation primitives) weekly and on any change to code
  that can alter a primitive. Agreement must be bit-identical locally, and
  across machines floats may differ within a declared relative tolerance of
  1e-9, while every flag, count and verdict must be exact. The first CI run
  found last-digit drift in one stage. R-13 applies to this branch: it was written
  by one implementer and needs a fresh reviewer.
* **Horizon.** M2's repair is verified at 1120 steps on the tested families.
  Other wall geometries, noise, missing observations near walls, and large
  accelerations at walls are untested.

## 8. What this means for the protocol

The outer path has now run once, for real, and it caught a real defect in
itself. A crash between claim and observation burned identities rather than
reusing them, exactly as designed. I recommend `aaa.loop.v1` after an
independent review of this branch (audit R-13). The version string stays
`v0-pilot` until then, because a promotion the implementer reviews alone is
the collapse R-13 warns about.

For v1, generalize the outer machinery only as far as this use showed.
`outer4/5/6` are copies differing only in their spec, and `recompute4` is
already iteration-agnostic. That argues for one spec-driven outer module and
nothing more.

## 9. Compute

About 7.5 CPU-minutes of wall time across all iteration 0004–0006 stages
(`compute_seconds` in each artifact, 16 workers), plus 5.5 minutes
single-threaded for the Champion 1 round-3 re-run. The full test suite is 700+
tests, about 100 s.
