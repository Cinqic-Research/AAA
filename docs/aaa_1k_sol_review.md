# Independent Sol review of AAA-1K

**Date:** 2026-09-20

**Reviewer:** GPT-5.6 Sol

**Pull request:** Cinqic/AAA#16

**Base:** `02c3b20dd163c11a99502e846a2816b3e6c761de`

**Submitted head:** `a04e2f28b1272fdbae381fcc6f7c4fad12a135ac`

**Corrected implementation reviewed:** `21264f8`

**Final reviewed PR head:** `f47580a30c6bff1a0cd4c11a3e1b03b910700cae`

**Historical PR #16 scientific fingerprint:** `89c687404b772d1dd11688297e2fb67446096103518fe10a59d1203da828b875`

**Current post-closure scientific fingerprint:** `5ce6e019bd5389c5d8d466cc2be809ff9777e3bd6acba10917bc6a7ce98f771e`

The current fingerprint differs because the later repository-closure review
repaired `AAA-161`: divergent development configurations had been serialized
with the non-standard JSON token `Infinity`. They now use `null` beside their
explicit divergence flags, and the writer rejects any unhandled non-finite
number. The full current evidence was regenerated and independently
recomputed; no headline statistic or scientific interpretation changed. The
historical fingerprint above remains the exact identity reviewed on PR #16.

## Scope and independence

This is an independent AI review of the Opus implementation, not independent
human peer review. Sol reproduced the submitted implementation's behavior,
identified material defects, modified the code and evidence, and then
adversarially re-reviewed the exact corrected head. The review covered the
mathematical implementation, temporal protocol, controls, serialization,
statistics, provenance, packaging, documentation, and repository merge gates.

The verdict applies to a normal merge of this research phase. It is not a
release approval, a claim of physical-world validation, or evidence of general
intelligence, autonomy, causal understanding, or consciousness.

## Material findings and repairs

The submitted head was not approval-ready. The review reproduced and repaired
four material issue groups recorded as `AAA-157` through `AAA-160` in the issue
ledger:

1. Checkpoint loading did not fail closed. Malformed or non-finite TBPTT state,
   invalid counters, incomplete tracker state, and incompatible model state
   could be accepted or incompletely restored. Strict schemas, shapes, numeric
   domains, counter invariants, model-class restoration, and complete agent
   restoration are now enforced and regression-tested.
2. Branch equality covered only model parameters. Round-3 branching now hashes
   the complete interaction state, excluding only the intentional arm identity
   fields, and verifies bit-identical initial conditions for every matched
   online/frozen cell.
3. The round-2 estimands and uncertainty model were insufficient for the stated
   learning claims. Q1 now uses a matched frozen counterfactual instead of an
   early-versus-late comparison. Q2 and Q5 fully cross five initializations with
   every evaluation identity and bootstrap at that crossed structure.
4. Evidence and documentation had stale paths and quantitative language. Round
   2 is explicitly preserved as superseded evidence, round 3 uses fresh
   identities, the erroneous 65% adaptation wording is corrected, Q7 is labeled
   exploratory, and the fixed-budget versus width-matched control distinction
   is explicit.

Additional characterization records the width-matched gated/ungated comparison
and the behavior of the occlusion construction, including its stationary-zero
ambiguity. These characterizations constrain interpretation; they are not
promoted to confirmatory gates.

## Definitive round-3 evidence

The final artifact contains 720 cells, 120 crossed adaptation trials, and 60
crossed retention trials. A fresh rerun against the final scientific
fingerprint reproduced the committed artifact exactly.

- Q1, online learning versus a matched frozen branch: `+3.2986274816493005e-03`,
  95% interval `[+3.0134690510009903e-03, +3.5897690088720915e-03]`. All five
  initialization means and all 144 stream means are favorable.
- Q2, adaptation to the changed regime after subtracting the unchanged-world
  control: `+5.120513746349793e-04`, 95% interval
  `[+1.6889095282344988e-04, +8.666670821397387e-04]`. All five initialization
  means are favorable. The estimated adaptation share is 43.15% under the
  canonical definition.
- Q3, recurrent model versus stateless MLP: `+2.6284882389902815e-04`, 95%
  interval `[+1.5093072725569148e-04, +3.573599145100915e-04]`; versus the
  direct state-reset ablation: `+1.4364645749785857e-04`, 95% interval
  `[+2.6447804027368595e-05, +2.2303403594608144e-04]`.
- Q4, all fixed-budget gated versus ungated cells: `-4.838933087386695e-05`,
  95% interval `[-1.157626155754866e-04, +1.3151905715991445e-05]`. The result
  is mixed: median behavior is favorable, but a minority of large losses flips
  the mean, and memory-family cells favor the ungated control.
- Q5, forgetting on the fixed probe bank: `-2.024749791292492e-04`, 95%
  interval `[-6.417536468928978e-04, +1.2572597984547324e-04]`. This supports
  only "no forgetting measured on this probe"; it does not establish immunity
  to forgetting.
- Q6, error-estimator calibration: mean Spearman correlation `0.45405788`, with
  monotone error bins in only 114 of 720 cells. Calibration is heterogeneous
  and poor.
- Q7 baseline wins and losses are symmetric and exploratory; no multiplicity-
  adjusted superiority claim is made.

The separate width-matched development characterization slightly favors the
ungated control but is inconclusive: ungated-minus-gated
`-3.82e-05`, 95% interval `[-7.35e-05, +1.20e-06]`.

## Verification performed

- All 570 repository unit tests at the reviewed PR #16 head passed, both
  normally and under coverage.
  Aggregate line coverage was 68%; the repository does not define an enforced
  coverage threshold.
- The focused AAA-1K phase suite at that head passed all 125 tests.
- Ruff lint and format checks passed; mypy passed for all 65 checked source
  files.
- The finite-difference gradient audit passed every parameter for the 994-
  parameter GRU at `lambda=0.5` and `lambda=0`, the 954-parameter vanilla RNN,
  and the 982-parameter stateless MLP. The largest observed absolute GRU
  discrepancy was `8.69e-10`.
- Independent recomputation passed all 21 round-3 checks, all 99 superseded
  round-2 checks, and all 59 superseded round-1 checks.
- The locked environment contains 24 matching distributions; the lock-file
  digest is `2b5a23094b0cd99df66fa534b2e979a7a3e8c8b393df71cd53f1345281ab45b6`.
- The built wheel contains the complete `research.aaa_1k` implementation,
  including round 3. Installing only that wheel into an otherwise empty
  environment correctly failed at runtime because dependencies were absent;
  after installing the locked dependencies, the packaged parameter audit,
  specification hash, and noise-protocol hash passed. The no-dependencies
  attempt is retained as a setup failure, not reported as a pass.
- The legacy 2x2 development benchmark and independent recomputation completed.
  Its stratum-coverage, constant-velocity-identification, and recovery gates
  truthfully remained `INSUFFICIENT_EVIDENCE`; other gates passed. This small
  development run is not release evidence.

## Remaining limitations

The evidence is CPU simulation over synthetic trajectory families. It does not
cover physical devices, real sensors, distribution families outside the stated
protocol, accelerators, long-horizon deployment, security, signing, release
artifacts, or publication-grade external archival. Q4 is mixed, Q5 is
inconclusive, Q6 is weak, and Q7 is exploratory. The characterization protocol
does not replace a preregistered external replication.

## Verdict

**APPROVE FOR NORMAL MERGE**, conditional on exact-head required CI remaining
green and the remote head matching the reviewed commit chain. Do not tag or
publish a release from this verdict.

## GitHub integration result

Both exact-head CPU CI runs passed on the final reviewed PR head:
`35494445673` and `35494447197`. Each passed the locked environment, fresh
installation, and Python 3.10 through 3.13 jobs.

GitHub refused to store the verdict as an `APPROVED` review because the
authenticated repository owner was also the PR author. The same substantive
verdict was therefore submitted as a `COMMENTED` review on the exact head. The
repository required zero approving reviews; this platform-enforced distinction
is not presented as a formal GitHub approval.

PR #16 was merged normally with expected-head protection at merge commit
`acbb52ee4d199c497a4c62517adcabdc3aa08421`. The reviewed head is an ancestor
of that merge, the remote feature branch was deleted, and post-merge main run
`35494986925` passed all six jobs on the exact merge SHA. No tag or release was
created.
