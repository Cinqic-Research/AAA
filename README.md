# AAA — Accurate Autonomous Adaptation

An experimental research project asking one deliberately small question:

> Can an autonomous system observe an environment, learn to predict what
> happens next, detect when its existing knowledge is no longer adequate, adapt
> from subsequent evidence, and improve its future predictions?

The environment is one moving dot on a line. The learner has three parameters.
Neither is an accident: the point of this version is not a capable system, it
is a **measuring instrument you can check**.

AAA is the research programme, not the dot. The long-term objective is a
persistent agent called Juniper; the dot is one benchmark family, and the
current learners are the smallest things that can be measured honestly on it.
See [the research charter](docs/aaa_charter.md) before assuming the benchmark
is the project.

AAA makes no claim to general intelligence, physical understanding, or
independent goal formation. "Autonomous" here means the observe / predict /
score / update loop runs unattended after launch, and nothing more.

## Why the benchmark looks the way it does

A previous version of this benchmark reported that all required gates passed.
Three independent reviews then found that several of those gates could not have
failed. Among other things:

- a **parameterless analytic formula** satisfied the gate named
  `straight_learning`;
- a run with **zero** required strata passed the coverage gate, because
  `all()` of an empty collection is `True`;
- the candidate was allowed to reflect its prediction into the public bounds
  while the baseline it was compared against was not — and essentially the
  entire reported bounce advantage was that transform, not learned parameters;
- the reproducibility gate compared a deterministic function with itself;
- a confirmation run that recorded failed gates still exited 0;
- `475 / 475` recovery counted only events whose 50-step average stayed
  elevated, so large, fast shocks were diluted out of the denominator;
- 26 values in the "frozen specification" were never read by the code.

Every one of those was reproduced against the old tree before anything was
changed. The probes are committed at
[`docs/evidence/pre_repair_probes.json`](docs/evidence/pre_repair_probes.json)
and each defect is tracked in
[`docs/issue_ledger.md`](docs/issue_ledger.md).

The current protocol is built so that it can say **no**:

| Status | Meaning |
|---|---|
| `PASS` | the required property was measured and holds |
| `FAIL` | it was measured and does not hold |
| `NOT_VERIFIED` | the check did not run |
| `INSUFFICIENT_EVIDENCE` | there was not enough evidence to decide |

Only `PASS` satisfies a required gate, and formal confirmation exits non-zero
on anything else. Absence of evidence is never turned into success.

**It has already said no.** The first confirmation round under the repaired
protocol failed: both independent fresh streams failed the required
`always_online_stability` gate and both exited non-zero. Thirteen of fourteen
gates passed in each. The failure was traced on development data to a specific
mechanism, repaired in the candidate, and the failed attempts are committed
alongside everything else. No threshold was touched. See `AAA-120` in
[`docs/issue_ledger.md`](docs/issue_ledger.md).

## What is implemented

- A seeded, bounded CPU simulator: constant velocity, reflection at the
  boundaries, unannounced speed changes, and a damped harmonic oscillator whose
  coefficients change mid-episode without notice.
- A strict temporal boundary — predict, record, advance, reveal, score, then
  update — with tests that fail if a scenario name, event flag, velocity,
  change schedule, hidden coefficient or future observation reaches a
  predictor.
- A baseline suite where each member isolates one source of predictive power,
  including a **reflected constant-motion** baseline that uses exactly the same
  public boundary map the candidate may use.
- A three-parameter square-root recursive-least-squares candidate with
  trace-bounded, self-triggered forgetting, verified against an independent
  batch least-squares reference.
- The historical v1 SGD learner, retained and reported as a named diagnostic
  arm rather than quietly replaced.
- Benchmark v2.1: a typed, hash-identified, fully executable specification;
  planned stratified coverage; a paired hierarchical bootstrap that recomputes
  each gate's own statistic; predeclared confirmation batches with a registry
  and a freeze manifest; an experiment registry with safe resume; and
  independent recomputation of every metric and gate from retained raw
  evidence.
- Observation-noise v1.1: a separately versioned sensor model with cached
  schedules, latent/observed field separation, causal noisy-target updates,
  matched intervention branches, deterministic sharded evidence, and an
  independent reference verifier. The corrected full 2 x 2 x 1 development
  search evaluated four candidate identities and retained the unchanged
  incumbent as an explicit no-refinement control. Formal A/B was not executed:
  the dedicated HDD removes the local capacity blocker, but a measured profile
  projects about 31.4 wall hours per formal batch and the replication budget
  has no quantitative precision justification. The declared v1.1 batches were
  retired unobserved. Development evidence is not a confirmation result. See
  [`benchmarks/observation_noise_candidate_ledger.json`](benchmarks/observation_noise_candidate_ledger.json).

## AAA-1K: a 994-parameter recurrent core

The current research phase, isolated in [`research/aaa_1k/`](research/aaa_1k/).
A hand-written NumPy GRU -- 3 inputs, 16 hidden units, 2 outputs, 994 trainable
parameters and no optimizer state -- that learns online by truncated BPTT,
keeps a persistent hidden state, is fed its own previous prediction error, and
emits an estimate of how wrong it expects to be. It is compared against a
matched-capacity stateless MLP (982 parameters), an ungated RNN (954
parameters), three ablations of itself, and the full analytic baseline suite
including the unmodified v2.1 RLS candidate.

The evaluation ran **three times**. Rounds 1 and 2 are retained as superseded
evidence. Independent Sol review found that round 2's Q1 time contrast did not
identify weight learning and that Q2/Q5 flattened trials sharing model
initializations. Round 3 uses fresh identities and corrected designs.

- it **learns online** on every family;
- **persistent hidden state helps on these families** -- it beats both the
  direct state-reset ablation and matched stateless control;
- **it adapts on the paired-change benchmark**: the change-specific component
  is `+5.12e-04` and 43% of the combined online advantage;
- **no forgetting was measured on the fixed probe bank**, but the corrected
  interval crosses zero; this is not general retention immunity;
- **whether gating pays for itself is inconclusive**. Round 1 said it loses;
  that did not survive giving the ungated control its own rule-selected
  learning rate. Gating does buy stability -- the ungated arm diverges at a
  learning rate the gated one survives;
- it **loses** to reflected constant motion, dead reckoning and the existing
  three-parameter RLS learner on smooth, fully observed motion, and wins only
  where memory or coarse observation actually matter;
- its **self-error head is weakly informative** and systematically
  over-predicts.

The full result is [`docs/aaa_1k_report.md`](docs/aaa_1k_report.md); read
[`docs/aaa_1k_self_review.md`](docs/aaa_1k_self_review.md) first, because it is
where the four repairs come from -- including the one that would otherwise have
published a hidden-state advantage thirty times too large.

## The improvement loop

`research/aaa_1k_loop/` pilots the process by which AAA is supposed to get
better: observe a measured weakness, classify it, diagnose it, state competing
hypotheses and try to falsify them, intervene minimally, attack the
intervention, confirm on fresh evidence, decide, preserve everything, repeat.
Development and confirmation evidence are separated mechanically, not by
convention.

Its historical v0 pilot, on AAA-1K's Q4 result, **promoted nothing**: eight candidates,
eight rejections, Champion 0 unchanged. It reclassified Q4's "minority tail"
as one benchmark family, supported a mechanism for it, found a long-horizon
instability round 3 never measured, and flagged a documented claim as
unsupported. It never reached fresh confirmation. Read
[`docs/loop_pilot_report.md`](docs/loop_pilot_report.md).

Iterations 0004–0006 answered an external audit. The online TBPTT rule is an
approximation, but a numerically negligible one. The long-horizon runaway (M2)
is a self-confirming target-unfolding frame lock, not an optimization
instability. Three candidate fixes were rejected at a precommitted,
uncertainty-aware screen. The fourth survived attack, **fresh confirmation**
and independent recomputation, and became **Champion 1**: the same 994-parameter
network with one target rule changed, 21% → 0% long-horizon divergence, and
bitwise identical to Champion 0 in 718 of 720 round-3 cells (the other two
improve). This was the loop's first real run through its outer path. Read
[`docs/loop_report_0004_0006.md`](docs/loop_report_0004_0006.md).
Independent review of that complete cycle established `aaa.loop.v1` for
future work. The version change does not relabel the retained v0 artifacts.

## The learner

Feature vector, from the last four observed positions only:

```text
phi = [ 1,  (x[t] - x[t-1]) / (dt * speed_max),  (x[t] - midpoint) / L ]
```

It predicts the next displacement normalized by the interval width `L` and adds
it to `x[t]`. Parameters start at zero, so before any learning it predicts
persistence.

Boundary reflection is **programmed public knowledge of the observation
format**, not a learned capability. That is precisely why the reflected
constant-motion baseline exists: so the transform is never counted as
intelligence. The report decomposes any apparent bounce advantage into analytic
extrapolation, public boundary handling, and learned parameters, separately.

The covariance is propagated as a square-root factor, so symmetry and positive
semidefiniteness hold by construction, and forgetting is suspended when the
covariance trace would exceed its declared bound — with suspensions counted and
reported. The previous covariance-form learner lost positive semidefiniteness
after 323 updates on a slow constant-velocity stream and overflowed at 6,642 on
a stationary one.

## Install

```bash
git clone https://github.com/Cinqic/AAA.git
cd AAA
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python tools/check_lock.py
```

CPU only. NumPy and Matplotlib are the runtime dependencies. No GPU, no
external API, no pretrained model, no paid service.

That describes what the implementation *requires*, which is not the same as
what the development workstation *contains*. FLOWBOX, the current primary
development machine, has a discrete GPU that no AAA code uses, and the project
carries a current planning ceiling on model size. Both are recorded in
[`docs/hardware.md`](docs/hardware.md).

## Commands

```bash
python -m unittest discover -s tests -t .        # the test suite
python -m aaa.cli spec-hash                      # canonical specification identity
python -m aaa.cli observation-noise-protocol-hash # separate noise protocol identity
python -m aaa.cli observation-noise-fingerprint    # scientific source identity
python -m aaa.cli observation-noise-development-select --quick \
    --output runs/development-selection/observation_noise_development_smoke.json
python -m aaa.cli observation-noise --role development --quick \
    --attempt-label noise-smoke-001 --output-root runs
python -m aaa.cli observation-noise-recompute runs/observation-noise-v1_1/noise-smoke-001
python -m aaa.cli observation-noise-confirmation-evaluate <A> <B> \
    --output docs/evidence/observation_noise_joint_evaluation.json
python -m aaa.cli benchmark --role development --attempt-label dev-001 \
    --replicas 2 --episodes 3 --output-root runs
python -m aaa.cli recompute runs/benchmark-v2_1/dev-001
python -m aaa.cli diagnose --output docs/evidence/diagnosis
python -m aaa.cli select-candidate
python -m aaa.cli animate --checkpoint runs/<attempt>/checkpoints/replica-00.json

# AAA-1K
python -m research.aaa_1k parameter-audit         # 994 / 982 / 954, recounted three ways
python -m research.aaa_1k gradient-check --full   # finite-difference every parameter
python -m research.aaa_1k select       --output docs/evidence/aaa_1k_development_selection.json
python -m research.aaa_1k characterize --selection docs/evidence/aaa_1k_development_selection.json \
    --output docs/evidence/aaa_1k_characterization.json
python -m research.aaa_1k round3      --selection docs/evidence/aaa_1k_development_selection.json \
    --characterization docs/evidence/aaa_1k_characterization.json \
    --output docs/evidence/aaa_1k_evaluation_round3.json
python -m research.aaa_1k recompute   --evidence docs/evidence/aaa_1k_evaluation_round3.json
python -m research.aaa_1k visualize --family occlusion_v1 --output runs/aaa_1k/dashboard.png

# improvement-loop pilot
python -m research.aaa_1k_loop validate           # iteration records 0001-0006, Champion 0, identity ledger
python -m research.aaa_1k_loop.champion1 verify   # Champion 1's record against the repository
python -m research.aaa_1k_loop.recompute4 --confirmation docs/evidence/aaa1k_loop_0006/confirmation_2.json \
    --freeze docs/evidence/aaa1k_loop_0006/freeze_2.json   # independent PROMOTE recomputation
python -m research.aaa_1k_loop.stages4 reproduce attack6  # rerun a 0004-0006 stage; primitives must reappear exactly
python -m research.aaa_1k_loop reproduce diagnose2 # rerun a stage; committed primitives must reappear exactly
```

Formal confirmation requires a predeclared batch, a committed freeze manifest,
the selected candidate ledger entry, the exact lock, and a matching
non-self-referential scientific fingerprint; see
[`docs/reproduction.md`](docs/reproduction.md). The confirmation evaluator
reconstructs the complete primary A+B claim family and applies one Holm
adjustment without trusting stored conclusions.

## Interpreting a result

These are five different claims, and none of them implies another:

1. the implementation runs correctly;
2. the model improves with experience;
3. what it learned generalizes to unfamiliar episodes;
4. continued updates help after a change;
5. it beats a baseline.

Constant-motion extrapolation is an extremely strong baseline in a
deterministic, noiseless world — with the public boundary map it is *exact* at
bounce transitions. A learned model losing to it is a valid and informative
result, and this repository is built to report that rather than to avoid it.

## Documentation

| Document | What it covers |
|---|---|
| [Research charter](docs/aaa_charter.md) | what AAA is, what Juniper is, and why the dot is one benchmark |
| [Development hardware](docs/hardware.md) | FLOWBOX, the CPU-only execution boundary, the current 125M planning ceiling, and future compute |
| [AAA-1K architecture](docs/aaa_1k_architecture.md) | the frozen 994-parameter specification |
| [AAA-1K literature review](docs/aaa_1k_literature_review.md) | what was adopted from the literature, and what was refused |
| [AAA-1K decisions](docs/aaa_1k_decisions.md) | every decision, including three departures from the phase brief |
| [AAA-1K report](docs/aaa_1k_report.md) | the measured result, with its claim boundaries |
| [AAA-1K self-review](docs/aaa_1k_self_review.md) | the attempt to break those results, and what it found |
| [AAA-1K handoff](docs/aaa_1k_handoff.md) | everything an independent reviewer needs |
| [Loop protocol](docs/loop_protocol.md) | the improvement loop's current `aaa.loop.v1` governance, what enforces it, and the pilot gaps that became rules |
| [Loop pilot report](docs/loop_pilot_report.md) | three iterations on AAA-1K, eight rejections, and an evaluation of the loop itself |
| [Loop pilot handoff](docs/loop_pilot_handoff.md) | reproduction commands and what an independent reviewer should challenge |
| [Loop iterations 0004–0006](docs/loop_report_0004_0006.md) | the audit response: TBPTT semantics, M2's mechanism, the first real confirmation, Champion 1 |
| [Loop 0004–0006 handoff](docs/loop_0004_0006_handoff.md) | reproduction commands and what an independent reviewer should challenge |
| [PR #20 independent review](docs/pr20_independent_review.md) | the review of iterations 0004–0006 and Champion 1 that established `aaa.loop.v1` |
| [Independent review, 2026-09-22](docs/independent_review_2026-09-22.md) | end-to-end repository review: verification performed, findings `AAA-174`–`AAA-178`, and settings recommendations |
| [Benchmark protocol](docs/benchmark_protocol.md) | the active v2.1 protocol, gates, statistics and confirmation discipline |
| [Observation-noise protocol](docs/observation_noise_protocol.md) | the separately versioned sensor study, causal boundary, schedules, replication and limits |
| [Issue ledger](docs/issue_ledger.md) | every defect: reproduction, root cause, repair, regression test, status |
| [Errata](docs/errata.md) | which earlier claims were affected, and why |
| [Candidate selection](docs/candidate_selection.md) | why this candidate, with the full table including what lost |
| [Diagnosis](docs/diagnosis.md) | controlled ablations on the legacy learner |
| [Reproduction](docs/reproduction.md) | exact commands from a clean checkout |
| [Evidence policy](docs/evidence_policy.md) | what is committed, what is regenerable, and the known limitation |
| [Experiment registry](docs/experiment_registry.md) | trial state, resume semantics, verification |
| [Limitations](docs/limitations.md) | current evidence boundaries, unexecuted work, and known limitations |
| [Research history](CHANGELOG.md) | v1, v2, v2.1, observation-noise v1/v1.1, and AAA-1K |
| [Self-review](docs/self_review.md) | what was checked after the repair, and what stayed weak |
| [Review handoff](docs/handoff_sol.md) | identity, confirmation outcomes, reproduction commands |

## Historical evidence

[`results/final/`](results/final/) is the v1 snapshot, preserved unchanged. Its
result was unfavourable — the original learner did not convincingly improve,
generalized poorly, and lost to constant motion — and that is exactly why it is
kept.

[`results/benchmark_v2/`](results/benchmark_v2/) holds the v2 confirmation
attempts. They are **superseded**: historical and provisional evidence produced
under a methodology since found defective. They are not acceptance evidence for
anything.

## License

Apache License 2.0. See [LICENSE](LICENSE).
