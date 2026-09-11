# AAA engineering issue ledger

Every entry was independently reproduced against the pre-repair tree at
`235ce28518ca4ec3a9a240066154090a33b33470` before anything was changed. The raw
probe output is committed at
[`evidence/pre_repair_probes.json`](evidence/pre_repair_probes.json).

Nothing in this ledger is marked repaired because it looked intentional or
because an earlier ledger said so. Every previously "repaired" item was
re-tested from scratch, and several were reopened.

**Status vocabulary**

| Status | Meaning |
|---|---|
| `repaired` | reproduced, fixed, and covered by a regression test that fails on the old behaviour |
| `superseded` | the underlying concept was replaced for a documented scientific reason |
| `disproven` | the reported claim did not reproduce |
| `open` | reproduced, not fully resolved; the limitation is stated |

**Severity**

`critical` — invalidates a scientific claim. `high` — can silently produce a
wrong acceptance decision. `medium` — wrong or misleading measurement.
`low` — correctness or hygiene with no current claim attached.

**Sources** — `Astra`, `Opus`, `Sol` (the three independent reviews), `self`
(found during this investigation).

---

## A. Gates that could not say no

### AAA-001 — formal confirmation could record failed gates and exit 0
- **Source** Astra, Sol · **Severity** critical · **Status** repaired
- **Affected** `aaa/cli.py`, `aaa/benchmark.py`
- **Claim** The confirmation command returns success regardless of gate outcomes.
- **Reproduction** `main()` had four unconditional `return 0` paths and never
  consulted `gates["all_required_gates_pass"]`.
- **Root cause** The exit status carried "the process did not crash", not "the
  evidence was accepted". CI checked only the former.
- **Consequence** A confirmation run could be reported as successful while
  recording required-gate failures.
- **Repair** An explicit exit-code contract: development returns 0 even with
  failures; confirmation roles return non-zero if any required gate is not
  `PASS` — including `NOT_VERIFIED` and `INSUFFICIENT_EVIDENCE`; a refusal
  before running returns 2. CI asserts it.
- **Regression** `tests/test_cli.py::ExitCodeTests` (7 cases);
  `tools/check_exit_codes.py` runs real subprocesses.

### AAA-002 — an empty or missing stratum set passed the coverage gate vacuously
- **Source** Astra, Opus · **Severity** critical · **Status** repaired
- **Affected** `aaa/benchmark.py`
- **Reproduction** Gate evaluation with `strata = {}` returned `PASS` and
  `all_required_gates_pass = True`; with one stratum out of 32, also `PASS`.
- **Root cause** `all(...)` over an empty collection is `True`, and nothing
  compared the observed strata against a required set.
- **Consequence** Complete absence of the required evidence was reported as
  satisfied.
- **Repair** Coverage is planned as an explicit Cartesian product; the gate
  compares observed strata against the required set and checks per-stratum
  minimums, per-replica spread, both walls and event minimums. A shortfall is
  `INSUFFICIENT_EVIDENCE`.
- **Regression** `tests/test_gates.py::CoverageTests` (7 cases: one stratum
  removed, all removed, thin stratum, replica concentration, missing wall, too
  few events, missing family).

### AAA-003 — a parameterless analytic rule satisfied `straight_learning`
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** superseded
- **Affected** `aaa/benchmark.py`
- **Reproduction** The criterion was ">= 0.99 relative reduction from a
  zero-weight control on constant-velocity motion", which a constant-motion
  formula with no parameters attains.
- **Root cause** Beating a deliberately bad control is not evidence of learning.
- **Consequence** A "learning" claim rested on a comparison that learning was
  not required to win.
- **Repair** Replaced by three separately named claims:
  `constant_velocity_identification` (absolute accuracy per stratum),
  `learning_progress` (frozen checkpoints at increasing cumulative update
  budgets scored on one fixed probe bank), `frozen_prediction_accuracy`
  (generalization). A rule with no parameters has no budgets and produces a
  flat curve.
- **Regression** `tests/test_gates.py::LearningGateTests`, including
  `test_a_parameterless_rule_cannot_satisfy_the_learning_gate`.

### AAA-004 — the bounce gate measured the reflection transform, not learning
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** superseded
- **Affected** `aaa/benchmark.py`, `aaa/predictors.py`
- **Claim** ~13.9% of the reported bounce advantage was candidate-only
  reflection.
- **Reproduction** Measured on development streams with four arms. Normalized
  bounce-event MAE: raw constant motion 3.14e-03; reflected constant motion
  3.70e-17; candidate 4.24e-17; candidate without reflection 3.14e-03. The
  candidate's improvement over raw constant motion is 100.0%; over the fair
  reflected baseline it is **-14.6%**.
- **Observed** The effect is larger than reported: not 13.9% of the advantage
  but effectively all of it. The candidate without reflection is
  indistinguishable from raw constant motion at bounce transitions.
- **Root cause** The candidate could reflect into the public bounds; the
  baseline could not. The comparison gave the two arms different information.
- **Consequence** The bounce claim, as stated, was an artifact of the
  comparison rather than a property of the learner.
- **Repair** A `constant_motion_reflected` baseline using the identical public
  map is a first-class predictor in every family. The gate is now an absolute
  bounce-accuracy requirement plus non-regression against that baseline with an
  absolute floor, since a ratio between two quantities near 1e-17 is noise. The
  report decomposes the advantage into analytic extrapolation, public boundary
  handling, and learned parameters, under separate names.
- **Threshold discipline** The 20% relative criterion was not lowered to let
  the candidate through. It was removed because *no* correct system can satisfy
  it against a fair baseline that is already exact — only an unfair comparison
  can. The reasoning is in [`benchmark_protocol.md`](benchmark_protocol.md) and
  [`errata.md`](errata.md). The decision was made on development evidence and
  frozen before any confirmation batch was generated.
- **Regression** `tests/test_gates.py::ComparisonGateTests::test_bounce_parity_fails_when_the_candidate_is_worse_than_the_fair_baseline`;
  `tests/test_predictors.py::BaselineTests::test_reflected_baseline_is_exact_at_a_wall`.

### AAA-005 — the reproducibility gate performed no computation
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** repaired
- **Affected** `aaa/benchmark.py`
- **Reproduction** The gate was
  `_trial_seed(...) == _trial_seed(...) and all(state["format_version"] == ...)`
  — a deterministic pure function compared with itself, plus a string check.
- **Consequence** "Reproducibility PASS" asserted something never tested.
- **Repair** Three executed checks: an independently initialized duplicate mini
  experiment compared trajectory-by-trajectory and on final state; a
  checkpoint/save/reload/resume equivalence comparison; and a committed golden
  seed-mapping fixture. Results are written to `verification/`. The gate is
  `NOT_VERIFIED` when they have not run.
- **Regression** `tests/test_gates.py::VerificationGateTests`;
  `tests/test_seeds.py::GoldenSeedTests`;
  `tests/test_rls_numerics.py::SerializationAndResumeTests`.

### AAA-006 — `speed_change` was named as adaptation but froze the candidate
- **Source** Opus, Sol · **Severity** medium · **Status** repaired
- **Reproduction** `_run_regular_family` built the candidate with
  `update_enabled=False` for the speed-change family.
- **Consequence** A robustness control read as evidence of online adaptation.
- **Repair** Renamed `speed_change_frozen_robustness`, documented as a frozen
  robustness control. An updating arm is reported alongside. A separate
  `always_online` family carries the autonomy claim.
- **Regression** `tests/test_gates.py`; the family's report section names the
  update mode explicitly.

### AAA-007 — the changed-law gate omitted its declared constant-motion requirement
- **Source** Sol · **Severity** high · **Status** repaired
- **Reproduction** The gate checked improvement over frozen and persistence; the
  specification also declared a constant-motion non-regression criterion that
  the code never evaluated.
- **Repair** The gate now evaluates online-vs-frozen, online-vs-persistence,
  and constant-motion non-regression, with an interval for each, plus the
  unchanged-world control as a separate required gate. Both average and
  cumulative post-change error are reported, including the unavoidable first
  surprise.
- **Regression** `tests/test_gates.py::ComparisonGateTests`.

### AAA-008 — no always-online deployment track existed
- **Source** Astra, Opus · **Severity** high · **Status** repaired
- **Consequence** Every candidate arm was frozen or unfrozen by the evaluator
  based on hidden scenario identity, so nothing measured autonomous operation.
- **Repair** An `always_online` family: one instance per replica keeps
  predicting and updating across a rotation of regimes with no evaluator mode
  switching, no reset and no event signal. Its gate is a non-regression
  requirement against the reflected baseline.
- **Remaining limitation** The rotation is a fixed sequence of the existing
  regimes, not an open-ended stream.

---

## B. Statistics

### AAA-010 — the bootstrap was not hierarchical
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** repaired
- **Reproduction** `_bootstrap_ci` and `_paired_improvement_ci` resampled
  replicas, then took **one** episode per selected replica.
- **Root cause** The second stage discarded the within-replica sample size, so
  the interval described a different estimand from the point estimate and
  understated the evidence available.
- **Repair** Each draw resamples replicas with replacement, then resamples that
  replica's episode summaries with replacement preserving its observation
  count, preserves pairing, and recomputes the gate's exact statistic.
- **Regression** `tests/test_stats.py::ResamplingStructureTests`.

### AAA-011 — the interval and the headline computed different quantities
- **Source** Astra, Sol · **Severity** critical · **Status** repaired
- **Reproduction** Point estimates were `1 - mean(candidate)/mean(baseline)`
  (a ratio of means) while intervals averaged per-episode
  `1 - left/right` (a mean of ratios).
- **Repair** Every estimand is named in the specification and in code, and each
  bootstrap draw recomputes the same statistic the gate uses.
- **Regression** `tests/test_stats.py::EstimandTests`, including
  `test_ratio_of_means_is_not_mean_of_ratios`.

### AAA-012 — event weighting differed between point estimate and interval
- **Source** Sol · **Severity** high · **Status** repaired
- **Reproduction** The bounce point estimate pooled every event; the interval
  used episode-balanced event means, under the same name.
- **Repair** `statistics.event_weighting` is declared once and used for both.
  Alternative summaries are reported under distinct names.

### AAA-013 — no-event episodes contaminated the event bootstrap
- **Source** Astra, Sol · **Severity** high · **Status** repaired
- **Reproduction** `event_values[name].append(mean(event_errors) if event_errors else mean(errors))`
  — an episode with zero bounce events contributed its ordinary whole-episode
  error as if it were an event error.
- **Repair** Such episodes are excluded from event resampling and counted
  separately; every family report states how many were excluded.
- **Regression** `tests/test_gates.py::ComparisonGateTests::test_bounce_gate_is_not_verified_without_event_episodes`.

### AAA-014 — a missing interval was treated as satisfied
- **Source** Astra, Opus · **Severity** critical · **Status** repaired
- **Reproduction** Every gate used the pattern
  `(ci is None or ci[0] > 0)` — an interval that could not be computed made the
  condition true.
- **Consequence** Absence of evidence was converted into `PASS`.
- **Repair** `Interval` carries a status and a reason. A required interval that
  could not be computed yields `INSUFFICIENT_EVIDENCE`.
- **Regression** `tests/test_stats.py::InsufficientEvidenceTests`.

### AAA-015 — the post-change gate resampled whole-episode error
- **Source** Sol · **Severity** high · **Status** repaired
- **Repair** Post-change gates resample and recompute the post-change window
  statistic itself.

### AAA-016 — non-regression claims had no uncertainty
- **Source** Sol · **Severity** medium · **Status** repaired
- **Repair** A single `margin_statistic`,
  `mean(candidate) - max_ratio * mean(baseline) - floor`, gives every
  non-regression claim an interval; the gate asks whether its upper end stays
  at or below zero.

### AAA-017 — multiple primary comparisons and repeated attempts were untreated
- **Source** Astra, Sol · **Severity** medium · **Status** repaired
- **Repair** Holm-Bonferroni, predeclared in the specification, over the
  primary comparisons. The family size includes previously spent confirmation
  attempts against the same specification hash, so re-rolling costs power.
  Holm was chosen as the standard uniformly-more-powerful improvement on
  Bonferroni that requires no dependence assumptions; the choice is recorded in
  the specification, not selected after seeing results.
- **Regression** `tests/test_stats.py::MultiplicityTests`, including
  step-down ordering and the repeated-attempt family enlargement.

### AAA-018 — 500 episodes from five checkpoints was described as 500 replicas
- **Source** Astra, Opus · **Severity** medium · **Status** repaired
- **Repair** Five training replicas is labelled the routine engineering
  minimum. A `high_replication` role runs 20 independent training lineages.
  Replica and episode counts are reported separately everywhere.

---

## C. The learner

### AAA-020 — covariance windup under weak excitation
- **Source** Astra · **Severity** critical · **Status** repaired
- **Claim** Failures around update 280 (repeated input) and 6,644 (stationary
  stream).
- **Reproduction** With the selected forgetting factor 0.90:

  | Probe | Observed |
  |---|---|
  | repeated identical input | non-finite covariance at update **6,642** |
  | stationary observation stream | non-finite covariance at update **6,642** |
  | slow constant velocity | min eigenvalue **-6.7e-02** at update **323** |

  The reported indices differ from Astra's because the probe details differ;
  the failure class reproduces, and positive semidefiniteness is lost far
  earlier than the overflow.
- **Root cause** Classical covariance windup: the `1/lambda` inflation is
  applied in directions that receive no information, so `P` grows without
  bound and symmetry and PSD are lost to rounding before it overflows
  (Astrom & Wittenmark, *Adaptive Control* 2e ch. 3; Ljung & Soderstrom,
  *Theory and Practice of Recursive Identification* ch. 2).
- **Repair** The learner propagates a square-root factor `S` with `P = S S^T`
  (Potter rank-one update), so symmetry and PSD hold by construction and the
  effective condition number is the square root of the covariance-form one;
  plus trace-bounded forgetting, where inflation is suspended when `trace(P)`
  would exceed the declared bound. Suspensions are **counted and reported**,
  not silently absorbed. Optional directional forgetting, a dead zone, and
  self-triggered forgetting are available and declared in the specification.
- **Verification** With no forgetting the recursion matches an independent
  ridge batch least-squares solution computed by SVD (not by re-running the
  same formula) to 1.7e-18. The pre-repair algebra was correct; the failure was
  forgetting, not the recursion.
- **Regression** `tests/test_rls_numerics.py` — 11 weak-excitation streams run
  to 20,000 updates, batch-reference comparison, square-root factor identities,
  resume equivalence, detector and dead-zone behaviour.
- **Development evidence** [`evidence/diagnosis/diagnosis.json`](evidence/diagnosis/diagnosis.json)
  runs the forgetting sweep with and without the trace bound. Unbounded
  lambda=0.90 on the legacy position basis reaches covariance trace 9e22 and
  loses PSD to reconstruction rounding; the bounded arm stays valid at the same
  holdout error.

### AAA-021 — invalid learner state was accepted, and asymmetry silently repaired
- **Source** Astra, Opus · **Severity** critical · **Status** repaired
- **Reproduction** Checkpoint loading accepted every one of: negative-definite,
  indefinite, asymmetric, nearly singular, all-zero, and badly conditioned
  covariance. Only non-finite entries were rejected. `_validate_state`
  symmetrized an asymmetric matrix in place, with no record.
- **Repair** `validate_covariance` checks finiteness, symmetry within a declared
  tolerance, positive semidefiniteness and condition number, and **raises**
  rather than repairing. Tolerances come from the specification rather than
  being hardcoded. Required checkpoint fields and the format version are
  checked. A numerical failure is visible evidence.
- **Regression** `tests/test_predictors.py::RLSStateTests::test_invalid_covariance_states_are_rejected_not_silently_repaired`
  covers all seven classes.
- **Note** An ill-scaled configuration (displacement scale 500x off) is
  detected and reported by `check_state()` rather than running on quietly:
  `tests/test_rls_numerics.py::test_a_badly_scaled_displacement_is_reported_not_hidden`.

### AAA-022 — the legacy learner had no finite-input validation
- **Source** Sol · **Severity** medium · **Status** repaired
- **Repair** `OnlineLinearPredictor` validates finite learning rate, weights and
  targets, uses the shared `_check_history` safeguard, and raises on a
  non-finite update result.

### AAA-023 — the legacy-learner diagnosis was incomplete
- **Source** Astra, Opus · **Severity** medium · **Status** repaired
- **Previous state** A single SVD fit that could not separate conditioning from
  optimization from regime mixing, while the prose implied it could.
- **Repair** `aaa/diagnosis.py` runs a controlled grid over four feature bases,
  five estimators (batch least squares, SGD, normalized LMS, square-root RLS
  bounded and unbounded), three regimes, three orderings including replay, and
  a forgetting and ridge sweep — 480 configurations. It reports per-basis
  conditioning, effective rank, singular values, column scales and
  correlations, update norms, order sensitivity, divergence, and covariance
  diagnostics. It explicitly does **not** assert a single root cause.
- **Evidence** [`evidence/diagnosis/`](evidence/diagnosis/). Development data
  only; no confirmation stream is touched.

---

## D. Measurement

### AAA-030 — normalized and raw units were mixed in one table
- **Source** Astra, Sol · **Severity** high · **Status** repaired
- **Reproduction** On `[0, 10]` with a raw error of 1.0: `aggregate_metrics`
  reported `mae = 0.1` (normalized) while `bounce_mae`, `change_mae` and
  `non_bounce_mae` reported `1.0` (raw), in the same table.
- **Root cause** Some paths used `normalized_absolute_error`, others read
  `absolute_error` directly. On the default unit-width interval the two are
  numerically identical, which hid it.
- **Repair** One canonical accessor; every benchmark aggregate routes through
  it. Raw values are available under explicitly raw names.
- **Regression** `tests/test_metrics.py::UnitContractTests` runs the contract on
  `[0, 1]`, `[0, 10]`, `[-5, 5]` and shifted intervals, including
  `test_unit_width_hides_nothing_that_a_wide_interval_reveals`.

### AAA-031 — recovery eligibility was diluted away by averaging
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** repaired
- **Reproduction** Pre-event MAE 0.02, a single-transition shock of 0.20 (a 10x
  spike), immediate recovery. Post-window mean 0.0236 against a meaningful
  threshold of 0.025, so `applicable = False`, reason "no meaningful
  post-change error increase". The event vanished from the denominator.
- **Consequence** `475 / 475` recovery rates counted only events whose 50-step
  average stayed elevated. Fast, large shocks — the ones a recovery metric most
  wants to count — were structurally excluded.
- **Repair** Eligibility is driven by the **peak** shock in a declared shock
  window against a recent pre-event reference. Recovery is a separate
  question. Every episode falls into exactly one of six statuses and
  unrecovered events are counted.
- **Regression** `tests/test_recovery.py::ShockDilutionRegressionTests` and
  `StatusPartitionTests` (every status), plus threshold-equality and
  window-boundary cases.

### AAA-032 — one learner's errors were the recovery reference for every predictor
- **Source** Astra, Sol · **Severity** high · **Status** repaired
- **Reproduction** `changed.add(records, pre_change_errors={name: prefix_errors for name in names})`
  gave persistence, constant motion, frozen and online the same pre-event error
  sequence — the prefix model's.
- **Consequence** Per-predictor recovery statistics were not about those
  predictors.
- **Repair** Each predictor's tolerance derives from its own pre-event errors.
- **Regression** `tests/test_recovery.py::test_predictor_specific_references_give_predictor_specific_results`.

### AAA-033 — the pre-event reference was the entire early prefix
- **Source** Opus · **Severity** medium · **Status** repaired
- **Root cause** Averaging over the whole prefix includes the early training
  transient, inflating the reference and making the tolerance too generous.
- **Repair** A declared recent pre-event reference window.
- **Regression** `tests/test_recovery.py::ReferenceWindowTests`.

### AAA-034 — latency benchmarked a model that was not the candidate
- **Source** Astra, Sol · **Severity** medium · **Status** repaired
- **Reproduction** `_measure_cpu_latency` constructed `forgetting=1.0` while the
  selected candidate used 0.90, on a repeated feature vector.
- **Repair** The selected candidate itself is benchmarked, on a representative
  input distribution, with warm-up excluded, separate predict / update /
  combined p50, p95 and p99, a failure count, and recorded hardware and
  numerical-library metadata. No plotting or disk I/O is inside the timed
  region.

### AAA-035 — the "learning curve" plotted within-episode behaviour
- **Source** Opus · **Severity** medium · **Status** repaired
- **Root cause** Rolling error within episodes reflects changing episode
  difficulty, not cumulative learning.
- **Repair** The within-episode view is retained and explicitly labelled "not a
  learning curve". A real curve scores frozen checkpoints taken at declared
  cumulative update budgets on one fixed development probe bank, with
  per-replica spread. Different reset models are never connected as if they
  were one learner.

---

## E. Protocol integrity

### AAA-040 — most of the specification file was decorative
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** repaired
- **Reproduction** 26 declared leaves were never read by the runner, including
  every `world` value, the entire `candidate` block, every `primary_gates`
  threshold, `families`, `resampling` and `post_change_window`. Editing them
  changed the recorded specification and nothing else.
- **Consequence** The document describing the experiment and the code running
  it could disagree without anything noticing.
- **Repair** A typed, strictly validated schema covering randomness, world,
  candidate, training, stratification, families, changed-law equations and
  ranges, confirmation, recovery, statistics, gates, latency, tolerances,
  artifacts and baselines. The runner consumes these values; nothing is
  re-hardcoded. Unknown keys, missing keys, unknown feature sets and unknown
  gate evaluators are rejected at parse time.
- **Regression** `tests/test_spec.py::ExecutableSpecTests::test_every_declared_leaf_is_either_consumed_or_explicitly_descriptive`
  fails if any declared value stops being read, plus per-section tests proving
  that changing a value changes behaviour.

### AAA-041 — confirmation minimums lived only in the CLI
- **Source** Astra, Sol · **Severity** critical · **Status** repaired
- **Reproduction** `run_benchmark_v2(role="confirmation_a", replicas=1, episodes=1)`
  was accepted; only `main()` checked minimums.
- **Repair** The core runner enforces every confirmation invariant.
- **Regression** `tests/test_cli.py::RunnerEnforcementTests`.

### AAA-042 — confirmation proceeded from a dirty working tree
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** repaired
- **Reproduction** Dirty state was recorded in metadata and never enforced.
- **Repair** `require_clean_source_tree` is declared and enforced by the runner
  for confirmation roles. Development may run dirty, labelled as such. Commit,
  tree hash, dirty flag, specification hash, dependency-lock hash, checkpoint
  hashes and benchmark version are all recorded.

### AAA-043 — changing only `--attempt-id` reused identical confirmation streams
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** repaired
- **Reproduction** `attempt_id` did not appear in `_trial_seed`'s signature. Two
  confirmations differing only by label drew the same numbers.
- **Consequence** "Independent confirmation B" could be a relabelled rerun of A.
- **Repair** A predeclared confirmation batch registry. The batch identity *is*
  the seed input. A batch has a lifecycle (planned / consumed / retired); a
  consumed batch cannot be reused as fresh; re-running one needs explicit
  `--reproduce`; a failed batch is retired permanently and stays recorded.
- **Regression** `tests/test_seeds.py::BatchRegistryTests` (14 cases).

### AAA-044 — the `role` parameter threaded into training was ignored
- **Source** Astra, Opus, Sol · **Severity** high · **Status** repaired
- **Reproduction** `_train_replica(world, root, role, replica)` accepted `role`
  and called `_trial_seed(root, "development", ...)`. Confirmation A and B
  trained from identical streams and produced identical checkpoints.
- **Assessment** Not automatically wrong — evaluating one frozen selected model
  set on two independent test streams is a valid design — but it was
  accidental and undeclared.
- **Repair** `confirmation.ab_relationship` is an explicit declared choice.
  `shared_frozen_checkpoints` (the default) states that A and B evaluate the
  same frozen checkpoints; `high_replication` uses independent lineages. The
  manifest records the relationship and the checkpoint hashes. No parameter is
  accepted and discarded.
- **Regression** `tests/test_seeds.py::StreamIdentityTests` covers both
  relationships.

### AAA-045 — no freeze manifest existed
- **Source** Astra, Opus · **Severity** high · **Status** repaired
- **Repair** `aaa freeze` writes a machine-readable manifest with the protocol
  version, specification hash, source commit and tree hash, dependency-lock
  hash, candidate definition and hyperparameters, training stream identities,
  checkpoint hashes, baselines, gate thresholds, sample counts, recovery
  definitions, statistical methods, planned batch ids and the intended A/B
  relationship. A confirmation that drifts from it is refused, with every
  mismatch reported at once.
- **Regression** `tests/test_recompute_and_manifest.py::ManifestTests`.

### AAA-046 — `change_step == steps_per_episode` was accepted
- **Source** Sol · **Severity** medium · **Status** repaired
- **Root cause** An implicit end-of-horizon sentinel meaning "no event".
- **Repair** Explicit semantics: `None` means no change event; an integer must
  satisfy `0 <= change_step < steps_per_episode`.
- **Regression** `tests/test_environment.py` covers zero, the last valid
  transition, the rejected endpoint and the no-change case.

### AAA-047 — `StepRecord.seed` carried a replica id
- **Source** Astra, Sol · **Severity** medium · **Status** repaired
- **Repair** A versioned `aaa.step_record.v2` schema with a `TrialIdentity`
  giving each concept its own field: trial id, role, family, branch,
  confirmation batch, replica id, episode, environment seed, training seed
  lineage, stratum, checkpoint hash and update mode. Loading an unrecognised
  schema version fails loudly.
- **Regression** `tests/test_experiment.py::RecordSchemaTests`.

### AAA-048 — multi-wall crossings were collapsed to one boolean and one wall
- **Source** Sol · **Severity** medium · **Status** repaired
- **Root cause** Reflection could cross several boundaries in one transition
  while the metadata recorded a single inferred wall, and the inference was
  duplicated inline in two environments.
- **Repair** One tested `resolve_reflection` helper returns the ordered tuple of
  walls crossed; records carry `bounce_walls` and `bounce_count`.
- **Regression** `tests/test_environment.py::ReflectionTests` — exact contacts
  in both directions, inward contact, one-, two- and many-wall overshoot,
  arbitrary bounds.

### AAA-049 — frozen-checkpoint integrity was a hardcoded `True`
- **Source** Astra, Opus, Sol · **Severity** high · **Status** repaired
- **Reproduction** `"checkpoint_weights_unchanged": True` and
  `"weights_after": weights_before` were literals.
- **Repair** Actual post-evaluation weights and update counts of every frozen
  copy are collected and compared; the report prints the measured comparison
  and the number of copies checked. The benchmark's correctness gate checks the
  same property for the frozen branch, and that the online branch really did
  update.
- **Regression** `tests/test_legacy_v1.py::test_frozen_integrity_is_measured_from_actual_state`.

### AAA-050 — evaluation wrote into the source tree
- **Source** Astra, Sol · **Severity** medium · **Status** repaired
- **Reproduction** `run_full_evaluation` wrote `reports/experiment_report_*.md`
  into the project root even when the caller chose another `--output-root`.
- **Repair** Nothing is written outside the requested output root.
- **Regression** `tests/test_legacy_v1.py::test_a_run_writes_nothing_outside_the_requested_output_root`.

### AAA-051 — `_episode_seed` raised `KeyError` for a supported scenario
- **Source** Sol · **Severity** low · **Status** repaired
- **Repair** The oscillator scenario has its own stream; an unknown scenario or
  phase raises a clear `ValueError` naming the offending value.

---

## F. Animation and visualization

### AAA-060 — the animation could only load the legacy checkpoint
- **Source** Astra, Opus, Sol · **Severity** medium · **Status** repaired
- **Repair** Dispatch on the checkpoint's declared format version, covering both
  the RLS candidate and the legacy learner; an unknown format is refused.

### AAA-061 — `event.key = None` crashed the key handler
- **Source** Sol · **Severity** low · **Status** repaired
- **Reproduction** `event.key.lower()` with Matplotlib's valid `None` key.
- **Repair** `handle_key(None)` returns `"ignored"`.
- **Regression** `tests/test_animation.py::KeyHandlingTests`.

### AAA-062 — animation temporal semantics were a second, divergent implementation
- **Source** Opus · **Severity** medium · **Status** repaired
- **Root cause** The ordering lived inside a Matplotlib callback, was untestable
  without a display, and passed the whole history to `update` rather than the
  window it predicted from.
- **Repair** `AnimationSession` is a plain object with no GUI dependency,
  following the same predict / advance / reveal / score / update ordering. The
  GUI wrapper only draws what the session reports. Pause stops the simulation
  state, not just the drawing; restart restores the environment, RNG, history,
  model, counters, displayed prediction and paused state.
- **Regression** `tests/test_animation.py` — 22 headless cases.

### AAA-063 — pyplot was imported, then the backend switched afterwards
- **Source** Astra, Opus, Sol · **Severity** medium · **Status** repaired
- **Repair** `aaa/visualization.py` never imports pyplot. Figures are built with
  the object-oriented `Figure` API and rendered by the Agg canvas. The
  interactive backend choice is left entirely to the animation.
- **Regression** `tests/test_visualization.py::BackendTests`.

### AAA-064 — plot axes assumed the interval was `[0, 1]`
- **Source** Sol · **Severity** low · **Status** repaired
- **Repair** Limits derive from the configured bounds and the data.
- **Regression** `tests/test_visualization.py::ArbitraryBoundsTests`.

### AAA-065 — no plots supported scientific auditing
- **Source** Opus · **Severity** low · **Status** repaired
- **Repair** Five benchmark figures: cumulative learning curve, per-stratum
  performance, recovery-time distribution, paired replica scatter, and RLS
  covariance diagnostics.

---

## G. Reproducibility and infrastructure

### AAA-070 — CI published a lock file and ignored it
- **Source** Astra, Opus, Sol · **Severity** high · **Status** repaired
- **Repair** The validated CI job installs `requirements-lock.txt` and then
  proves the installed set matches it (`tools/check_lock.py`), recording the
  lock hash. The lock now pins the tooling CI actually runs.

### AAA-071 — `requires-python >= 3.10` with only 3.12 tested
- **Source** Astra, Sol · **Severity** medium · **Status** repaired
- **Repair** A CI matrix over 3.10, 3.11, 3.12 and 3.13, matching the declared
  classifiers exactly.

### AAA-072 — no fresh-install verification
- **Source** Astra · **Severity** high · **Status** repaired
- **Root cause** The benchmark specification was loaded from a
  repository-relative path, which disappears when the package is installed
  elsewhere.
- **Repair** The canonical specification ships as package data. A CI job builds
  a wheel, installs it, and runs from an unrelated directory.
- **Regression** `tests/test_spec.py::test_the_spec_ships_inside_the_installed_package`.

### AAA-073 — workflow dispatch inputs were interpolated into shell text
- **Source** Astra, Sol · **Severity** high · **Status** repaired
- **Repair** Inputs are passed through the environment, validated against a
  character-class pattern before use, and never interpolated into a command
  string. Workflows declare `permissions: contents: read`, pin actions to
  commit SHAs, keep timeouts, and retain evidence with `if: always()`.

### AAA-074 — the confirmation workflow could not fail
- **Source** Astra, Opus, Sol · **Severity** critical · **Status** repaired
- **Repair** The workflow takes the CLI exit status as its verdict, and
  recomputes every metric and gate from the retained raw evidence afterwards.
  Evidence is retained whether the attempt passed or failed.

### AAA-075 — no experiment registry or safe resume
- **Source** Astra, Opus · **Severity** medium · **Status** repaired
- **Root cause** Output-directory existence was the only notion of experiment
  state.
- **Repair** A registry with atomic per-trial transitions (`PLANNED`, `RUNNING`,
  `COMPLETE`, `FAILED`, `INTERRUPTED`, `SUPERSEDED`) storing identity, seed,
  outputs, checksums, checkpoint hash, timings, failure stage and error. A
  resume does not re-run completed trials, cannot change their seeds, does not
  overwrite evidence and preserves failure records.
- **Regression** `tests/test_evidence.py::RegistryStateMachineTests`.

### AAA-076 — evidence could not be independently recomputed
- **Source** Astra, Opus · **Severity** high · **Status** repaired
- **Repair** `aaa recompute` rebuilds per-step metrics, episode and replica
  summaries, intervals, gates and the report from retained raw evidence without
  retraining or re-simulating. It verifies checksums and the specification hash
  first and exits non-zero if a stored gate status fails to reproduce. The
  oscillator experiment additionally retains the common prefix, the intervention
  state, the hashed branch state and all three branch outputs.
- **Regression** `tests/test_recompute_and_manifest.py::RecomputeTests`,
  including corrupted and deleted raw files.

### AAA-077 — evidence durability
- **Source** Astra, Opus, Sol · **Severity** medium · **Status** open
- **Reproduction** Tens of megabytes of generated JSON were committed while
  being described as compact, and the sole copy of the acceptance evidence
  otherwise lived in a git-ignored directory or a 30-day CI artifact.
- **Repair** An explicit artifact policy: compact summaries, reports, metadata,
  manifests, checksums, small checkpoints and the registry are committed; raw
  per-step arrays are not. Raw evidence is regenerable from public source, the
  immutable specification, the immutable seed mapping and the checkpoint
  lineage, and CI retention is raised to 90 days.
- **Remaining limitation** Raw step logs are regenerable rather than
  independently archived. Git LFS or an external archive would be stronger and
  is a documented recommendation, not something claimed to be in place. See
  [`evidence_policy.md`](evidence_policy.md).

### AAA-078 — private filesystem paths in public documentation
- **Source** Sol · **Severity** low · **Status** repaired
- **Repair** `/home/cinqic/Documents/AAA` is gone from instructional
  documentation, replaced with `git clone` / `cd AAA`. Historical result files
  retain their original provenance strings, which is what provenance is for.
  Run metadata never records a home directory.
- **Regression** `tests/test_evidence.py::test_hardware_metadata_never_leaks_a_home_directory`.

---

## H. Distribution and selection

### AAA-080 — training and confirmation speed ranges did not match
- **Source** Astra, Opus, Sol · **Severity** high · **Status** repaired
- **Reproduction** Training used 0.02-0.06; confirmation used 0.08-0.20.
- **Consequence** A deliberate-looking extrapolation was reported as ordinary
  held-out generalization.
- **Repair** Training is in-domain on the confirmation range, declared
  explicitly in `training.distribution_relationship`. Extrapolation above the
  training range is measured by a separate family labelled
  `speed_extrapolation`.

### AAA-081 — candidate selection had no auditable evidence
- **Source** Astra, Opus · **Severity** high · **Status** repaired
- **Repair** `aaa select-candidate` compares 36 variants across feature set,
  forgetting factor and mode, ridge, reflection, trace bound, dead zone,
  unfolding, detector and training budget on development streams only, and
  records the full table including unsuccessful alternatives, the stated
  selection rule, tie handling, and a numerical stress verdict per variant.
- **Evidence** [`evidence/candidate_selection.json`](evidence/candidate_selection.json);
  the earlier round is retained as
  [`evidence/candidate_selection_first_pass.json`](evidence/candidate_selection_first_pass.json).
- **Discipline** Selection touches no confirmation stream. Every previously
  inspected v2 confirmation stream is treated as contaminated and is not used
  for selection.

---

## I. Found during this investigation

### AAA-090 — a zero `--replicas`/`--episodes` override silently became the full budget
- **Source** self · **Severity** high · **Status** repaired
- **Reproduction** `replicas or default` treats an explicit `0` as falsy, so
  `run_benchmark(replicas=0)` ran the full 5x100 confirmation budget instead of
  raising.
- **Repair** Overrides are validated before defaulting; zero, negative,
  fractional and boolean values are rejected.
- **Regression** `tests/test_cli.py::test_a_zero_override_is_refused_rather_than_silently_defaulted`.

### AAA-091 — a resume could silently change a recorded trial's seed
- **Source** self · **Severity** high · **Status** repaired
- **Reproduction** `ExperimentRegistry.plan` returned any existing record
  without checking that the identity matched, so a resumed run with a changed
  seed mapping would continue as if it were the same experiment.
- **Repair** Replanning an existing trial with a different family, branch,
  replica, episode or environment seed raises.
- **Regression** `tests/test_evidence.py::test_replanning_must_not_change_a_recorded_seed`.

### AAA-092 — the declared numerical tolerances were never used
- **Source** self · **Severity** medium · **Status** repaired
- **Reproduction** `tolerances.covariance_symmetry`, `covariance_psd` and
  `max_condition_number` were in the specification while the learner used
  hardcoded module constants — the same class of defect as AAA-040.
- **Repair** Tolerances are validation policy supplied by the caller, passed
  from the specification, and deliberately excluded from the checkpoint hash.

### AAA-093 — `baselines.legacy_linear_sgd` was declared but never run
- **Source** self · **Severity** medium · **Status** repaired
- **Repair** The legacy learner is trained on the identical training stream and
  reported as a named diagnostic arm in every motion family. It reproduces the
  historical v1 behaviour under the repaired protocol: in the bouncing family
  it is worse than persistence.

### AAA-094 — the v1 report generator referenced removed metric keys
- **Source** self · **Severity** medium · **Status** repaired
- **Reproduction** `_seed_mae_sd` read `summary["experiments"][x]["seeds"]`,
  which the repaired metrics module no longer produces; report generation
  raised `KeyError`.
- **Repair** Updated to the replica structure, with recovery status counts and
  four-significant-figure human output.

### AAA-095 — the early/late summary did arithmetic on possibly-missing values
- **Source** self · **Severity** low · **Status** repaired
- **Repair** A missing side stays `None` rather than becoming a fabricated zero.

### AAA-096 — the matched branch state was not hashed into the evidence
- **Source** self · **Severity** medium · **Status** repaired
- **Repair** `metrics/changed_law_interventions.json` records a hash of the
  exact state both branch copies were created from, so the matched design is
  auditable without trusting the runtime.

### AAA-097 — one failing optional plot discarded all of them
- **Source** self · **Severity** low · **Status** repaired
- **Repair** Each figure is rendered independently; a summary section that is
  absent skips its figure and costs nothing else.

### AAA-098 — the spec schema accepted missing keys and unknown enumerations
- **Source** self · **Severity** medium · **Status** repaired
- **Reproduction** Removing a whole section raised `KeyError` rather than
  `SpecError`; an unknown `feature_set` or gate `evaluator` was accepted at
  parse time and only failed much later, or never.
- **Repair** Strict validation rejects unknown keys, missing required keys,
  unknown feature sets, unknown forgetting modes and unknown gate evaluators,
  and validates against the model's own declared option sets so the two cannot
  drift.

### AAA-099 — `WorldConfig`'s default `change_step` was invalid for short episodes
- **Source** self · **Severity** low · **Status** repaired
- **Repair** The default is `None` (no change event). The historical v1
  configuration declares its change step explicitly.

### AAA-100 — misleading `# type: ignore` comments and `object` annotations
- **Source** Sol, self · **Severity** low · **Status** repaired
- **Reproduction** 35 `# type: ignore` comments, several suppressing an error
  code other than the one they named; `__import__("math")` inside a metrics
  function; unused imports, parameters and assignments; a `while` loop whose
  body was an unconditional `raise`; duplicated wall-classification
  expressions; broad `object` environment typing.
- **Repair** Every ignore removed rather than re-aimed, the `dict[str, object]`
  annotations that made them necessary corrected, an `Environment` Protocol
  introduced, dead parameters deleted (not merely renamed), and ruff plus mypy
  now pass with no blanket suppressions.

---

## J. Found by the repaired benchmark itself

### AAA-120 — continued updating regresses in continuous deployment
- **Source** the v2.1 `always_online` gate · **Severity** critical · **Status** repaired
- **Affected** `aaa/predictors.py`

**What happened.** The first formal confirmation round under protocol v2.1
**failed**. Confirmation A and confirmation B, on independent fresh streams,
both failed the required `always_online_stability` gate and both exited
non-zero. Thirteen of fourteen gates passed in each.

| | A | B |
|---|---:|---:|
| `candidate_online` normalized MAE | 4.427e-05 | 3.414e-05 |
| `constant_motion_reflected` | 6.621e-06 | 7.012e-06 |
| limit (1.1x baseline + 1e-5) | 1.728e-05 | 1.771e-05 |
| worst replica | 1.519e-04 | 8.784e-05 |

Both batches are retired permanently. **No threshold was altered.** The failed
attempts are committed at [`../results/benchmark_v2_1/`](../results/benchmark_v2_1/).

This gate did not exist before the repair. Every frozen track passed; only the
continuously-updating deployment arm failed, which is precisely the question the
always-online family was added to ask.

**First, not a software bug.** The trained checkpoints were numerically healthy
(condition number ~31, positive semidefinite, zero forgetting suspensions). The
frozen copy of the same model was exact. The failure was concentrated in
particular replicas (one at 1.5e-04 against four at ~1.2e-05), and the
per-segment breakdown showed straight-motion segments carrying a heavy tail —
mean 8.2e-05 against a p95 of 1.2e-05 — while the reflected baseline was exact
at 2.2e-19 there.

**Root cause, located on development streams.** Instrumenting weight drift
against the exactly-identified solution through a single wall contact:

| step | | scored error | weight drift from exact | change |
|---:|---|---:|---:|---:|
| 167 | wall crossing | 3.640e-11 | 9.525e-12 | +1.211e-13 |
| 168 | **next window** | 4.222e-03 | 2.952e-05 | **+2.952e-05** |
| 169 | | 1.664e-05 | 2.928e-05 | -2.450e-07 |
| ... | | | | slow decay over 30+ steps |

The wall transition itself is harmless: `unfold_target` already inverts the
public reflection map, so that sample is an ordinary one. The damage is entirely
the **next** window, whose displacement feature `x[t] - x[t-1]` is a *folded*
difference and therefore not a sample of the linear law at all. Fitting that one
sample moves the weights by ~3e-5 and costs tens of transitions to recover.
A frozen copy never updates, so it is never damaged; a continuously updating
instance accumulates the damage across every wall contact it ever sees.

**Repair.** The learner does not update on a window whose displacement feature
straddles a wall. The trigger is causally clean: the learner's **own** previous
raw prediction having required reflection to stay in bounds. That is the same
programmed public knowledge of the observation format that licenses reflection
in the first place, applied to its own output. No evaluator bounce label,
scenario, event flag or change schedule is involved. Skipped steps are counted
in `reflection_skips` and serialized.

Measured on development streams, three replicas, always-online rotation:

| | mean | worst replica | p99 | max weight drift |
|---|---:|---:|---:|---:|
| update on every step | 4.732e-05 | 1.133e-04 | 1.549e-04 | 3.541e-01 |
| skip the straddling window | **7.862e-06** | **9.062e-06** | **1.501e-05** | **1.564e-04** |

It costs nothing where the candidate already succeeded: the changed-law
adaptation improvement over the identical frozen copy is 0.6964 with the policy
and 0.6964 without it, because the oscillator branch rarely reaches a wall.

**An alternative that did not work, recorded because it did not.** The first
idea was to maintain a continuous unfolded coordinate frame and run the linear
law there. The prototype was far worse than the current candidate (mean 2.84e-01
against 1.68e-05): the offset bookkeeping across successive reflections is
subtler than it looks, and each reflection flips the direction of the unfolded
axis. It was abandoned on development evidence, not adopted and quietly dropped.

- **Regression** `tests/test_rls_numerics.py::StraddlingWindowTests` — six cases
  covering the skip, its absence, the causal trigger, interaction with
  `reflect=False`, checkpoint survival and that a skipped step moves no learned
  state.
- **Discipline** The change is a candidate change, so it requires a fresh
  predeclared confirmation batch pair, a new specification hash and a new freeze
  manifest. The spent attempts are counted in the multiplicity family. The
  motivation to look came from a confirmation failure; the mechanism, the fix
  and its validation all come from development streams, and the failed streams
  were not used to choose anything.

---

## Not repaired, and why

### AAA-110 — repository settings could not be verified or changed from here
- **Source** Astra · **Status** open
- Branch protections, required status checks, rulesets, Dependabot
  configuration and default action permissions are administrative settings.
  This environment cannot read or change them reliably, so nothing is claimed
  about their state. The recommended settings are written out in
  [`../SECURITY.md`](../SECURITY.md) as recommendations, not as completed work.

### AAA-111 — parallel execution was profiled and not adopted
- **Source** Astra, Opus · **Status** superseded
- A full confirmation attempt completes in minutes of CPU time. Introducing
  workers would add an RNG-allocation and BLAS-oversubscription risk surface to
  save time the project does not need. Seeds already map to trial identity
  rather than scheduling order, so parallelism remains available without a
  reproducibility change if the workload ever grows. Not adopted, and the
  reason is recorded rather than the option being silently dropped.
