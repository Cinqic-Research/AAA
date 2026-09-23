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
- **Development evidence** is generated by the reproducible procedure in
  [`diagnosis.md`](diagnosis.md). Its ignored
  `docs/evidence/diagnosis/diagnosis.json` output is not a committed evidence
  artifact. The forgetting sweep with and without the trace bound found that
  unbounded lambda=0.90 on the legacy position basis reaches covariance trace
  9e22 and loses PSD to reconstruction rounding; the bounded arm stays valid at
  the same holdout error.

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
- **Evidence** is generated by [`diagnosis.md`](diagnosis.md) into the ignored
  `docs/evidence/diagnosis/` directory. It is development data only; no
  confirmation stream is touched.

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
- **Note (2026-09-22)** Since `AAA-125`, formal confirmation does not run in
  Actions, so the 90-day retention above covers only development and
  high-replication dispatches. Confirmation raw bytes have no CI copy
  (`AAA-178`).

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

### AAA-110 — no branch protection or ruleset on `main`
- **Source** Astra · **Severity** medium · **Status** repaired
- **Reproduction** Independently confirmed rather than taken on report:
  `GET /repos/Cinqic/AAA/rulesets` returned `[]` and
  `GET /repos/Cinqic/AAA/branches/main/protection` returned
  `404 Branch not protected`. `delete_branch_on_merge` was `false` and
  Dependabot security updates were `disabled`. Secret scanning and push
  protection were already enabled.
- **Consequence** Nothing required the CI that this repair made meaningful. A
  direct push to `main` could bypass every check.
- **Repair** Applied through the API and read back to confirm: a pull request
  is required before merging; the six CI jobs are required status checks with
  `strict` up-to-date enforcement; force pushes and deletion of `main` are
  blocked; conversation resolution is required; head branches are deleted after
  merge; Dependabot alerts and security updates are enabled; and
  `.github/dependabot.yml` schedules weekly `github-actions` and `pip` updates
  so the pinned action SHAs cannot go stale unnoticed.
- **Deliberately not applied** `enforce_admins` is left off so the maintainer
  keeps a recovery path if a required check is misconfigured or a runner is
  unavailable, and the read-only default workflow permission is an
  account-level setting — every workflow here already declares
  `permissions: contents: read` explicitly, which does not depend on an
  inherited default. Both are stated with their exact commands in
  [`../SECURITY.md`](../SECURITY.md) rather than quietly omitted.
- **Verification** `gh api repos/Cinqic/AAA/branches/main/protection` and
  `gh api repos/Cinqic/AAA --jq '.delete_branch_on_merge, .security_and_analysis'`.
- **Note (2026-09-22)** The `pip` entry described above ignored only patch
  updates, so it could still open minor, major and security pull requests
  against `requirements-lock.txt`, a fingerprinted file. It now ignores every
  `pip` update; see `AAA-175`. The repository settings listed above were read
  back unchanged on 2026-09-22.

### AAA-112 — confirmation status edits could erase prior-use eligibility
- **Source** independent Sol reproduction · **Severity** blocking · **Status** repaired
- **Reproduction before repair** Changing only a retired batch's serialized
  status to `planned`, while retaining its prior run and failure, allowed a
  fresh claim. Conversely, changing a fresh status to `consumed` established
  reproduction eligibility without execution evidence.
- **Repair** Loading, saving and claiming validate status against durable
  consumption and outcome evidence. Planned batches require both an empty
  consumption list and null outcome; completed states require nonempty unique
  run identities and their corresponding outcome. Invalid serialized types,
  empty identities and contradictory in-memory edits are refused.
- **Regression** `tests/test_batch_integrity.py` reproduces serialized tampering
  for failed and passed batches, fabricated prior execution, and in-memory
  tampering. The historical registry loads unchanged. Tests failed before the
  repair; all 381 tests passed afterward (68.775 seconds).
- **Boundary** This prevents contradictory records, not malicious rewriting of
  every field and Git history. Interrupted-run reservation and concurrent use
  require separate review; this change does not claim to solve them.

### AAA-113 — multiplicity is scoped to a specification hash
- **Source** self · **Status** open, deliberate
- Repeated confirmation attempts against the same frozen specification enlarge
  the Holm family. A candidate change changes the specification hash and starts
  a new family, so the counter does not by itself bound how many *systems* may
  be tried. That scoping is deliberate — a different system under a different
  frozen protocol is a different hypothesis — and the rule was declared before
  the round-1 failure and not adjusted after it.
- What actually bounds system-shopping is the surrounding discipline: a failed
  batch is retired permanently, a candidate change requires a fresh freeze and
  fresh batches, and every attempt stays on the record. A reviewer should count
  attempts in `benchmarks/confirmation_batches.json` directly rather than read
  the family size as the whole story.

### AAA-111 — parallel execution was profiled and not adopted
- **Source** Astra, Opus · **Status** superseded
- A full confirmation attempt completes in minutes of CPU time. Introducing
  workers would add an RNG-allocation and BLAS-oversubscription risk surface to
  save time the project does not need. Seeds already map to trial identity
  rather than scheduling order, so parallelism remains available without a
  reproducibility change if the workload ever grows. Not adopted, and the
  reason is recorded rather than the option being silently dropped.


### AAA-121 — frozen source identity was recorded but not enforced
- **Source** independent Sol reproduction · **Severity** blocking · **Status** repaired
- **Reproduction** A clean committed implementation edit passed `check_manifest`
  while the specification, checkpoint hashes, training seeds and lock were
  unchanged. New regression tests also exposed unguarded test, tool, seed-fixture
  and batch-identity edits.
- **Repair** Manifest schema v2 enforces a canonical scientific fingerprint over
  file paths, executable bits and contents. It covers all tracked and nonignored
  untracked files, including implementation, tests, tools, workflows, documents,
  specification and golden seeds. Missing files and symlinks are refused.
  The runner and freeze CLI bind the root to the loaded AAA implementation;
  formal confirmation also requires a real Git commit.
- **Precisely excluded** `results/`, `benchmarks/freeze_manifest.json`,
  `docs/final_audit.md`, `docs/handoff_sol.md`, and `docs/sol_review.md` contain
  confirmation evidence or generated final-review bookkeeping.
  In `benchmarks/confirmation_batches.json`, only each batch's `status`,
  `consumed_by`, `outcome`, `claimed_by`, and `claim_started_at` fields are
  omitted; declarations, roles, specification identities and notes remain
  covered. No implementation may be put in excluded paths. Git commit and
  full-tree provenance are still recorded separately.
- **Regression** `tests/test_source_freeze.py` verifies clean committed source,
  test, tool, seed and declaration drift is rejected, while a legitimate A-to-B
  evidence/status commit is accepted. Existing manifest tests cover specification,
  checkpoint, training-stream and undeclared-batch drift. Old v1 manifests are
  preserved as historical evidence and cannot authorize new confirmation.

### AAA-122 — non-finite recomputation values disappeared behind tolerance arithmetic
- **Source** independent Sol reproduction · **Severity** blocking · **Status** repaired
- **Reproduction before repair** `compare_results({"x": NaN}, {"x": 1.0},
  tolerance=1e-10)` returned `equivalent: true`; matching infinities and malformed
  tolerances were also accepted or failed with incidental type errors.
- **Repair** Comparison now requires a finite non-negative numeric tolerance and
  records every non-finite value as a scientific disagreement, including equal
  infinities. The CLI compares the full recomputed result and gate trees rather
  than only gate status labels, so altered numbers that preserve a status fail.
- **Regression** `CompareResultsTests` covers NaN on either side, equal NaN and
  infinities, booleans, malformed tolerances and ordinary finite tolerance.
  `RecomputeCommandTests` exercises the CLI acceptance path with an injected NaN.

### AAA-123 — replica records carried replica 0 training provenance
- **Source** independent Sol reproduction · **Severity** high · **Status** repaired
- **Reproduction before repair** A two-replica development run recorded replica
  0's training seed tuple in every raw trajectory although checkpoint hashes and
  summary training seeds differed by replica.
- **Repair** Family runners now receive the complete replica-to-seed mapping and
  attach the lineage selected by each episode plan's replica id across motion,
  always-online, changed-law prefix and both changed-law branches.
- **Regression** A two-replica end-to-end benchmark reloads every compressed raw
  record and matches its lineage to the corresponding summary checkpoint entry.

### AAA-124 — historical v2.1 checksum manifests name absent evidence
- **Source** independent Sol reproduction · **Severity** blocking for historical
  clean-clone recomputation · **Status** documented; new evidence required
- **Reproduction** `verify_checksums` on each of the four committed v2.1 attempt
  directories reports thousands of missing paths. The repositories retain the
  compact summaries and checkpoints but omit the raw, metric, registry, plot and
  verification bytes that their manifests cover.
- **Disposition** The historical summaries remain identifiable negative and
  positive evidence, but are not called byte-verifiable archives or fresh
  clean-clone recomputations. New confirmation is required after the scientific
  repairs. The evidence policy now distinguishes original-byte integrity from
  semantic reproduction and retains the existing `AAA-077` archive limitation.

### AAA-125 — ephemeral remote checkouts could claim one batch concurrently
- **Source** independent Sol review · **Severity** blocking · **Status** repaired
- **Finding** A local sidecar lock can serialize processes in one checkout, but
  two manually dispatched Actions jobs have separate filesystems and cannot
  persist a shared registry claim. Both could therefore start the same planned
  formal stream.
- **Repair** Formal reservation now persists `running`, claimant and claim time
  under a checkout lock before the first observation. The manual remote workflow
  accepts only development and high-replication roles and refuses any batch id;
  formal A/B runs are made in the canonical maintained checkout where the claim
  and resulting evidence can be committed between attempts.

### AAA-126 — wheel and runtime reported different versions
- **Source** independent Sol package audit · **Severity** medium · **Status** repaired
- **Reproduction** A freshly built wheel was named and described as version
  `0.2.0`, matching `pyproject.toml` and `CITATION.cff`, while importing it
  reported `aaa.__version__ == "0.1.0"`.
- **Repair** The runtime constant is `0.2.0`. The wheel is rebuilt, its payload
  inspected, installed after the exact dependency lock with `--no-deps`, and
  exercised outside the source checkout before package evidence is accepted.

### AAA-127 — a durable claim made clean confirmation look dirty
- **Source** fresh Sol confirmation A-0003 · **Severity** blocking · **Status** repaired
- **Reproduction** The runner correctly verified a clean tree, then persisted
  its `running` batch claim. Metadata was collected afterward, saw that expected
  registry edit, and failed `source_tree_clean_when_confirming`. All other 13
  required gates passed; the batch exited 1 and remains retired evidence.
- **Repair** Git identity and cleanliness are captured once, before reservation,
  and that exact pre-observation snapshot is retained in metadata. Reservation
  remains durable before the first observation. The unobserved paired B stream
  is cancelled explicitly, without a fabricated consumer or outcome.

### AAA-128 — observation-noise phase needed a separate evidence boundary
- **Source** observation-noise v1 design audit · **Severity** high · **Status** implemented; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Finding** Reusing the v2.1 record, metric, or gate namespace would make a
  noisy rerun look like a deterministic reference result and would permit
  clean latent targets to enter a predictor through the existing environment
  interface.
- **Repair** The phase has its own protocol hash, schedule generator, cached
  observation wrapper, primitive record schema, output root, lifecycle files,
  independent arithmetic verifier, direct stratum identities, and source-freeze
  manifest command. The predictor update target is the newly revealed noisy
  observation; latent truth is evaluator-side only. v2.1's specification and
  historical evidence are unchanged.
- **Regression** Focused tests cover schedule determinism, distribution
  validation, unbounded sensor values, idempotent observation calls, causal
  update order, strict boolean/integer handling, stale cached errors, unknown
  fields, and the independent recomputation of the all-cell development smoke.
- **Remaining limitation** The fixed ten-lineage A/B confirmation plan and
  durable external archive/retrieval workflow have not been executed or
  claimed. Sol's current AI review is separate from human review.

The smoke implementation also executes the four predeclared factorial control
cells (unchanged/changed dynamics crossed with unchanged/shifted noise), both
shift directions, and aligned/staggered timing. These controls remain
engineering evidence only until the frozen ten-lineage confirmation plan is
run. The phase verifier now binds every record to its saved schedule, checks
the raw sensor equation and units, and rejects rehashed cached-error
corruption, missing scientific gates, and strict-type substitutions.

---

## Observation-noise v1/v1.1 completion audit

### AAA-129 — confirmation freeze used a self-referential commit equality
- **Source** fresh observation-noise completion audit · **Severity** critical · **Status** repaired
- **Reproduction** The prior confirmation admission required the freeze's
  recorded commit and tree hash to equal the current checkout. Generating and
  committing `observation_noise_freeze.json` necessarily changed the commit,
  so a valid freeze could not admit its own clean checkout.
- **Repair** `aaa.noise.scientific_identity` enumerates the Git scientific file
  map, hashes normalized content plus executable-bit metadata, excludes only
  explicit generated result/freeze/review files, normalizes only mutable batch
  lifecycle fields, and fails closed on nonignored untracked files, unsafe
  paths and symlinks. Git metadata remains provenance. Confirmation compares
  the complete fingerprint map instead of commit equality.
- **Regression** `tests/test_observation_noise_identity.py` edits scientific
  code, candidate, analysis, verifier and batch declarations; tests mutable
  registry transitions, generated output, exact freeze commits, untracked
  inputs and symlinks.

### AAA-130 — observation-noise candidate ledger had no selected first-class candidate
- **Source** fresh observation-noise completion audit · **Severity** critical · **Status** repaired
- **Reproduction** The design ledger was empty and the runner had only named
  incumbent/comparator predictors; no immutable selected candidate ID,
  configuration hash, development disposition or confirmation resolver existed.
- **Repair** Four catalogued identities now carry parent, mechanism, exact
  parameters, trainable/fixed state, training condition, mechanism count,
  configuration hash, attempt IDs and outcome reason. The committed bounded
  selection plan evaluated the unchanged incumbent plus three causal
  innovation-clipping variants. All refinements were rejected; the unchanged
  incumbent is selected as a valid no-refinement control. Confirmation reads
  only the committed freeze and selected ledger entry.
- **Evidence** `benchmarks/observation_noise_candidate_ledger.json` and
  `docs/evidence/observation_noise_development_selection.json`.

### AAA-131 — observation-noise scientific gates were permanent placeholders
- **Source** fresh observation-noise completion audit · **Severity** critical · **Status** partially repaired; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Reproduction** The runner emitted `INSUFFICIENT_EVIDENCE` for scientific
  endpoints on every role and had no machine-readable primary bound evaluator.
- **Repair** Exact acceptance formulas, one-sided 95% bound direction,
  thresholds, practical margin, first-surprise adaptation definition and
  no-refinement promotion rule are now protocol fields. The joint evaluator
  reconstructs the final claims from primitive A+B records. Development smoke
  remains `INSUFFICIENT_EVIDENCE` by design; no confirmation result is claimed.
- **Remaining boundary** Full ten-lineage A/B evidence, durable archive and
  independent review have not been executed.

### AAA-132 — separate A/B analyses did not implement one joint multiplicity family
- **Source** fresh observation-noise completion audit · **Severity** high · **Status** repaired in code; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Reproduction** Existing per-attempt paired statistics applied their own
  local comparison output and could not enumerate the combined A+B endpoint,
  cell and batch family.
- **Repair** `observation-noise-confirmation-evaluate <A> <B>` independently
  verifies both archives, checks compatibility and shared checkpoints,
  reconstructs all primary claims, records a deterministic Holm order/family
  size, and recomputes the outcome without stored conclusions.
- **Regression** The joint evaluator's hierarchy, claim-family and archive
  compatibility paths are covered by unit-level fixtures and fail-closed
  archive admission tests.

### AAA-133 — the observation-noise verifier did not bind candidate identity
- **Source** fresh observation-noise completion audit · **Severity** high · **Status** repaired
- **Reproduction** A v3 summary could be paired with primitive records while
  omitting or changing the selected candidate identity and configuration hash.
- **Repair** The independent verifier now requires matching canonical candidate
  ID/configuration hashes in metadata, run manifest and summary; confirmation
  metadata must also contain the scientific fingerprint. Cached outcomes remain
  insufficient without primitive arithmetic.

### AAA-134 — no approved durable archive locator exists for observation-noise A/B
- **Source** fresh observation-noise completion audit · **Severity** publication-grade evidence limitation · **Status** retained limitation; not required for current internal engineering phase
- **Reproduction** The protocol requires a full archive and immutable locator,
  but this workspace has no approved durable storage mechanism for the new
  observation-noise records. Temporary development roots cannot be presented
  as durable evidence.
- **Disposition** No locator is fabricated. The dedicated HDD provides verified
  local working capacity but not immutability, off-site retention, or an
  independent failure domain. External archival is reclassified as a
  publication-grade evidence requirement rather than a merge gate for this
  internal engineering phase. Any future claim of durable independent
  retrieval still requires a genuine external mechanism established before
  observation.

### AAA-135 — verifier accepted an archive with no checksum manifest
- **Source** independent Sol reproduction · **Severity** critical · **Status** repaired; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Reproduction** A complete synthetic attempt with `checksums.json` omitted
  returned `PASS`. Structural and arithmetic checks therefore did not establish
  complete byte coverage.
- **Evidence** `docs/evidence/sol_observation_noise_pre_repair_probes.json`.
- **Repair** Archive verification now requires `checksums.json`, requires exact
  coverage of every retained file other than the checksum manifest itself, and
  rejects missing, extra, or mismatched entries.

### AAA-136 — stored NaN values bypassed verifier comparisons
- **Source** independent Sol reproduction · **Severity** critical · **Status** repaired; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Reproduction** Replacing a stored summary MAE with JSON `NaN` left
  `abs(stored - recomputed) > tolerance` false, and the verifier returned
  `PASS`.
- **Evidence** `docs/evidence/sol_observation_noise_pre_repair_probes.json`.
- **Repair** All stored scalar comparisons pass through a strict finite-number
  check before tolerance arithmetic; a regression retains the original NaN
  bypass as a negative fixture.

### AAA-137 — adaptation implementation used the wrong estimand
- **Source** independent Sol reproduction · **Severity** critical · **Status** repaired; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Reproduction** For heterogeneous frozen denominators 1 and 3, with online
  values 1 and 1.5, the implemented mean of per-identity ratios is 0.25 while
  the declared ratio of means is 0.375.
- **Evidence** `docs/evidence/sol_observation_noise_pre_repair_probes.json`.
- **Repair** Adaptation now bootstraps the declared ratio of aggregate frozen
  and online means at every hierarchy draw. A heterogeneous-denominator
  regression distinguishes it from the rejected mean-of-ratios implementation.

### AAA-138 — quick smoke was recorded as complete candidate selection
- **Source** independent Sol reproduction · **Severity** critical · **Status** repaired and verified in full development selection
- **Reproduction** The frozen plan requires 2 x 2 x 1 hierarchy counts, but the
  committed selection evidence records `quick: true` and executed 1 x 1 x 1.
  Refinement eligibility also omits retention, incumbent practical gain,
  complete-cell coverage, and stability, and treats legitimate zero values as
  missing through truthiness fallbacks.
- **Evidence** `docs/evidence/sol_observation_noise_pre_repair_probes.json`.
- **Repair** Quick mode is explicitly `development_smoke`, cannot select a
  candidate or mutate the candidate ledger, and records only a provisional
  point-estimate ordering. Full selection enforces the frozen 2 x 2 x 1 sample,
  complete primary coverage, stability, utility, baseline, retention,
  adaptation, practical incumbent gain, and adjusted positive evidence.
- **Verification** Four fresh 2 x 2 x 1 archives each retained 235,776 scored
  records across 1,024 trials plus 3,552 training records and independently
  verified `PASS`. No refinement met the conjunctive eligibility rule; the
  incumbent was retained explicitly as the no-refinement control.

### AAA-139 — formal joint evaluation admitted mutable analysis identity
- **Source** independent Sol reproduction · **Severity** critical · **Status** repaired; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Reproduction** The public formal evaluator accepts a caller-selected draw
  count and hard-codes the first A/B batch IDs rather than resolving the frozen
  registry identities.
- **Evidence** `docs/evidence/sol_observation_noise_pre_repair_probes.json`.
- **Repair** The formal evaluator uses the protocol-declared 4,000 draws with no
  public override and resolves the one A/B pair from the exact committed
  confirmation freeze and registry rather than hard-coded IDs.

### AAA-140 — batch consumption was local to an output directory
- **Source** independent Sol reproduction · **Severity** critical · **Status** repaired; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Reproduction** Confirmation admission reads `planned` from the repository
  registry, but the visible `consumed` transition is written only to the
  attempt-local lifecycle. Two output roots or checkouts can therefore spend
  the same planned batch.
- **Evidence** `docs/evidence/sol_observation_noise_pre_repair_probes.json`.
- **Repair** Confirmation claims an append-only remote Git ref before creating
  an attempt. Normal atomic ref creation permits one owner across separate
  clones; only that exact owner may resume. No force update or deletion path is
  provided. A two-clone regression proves that the second claimant is rejected.

### AAA-141 — full plan cannot run through whole-collection materialization
- **Source** independent Sol measured pilot · **Severity** critical · **Status** repaired in code; formal scale unexecuted
- **Reproduction** A valid 58,944-record quick run took 103.34 seconds,
  peaked at 2.67 GiB RSS, and retained about 752 MiB because primitive records
  were simultaneously materialized in memory and duplicated as trial shards,
  uncompressed combined JSONL, and gzip. The declared formal plan contains
  52,945,920 scored records; linear sizing projects about 332 GB of uncompressed
  combined JSONL, 34.6 GB gzip, and about 706 GB with the current duplicate
  layout for one batch. Exact plan enumeration yields 153,600 planned trials
  and 59,043,840 scored records; the corrected linear projection is about
  370.5 GB uncompressed, 38.5 GB gzip, and 788 GB with current duplication.
- **Evidence** `docs/evidence/observation_noise_scale_pilot.json` records the
  exact command, measured manifest values, `/usr/bin/time -v` results, formulas,
  and the approximately 65 GB available-disk boundary.
- **Required repair** Stream production, verification, and sufficient-statistic
  aggregation; avoid retaining duplicate primitive copies; preserve every
  declared record, cell, formal draw, and scientific estimand.
- **Repair evidence** A fresh 58,944-record run with deterministic per-trial
  gzip shards peaked at 155,054,080 bytes RSS and occupied 47,281,979 bytes.
  Independent verification passed. Its timing-neutral primitive digest,
  metrics, and coverage exactly match the pre-shard streaming run. Formal
  runtime, 38.5 GB projected compressed size, and durable transfer remain
  unverified projections rather than completed evidence.

### AAA-142 — frozen adaptation prose and machine operator disagreed at 10%
- **Source** independent Sol protocol audit · **Severity** critical · **Status** repaired by protocol supersession
- **Reproduction** v1 prose admitted an adjusted lower bound equal to 10% by
  saying "at least", while `machine_criteria` required the bound to be strictly
  greater than 0.10. A boundary result therefore had two incompatible verdicts.
- **Repair** v1 is retained in the registry by exact protocol hash and source
  commit as superseded without observed confirmation data. v1.1 makes the prose
  and `>` operator identical and allocates fresh unobserved A/B batch IDs.

### AAA-143 — zero-noise reference executed current code instead of a pinned checkout
- **Source** independent Sol reference-isolation audit · **Severity** critical · **Status** repaired; confirmation not executed (v1.1 A/B retired unobserved on 2026-09-19; reworded 2026-09-22 from "confirmation pending", `AAA-177`)
- **Reproduction** The protocol recorded v2.1 file identities but no commit, and
  `_zero_noise_fixture` imported both sides from the current checkout. Shared
  drift could therefore pass as reference preservation.
- **Repair** v1.1 pins v2.1 commit
  `25b6c32c9040d0f934314a2139993d12763afc99`. The fixture exports that commit
  into an isolated temporary checkout and compares histories, trajectories,
  bounce/change behavior, forecasts, update decisions, final state and counters.
  The isolated fixture passes both recorded v2.1 content identities.

### AAA-144 — formal archive remains without an approved durable destination
- **Source** independent Sol archive audit · **Severity** publication-grade evidence limitation · **Status** retained limitation; not required for current internal engineering phase
- **Reproduction** The repaired single-copy format projects one formal batch at
  about 38.5 GB of compressed primitive records plus schedules and metadata.
  Direct 2026-09-19 inspection found 425,141,489,664 available bytes on the
  dedicated ext4 HDD, so local capacity is no longer the blocker. The
  repository still has no Git LFS configuration, release asset, or approved
  object-store credentials. A local run therefore cannot be truthfully called
  durably archived or independently retrieved.
- **Disposition** No locator is fabricated and A/B remain unexecuted. External
  immutable archival is not required to merge validated infrastructure in the
  current internal phase. It remains required before any publication-grade
  durability or independent-retrieval claim. Formal execution was separately
  deferred because the replication budget lacks a quantitative precision
  justification; both declared v1.1 batches were retired unobserved.

### AAA-145 — scalar bootstrap traversal made the frozen draw count impractical
- **Source** independent Sol full-selection timing probe · **Severity** critical · **Status** repaired and verified
- **Reproduction** Candidate one completed its primitive archive, then remained
  at 100% CPU inside `_draw_hierarchical` for more than 11 minutes without
  finalizing a summary. The partial unobserved development archive is retained
  under `/tmp/aaa-noise-v1_1-full-selection-20260914`; it did not update the
  candidate ledger or produce a selection conclusion.
- **Repair** Balanced hierarchy draws are vectorized across the declared 4,000
  samples while retaining lineage, episode-within-lineage, and
  sensor-realization-within-episode resampling. Ragged inputs retain the scalar
  fallback. The repaired path computed 1,888 hierarchical cells in 18.71 seconds
  and 160 paired cells in 15.21 seconds with the draw count unchanged.
- **Regression** A deterministic 100,000-draw fixture verifies repeatability and
  agreement between the vectorized sample mean and the exact balanced
  hierarchical point estimand.
- **Verification** Four full candidate runs completed concurrently in 470.75 to
  476.65 seconds each, including their 4,000-draw statistics. Selector-owned
  independent recomputation completed in 509.82 seconds and wrote the full
  development evidence and ledger decision.

### AAA-146 — schedule index admitted paths outside the archive
- **Source** independent Sol final self-review · **Severity** high · **Status** repaired and verified
- **Reproduction** The verifier joined each schedule-index `path` to the archive
  root and checked only `is_file()` plus its digest. A rehashed `../` path or a
  symlink could therefore supply schedule bytes from outside the retained
  attempt while still passing the local checksum comparison.
- **Repair** Every flat record, shard-index, shard, and schedule path now passes
  a shared containment check that rejects absolute paths, parent traversal,
  symlinks in any path component, missing members, and ambiguous simultaneous
  flat/sharded record layouts before bytes are accepted.
- **Regression** `test_schedule_index_cannot_escape_archive_or_follow_symlink`
  proves both a parent traversal and an in-archive symlink fail independent
  verification even when the checksum manifest requirement is bypassed for the
  mutation probe.

### AAA-147 — quick selection could overwrite canonical full evidence
- **Source** independent Sol documentation self-review · **Severity** high · **Status** repaired and verified
- **Reproduction** The CLI default and documented quick-selection commands
  targeted `docs/evidence/observation_noise_development_selection.json`. Quick
  mode correctly avoided ledger mutation but still wrote its smoke payload over
  the retained full 2 x 2 x 1 selection evidence.
- **Repair** Quick output now defaults under ignored `runs/`, active commands use
  that location, and the selection API refuses a quick write to the canonical
  full-evidence path even when a caller requests it explicitly.
- **Regression** `test_quick_selection_cannot_replace_canonical_full_evidence`
  proves smoke output is rejected at the canonical path while a full selection
  remains permitted there.

### AAA-148 — installed noise smoke required unavailable Git history
- **Source** independent Sol fresh-wheel verification · **Severity** high · **Status** repaired and verified
- **Reproduction** A clean wheel installed outside the checkout loaded both
  packaged specifications, then failed its advertised development noise smoke
  because the isolated v2.1 fixture unconditionally ran `git archive` against
  `site-packages`, which is not a Git repository. No attempt summary was
  finalized, so recomputation also failed.
- **Repair** A maintained checkout still requires the exact pinned reference
  commit and fails loudly if it is missing. An installed-package development
  run now records the checkout-only isolated replay as `NOT_VERIFIED`, including
  the reference gate and reproduction field, while continuing the package and
  mechanics smoke. It never substitutes an in-process comparison or reports a
  false reference `PASS`.
- **Regression** `test_installed_package_reports_checkout_only_reference_as_not_verified`
  exercises the no-checkout boundary. Fresh-wheel execution and recomputation
  are repeated separately in the final validation record.

### AAA-149 — hosted CI omitted the pinned reference commit
- **Source** final-head GitHub CPU CI runs `34922456399` and `34922458676` · **Severity** high · **Status** repaired; hosted CI verified (2026-09-22, `AAA-177`; was "replacement CI pending")
- **Reproduction** The Python matrix failed across versions at
  `test_zero_noise_reference_runs_from_the_pinned_v21_commit`. Actions checked
  out only the PR head, so `git cat-file -e 25b6c32...^{commit}` failed and the
  maintained-checkout fixture correctly refused to claim a pinned replay.
- **Repair** Every CPU workflow checkout now uses `fetch-depth: 0`. This makes
  the exact protocol-pinned historical commit available to the isolated
  `git archive` fixture in the locked, version-matrix and fresh-wheel jobs.
- **Regression** The fail-closed unit test remains unchanged. Replacement
  final-head GitHub checks must pass all required Python versions before this
  repair is considered hosted-CI verified.
- **Hosted verification (2026-09-22)** Satisfied. Exact PR #11 head
  `e463773` passed all six jobs in push run `35451372899` and pull-request run
  `35451375484`, and integration commit `ff7199d` passed run `35451862857`
  (recorded in [`handoff_sol.md`](handoff_sol.md) and read back through the
  API). On `main` at `efa1fdb`, run `35775776610` passed all six jobs. Its
  locked job reports
  `test_zero_noise_reference_runs_from_the_pinned_v21_commit ... ok`, and all
  701 tests passed on Python 3.10, 3.11, 3.12 and 3.13.

### AAA-150 — schedule filenames were not portable to GitHub artifact storage
- **Source** final-head GitHub CPU CI runs `34922822759` and `34922825001` · **Severity** high · **Status** repaired; hosted artifact upload verified (2026-09-22, `AAA-177`; was "repaired locally; final hosted artifact upload pending")
- **Reproduction** Both the locked and fresh-install jobs completed their
  environment, static, test, benchmark, recomputation, and exit-code work, then
  `actions/upload-artifact` rejected schedule paths such as
  `development:clean_trained:bouncing:correlated:...json` because `:` is not a
  portable artifact filename character.
- **Repair** Schedule files now use `trial-<full lowercase SHA-256>.json`, where
  the digest is computed over the canonical UTF-8 trial identity. The complete
  authoritative identity remains in the schedule payload index and primitive
  records; no verifier reconstructs it from the filename.
- **Regression** The filename regression covers 3,459 representative and
  adversarial identities across every role, family, training condition,
  channel, scale, lineage, episode, realization, and shift form. It checks
  determinism, uniqueness, fixed length, and exclusion of all forbidden path
  characters. Final hosted CI must also pass the real artifact-upload step.
- **Hosted verification (2026-09-22)** Satisfied. Runs `35451372899`,
  `35451375484` and `35451862857` each retain unexpired `ci-locked-evidence`
  and `ci-fresh-install-evidence` artifacts. On `main` at `efa1fdb`, run
  `35775776610` logged both uploads as successful: 42,132,387 bytes including
  the observation-noise smoke archive, and 40,451,944 bytes.

### AAA-151 — formal execution had no output-filesystem capacity preflight
- **Source** 2026-09-19 storage audit · **Severity** high · **Status** repaired and locally verified
- **Reproduction** The runner could reserve a formal batch before establishing
  that the explicitly selected output filesystem had enough space for both
  retained A/B evidence and working headroom. On this host `/` had only
  24,722,862,080 available bytes while the dedicated HDD had
  425,141,489,664.
- **Repair** Before any remote reservation, a formal run resolves the target
  filesystem and requires at least 100,000,000,000 available bytes. It records
  target, existing probe path, device ID, total bytes, available bytes, and the
  threshold in its manifest and metadata. Development runs remain portable and
  do not require the machine-specific HDD path.
- **Regression** Focused tests inject insufficient and sufficient filesystem
  states and verify refusal and provenance without hard-coding a deployment
  path into AAA source.

### AAA-152 — a new research phase necessarily changes both repository-wide source fingerprints
- **Source** AAA-1K phase construction, 2026-09-19 · **Severity** medium · **Status** open; accepted and documented
- **Reproduction** `aaa/benchmark/source_identity.py` and
  `aaa/noise/scientific_identity.py` both hash every tracked, non-generated
  file. Adding `research/aaa_1k/` therefore changes both digests, so the
  working tree no longer reproduces the fingerprints recorded in
  `benchmarks/freeze_manifest.json` and
  `benchmarks/observation_noise_source_freeze.json`.
- **Rejected repair** Adding `research/` to either module's excluded prefixes.
  That is a convenience exclusion whose only purpose is to make an old hash
  keep matching, and `AAA-121` already established what incomplete source
  identity costs.
- **Accepted disposition** AAA-1K is a separate versioned phase with its own
  non-self-referential fingerprint over its own source, the shared `aaa/`
  modules it executes, its tests, its protocol documents and the dependency
  lock (`python -m research.aaa_1k fingerprint`). Historical evidence stays
  reproducible from its historical checkout: the annotated tags
  `aaa-pre-next-phase-2026-09-19` and
  `aaa-pre-next-phase-closure-2026-09-19` mark those trees exactly. Every
  v2.1 confirmation batch is already spent or retired, so no pending
  confirmation is blocked. One hash is not asked to describe two worlds.
- **Regression** `tests/test_aaa_1k.py::IdentityTests` asserts the phase
  fingerprint covers the phase source and the shared modules, and excludes the
  documents this phase generates so the identity cannot become
  self-referential. No existing fingerprint test was weakened.

### AAA-153 — the AAA-1K online/frozen comparison does not isolate adaptation
- **Source** AAA-1K adversarial self-review, 2026-09-19 · **Severity** medium · **Status** repaired
- **Reproduction** `python -m research.aaa_1k adversarial-probes`. Branching an
  online/frozen pair at step 60, where nothing happens, reproduces 95% of the
  advantage measured at the declared change point at step 100
  (+7.65e-04 against +8.07e-04).
- **Root cause** The model improves throughout every stream, so "online beat
  its frozen twin after a change" is equally consistent with "continued
  learning helps everywhere". The single-branch design cannot separate them.
- **Repair** A new `paired_change_v1` family emits two streams that are
  bit-identical until a declared step, after which one changes speed and the
  other does not. One model is driven through the shared prefix, so both
  variants branch from the same model state — asserted equal by complete
  state hash, not assumed. The estimator is
  `advantage(changed) - advantage(control)`, so the ordinary benefit of
  continuing to learn cancels. Measured on fresh evaluation identities in
  round 2: the adaptation component is `+9.50e-04` with a 95% interval of
  `[+5.63e-04, +1.33e-03]`, about 49% under the canonical
  `adaptation / (adaptation + continued learning)` definition. Earlier 65%
  wording was an arithmetic/documentation error. The
  round-1 claim was therefore *directionally* right and *quantitatively*
  overstated by roughly a third.
- **Regression** `Round2FailureInjectionTests` substitutes an unpaired control
  stream and proves the identical-trunk assertion fires. The original probe
  remains committed at `docs/evidence/aaa_1k_adversarial_probes.json`, and
  round 1 is retained at
  `docs/evidence/aaa_1k_evaluation_round1_superseded.json`.

### AAA-154 — the AAA-1K A/B/A benchmark does not measure retention
- **Source** AAA-1K adversarial self-review, 2026-09-19 · **Severity** medium · **Status** repaired
- **Reproduction** In `docs/evidence/aaa_1k_evaluation_round1_superseded.json`, every learning arm
  has a *lower* error in the final A segment than in the first, which reads as
  "no catastrophic forgetting" but is confounded: by A2 the model has had three
  times as much total experience. The non-learning control
  (`constant_motion_reflected`) is flat across A1 and A2, as it must be.
- **Repair** A fixed bank of eight held-out regime-A episodes, generated once
  from the benchmark-generation namespace and never trained on, is evaluated by
  a frozen clone with its hidden state reset at the end of each of A1, B and
  A2. Because the same questions are asked at every checkpoint, accumulated
  experience cannot flatter the later ones. Measured in round 2: probe error
  after B minus after A1 is `-3.74e-04`, interval `[-7.99e-04, -9.52e-07]` —
  regime-A ability *improved* while the model trained on regime B, so there is
  no forgetting to target and no replay is warranted.
- **Regression** `Round2FailureInjectionTests` hands the probe an agent whose
  clones still learn and proves the read-only assertion fires. Round 1's
  confounded comparison is retained in the superseded evidence file.

### AAA-155 — AAA-1K intervals resampled streams but not initializations
- **Source** AAA-1K adversarial self-review, 2026-09-19 · **Severity** medium · **Status** repaired
- **Reproduction** Round 1 gave every arm one `model_init` seed, deliberately,
  so that an ablation differed from the primary in exactly one mechanism. The
  bootstrap then resampled streams only. Initialization variance was therefore
  entirely unsampled, and a five-seed probe found effect magnitudes varying by
  up to a factor of two while the intervals claimed a precision that did not
  account for it.
- **Repair** Round 2 runs the whole design from five initializations and uses a
  crossed bootstrap that resamples initializations and streams independently,
  recomputing the mean over the selected cells. Per-initialization differences
  are reported alongside the mean, and the record states whether every
  initialization agreed on the sign. Achieved precision is now reported as the
  interval half-width against the effect that was actually measured, replacing
  a target sized from a pilot estimate of an effect nobody had seen.
- **Regression** `CrossedBootstrapTests` builds a design whose initializations
  genuinely disagree and asserts the crossed interval is more than three times
  wider than the stream-only interval on the same data, so the old estimator's
  optimism is a live check rather than a remembered argument.

### AAA-156 — one architecture's hyperparameters were imposed on the others
- **Source** AAA-1K round-2 construction, 2026-09-19 · **Severity** high · **Status** repaired
- **Reproduction** The gradient-clip threshold was *asserted* at 1.0 rather
  than selected. A development probe found it activating on 24% of updates and
  costing 29% of development error against the best stable threshold — at that
  rate it is a hyperparameter deciding what is learned, not a guard. Worse,
  when the threshold was first selected on the gated model and then applied to
  every arm, the stateless control destabilized on `coarse_speed_v1`: its mean
  error reached `1.6e-01` against the gated model's `2.2e-03`, and the crossed
  comparison reported a hidden-state advantage of `+1.28e-02`, thirty times the
  true effect. That would have been published as "hidden state helps".
- **Repair** Stage 3 of the development selection now chooses the clip
  threshold by the same rules as every other hyperparameter, preferring the
  most conservative threshold within the practical margin of the best. Stage 3a
  runs the whole selection — learning rate *and* clip — **separately for each
  architecture**, because imposing one architecture's hyperparameters on
  another turns a comparison into a handicap. Ablations of the gated model
  still share its hyperparameters exactly, since an ablation is the same
  architecture with one mechanism removed. With fair hyperparameters the
  hidden-state effect is `+2.49e-04` and the gating result moves from
  `NEGATIVE` to `INCONCLUSIVE`.
- **Regression** `PerArchitectureSelectionTests` asserts that each control
  receives its own rule-selected learning rate and clip, that the primary and
  its three ablations share one configuration and bitwise-identical initial
  parameters, and that each ablation differs from the primary in exactly one
  declared mechanism.

### AAA-157 — AAA-1K checkpoint restoration accepted malformed persistent state
- **Source** independent Sol review of PR #16, 2026-09-20 · **Severity** high · **Status** repaired and verified
- **Reproduction** The reviewed head accepted wrong-shaped and non-finite TBPTT
  cache entries, silently truncated an oversized cache through `deque(maxlen=...)`,
  accepted negative or boolean counters, and loaded impossible observation-gap
  states. `NeuralAgent.load_state` ignored the serialized model entirely, so its
  advertised complete-resume boundary was not actually complete.
- **Scientific impact** An exact-resume or identical-branch claim is not valid
  when malformed state can enter the claim-critical path and fail only later.
- **Repair** Model, cache, counter, tracker, and adapter state now validate
  schemas, types, shapes, finiteness, ranges, and cross-field invariants before
  mutation. Agent restoration restores the serialized model and previously
  omitted error-estimate/update-diagnostic state.
- **Regression** `CheckpointBoundaryTests` injects oversized, non-finite,
  wrong-shaped, negative, boolean, and impossible state and requires fail-closed
  rejection; it also verifies exact model restoration.

### AAA-158 — branch identity covered the model but not the complete interaction state
- **Source** independent Sol review of PR #16, 2026-09-20 · **Severity** high · **Status** repaired and verified
- **Reproduction** `run_online_frozen_branch` recorded and compared only
  `model.state_hash()`, despite documentation claiming equality of weights,
  hidden/cache state, previous error, observation tracker, pending prediction,
  and adapter counters. Mutating `previous_signed_error` did not affect the
  asserted branch identity.
- **Scientific impact** Online/frozen and changed/control comparisons could
  claim identical starting agents while adapter state differed.
- **Repair** The branch hash now canonically covers every future-affecting model
  and interaction field, excluding only the intentional treatment labels
  `name` and `update_enabled`.
- **Regression** A failure-injection test mutates adapter state with identical
  weights and proves the complete-interaction hash changes.

### AAA-159 — Q1's time contrast did not causally identify online learning
- **Source** independent Sol review of PR #16, 2026-09-20 · **Severity** high · **Status** repaired and measured in round 3
- **Reproduction** Round 2 called first-quarter minus last-quarter error
  "learning." A stream whose later portion is easier produces the same sign
  even when no weights update; no matched non-learning arm removed this time
  structure.
- **Scientific impact** The headline Q1 claim was stronger than its estimator.
- **Repair** Round 3 compares the online model with an otherwise identical
  frozen-weight copy on every initialization-stream cell. Both arms see the
  same temporal structure and start from the same complete interaction state.
  Positive frozen-minus-online error now identifies the effect of updating
  weights under this benchmark.
- **Regression** The round-3 Q1 test requires the matched frozen-arm statistic
  and complete crossed design; the old quarter contrast remains only in
  superseded evidence.
- **Outcome** Matched frozen-minus-online error is `+3.299e-03`, 95% interval
  `[+3.013e-03, +3.590e-03]`; every initialization mean and all 144 stream
  averages have the favorable sign.

### AAA-160 — Q2 and Q5 treated reused initializations as independent trials
- **Source** independent Sol review of PR #16, 2026-09-20 · **Severity** high · **Status** repaired and measured in round 3
- **Reproduction** Round 2 cycled `trial_index % 5` across 24 adaptation and 12
  retention trials, then applied a one-dimensional paired bootstrap. Trials
  sharing starting weights were treated as independent even though the general
  round-2 description claimed initialization resampling.
- **Scientific impact** Q2/Q5 uncertainty could be materially understated and
  was inconsistent with the estimator used for Q1/Q3/Q4.
- **Repair** Round 3 fully crosses every declared initialization with every
  adaptation or retention environment and independently resamples both factors.
  A ragged or incomplete crossing is rejected rather than flattened.
- **Regression** `_trial_matrix` refuses an incomplete grid; independent
  recomputation reconstructs all Q2/Q5 interval endpoints from retained trials.
- **Outcome** Q2 adaptation is `+5.121e-04`
  `[+1.689e-04, +8.667e-04]`; its share of the combined adaptation plus
  continued-learning advantage is 43.15%. Q5 forgetting is inconclusive at
  `-2.025e-04` `[-6.418e-04, +1.257e-04]`. The truthful claim is that no
  forgetting was measured on this probe bank, not that retention immunity was
  established.

### AAA-161 — divergent AAA-1K development records were not strict JSON
- **Source** final repository audit, 2026-09-20 · **Severity** medium · **Status** repaired and verified
- **Reproduction** Strict parsing with rejection of non-standard numeric
  constants failed on four tracked AAA-1K artifacts. Divergent development
  configurations stored `mean_mae: Infinity`, which Python's permissive JSON
  encoder and decoder accept even though RFC 8259 JSON does not.
- **Scientific impact** The explicit divergence flags and selected
  configurations were correct, so headline results and model behavior were
  unaffected. The artifacts were nevertheless not portable JSON and could be
  rejected by standards-compliant independent tooling.
- **Repair** A divergent configuration now records `mean_mae: null` alongside
  its existing explicit divergence flag and diagnostic. The evidence writer
  uses `allow_nan=False`, so any future unhandled `NaN` or infinity fails
  before replacing an artifact. Current selection, characterization, and
  round-3 evidence were regenerated from the repaired source. The three
  affected tokens in the explicitly superseded round-2 artifact were
  syntax-normalized to `null`; no primitive measurement, summary, claim, or
  historical status changed.
- **Regression** `SerializationTests` injects positive infinity into an
  evidence payload and requires a fail-closed `ValueError` with no output file.
  The repository integrity audit strictly parses every tracked JSON file.

## AAA-1K improvement-loop pilot (`aaa.loop.v0-pilot`, 2026-09-21)

Found while running the first iterations of the loop in
[`loop_pilot_report.md`](loop_pilot_report.md). Model limitations the pilot
measured (the coarse-family operating point, the long-horizon runaway) are
*known limitations*, recorded in [`limitations.md`](limitations.md), not
defects; failed candidates are *failed hypotheses*, recorded in the iteration
records. Only actual defects are entered here.

### AAA-162 — the coarse_speed_v1 characterization attributes a gated-model limitation to the benchmark
- **Source** loop pilot diagnosis, 2026-09-21 · **Severity** medium · **Status** resolved in aaa.1k.v2 (diagnostic level); historical text unchanged
- **Claim** `docs/limitations.md`, the round-3 report and the AAA-1K handoff
  and self-review state that `coarse_speed_v1` "genuinely tests hidden-regime
  inference" because, with the speed held fixed, "the quantizer alone hands the
  advantage to the stateless arm". The report and handoff describe Q4's
  unfavourable mean as a minority of streams with large gated losses.
- **Reproduction** `python -m research.aaa_1k_loop observe` recomputes, from
  round-3 primitives, that all 31 ungated-favouring stream means are 24
  `coarse_speed_v1` streams (every one of that family, in every
  initialization) and 7 `aba_v1` streams; without `coarse_speed_v1` the Q4
  aggregate is `+8.3e-05`. `python -m research.aaa_1k_loop diagnose2` measures,
  on fresh diagnostic streams, the ungated control's memory advantage over the
  stateless control at fixed speed at 99% of its switching size (H15), while
  the champion is worse than the stateless control at fixed speed.
- **Root cause** The decomposition probe (`coarse_speed_decomposition` in
  `research/aaa_1k/characterization.py`) compared only the gated champion with
  the stateless control, then drew a conclusion about the benchmark. A model
  that cannot use memory on a family cannot show whether the family rewards
  memory.
- **Scientific impact** The explanation of Q4's memory-family result and of
  what `coarse_speed_v1` measures is unsupported. No round-3 number is wrong;
  its interpretation is.
- **Repair attempted** Iteration `aaa1k-loop-0003` took a corrected
  interpretation through the loop. Claim v2 failed attack T1: the fixed-speed
  memory advantage is construction-dependent (about 30% survives at quantum
  0.006 or speeds 0.10/0.25). Scoped claim v3 failed attack R4 on an unresolved
  interval. Neither is promoted. The original statements are flagged in
  `docs/limitations.md` and `docs/errata.md`; the historical reports are
  unchanged.
- **Regression** `tests/test_aaa_1k_loop.py` re-verifies the observation's
  source hash; the evidence is `docs/evidence/aaa1k_loop_0001/observation.json`,
  `diagnosis_2.json` and `docs/evidence/aaa1k_loop_0003/attack*.json`.
- **Outcome** open: the existing claim is contradicted on diagnostic and attack
  identities; a replacement needs a better-designed attack (see the protocol's
  known gap 4) and fresh confirmation.
- **aaa.1k.v2 update (2026-09-23)** a model-independent test on fresh
  diagnostic identities (V2-D16, `docs/evidence/aaa_1k_v2/diagnostics.json`)
  settles what the family rewards: at fixed speed an 8-observation
  least-squares window has 0.49 times the error of a 2-observation window,
  and regime switching adds only 12%. The family mainly measures
  coarse-observation integration; hidden-regime inference is secondary. The
  original characterization is refuted, not replaced by a promoted claim.
  The v2 benchmark protocol names the family accordingly.

### AAA-163 — AAA-1K seed namespaces can collide, contrary to the module's claim
- **Source** loop pilot identity work, 2026-09-21 · **Severity** low · **Status** accepted for aaa.1k.v1; resolved by construction in aaa.1k.v2
- **Claim** `research/aaa_1k/seeds.py`: "two different labels cannot collide
  except by a SHA-256 collision".
- **Reproduction** Enumerating every AAA-1K namespace at indices below 100,000
  yields 499,927 distinct seeds from 500,000 derivations: 73 cross-index
  collisions (for example `model_init` 1556 and `development_env` 35877). Seeds
  are SHA-256 digests reduced modulo `2^31 - 1`, so birthday collisions are
  expected at this scale.
- **Scientific impact** None measured: no collision exists among the indices
  AAA-1K actually used (checked across every declared range). The docstring
  overstates a guarantee.
- **Repair** Not applied to `research/aaa_1k/`, because editing it would change
  Champion 0's scientific fingerprint for a comment. The loop does not rely on
  the claim: `research/aaa_1k_loop/identities.py` proves disjointness by set
  intersection against every AAA-1K seed below index 100,000 and every seed
  recorded in AAA-1K evidence.
- **Regression** `IdentityLedgerTests` inject a collision and require the
  freshness proof to fail.
- **aaa.1k.v2 update (2026-09-23)** v2 identities are 64-bit SHA-256 seeds
  over `aaa.1k.v2:<role>:<namespace>:<index>` (`research/aaa_1k_v2/identities.py`),
  and `python -m research.aaa_1k_v2 prove-fresh` proves every registered v2
  seed disjoint from each other and from every AAA-1K and loop seed, failing
  closed on any collision. `research/aaa_1k/seeds.py` is untouched.

### AAA-164 — the first claim attack's command stamped the current claim id
- **Source** loop pilot reproduction check, 2026-09-21 · **Severity** low · **Status** repaired
- **Reproduction** After claim v3 was declared, `python -m research.aaa_1k_loop
  reproduce attack3` reported one mismatch: the committed artifact records
  `claim_id: aaa1k-claim-q4-coarse-v2` (the claim that attack judged), but the
  command now wrote `...-v3`. Every measured value reproduced.
- **Root cause** `command_attack3` labelled its output with the module's
  current `CLAIM_ID` rather than the claim that attack was declared to judge.
- **Repair** The command uses `CLAIM_ID_V2`.
- **Regression** `ReproductionTests` replay the committed attack records through
  the command and require zero mismatches; it fails on the defective code.

### AAA-165 — loop freeze could attest scientific bytes absent from its commit
- **Source** independent PR #19 final review · **Severity** high · **Status** repaired
- **Reproduction** Modify a confirmation-affecting file, build a freeze from
  the dirty bytes, commit only the freeze and ledger, and run confirmation.
  The old admission checked only that the manifest matched `HEAD`; live dirty
  source still matched the manifest even though no durable commit contained it.
- **Root cause** `require_committed()` proved the freeze was committed but did
  not prove the freeze's file map existed in that commit.
- **Repair** Confirmation now compares every frozen source byte and executable
  bit with `HEAD`. The frozen set includes the complete loop package, including
  CLI admission and freeze logic, plus the AAA-1K phase files.
- **Regression** `test_dirty_frozen_source_that_is_absent_from_head_is_refused`
  implements the adversarial commit sequence and requires refusal.

### AAA-166 — locally spent loop identities could become fresh after reset or reclone
- **Source** independent PR #19 final review · **Severity** high · **Status** repaired
- **Reproduction** The old `confirm3` wrote `spent` only to the working-tree
  ledger immediately before execution. A crash followed by reset or a fresh
  clone recovered the committed `reserved` ledger.
- **Root cause** a mutable local file was treated as durable ownership.
- **Repair** Before a confirmation cell can run, `confirm3` atomically creates
  immutable remote Git ownership refs for both confirmation blocks using the
  repository's existing reservation mechanism. A crash, clone, worktree or
  competing actor sees the same durable owner; partial reservation fails safe
  by consuming rather than reusing identities.
- **Regression** Existing reservation failure-injection tests cover atomic
  races, mismatched resume and immutable ownership; loop tests continue to
  require a spent local ledger for recomputation.

### AAA-167 — iteration records accepted malformed and unsafe artifact structures
- **Source** independent PR #19 final review · **Severity** medium · **Status** repaired
- **Reproduction** Artifact mappings were indexed before shape validation and
  accepted traversal, symlinks, malformed hashes, duplicate paths with
  contradictory roles, and unknown roles; candidate IDs could repeat.
- **Repair** The validator now checks exact artifact fields, safe relative
  paths confined to the repository, regular non-symlink files, known roles,
  unique paths, lowercase SHA-256 shape and unique non-empty candidate IDs.
  Iteration 0003 now cites its predeclared source as the development artifact
  instead of assigning one diagnosis JSON two contradictory roles.
- **Regression** The loop validator suite exercises the hardened shape checks
  while all three committed records continue to validate.

### AAA-168 — H13 operational tests were reported as a proved mechanism
- **Source** independent PR #19 final review · **Severity** medium · **Status** repaired in active interpretation
- **Reproduction** Four declared subtests passed, but the Jacobian result is a
  per-cell median of per-step spectral summaries and the bias perturbations
  correlate the operating point with performance. They do not identify the
  claimed sign-alternating mode or exclude all alternative mechanisms; the
  same intervention later showed tradeoffs and initialization instability.
- **Repair** Active report and handoff classify M1/H13 as partial mechanistic
  support while preserving the observed evidence and its computed operational
  verdict. M2's closed-loop-gain explanation remains explicitly a hypothesis.
- **Outcome** No model or historical evidence changed.

### AAA-169 — AAA1KGRU's docstring misnames its online TBPTT rule
- **Source** external research audit 2026-09-21 (R-02) · **Severity** low (numerically) · **Status** documented; code intentionally unchanged
- **Reproduction** `research/aaa_1k_loop/tbptt.py`: after interleaved SGD
  updates, `AAA1KGRU.backward()` equals neither the realized-trajectory
  gradient (`snapshot`, which backpropagates each cache through the matrices
  it was computed with) nor the current-parameter chunk gradient (`replay`).
  Both references are proven against finite differences; the tests fail if
  live equals either one once parameters drift.
- **Root cause** cached activations from older parameters are multiplied
  through the current recurrent matrices. The model docstring calls the
  result "exact for the realized trajectory up to the truncation horizon".
- **Measured impact** (iteration 0004, H20–H23, fresh diagnostic block): the
  median per-step update difference from `snapshot` is 7.1e-5 (q99 8.9e-3);
  snapshot is EQUIVALENT to live on every standard family; M2 is identical
  under live, snapshot, replay and T = 1.
- **Disposition** the rule is a standard approximation (online TBPTT with
  cached activations and current weights). The docstring is corrected in
  `errata.md`, not in `model.py`, because editing `model.py` changes the AAA-1K
  phase fingerprint that every AAA-1K result and both loop champions carry.

### AAA-170 — M2 is a self-confirming target-unfolding frame lock
- **Source** loop iteration 0004, diagnosis rounds 2–4 · **Severity** high · **Status** repaired in Champion 1 (loop); AAA-1K core unchanged
- **Reproduction** `python -m research.aaa_1k_loop.stages4 unfold`: on
  1120-step quantized coarse streams, every diverged champion cell (54/54)
  first holds a run of at least 10 consecutive mirrored unfolded targets,
  11–20 steps before onset. No stable cell does.
- **Root cause** `NeuralAgent._training_target` unfolds the revealed
  observation onto the branch nearest the learner's *own* raw prediction
  (`aaa.predictors.unfold_observation`). After a slow bounce the raw
  prediction can remain past the wall. Later observations are then mirrored,
  and the target, relative to the real-frame input position, is about −2 ×
  distance-from-wall / scale. That keeps the prediction past the wall and
  grows as the dot moves away.
- **Corrects** the pilot's M2 attribution to a learned previous-error loop
  gain (falsified: H24, H27) and the implicit overshoot reading (falsified:
  H29). The previous-error input sustains the lock but does not cause it. The
  pilot's "likely source of round 3's two burst cells" is confirmed: those
  two cells differ, and only those, when round 3 is re-run with the repair.
- **Repair** Champion 1 (`docs/evidence/aaa1k_loop_0006/champion_1.json`):
  refuse a mirrored branch when the input position is farther from the crossed
  wall than |velocity estimate| + |observed displacement|. Promoted through
  fresh confirmation (21.3% → 0% long-coarse divergence; bitwise identical on
  every standard family).
- **Not repaired** `research/aaa_1k/agents.py` itself: porting the rule
  there is a new AAA-1K phase version.

### AAA-171 — loop confirmation refused its own freshly spent identities
- **Source** iteration 0006 confirmation attempt 1 · **Severity** high (burned identities) · **Status** repaired
- **Reproduction** `outer6 confirm` created both remote claims and marked the
  blocks spent (the spent-before-run rule), then `confirmation_cells()`
  called `require_usable(purpose="confirmation")`, which demands `reserved`,
  and raised before the first cell was built.
- **Consequence** nothing was observed. Blocks
  `aaa1k-loop-0006/confirmation/{env,init}` stay spent and are never reused
  (`confirmation_attempt_1.json`). The unchanged challenger was refrozen on
  new blocks (`freeze_2.json`).
- **Repair** `identities.require_claimed` (confirmation role, status spent,
  observed by *this* observer). `spend_and_build()` puts admission-to-cells
  in one function for every outer module.
- **Regression** `ConfirmationAdmissionTests` and `ConfirmationEndToEndTests`
  cover admission → cells → primitives → payload → decision → independent
  recomputation. The gap was that the outer path had only ever been
  unit-tested in pieces (audit R-12).

### AAA-172 — the v2.1 core RLS learner unfolds around its own prediction too
- **Source** AAA-170 follow-up · **Severity** low · **Status** characterized in aaa.1k.v2; no defect found on the tested streams
- **Observation** `OnlineRLSPredictor.update` (`aaa/predictors.py`) builds
  its target with `unfold_observation(target, raw, …)`, the same
  self-reference as AAA-170. It also has `skip_after_reflected_prediction`,
  which might prevent the chase, or might instead stall learning while its
  raw prediction stays past a wall.
- **Next** instrument without modifying it, on long quantized and smooth
  streams, with predeclared lock and stall thresholds. The v2.1 core and
  round 3's `rls_online` baseline both use it.
- **aaa.1k.v2 update (2026-09-23)** instrumented without modification on
  fresh diagnostic identities (V2-D16): the longest run of applied mirrored
  targets is 1 step and the longest run of reflection skips 8 steps, against
  predeclared thresholds of 10 and 50. No lock, no stall, no divergence; the
  learner beat persistence on every family. `aaa/predictors.py` is unchanged.

### AAA-173 — Champion 0's diverged cells are not bit-reproducible across CPUs
- **Source** first CI runs of `loop-reproduction.yml` (PR #20) · **Severity** medium (reproducibility semantics) · **Status** characterized; reproduction policy made explicit
- **Observation** on the implementer's machine every iteration 0004–0006
  stage reproduces bit for bit. On GitHub runners, results depend on the
  runner's CPU. Some jobs were bit-identical. Others differed in the last
  digit of stable-cell MAEs (relative ~4e-15). In cells where the champion
  diverges (the frame lock, M2), the difference amplified into different
  trajectories: MAE differences up to ~30%, and different mirror-step counts.
- **Cause** long online-learning trajectories in a runaway regime are
  chaotic, so ordinary cross-platform float differences (SIMD width, library
  build) do not stay small. Stable cells, including every Champion 1 cell,
  stay within ~1e-9.
- **Platform** on the implementer's Ryzen 7 5700G and on AMD EPYC 7763
  runners (both Zen 3), every stage is bit-identical. On AMD EPYC 9V74
  runners (Zen 4, AVX-512 kernels), results drift: up to ~30% in diverged
  cells, up to ~2e-3 in long (3360-step) non-diverged gain cells, and ≤1e-10
  in 1120-step stable cells. The drift is an instruction-set effect, not
  randomness.
- **Policy** `stages4 reproduce` gates on cell identities, artifact
  structure and every adjudicated verdict (recomputed from the rerun's own
  primitives), which must be identical. Per-cell numeric drift is reported
  (cells that are bit-different, cells beyond relative 1e-9, chaotic cells,
  divergence-status flips, largest drift) but does not gate, because no fixed
  tolerance separates instruction-set noise from a defect. `--exact` gates on
  bitwise identity, which is the guarantee on the evidence platform. The
  workflow logs each runner's CPU.
- **Consequence** per-cell magnitudes of Champion 0's diverged cells (and
  aggregates over them, such as median gains or divergence counts) are
  platform-dependent at the level of about one cell. The claims rest on the
  verdicts, which are held exact.

---

## Independent repository review (2026-09-22)

Found by the independent end-to-end review of `main` at `efa1fdb`. Each
entry was reproduced against that tree before it was changed; the review
record is [`independent_review_2026-09-22.md`](independent_review_2026-09-22.md).

### AAA-174 — `champion1 verify` did not recompute the phase fingerprint
- **Source** independent review 2026-09-22 · **Severity** medium · **Status** repaired
- **Reproduction** In a throwaway copy of `efa1fdb`, appending one byte to the
  fingerprinted `research/aaa_1k/model.py` changed the `aaa.1k.v1` fingerprint
  to `e9e65b0c…`. `champion --verify` failed, but
  `python -m research.aaa_1k_loop.champion1 verify` still printed
  `champion 1: VALID` and exited 0. Changing a byte of `requirements-lock.txt`
  gave the same result.
- **Cause** `champion1.build_record` copies `phase_fingerprint` from Champion
  0's record, and `verify` compares the record with that rebuilt copy. It
  never hashed the live tree, so the check compared the record with itself.
  `CONTRIBUTING.md` says that `champion1 verify` fails when a fingerprinted
  file changes.
- **Consequence** No wrong acceptance occurred. `champion --verify`,
  `validate`, `tests.test_aaa_1k_loop` and CI all recompute the fingerprint and
  caught the change. Champion 1's record itself is unchanged and correct.
- **Repair** `champion1.verify` now recomputes `phase_fingerprint(root)` and
  reports any disagreement with the recorded value. `build_record` and the
  Champion 1 record are unchanged.
- **Regression** `ChampionOneTests.test_champion_1_verification_recomputes_the_live_phase_fingerprint`
  substitutes a drifted fingerprint. It errors on `efa1fdb` and passes after
  the repair.
- **Verification** On the review branch the same one-byte injection makes
  `champion1 verify` print `STALE: phase_fingerprint: …` and exit 1. On the
  unmodified tree it prints `VALID`, and the fingerprint is still `5ce6e019…`.

### AAA-175 — Dependabot could open pull requests that rewrite the fingerprinted lock
- **Source** independent review 2026-09-22 · **Severity** low · **Status** repaired in configuration; policy confirmed 2026-09-23
- **Reproduction** On `efa1fdb`, the `pip` entry in `.github/dependabot.yml`
  ignored only `version-update:semver-patch`, although its comment says the
  lock "is regenerated deliberately, not by a bot". Dependabot has already
  rewritten the lock once: commit `1d1ccfe` (fonttools 4.64.0 → 4.65.0) is a
  `semver-minor` bump, which that rule does not ignore. `requirements-lock.txt`
  is one of the 32 `aaa.1k.v1` fingerprint files. In a throwaway copy, one
  bumped pin changed the fingerprint and made `champion --verify` fail.
  GitHub's options reference marks `ignore` as applying to version *and*
  security updates, and `open-pull-requests-limit` as version updates only.
  So repository-level security updates for `pip` were also constrained only by
  the patch rule.
- **Consequence** No bad change could reach `main`, because such a pull
  request fails required CI. But each one would ask the maintainer to merge a
  change to scientific identity through a bot pull request, which the
  project's rules reserve for a recorded decision (`AAA-152`).
- **Repair** The `pip` entry sets `open-pull-requests-limit: 0` and ignores
  `dependency-name: "*"` with no `update-types`. `github-actions` updates are
  unchanged. `SECURITY.md`, `docs/dependencies.md` and a dated note on
  `AAA-110` record the policy. Dependabot alerts remain enabled. Whether to
  also change repository-level security-update settings is left to the
  maintainer and is not changed here.
- **Regression** `tests/test_repository_automation.py` fails on `efa1fdb`
  (two failures) and passes after the repair.
- **Remaining limitation** Dependabot's actual behaviour can only be observed
  on GitHub, the next time it runs. This entry relies on GitHub's
  documentation.
- **Decision (2026-09-23)** The owner delegated the policy question that the
  review left open, and the policy is kept: no `pip` pull requests of any
  kind, Dependabot alerts on, `github-actions` updates unchanged. A bot pull
  request against the lock can never pass the required checks, because the
  lock is fingerprinted, so it could only be merged by an admin override,
  which the project reserves for recovery. A lock change is a
  scientific-identity event that needs a recorded decision (`AAA-152`). The
  realistic exposure is small (see the threat model in `SECURITY.md`), and
  alerts still report any vulnerable pin. When one arrives, the maintainer
  decides whether it matters under that threat model and, if it does,
  regenerates the lock as a deliberate, recorded identity change.
- **Observation (2026-09-23)** The first Dependabot run after the merge
  (`35815074694`, triggered by the configuration change) ran only the
  `github-actions` job and opened no pull request. No `pip` version job ran,
  as `open-pull-requests-limit: 0` intends. Suppression of security pull
  requests remains unobserved while there are no open alerts.

### AAA-176 — active loop documentation still described the pre-review state
- **Source** independent review 2026-09-22 · **Severity** low · **Status** repaired
- **Reproduction** On `efa1fdb`, `docs/loop_protocol.md` says in its title
  and status note that `aaa.loop.v1` is current, and
  `LOOP_PROTOCOL_VERSION == "aaa.loop.v1"`. Its "Status after iterations
  0004–0006" section still said "The version remains `v0-pilot` pending an
  independent review of that work", and the PR #20 reconciliation did not
  catch it. The README described the protocol as "the pilot improvement loop"
  and did not list `docs/pr20_independent_review.md`, the record that
  established v1. The historical pilot report links to
  `loop_protocol.md#known-gaps-to-close-before-v1`. That heading was renamed
  in `0fbf20e`, so the link no longer resolved.
- **Repair** A dated, additive note after the stale paragraph. The paragraph
  itself is kept as the record of the pre-review state. The README row now
  describes current governance, and a row lists the PR #20 review. An explicit
  HTML anchor with the old name sits before the renamed heading, so the
  historical report resolves without being edited.
- **Verification** A local Markdown link and anchor check reports no broken
  relative link or anchor in the tracked tree.

### AAA-177 — ledger statuses still read "pending" after the pending event resolved
- **Source** independent review 2026-09-22 · **Severity** low · **Status** repaired
- **Reproduction** On `efa1fdb`, `AAA-149` read "replacement CI pending" and
  `AAA-150` read "final hosted artifact upload pending". The GitHub API shows
  the replacement runs that [`handoff_sol.md`](handoff_sol.md) cites
  (`35451372899`, `35451375484`, `35451862857`) as successful in all six jobs,
  with both evidence artifacts uploaded, and current `main` CI passes the same
  steps. `AAA-128`, `-131`, `-132`, `-135`, `-136`, `-137`, `-139`, `-140` and
  `-143` read "confirmation pending". Both declared v1.1 batches, however, are
  `retired_unobserved` in `benchmarks/observation_noise_registry.json`, and
  the README, limitations and handoff say formal A/B was not executed.
- **Repair** Each status line now states the current fact and keeps its
  earlier wording in quotation marks with the date. `AAA-149` and `AAA-150`
  also gain a dated hosted-verification line. No observation-noise protocol,
  registry, evidence or conclusion changed. "Not executed" is not a negative,
  inconclusive or positive result, as [`handoff_sol.md`](handoff_sol.md)
  already says.

### AAA-178 — the evidence policy implied a 90-day CI copy of confirmation evidence
- **Source** independent review 2026-09-22 · **Severity** low · **Status** repaired
- **Reproduction** On `efa1fdb`, the "How raw evidence stays recoverable"
  section of `docs/evidence_policy.md` said "CI additionally retains the full
  attempt directory for 90 days". Its known limitation needed "the recorded
  commit lost *and* the CI artifact expired". The sentence dates from
  `a85b506` (2026-09-11). Since `2e54cc3` (2026-09-12, `AAA-125`),
  `benchmark.yml` accepts only `development` and `high_replication` and
  refuses any batch id, so no formal confirmation attempt has a CI artifact.
  Routine CPU CI retains its smoke artifacts for 14 days.
- **Consequence** The policy overstated how durable confirmation raw evidence
  is. No evidence was lost, and `AAA-077` already records that
  regenerability is weaker than archival.
- **Repair** The policy now says which runs the 90-day retention covers, that
  confirmation attempts have no CI copy, and that routine CI keeps artifacts
  for 14 days. The known limitation is restated for each case, and `AAA-077`
  has a dated note. No workflow retention value changed.

---

## AAA-1K v2 final 1K pass (2026-09-23)

### AAA-179 — the first v2 development run could have recorded code that did not produce its evidence
- **Source** self-found during the v2 development run, 2026-09-23 · **Severity** medium (provenance) · **Status** repaired before any v2 evidence was written
- **Reproduction** `python -m research.aaa_1k_v2 develop` at `621e2a9` captured
  the environment (commit, dirty flag, v2 fingerprint) only *after* the run.
  `run_jobs` also spawns a fresh worker pool for every call, and each worker
  re-imports the package from disk. Stage code edited in the working tree while
  the run was in progress was therefore imported by later workers, and would
  have been fingerprinted as the producing code.
- **Scientific impact** none recorded: the run was stopped and nothing it
  produced became evidence (decision V2-D14). The edits made during it were
  designed to leave the SGD path unchanged, but that was not treated as proof.
- **Repair** the CLI captures provenance before any stage runs, and every formal
  v2 stage runs from a separate `git worktree` checked out at a named commit
  (`docs/reproduction.md`), so no edit can reach a running stage.
- **Regression** the rerun of development reproduces the aborted run's one
  completed raw archive (`gru_v1_retuned`) exactly; this check is recorded with
  the development evidence.

### AAA-180 — the v2 confirmation's K5 omitted Monash, and its two K5 implementations disagree on a single-series dataset
- **Source** self-found while writing the v2 report, 2026-09-23 · **Severity** high had a challenger existed; none in this phase · **Status** repaired prospectively; frozen aaa.1k.v2 defect retained and forbidden for future promotion (2026-09-23; was "open; latent, documented, not repaired (the v2 source is frozen)")
- **Reproduction** `python tools/write_aaa_1k_v2_report.py` applies the frozen
  K1-K5 criteria to every confirmation arm twice. (1) `run_confirmation`
  (`research/aaa_1k_v2/confirmation.py`) calls `decide` without
  `monash_primitives`, although the frozen K5 text and `recompute` include the
  four Monash datasets. (2) With Monash included, the two implementations
  disagree for all six arms that have Monash cells: `decide` returns K5
  `INCONCLUSIVE` and `recompute` returns `FAIL`. Monash `saugeen` has one
  series, so `stats.geometric_relative` reports `INSUFFICIENT_EVIDENCE` for the
  whole geometric mean, while `recompute`'s own bootstrap measures an interval
  anyway.
- **Scientific impact** none on recorded evidence: the frozen confirmation had
  no challenger, so no K5 decision was taken (`NO_CHALLENGER`, recomputed).
  Had a challenger reached confirmation, K5 would have been decided without
  Monash, and the independent recompute's point check would have flagged the
  difference, blocking promotion rather than silently passing it.
- **Next** a future phase must (a) pass the Monash primitives to `decide`, (b)
  predeclare how a single-series dataset enters a crossed bootstrap (resample
  initializations only, or exclude it from interval estimation), and (c) share
  that rule between both implementations, with a regression test that runs
  both on one fixture containing a single-series group. **Hard precondition:**
  the frozen v2 K5 path may not be used to promote any future candidate. A
  versioned successor must fail closed on omitted groups or decision/recompute
  disagreement before its held-out evidence is observed. See
  [`docs/scaling_readiness.md`](scaling_readiness.md).
- **Prospective repair (2026-09-23)** Independently reproduced before any change
  (`tools/check_promotion_successor.py` shows it on every run): the frozen
  `run_confirmation` passes `decide` only `persistence, external_base,
  bootstrap, families, external`; K5 is computed on 15 groups without Monash
  and on 19 with it; `monash:saugeen` is an 8 x 1 grid, so frozen `decide`
  returns `INCONCLUSIVE` (`INSUFFICIENT_EVIDENCE` interval) while frozen
  `recompute` returns `FAIL`, for all six arms with Monash cells. The frozen
  v2 source and evidence are **unchanged** and still reproduce this. The
  successor `aaa.promotion.crossed.v1` (`aaa/promotion/`,
  [`promotion_contract.md`](promotion_contract.md)) implements (a) mandatory
  presence of every declared group, (b) a predeclared per-group design, where a
  single-series group is `conditional_on_single_series` (sole series fixed,
  shared initializations resampled, scope reported as conditional on that
  series, no series dimension fabricated), and (c) a primary index-resampling
  evaluator plus a structurally independent count-weighted recomputation that
  must agree on point value, interval status, bounds, criteria and verdict,
  failing closed as `DISAGREEMENT` or `INVALID_EVIDENCE`. It does not share
  implementation code between the two sides. `aaa.1k.v2.k1-k5` is in
  `FORBIDDEN_CONTRACTS`, and the v2 registry's spent confirmation blocks
  cannot be re-frozen. On the retained v2 K5 groups, descriptively, both
  successor implementations agree for all six arms.
- **Regression** `tests/test_promotion.py` (28 cases): single- and multi-series
  agreement on point value, interval status, bounds, criteria and verdict; an
  explicit initialization-only check of the single-series interval; omitted
  groups (including one arm only) cannot promote; extra groups, mismatched
  initialization or series identities, missing and duplicate cells,
  non-finite, non-positive, `None`, string and boolean values, and malformed
  records are refused by each implementation separately; injected point,
  bound, status, criterion and verdict disagreements; the frozen defect is
  asserted to remain in the frozen source; the forbidden contract is refused.

### AAA-181 — static v2 report prose overstated the family and error-head readings
- **Source** independent pre-scale review, 2026-09-23 · **Severity** recommendation · **Status** corrected in current report generator and regenerated report; frozen results unchanged
- **Reproduction** the v2 family table contains small favourable differences
  for an ablation outside the coarse/noisy families, while its static sentence
  said those were the only families where *anything* beats Champion 1. The K3
  error-head-loss interval is [0.998, 1.002], so a literal claim of unchanged
  prediction or no effect is stronger than the measured result. The Monash
  sentence attributed a seasonal-naive advantage to missing seasonal memory
  without isolating that mechanism, and the capacity paragraph rounded a
  measured range into a fixed verbal range.
- **Scientific impact** no primitive, decision, verdict or frozen identity
  changes. The revised interpretation distinguishes small ablation effects,
  generates the auxiliary-loss interval, Monash comparisons and capacity range
  from retained values, and no longer asserts an untested Monash mechanism.
  `tools/write_aaa_1k_v2_report.py --check` guards the regenerated report bytes.

---

## Python-first transition (2026-09-23)

Found while moving AAA's active research to `aaa.python.v0`. Each was
reproduced before it was changed. The development-construction defects
(`AAA-184`, `AAA-185`) were all found and repaired **before** any development
evidence was produced, so no retained evidence carries them.

### AAA-182 — frozen v2 recomputation silently skipped Monash for a challenger without Monash primitives
- **Source** self, while reproducing `AAA-180` · **Severity** high had a challenger existed; none did · **Status** frozen defect retained; repaired prospectively by `aaa.promotion.crossed.v1`
- **Reproduction** `research/aaa_1k_v2/recompute.py` adds the four Monash
  groups to K5 only `if name in monash`. A challenger whose Monash primitives
  are absent is therefore adjudicated on 15 of the 19 declared K5 groups, with
  no problem recorded. The declared-group set is not checked.
- **Scientific impact** none on recorded evidence (`NO_CHALLENGER`). It is the
  independent side's counterpart of `AAA-180`'s omission: both implementations
  could have agreed on an incomplete group set.
- **Repair** prospective only; the frozen file is unchanged. The successor
  requires every declared group for both arms, in both implementations, and
  returns `INVALID_EVIDENCE` otherwise.
- **Regression** `FailClosedTests.test_omitting_a_required_group_cannot_promote`
  and `test_omitting_a_group_for_one_arm_only_cannot_promote`, against a
  challenger that would otherwise be promoted.

### AAA-183 — current-facing documents still presented the dot as AAA's active research
- **Source** self, transition audit · **Severity** low (claims) · **Status** repaired
- **Reproduction** At `f56caaf`: the README's first sentences described "one
  moving dot on a line" and a three-parameter learner as the project and
  called AAA-1K "the current research phase"; `CONTRIBUTING.md`'s scope named
  only dot-era work; `CITATION.cff`'s abstract described only the dot
  prototype; `SECURITY.md`'s threat model said AAA runs one-dimensional
  simulations with no `eval`, which stopped being true when the Python phase
  began executing programs; `docs/hardware.md` still called the 105M planning
  goal a ceiling in its introduction.
- **Repair** Those documents now lead with the research problem and the
  Python specialization. The dot era is mapped in `dot_benchmark_archive.md`,
  and SECURITY describes the sandbox and what it is not. Fingerprinted and
  protected historical documents (the charter, protocols, reports, reviews,
  handoffs) are **unchanged**; `errata.md` carries the notice instead.
- **Verification** a relative-link and anchor check over every tracked Markdown
  file; `tools/check_protected_identities.py` shows no protected byte moved.

### AAA-184 — an AST representation would have handed the syntax answer to the learner
- **Source** self, first `aaa.python.v0` development smoke · **Severity** critical for that family had it been retained · **Status** repaired before any evidence
- **Reproduction** With `ast_nodes`, programs that fail `ast.parse` received a
  single `<unparsable>` feature. The online and frozen `ast_nodes` arms then
  scored 1.0 on `syntax` in the quick smoke, against about 0.5 for every other
  representation.
- **Root cause** whether CPython's parser accepts the source *is* the
  `syntax` family's answer key. A representation built on that parser carries
  the evaluator's verdict into the learner before the action.
- **Repair** `representation.effective_representation` refuses `ast_nodes` for
  `syntax` (and for lone repair-candidate fragments, which do not parse) and
  falls back to `lexical`; `vector()` raises rather than emit a parse verdict.
  After the repair, `ast_nodes` syntax accuracy is at chance-level 0.58 in the
  smoke.
- **Regression** `RepresentationTests.test_the_ast_representation_never_sees_a_syntax_verdict`.
  `CONTRIBUTING.md` names this as the pattern to avoid.

### AAA-185 — `aaa.python.v0` construction defects found by its own tests and smoke
- **Source** self, during construction · **Severity** medium (each would have biased or broken development evidence) · **Status** repaired before any evidence
- **Findings and repairs**
  - *Construction cues.* Every fault assigned to a variable named `t`, and each
    faulty program had one risky line, so a heuristic localized 92% of
    failures from surface form in the quick smoke (24 tasks). Faults now assign a name the program never
    uses, fire only on value-dependent conditions over earlier-bound names,
    and half carry a decoy fault. The heuristic still wins, at 0.73, and that
    is reported.
  - *Cross-family memory.* The lookup baseline keyed memory by source alone;
    an identical program in two families returned an output value for a
    localization task. The causal boundary refused the out-of-space answer,
    which is how it was found. The key now includes the family.
  - *Out-of-pool identities.* The first runner requested development indices
    past the declared 400-item pool, where probe items are not excluded.
    Runs now use exactly the declared pool, and `build()` refuses any index
    outside a split's pool.
  - *Resume that could never resume.* Checkpoints compared the stored plan
    (JSON lists) with `asdict(plan)` (tuples), so every resume was refused.
    Found by `test_resuming_from_checkpoints_reproduces_the_run_exactly`;
    the comparison is now JSON-normalized.
  - *Incomplete memory reset.* The memory-disabled control restored weights
    but not its update counter, which is part of persistent state. Found by
    a state-hash test; it now restores both.
  - *Decorative specification values.* Repair counts, the adaptation's changed
    slice and two pool sizes were declared but hardcoded or unread. The
    spec-consumption test's first version passed vacuously, because
    `json.dumps` of the specification recorded every leaf. It now fails on
    an injected unread leaf, and the code reads every non-descriptive value.
- **Regression** `tests/test_aaa_python_generation.py`,
  `tests/test_aaa_python_learning.py` and `tests/test_aaa_python_spec.py`,
  each written to fail on the defective behaviour. The golden answer keys and
  the 250-task generation digest were unchanged by the late spec-consumption
  repairs, so no task changed.

