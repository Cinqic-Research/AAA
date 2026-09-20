# AAA-1K independent-review handoff

For a reviewer who should be able to verify every claim without trusting the
implementer's summary. Written by the implementer; **not** an approval, and not
a substitute for independent review.

## 1. Identity

| Item | Value |
|---|---|
| phase | `aaa.1k.v1` |
| base commit (`main`) | `02c3b20dd163c11a99502e846a2816b3e6c761de` |
| branch | `opus/aaa-1k-recurrent-core` |
| head commit | recorded in the pull request; `git log -1 --format=%H` |
| phase scientific fingerprint | `python -m research.aaa_1k fingerprint` |
| model format | `aaa.1k.gru.v1`, architecture `AAA1KGRU-3x16x2` |
| benchmark families | `aaa.1k.benchmarks.v1` |

Baseline before any change: the full suite was 445 tests, green, on
`02c3b20`. After this phase it is 538 tests, green.

## 2. What to read, in order

| Document | Why |
|---|---|
| [`aaa_charter.md`](aaa_charter.md) | what AAA is, what Juniper is, and why the dot is one benchmark |
| [`aaa_1k_architecture.md`](aaa_1k_architecture.md) | the frozen specification: equations, counts, loss, temporal order |
| [`aaa_1k_literature_review.md`](aaa_1k_literature_review.md) | what was adopted, what was refused, and why |
| [`aaa_1k_decisions.md`](aaa_1k_decisions.md) | every decision, including three departures from the phase brief |
| [`aaa_1k_report.md`](aaa_1k_report.md) | the result |
| [`aaa_1k_self_review.md`](aaa_1k_self_review.md) | the implementer's attempt to break it — **read this before the report** |

## 3. File inventory

Implementation, `research/aaa_1k/`:

| File | Contents |
|---|---|
| `model.py` | the 994-parameter GRU: forward, TBPTT backward, SGD, serialization, hashing |
| `controls.py` | 982-parameter stateless MLP, 954-parameter ungated RNN |
| `features.py` | the three inputs and the public normalization constants |
| `agents.py` | the causal adapter, branching, and every analytic baseline |
| `streams.py` | four benchmark families; latent truth lives here and goes no further |
| `runner.py` | the temporal boundary and the online/frozen branch construction |
| `selection.py` | the frozen development plan and its selection rule |
| `experiments.py` | Q1–Q7, replication planning, the evaluation pass |
| `stats.py` | paired stream-level bootstrap, calibration, capability vector |
| `gradcheck.py` | finite-difference verification |
| `adversarial.py` | the two probes that try to break the conclusions |
| `report.py`, `visualize.py`, `identity.py`, `seeds.py`, `cli.py` | reporting, dashboard, identity, seeds, entry point |

Tests: `tests/test_aaa_1k.py`, 93 tests. Evidence:
`docs/evidence/aaa_1k_development_selection.json`,
`docs/evidence/aaa_1k_evaluation.json`,
`docs/evidence/aaa_1k_adversarial_probes.json`.

Nothing under `aaa/`, `benchmarks/` or `results/` is modified. `pyproject.toml`
is modified only to bring `research/` under mypy, coverage and packaging.

## 4. The parameter-count formula, to check independently

```text
gates = 3 * (I*H + H*H + H) = 3 * (3*16 + 16*16 + 16) = 960
head  = H*O + O             = 16*2 + 2                =  34
total                                                 = 994
```

One bias vector per gate, not PyTorch's two. Controls:
`3*28 + 28 + 28*28 + 28 + 28*2 + 2 = 982` and
`3*28 + 28*28 + 28 + 28*2 + 2 = 954`.

```bash
python -m research.aaa_1k parameter-audit
```

recomputes all three from the array sizes, from the formula, and against the
declared constants, and exits non-zero on any disagreement.

## 5. Model-state schema

`AAA1KGRU.state_dict()`:

```text
format_version, architecture_id, parameter_count, parent_model_id
config { seed, learning_rate, tbptt_steps, error_loss_weight, gradient_clip,
         freeze_recurrent, reset_state_every_step, zero_error_input }
parameters { W_z U_z b_z  W_r U_r b_r  W_n U_n b_n  W_o b_o }
hidden          16 values
tbptt_buffer    up to `tbptt_steps` entries of { x h_prev z r n hr h output }
counters { update_count, forward_count, clip_events, nonfinite_events }
```

`NeuralAgent.state_dict()` adds the interaction state the network does not own:
`previous_signed_error`, the observation tracker, the last raw and scored
predictions, and the trained/skipped counters. `state_hash()` is SHA-256 over
the canonical complete state.

## 6. Research questions, frozen before evaluation

Q1 online learning · Q2 online versus frozen after change · Q3 persistent
hidden state · Q4 gating beyond plain recurrence · Q5 retention across A,B,A ·
Q6 self-error estimation · Q7 baseline competitiveness. Stated in full at the
top of `research/aaa_1k/experiments.py`.

## 7. Seed identities

Five namespaces, derived as `SHA-256("aaa.1k.v1:<namespace>:<index>")`, asserted
pairwise disjoint over 200 indices each:

`model_init` · `development_env` · `evaluation_env` · `benchmark_generation` ·
`bootstrap`

Development uses `development_env` only. Evaluation uses `evaluation_env` only.
The bootstrap draws from `bootstrap`. All model initialization comes from
`model_init`. Every stream's realized seed is recorded in the evidence files.

## 8. Hyperparameter-selection history

Complete, in `docs/evidence/aaa_1k_development_selection.json`:

- **stage 0**, divergence probe, 6 learning rates with clipping disabled;
  boundary found at `lr = 0.3`;
- **stage 1**, 24 configurations (6 learning rates x 4 horizons) at
  `lambda = 0.25`; every attempt recorded, 8 eliminated by the stability
  margin with their scores retained;
- **stage 2**, 4 auxiliary weights at the selected `(lr, T)`.

Selected: `lr = 0.03`, `T = 4`, `lambda = 0.25`.

**The margin rule eliminated the best development configurations**, at a 26%
development cost (1.138e-03 → 1.434e-03). Check this: the eliminated entries
are in `stage_one_eliminated` with their `mean_mae` intact.

## 9. Commands

Install (from a clean checkout):

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python tools/check_lock.py
```

Tests, lint, types:

```bash
python -m unittest discover -s tests -t .
python -m unittest tests.test_aaa_1k
python -m ruff check . && python -m ruff format --check .
python -m mypy
```

Verify the architecture and the gradients:

```bash
python -m research.aaa_1k parameter-audit
python -m research.aaa_1k gradient-check --full
```

Regenerate every piece of evidence (about four minutes on the reference CPU):

```bash
python -m research.aaa_1k fingerprint
python -m research.aaa_1k select   --output docs/evidence/aaa_1k_development_selection.json
python -m research.aaa_1k evaluate --selection docs/evidence/aaa_1k_development_selection.json \
                                   --output docs/evidence/aaa_1k_evaluation.json
python -m research.aaa_1k adversarial-probes \
    --selection docs/evidence/aaa_1k_development_selection.json \
    --output docs/evidence/aaa_1k_adversarial_probes.json
python -m research.aaa_1k report --evidence docs/evidence/aaa_1k_evaluation.json \
    --selection docs/evidence/aaa_1k_development_selection.json \
    --output docs/aaa_1k_report.md
```

Independently recompute the headline statistics from the retained primitives,
without rerunning a model:

```bash
python -m research.aaa_1k recompute --evidence docs/evidence/aaa_1k_evaluation.json
```

It rebuilds 59 stored values — Q1, Q3, Q4, every per-family mean and the
stability counters — from the per-stream entries beside them and exits non-zero
on any disagreement.

Dashboard:

```bash
python -m research.aaa_1k visualize --family occlusion_v1 --output runs/aaa_1k/dashboard.png
```

Confirm the legacy paths are unaffected:

```bash
python -m aaa.cli spec-hash
python -m aaa.cli observation-noise-protocol-hash
python -m aaa.cli benchmark --role development --attempt-label review --replicas 2 --episodes 2 --output-root runs
python -m aaa.cli recompute runs/benchmark-v2_1/review
python -m aaa.cli smoke --output-root runs
python tools/check_exit_codes.py
```

## 10. Known failures, confounds and accepted limitations

Ranked by how much they should change a reader's confidence.

1. **Q2 measures continued learning, not adaptation to change.** Branching at a
   quiet point reproduces 95% of the effect. `SR-1`.
2. **Q5 did not measure retention.** A2 is confounded with three times as much
   total experience. Read `retention_exists` as `NOT MEASURED`. `SR-3` in the
   report, item 3 in the self-review.
3. **The precision objective was missed by a factor of nearly twenty.** 599
   replicas per family were needed; 32 were run under a declared bound. Every
   interval is wider than the design asked for. `D-13`.
4. **Every effect is conditional on one model initialization.** Directions are
   stable across five seeds; magnitudes vary by up to 2x. `SR-2`.
5. **Q4's capacity match gives the ungated control 28 hidden units to the
   gated model's 16.** Honest for a fixed parameter budget, not a width-matched
   comparison. `SR-3`.
6. **The missingness code is weak.** During a gap inputs 2 and 3 are both
   exactly zero, which a genuine zero displacement and zero error would also
   produce. A clean flag needs a fourth input and a new architecture. `D-5`.
7. **Gradient clipping activated 12,234 times** across 55,040 scored
   transitions (22%). It is a declared mechanism with a declared threshold, but
   at that rate it is shaping the optimization, not just guarding it.
8. **TBPTT's truncation bias is real but measured to be negligible here**, and
   this benchmark does not exercise long-range credit assignment at all.
   `D-12`.
9. **`coarse_speed_v1` and `occlusion_v1` are new and unvalidated by anyone
   else.** They have never been reviewed, and a benchmark that its own author
   designed to show a mechanism is the weakest kind.
10. **Beating `rls_online` on `coarse_speed_v1` is beating it outside its
    declared operating envelope**, since the v2.1 candidate was selected and
    frozen for smooth, fully observed motion.

## 11. Effect on the existing repository-wide fingerprints

Adding `research/aaa_1k/` changes both `aaa/benchmark/source_identity.py` and
`aaa/noise/scientific_identity.py` fingerprints, because both cover every
tracked non-generated file. **This is intended and no exclusion was added to
prevent it.**

Consequences, stated rather than hidden:

- the working tree no longer reproduces the fingerprints recorded in
  `benchmarks/freeze_manifest.json` and
  `benchmarks/observation_noise_source_freeze.json`;
- those manifests describe the trees that produced their evidence, and those
  trees are reachable at the annotated tags `aaa-pre-next-phase-2026-09-19` and
  `aaa-pre-next-phase-closure-2026-09-19`;
- every v2.1 confirmation batch is already spent or retired, so no pending
  confirmation is blocked;
- `docs/final_audit.md` is a generated inventory of the pre-AAA-1K tree and has
  deliberately **not** been regenerated: it is a record of what was reviewed,
  not a live file listing.

Tracked as `AAA-152`.

## 12. Comparison tables

In [`aaa_1k_report.md`](aaa_1k_report.md): per-family per-arm accuracy, Q1–Q7
with 95% intervals and per-stream sign counts, the A/B/A segment table, the
calibration summary, and the claim ladder with a status and supporting number
on every rung.

## 13. Unresolved concerns I could not settle

- Whether `coarse_speed_v1` measures memory or measures tolerance to
  quantization. The two are entangled in its design and I did not separate
  them.
- Whether the GRU losing to the ungated RNN is a property of gating at this
  scale, of these particular streams, or of the shared learning rate being
  better suited to the ungated arm. A per-architecture learning-rate selection
  would answer it and would also open a tuning-fairness question of its own.
- Whether the 22% clip rate is masking a learning-rate problem the stability
  margin pushed me into.

## 14. What I did not do

Self-approve, merge, regenerate historical evidence, modify anything under
`aaa/` or `results/`, weaken an existing test, add an exclusion to preserve an
old hash, implement growth or replay, or claim that any of this is a step
toward general intelligence.
