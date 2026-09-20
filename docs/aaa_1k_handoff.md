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
`02c3b20`. After this phase it is 562 tests, green.

**The evaluation ran twice.** Round 1 was completed, then probed, and four
design defects were found in it. All four are repaired and round 2 is the
current result, on fresh stream identities. Round 1 is retained, superseded, at
`docs/evidence/aaa_1k_evaluation_round1_superseded.json` with its report
alongside. Read `aaa_1k_self_review.md` for what was wrong and what changed.

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
| `measurements.py` | difference-of-differences adaptation; retention against a frozen probe bank |
| `round2.py` | the corrected evaluation round on fresh identities |
| `characterization.py` | development probes for the clip, the coarse family and tuning fairness |
| `report.py`, `visualize.py`, `identity.py`, `seeds.py`, `cli.py` | reporting, dashboard, identity, seeds, entry point |

Tests: `tests/test_aaa_1k.py`, 117 tests. Evidence:
`docs/evidence/aaa_1k_development_selection.json`,
`docs/evidence/aaa_1k_evaluation_round2.json` (current),
`docs/evidence/aaa_1k_characterization.json`,
`docs/evidence/aaa_1k_adversarial_probes.json`,
`docs/evidence/aaa_1k_evaluation_round1_superseded.json` (retained).

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
- **stage 2**, 4 auxiliary weights at the selected `(lr, T)`;
- **stage 3**, 5 gradient-clip thresholds averaged over 3 initializations;
- **stage 3a**, the whole selection re-run independently for each of the three
  architectures.

Selected for the gated model: `lr = 0.03`, `T = 4`, `lambda = 0.25`,
`clip = 10.0`. The two controls get `lr = 0.01` and no clip, by the same rules.
Ablations share the gated model's row exactly.

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

Regenerate every piece of evidence (about eight minutes on the reference CPU):

```bash
python -m research.aaa_1k fingerprint
python -m research.aaa_1k select      --output docs/evidence/aaa_1k_development_selection.json
python -m research.aaa_1k characterize \
    --selection docs/evidence/aaa_1k_development_selection.json \
    --output docs/evidence/aaa_1k_characterization.json
python -m research.aaa_1k round2 \
    --selection docs/evidence/aaa_1k_development_selection.json \
    --characterization docs/evidence/aaa_1k_characterization.json \
    --output docs/evidence/aaa_1k_evaluation_round2.json
python -m research.aaa_1k adversarial-probes \
    --selection docs/evidence/aaa_1k_development_selection.json \
    --output docs/evidence/aaa_1k_adversarial_probes.json
python -m research.aaa_1k report2 \
    --evidence docs/evidence/aaa_1k_evaluation_round2.json \
    --selection docs/evidence/aaa_1k_development_selection.json \
    --characterization docs/evidence/aaa_1k_characterization.json \
    --output docs/aaa_1k_report.md
```

Independently recompute the headline statistics from the retained primitives,
without rerunning a model:

```bash
python -m research.aaa_1k recompute --evidence docs/evidence/aaa_1k_evaluation_round2.json
```

It rebuilds 99 stored values — Q1, Q3, Q4, every adaptation and retention
trial, every per-family mean and the stability counters — from the cells beside
them and exits non-zero on any disagreement. The same command on the superseded
round-1 file rebuilds its 59.

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

## 10. Known limitations, ranked by how much they should move confidence

Round 1's top four entries were defects and are now repaired; see
`aaa_1k_self_review.md` and `AAA-153` through `AAA-156`. What remains:

1. **Q4 is inconclusive overall but negative on the memory families.** Gating
   does not clearly help, and looks actively unhelpful where memory matters —
   but the ungated control is the less stable architecture, diverging at a
   learning rate of 0.1 where the gated model survives to 0.3. Three claims,
   two measured.
2. **Q4's mean and median disagree in sign.** The gated model is slightly better
   on 117 of 144 streams and much worse on the rest. That structure wants a
   distributional analysis this phase does not have.
3. **The two memory benchmarks were designed by the implementer whose model
   they evaluate**, and reviewed by nobody. The decomposition probe addresses
   `coarse_speed_v1` — with the speed fixed the recurrent model is *worse*, so
   the family genuinely tests regime inference — but `occlusion_v1` has no
   equivalent check.
4. **Q6 is weak.** Mean rank correlation 0.44; 106 of 720 cells have a monotone
   quintile table; the head over-predicts.
5. **The missingness code is weak.** During a gap inputs 2 and 3 are both
   exactly zero, which a genuine zero displacement and zero error would also
   produce. A clean flag needs a fourth input and a new architecture (`D-5`).
6. **Five initializations is a small second bootstrap level**, and the crossed
   interval is itself estimated from five points.
7. **Beating `rls_online` on `coarse_speed_v1`** is beating it outside its
   declared operating envelope.
8. **TBPTT's truncation bias is real but measured to be negligible here**, and
   this benchmark does not exercise long-range credit assignment at all
   (`D-12`).
9. **No generalization claim.** Evaluation streams are held out from
   development and from round 1, which supports a claim about unseen
   trajectories of the same families only. No unseen family was tested.

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

## 13. The three concerns round 1 could not settle, and what settled them

All three were open in the round-1 handoff. Each now has a measurement.

- **Does `coarse_speed_v1` measure memory or tolerance to quantization?**
  Settled: memory. With the speed held fixed — the quantizer alone — the
  recurrent model is 6.32e-04 *worse* than the stateless control, and becomes
  better only once the speed starts switching.
- **Is the ungated control's win an artefact of a shared learning rate?**
  Settled: largely yes. Tuned separately by the same rules, the comparison is
  inconclusive, and the ungated arm was winning at a rate its own stability
  rule forbids.
- **Is the 22% clip rate masking a problem?** Settled: it was the problem. The
  asserted threshold cost 29% of development error. It is now selected, per
  architecture, and fires on under 1% of updates.

Concerns I still cannot settle are in section 10; the sharpest is that I
designed the benchmarks my own model is measured on.

## 14. What I did not do

Self-approve, merge, regenerate historical evidence, modify anything under
`aaa/` or `results/`, weaken an existing test, add an exclusion to preserve an
old hash, implement growth or replay, or claim that any of this is a step
toward general intelligence.
