# Evidence and artifact policy

The [backup policy](backup_policy.md) covers committed Git history and evidence
through a local Git bundle and a planned private off-device copy. It does not
change the raw-artifact caveats below.

## What lives in git

| Committed | Why |
|---|---|
| compact per-attempt `summary.json` and `report.md` | the result, readable without regenerating anything |
| `metadata.json`, `benchmark_spec.json` | provenance and the exact resolved specification |
| `checksums.json` | integrity of every byte retained in the compact archive |
| `full_attempt_checksums.json`, when present | identity of the complete local attempt, including omitted bytes; not clean-clone verification |
| `experiment_registry.json` | per-trial state, seeds and completion |
| selected checkpoints (a few kB of JSON) | the model the claim is about |
| `benchmarks/freeze_manifest.json`, `benchmarks/confirmation_batches.json` | what was frozen, and which batches were spent |
| `benchmarks/golden_seeds.json` | a fixture that detects accidental RNG-mapping changes |
| development evidence under `docs/evidence/` | diagnosis and candidate-selection tables, including unsuccessful variants |

## What does not

Compressed per-step record files (`raw/**/*.jsonl.gz`) and the plot images.
A full confirmation attempt writes roughly 100 MB of per-step records. The
previous repository committed tens of megabytes of generated JSON while calling
it compact evidence; this policy is the correction.

## How raw evidence stays recoverable

Raw step records are **semantically regenerable**, not archived, from four committed things:

1. public source at a recorded commit and tree hash;
2. the immutable specification, identified by hash;
3. the immutable seed mapping, keyed by batch identity;
4. the checkpoint lineage, identified by checkpoint hash.

```bash
git checkout <recorded commit>
python -m aaa.cli benchmark --role confirmation_a \
  --batch-id <recorded batch id> --reproduce --output-root runs
python -m aaa.cli recompute runs/benchmark-v2_1/<batch id>
```

`recompute` verifies byte checksums only when the files named by the manifest
are present. A regenerated run is then compared semantically (schemas, trial
identities, metrics and gates); byte equality is not promised for timestamps or
other runtime provenance. Byte integrity and semantic reproducibility are
deliberately reported as different claims.

The manual benchmark workflow (`.github/workflows/benchmark.yml`) retains
the full attempt directory of each run it makes for 90 days. That workflow
runs only development and high-replication roles (see below and `AAA-125`).
A formal confirmation attempt therefore has **no** CI copy. Its omitted raw
bytes survive only in whatever local copy the maintainer keeps, or by
regeneration.
Routine CPU CI keeps its smoke-run artifacts for 14 days (`AAA-178`).

`tools/archive_attempt.py` creates the compact Git archive. It omits `raw/` and
`plots/`, retains the original complete-attempt manifest under the explicit
`full_attempt_checksums.json` name, and writes a new `checksums.json` that covers
every retained file. Thus a clean clone can verify the compact bytes without
pretending the omitted bytes are present.

Formal confirmation is run only in the canonical maintained checkout. The
manual Actions workflow accepts development and high-replication roles, but
refuses confirmation batch IDs: independent ephemeral checkouts cannot safely
persist the repository's single durable batch claim. This prevents two remote
jobs from spending the same planned stream while both believe they own it.

## Known limitation

Regenerability is weaker than durable archival. For a formal confirmation
attempt there is no CI artifact to fall back on: losing the recorded commit
and the maintainer's local raw records would leave the summaries and
checksums without the bytes they describe. For a CI-run development or
high-replication attempt, it would take the recorded commit being lost *and*
the 90-day artifact expiring.

Git LFS, or an external archive of the omitted raw bytes with content addresses
recorded here, would be stronger. Neither raw-artifact mechanism is in place.
This remains a recommendation tracked as `AAA-077` in the issue ledger; a Git
bundle cannot supply bytes that were never committed.

The four historical v2.1 attempt directories committed before the Sol review
have checksum manifests that name omitted raw and generated files. In a clean
clone, those manifests therefore fail integrity verification because the named
bytes are absent. Their summaries remain historical evidence, not currently
byte-verifiable archives. This reproduced limitation is tracked as `AAA-124`.

## Failed attempts

A confirmation that records a required-gate failure is evidence and is kept on
exactly the same terms as one that passes: the same compact summary, report,
provenance, checksums and checkpoints. Its batch is retired permanently in
`benchmarks/confirmation_batches.json`.

This has happened. The round-1 v2.1 pair failed `always_online_stability` on
both fresh streams and is committed at
[`../results/benchmark_v2_1/`](../results/benchmark_v2_1/) alongside the
round-2 pair.

## Historical evidence

`results/final/` and `results/benchmark_v2/` are preserved unchanged. They are
historical provenance. See [`errata.md`](errata.md) for which of their claims
were affected and why.

## Observation-noise v1.1

The observation-noise phase is additive and separately addressed. Its protocol,
source-freeze manifest, candidate ledger, registry, attempt metadata, schedule
files, primitive records, plots, reports, and checksums live under the names
declared in [`observation_noise_protocol.md`](observation_noise_protocol.md).
An attempt retains deterministic per-trial `records/*.jsonl.gz` shards and a
canonical index rather than duplicate combined compressed and uncompressed
copies. The manifest hashes the concatenated uncompressed record bytes in
canonical trial/step order, while every shard also has a compressed-byte hash.
A compact summary without the full schedule and primitive archive cannot
support an independent-reproduction claim.

Noise records contain both latent truth and predictor-visible observations, but
the predictor-generation boundary receives only the latter. The presence of a
truth field in evaluator evidence is not permission for a predictor or
calibration callback to read it. A report must state whether a result is
engineering complete, scientifically supported, negative, inconclusive, or
blocked by missing evidence.

### Observation-noise completion boundary

The scientific identity used for a future confirmation freeze is the output of
`observation-noise-fingerprint`, not a commit/tree equality check. It covers
the scientific source map, including the protocol, candidate ledger and plan,
runner, statistics, verifier, tests, lock and relevant documentation. It
normalizes only mutable registry lifecycle fields and fails closed on
nonignored untracked scientific files, unsafe paths and symlinks. The exact
confirmation freeze, design/source freeze, result archives and review handoff
are generated provenance and are excluded so their own commit does not create a
self-reference.

The committed development selection evidence is
[`evidence/observation_noise_development_selection.json`](evidence/observation_noise_development_selection.json),
and the full candidate ledger retains every attempted identity and outcome.
The selected result is the unchanged incumbent control. Four full 2 x 2 x 1
development archives were independently verified before that decision. Their
locator is still explicitly transient local evidence; it is not a durable
confirmation archive.

The joint command
`python -m aaa.cli observation-noise-confirmation-evaluate <A> <B>` is the only
supported route to a final A+B conclusion. It independently verifies both
primitive archives, checks shared training/checkpoint and candidate identity,
recomputes the endpoint family, records Holm order and adjusted bounds, and
does not trust a stored summary conclusion. No external durable archive
mechanism is approved or recorded. The dedicated Cinqic HDD is verified local
working storage only; it is not immutable, off-site, or an independent failure
domain. External archival is publication-grade evidence rather than an
engineering merge gate for this internal phase. Formal A/B remains unobserved
for a separate reason: the large replication budget has no quantitative
precision justification. Its declared v1.1 batches were retired unobserved.
Any successor must be versioned and receive fresh batch identities before
observation, and any claim of publication-grade durability still requires a
genuine external archive plus independent retrieval.

## AAA-1K

The phase commits its development selection (every attempted configuration,
including the eliminated ones and their scores), its held-out evaluation (the
per-cell primitives every headline statistic is recomputed from), its
development characterization probes, and its adversarial probes.

**Rounds 1 and 2 are retained, superseded.** Round 1 contained four design
defects. Independent review then found round-2 Q1 identification and Q2/Q5
uncertainty defects. Round 3 uses fresh identities. Earlier evidence remains at
`aaa_1k_evaluation_round1_superseded.json`,
`aaa_1k_report_round1_superseded.md`,
`aaa_1k_evaluation_round2_superseded.json`, and
`aaa_1k_report_round2_superseded.md`. A
round that produced a misleading result is evidence on exactly the same terms as
one that did not — particularly this one, since the largest defect inflated a
headline effect thirty-fold and was caught by reading a per-arm table.

Together the three evaluations are about 8.8 MB of JSON: 4.35 MB current round
3, 3.21 MB superseded round 2, and 1.23 MB superseded round 1. That is the price of
independent recomputation without rerunning a model, and it is paid
deliberately: the per-cell table is what exposed the stateless control's
instability that a summary had already averaged away.

It does **not** commit dashboard images or per-step traces. A 994-parameter
model has no excuse to generate another multi-gigabyte archive, and it does
not: the current evaluation is deterministic and takes about five
minutes of CPU, so regenerating a trace is far cheaper than storing one.

The committed evidence is sufficient for independent recomputation without
rerunning a model:

```bash
python -m research.aaa_1k recompute --evidence docs/evidence/aaa_1k_evaluation_round3.json
```

Failed and eliminated configurations are retained on the same terms as
successful ones. The four development configurations that scored best and were
then eliminated by the stability margin are in `stage_one_eliminated` with
their scores intact, which is the only way a reader can check what the rule
cost.

## AAA-1K improvement loop

The original v0 pilot (iterations 0001--0003) writes strict-JSON artifacts under
`docs/evidence/aaa1k_loop_000N/`, holding the per-cell primitives its verdicts
are computed from, and each iteration has a record (`iteration.json`) that
references its artifacts by SHA-256. Together they are about 6 MB, most of it
diagnosis round 1 (2 MB) and the first attack (0.7 MB). Failed and rejected
candidates are kept on the same terms as anything else: they are most of the
evidence. Per-step traces are not committed; the historical pilot's
`python -m research.aaa_1k_loop reproduce <stage>` path retains its original
exact comparison.

Iterations 0004--0006 add roughly 27 MB because the retained diagnostic,
screen, attack and confirmation primitives support independent adjudication of
TBPTT semantics, two falsified M2 explanations, the frame-lock mechanism and
the first promotion. These observed artifacts, including the aborted attempt-1
record and the original `recomputation_2.json`, are immutable. The current
`recompute4` path is stricter than that historical recomputation artifact and
compares underlying numbers as well as statuses.

`python -m research.aaa_1k_loop.stages4 reproduce <stage>` reproduces the nine
post-audit stages. `--exact` requires bitwise agreement on the matching Zen 3
evidence platform. Default cross-platform reproduction requires the same cell
identities, structure, divergence classifications and adjudicated verdicts,
and reports numerical drift in chaotic/diverged trajectories (`AAA-173`).
This distinction does not weaken artifact hashes: stored evidence bytes remain
bound by SHA-256. The identity ledger,
`benchmarks/aaa1k_loop_identity_ledger.json`, is append-only in the sense that
matters: blocks never overlap and a spent block never becomes usable again.

## `aaa.python.v0`

| Committed | Why |
|---|---|
| `docs/evidence/aaa_python_v0/development.json` | provenance captured before the run, the plan, parameter accounting, trained-state hashes, cell-level primitives (counts, calibration bins, per-class recall counts) and the summary derived from them |
| `docs/evidence/aaa_python_v0/development_records.jsonl.gz` | every scored action (27,840): arm, task identity, answer, recorded truth, confidence, abstention, correctness, update flag; 373 KB |
| `docs/aaa_python_development_report.md` | generated from the evidence; `--check` guards it |
| `research/aaa_python/data/golden_answers_v0.json` | cross-version answer keys for 100 development tasks |

**Regenerable, not committed:** the task programs themselves (a pure function
of identity and generator version), trained checkpoints (42 MB, deterministic
from the plan) and quick smoke runs. Regenerate them from the recorded commit
([reproduction](reproduction.md)); put large outputs under `$AAA_DATA_ROOT`.

**Derivable, never trusted:** `recompute` re-aggregates cell counts from the
records, recomputes the summary from the cells, regenerates every scored task,
re-executes it with CPython and re-derives its answer with its own mapping.
Any disagreement fails. The strict JSON writer rejects NaN and infinity.
Reports are generated from primitives; report prose is never the only copy of a
result.

**Development only.** This evidence is labeled development evidence. v0
generates no confirmation identities, and any later confirmation must be frozen
and claimed before observation, under a successor protocol.

## `aaa.python.v1`

The v1 evidence directory retains ten development stages, the superseded
censored encoder diagnostic, the committed `freeze.json`, the spent
`confirmation.json`, the compute benchmark, and pre-design diagnostics.
The [generated report](aaa_python_v1_development_report.md) names the stage
files, producing commits, source fingerprints, and measured outcomes. A stage
summary is regenerated from per-task correctness bits with `summarize` and
independently recounted with `recompute`. The post-freeze
`tools/check_aaa_python_v1_decisions.py` additionally checks stored Holm,
capacity, and promotion verdicts against those bits and the committed freeze.

The freeze was committed at `a73765b` before the confirmation file first
appeared at `7c46e48`. Its held-out identities have been observed and cannot
be reused as fresh evidence. Fingerprinted scientific source and protocol
files are retained at that identity; later audit tools and interpretation
errata sit outside it. The v1 evidence files are covered by the protected-file
manifest.

## Protected historical identities

`benchmarks/protected_identities.json` records the dot-era identities and the
SHA-256 of 493 retained files, including the new Python evidence.
`python tools/check_protected_identities.py`
fails on any change. New evidence may be added; retained evidence may not
change.
