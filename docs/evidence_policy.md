# Evidence and artifact policy

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

CI additionally retains the full attempt directory for 90 days.

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

Regenerability is weaker than durable archival. A sufficiently unlucky
combination — the recorded commit lost and the CI artifact expired — would
leave the summaries and checksums without the bytes they describe.

Git LFS, or an external archive with content addresses recorded here, would be
stronger. Neither is in place. This is stated as a recommendation rather than
described as done, and it is tracked as `AAA-077` in the issue ledger.

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

**Round 1 is retained, superseded.** It was completed, probed, and found to
contain four design defects; all four are repaired in round 2 on fresh stream
identities. Round 1's evidence and its rendered report are kept unchanged at
`aaa_1k_evaluation_round1_superseded.json` and
`aaa_1k_report_round1_superseded.md`, with a banner naming what was wrong. A
round that produced a misleading result is evidence on exactly the same terms as
one that did not — particularly this one, since the largest defect inflated a
headline effect thirty-fold and was caught by reading a per-arm table.

It does **not** commit dashboard images or per-step traces. A 994-parameter
model has no excuse to generate another multi-gigabyte archive, and it does
not: the entire evaluation is 55,040 scored transitions and about 73 seconds of
CPU, so regeneration is cheaper than storage by a wide margin.

The committed evidence is sufficient for independent recomputation without
rerunning a model:

```bash
python -m research.aaa_1k recompute --evidence docs/evidence/aaa_1k_evaluation.json
```

Failed and eliminated configurations are retained on the same terms as
successful ones. The four development configurations that scored best and were
then eliminated by the stability margin are in `stage_one_eliminated` with
their scores intact, which is the only way a reader can check what the rule
cost.

