# Observation-noise v1.1 Sol review handoff — PR #11

This is the current AI-review handoff for the separately versioned
`aaa.observation_noise.v1.1` phase. It supersedes the old v1 handoff as an
active instruction, while the prior v2.1 review remains preserved in Git
history and its dedicated evidence. This document is not human review,
scientific confirmation, merge approval, or release approval.

## Frozen identities

| Item | Value |
|---|---|
| PR | [#11](https://github.com/Cinqic/AAA/pull/11) |
| current local head | `5690aa94802a4f1e4bd3471618a456f45e8d28d2` |
| scientific source commit | `c0c3b0645cef3b94f82a104b728c7c56e1959499` |
| source dirty when frozen | `False` |
| protocol | `aaa.observation_noise.v1.1` |
| protocol hash | `546e2434cc15850779107c1a329af1e2f8581cd6a1c3b31807eecf5f6e7b427c` |
| scientific fingerprint | `369080bf21a43014561322d19e42c288d68ae9ccbc486b04f4d6791778c62311` |
| v2.1 reference commit | `25b6c32c9040d0f934314a2139993d12763afc99` |
| v2.1 raw / resolved hash | `4993c5e6e173f9dd5ef002bc84ff4c484da853b066ea45662d41ad826d10d48e` / `f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37` |
| dependency lock | `2aa52a7deefd05403b9fb6441bef42e13e47a4384baaf780f3b4b879bfcf3bc7` |
| selected candidate | `incumbent-no-refinement-v1` |
| candidate configuration | `dcc82cb0396985976419733a817d02027c45683db4f76d3ea37ac48290f6c649` |
| confirmation freeze | prepared at `369080bf21a43014561322d19e42c288d68ae9ccbc486b04f4d6791778c62311`; not authorization to run |

## Independent findings and repairs

Sol reproduced and retained the failures recorded as `AAA-135` through
`AAA-148` in `docs/issue_ledger.md`. The repairs include mandatory complete
checksums, finite comparisons, ratio-of-means adaptation, complete selection
constraints, fixed formal draw/batch identity, atomic cross-clone reservation,
bounded-memory sharded evidence, isolated pinned-reference replay, causal
prediction intervals, vectorized hierarchical bootstrap, and archive-path
containment. Negative and interrupted probes remain retained.

## Development selection

The corrected full 2 x 2 x 1 search completed with `quick: false`.
Eligible refinements: `[]`. Selected: `incumbent-no-refinement-v1`. No-refinement
remained possible and won because none of the three clipping mechanisms met
every preregistered utility, baseline, retention, adaptation, stability,
practical-gain, and adjusted-evidence condition. This is a development
decision only; it establishes no formal endpoint.

Each of four candidate archives contained 235,776 scored records,
1,024 trials, and 3,552 training
records and independently verified `PASS` before selection.

## Scale and archive boundary

The old quick layout peaked at 2,731,806,720 bytes RSS.
The repaired sharded pilot retained 58,944 records, peaked at
155,054,080 bytes RSS, occupied 47,281,979 bytes,
and independently verified. Timing-neutral primitive digests and computed
metrics matched the prior streaming implementation exactly.

One formal batch is exactly 59,043,840 scored records
and is projected at about 35.9 GiB compressed,
plus schedules and metadata. There is no approved immutable object-store, Git
LFS, or other durable locator and no demonstrated independent retrieval.
Consequently neither fresh batch has been observed:

| Batch | Role | Status |
|---|---|---|

## Review and scientific verdict

**Engineering review verdict: BLOCKED for completion and normal merge.**

**Scientific outcome: NOT ESTABLISHED.** No A/B observation, durable upload,
fresh-location retrieval, joint recomputation, or primary endpoint decision
exists. Missing confirmation is not an unfavorable or inconclusive result.
The exact blocker is `AAA-144`: an approved immutable destination with at
least roughly 80 GiB for the two projected compressed record streams, plus
schedule/metadata overhead and practical retrieval headroom.

The code and development evidence may be pushed to PR #11 for review and CI,
but the brief forbids merge until the full confirmation and durable retrieval
requirements are actually satisfied. No tag or release is authorized.

## Reproduction entry points

```bash
python -m aaa.cli observation-noise-protocol-hash
python -m aaa.cli observation-noise-fingerprint
python -m aaa.cli observation-noise-development-select --reuse-root <selection-root> \
  --output docs/evidence/observation_noise_development_selection.json
python -m aaa.cli observation-noise-recompute <attempt>
python -m aaa.cli observation-noise-confirmation-evaluate <A-archive> <B-archive> \
  --output docs/evidence/observation_noise_joint_evaluation.json
```

The formal runner additionally requires the committed confirmation freeze, an
explicit attempt label, the declared batch, and successful atomic remote
reservation. Running it is intentionally deferred until durable storage and
retrieval are real rather than aspirational.
