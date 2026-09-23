# AAA-1K v2 independent-review handoff

Everything a reviewer needs to check `aaa.1k.v2` without trusting its author.
Read the [self-review](aaa_1k_v2_self_review.md) first. It lists where the
implementer thinks the work is weakest, and it is not independent.

## 1. Identity

| Item | Value |
|---|---|
| Phase | `aaa.1k.v2`, protocol `aaa.1k.v2.protocol.v1` |
| v2 scientific fingerprint | `e5b11bdba08c6db77a1b2fb494bbea26b3a6c5352b54523c166c9788b59026eb` (74 files; `python -m research.aaa_1k_v2 fingerprint`) |
| `aaa.1k.v1` fingerprint | `5ce6e019bd5389c5d8d466cc2be809ff9777e3bd6acba10917bc6a7ce98f771e`, unchanged |
| Freeze | `docs/evidence/aaa_1k_v2/freeze.json`, sha256 `1d9ee722...`, source commit `47cec71`, committed in `94a0703` |
| Claim commit | `65a3bb1`: every confirmation block spent by `confirmation:1d9ee72295642a0f` before observation |
| Confirmation | `confirmation.json`, sha256 `30db7fbb...`, outcome `NO_CHALLENGER` |
| Final 1K system | Champion 1, unchanged (`research/aaa_1k_loop/champion1.py`) |
| Evidence platform | FLOWBOX, AMD Ryzen 7 5700G (Zen 3), CPU float64, 8 workers ([hardware](hardware.md)) |

Stage commits: development `ac905e7` → `2fc01a5`; diagnostics `2fc01a5` →
`3bd6047`; qualification `0eb0f2a` → `47cec71`; confirmation `94a0703` →
`2f96faf`; capacity `2f96faf` → `c9f654a`. Each stage ran from a detached
`git worktree` at the first commit (V2-D14). Its artifact was committed in the
second.

## 2. What to read, in order

1. [Research brief](aaa_1k_v2_research_brief.md) and
   [benchmark protocol](aaa_1k_v2_benchmark_protocol.md): what was
   preregistered (fingerprinted).
2. [Decisions](aaa_1k_v2_decisions.md) V2-D1 to V2-D17, especially V2-D15 (why
   there is no challenger) and V2-D17 (freeze to capacity).
3. [Report](aaa_1k_v2_report.md), generated from the evidence.
4. [Compute report](aaa_1k_v2_compute_report.md) and
   [compute strategy](aaa_1k_v2_compute_strategy.md).
5. [Self-review](aaa_1k_v2_self_review.md), then `AAA-179` and `AAA-180` in
   the [issue ledger](issue_ledger.md).
6. [Architecture](aaa_1k_v2_architecture.md),
   [external benchmarks](aaa_1k_v2_external_benchmarks.md),
   [literature review](aaa_1k_v2_literature_review.md).

## 3. File inventory

| Path | Role |
|---|---|
| `aaa/compute/` | device grammar, backend provenance, hardware probe |
| `research/aaa_1k_v2/cores.py`, `engine.py`, `adapters.py`, `runner.py` | batched NumPy/CuPy online learner (GRU, Elman, MGU, MLP, LRU with exact RTRL) |
| `research/aaa_1k_v2/arms.py` | Champion 0/1, candidates, controls, ablations; `audit` counts every adaptive-state scalar |
| `research/aaa_1k_v2/stress.py` | the 26 stress families and their roles |
| `research/aaa_1k_v2/external/` | NARMA, Mackey-Glass, dysts subset (vendored metadata), Monash (checksummed) |
| `research/aaa_1k_v2/identities.py`, `plan.py` | 64-bit identity registry, disjointness proof, the declared plan |
| `research/aaa_1k_v2/tournament.py`, `diagnostics.py`, `attack.py`, `confirmation.py`, `capability.py` | stages |
| `research/aaa_1k_v2/freeze.py`, `recompute.py` | freeze manifest and verification; independent recompute |
| `research/aaa_1k_v2/qualify.py`, `capacity.py`, `report.py` | compute qualification, capacity diagnostic, compute report |
| `benchmarks/aaa1k_v2_identity_registry.json` | 270 blocks, 2,162 seeds, with status and observer |
| `docs/evidence/aaa_1k_v2/` | every v2 artifact |
| `tools/write_aaa_1k_v2_report.py` | phase report generator (post-freeze, outside the fingerprint) |
| `tools/reproduce_aaa_1k_v2_confirmation.py` | confirmation rerun and cell-by-cell comparison |
| `tests/test_aaa_1k_v2.py`, `tests/test_compute.py` | unit, parity and pipeline tests (fingerprinted) |
| `tests/test_aaa_1k_v2_evidence.py` | retained-evidence guard (post-freeze) |
| `requirements-cuda-lock.txt` | the optional CUDA environment |

Raw per-grid-point development archives live outside the repository. Their
hashes are recorded in `development.json`.

## 4. Commands

```bash
# identity and accounting
python -m research.aaa_1k_v2 audit
python -m research.aaa_1k_v2 fingerprint
python -m research.aaa_1k_v2 registry
python -m research.aaa_1k_v2 prove-fresh
python -m research.aaa_1k fingerprint
python -m research.aaa_1k_loop.champion1 verify

# evidence
python -m research.aaa_1k_v2 recompute --freeze docs/evidence/aaa_1k_v2/freeze.json \
    --confirmation docs/evidence/aaa_1k_v2/confirmation.json
python -m unittest tests.test_aaa_1k_v2_evidence
python tools/write_aaa_1k_v2_report.py --check

# rerun the confirmation (about 25 minutes on 8 cores; Monash data under $AAA_DATA_ROOT)
python tools/reproduce_aaa_1k_v2_confirmation.py --workers 8

# compute (optional CUDA environment)
python -m aaa.compute probe
python -m research.aaa_1k_v2 qualify --output /tmp/qualification.json
```

`docs/reproduction.md` has the full stage sequence.

## 5. What to challenge

Ranked by how much they should move confidence:

1. **The stability rule decided the tournament** (V2-D15, SR-5). Judge whether
   a margin measured on unclipped references was a reasonable preregistration,
   and whether "no challenger" should be read as narrowly as the report reads
   it.
2. **Round 3's adaptation result did not replicate** (SR-6). The batched
   capability designs have no cell-for-cell parity check against
   `research/aaa_1k/measurements.py`. Running the v2 design on round 3's
   identities would settle whether this is a failure to replicate or a design
   difference.
3. **`AAA-180`**: K5's Monash handling. It had no effect here and must be
   fixed before any future promotion.
4. **The hand-written interpretive sentences** in the report generator (SR-4).
5. **Monash comparability** (SR-10) and the name-only `aus_elec_demand`
   mapping.
6. **Cross-backend parity** (SR-8): numerical agreement for stable
   configurations, verdict-level agreement for chaotic ones.

## 6. What I did not do

- No independent review: this handoff is for one.
- No remote claim ref for the confirmation batch (an outward push).
- No attack stage: there was no challenger.
- No CUDA stage in formal evidence: the qualification shows the CPU is the
  right processor at these batch sizes.
- No repair of `AAA-180` or of `report.py`'s docstring, which names a
  `phase_report` that was never written: both are in frozen source.
