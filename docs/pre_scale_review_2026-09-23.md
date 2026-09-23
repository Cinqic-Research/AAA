# Independent pre-scale review, 2026-09-23

## Starting state and scope

The remote `main` at review start was
`8cb752221ee2bbe0ceb8da361ba607426e868a55` (tree
`ecb77b8f610cf990599399dfe59eb00449b156d0`), with a clean working tree.
The only open PR was [#26](https://github.com/Cinqic/AAA/pull/26), head
`8f43da723fb26bd893edb2bb605d5af2b01a7256`, base `main`. Its two
documentation commits were checked against the retained v2 results; exact-head
CPU CI was green, no review threads were open, and it was merged with a merge
commit as `a372a759b60031d475eb90edc91af61e9d814f4a`. GitHub refused an
approval from the authenticated PR owner, so this is a documented independent
technical review, not a GitHub approval. The closure branch began from that
merged `main`.

At start, the remote carried four loop confirmation-claim refs for iteration
0006 (`confirmation-init`, `confirmation-env`, `confirmation-2-init`,
`confirmation-2-env`) and the existing pre-next-phase tags. The benchmark
v2.1 specification hash is
`f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37`;
the observation-noise v1.1 protocol hash is
`546e2434cc15850779107c1a329af1e2f8581cd6a1c3b31807eecf5f6e7b427c`.

The starting tree had 690 tracked regular files: source, protocols, tests,
workflows, current and retained evidence, historical and superseded results,
and generated reports as classified by `docs/final_audit.md`. The final
inventory has 694 tracked regular files (262 active, 392 retained evidence,
10 current evidence, 24 historical, 2 superseded evidence, 4 generated), with
an exact match to the tracked path set and SHA-256 bytes. All
tracked JSON parses with non-finite constants rejected, all tracked Python
parses as AST, and no tracked symlink was found. Large evidence was assessed by
schema, identity, recomputation and sampled primitives rather than by reading
each floating-point value. The source review concentrated on claim-critical
learning, decision, identity, report, compute, packaging and automation paths;
it does not turn routine unit coverage into independent scientific evidence.

## Findings against the starting tree

| Finding | Consequence | Disposition |
|---|---|---|
| `AAA-180`: K5 omitted Monash, and the two implementations disagree when Saugeen contributes one series | High severity for a future promotion; no v2 decision changed because there was no challenger | Frozen v2 preserved; hard successor precondition in [scaling readiness](scaling_readiness.md) |
| Round-3 positive Q2 versus v2 fresh-identity non-replication | Earlier adaptation claim cannot be generalized | Replayed all 120 historical cells through both designs; every branch MAE and effect matched exactly; keep both results and the fresh-evidence limitation |
| V2 stability margin used unclipped references and capped candidate rates | The no-challenger result is bounded by the preregistered search | Verified rule and development ledger; future actual-configuration stability rule must be frozen before observation |
| `AAA-181`: static report interpretation used absolute wording beyond its cells | Prose overstatement and an unisolated Monash mechanism; no primitive or verdict changed | Narrowed wording; generated the error-head interval, Monash comparison lists and capacity range from computed values |
| Current 125M ceiling prose and no canonical coding specialization | Current project guidance was stale or absent | Added Python-first direction and the revisable ~105M AAA 1 planning goal outside frozen scientific source |

The `AAA-180` control flow was reproduced directly: frozen
`run_confirmation` calls `decide` without `monash_primitives`; the report's
descriptive cross-check passes them and finds six K5 disagreements. Saugeen
has eight initialization cells but one distinct series. The primary crossed
bootstrap returns `INSUFFICIENT_EVIDENCE`, while the independent count-weighted
bootstrap resamples that fixed series and yields `FAIL`. A future single-series
interval must declare whether it conditions on the observed series. It may
not be chosen to make a candidate pass. The current frozen result remains
`NO_CHALLENGER`.

The Q2 diagnostic uses already observed round-3 identities only. It reran
the historical 5-by-24 grid and both batched Champion 0 and Champion 1
measurement paths. Maximum absolute differences for changed/unchanged,
online/frozen MAEs and the difference-of-differences were all **0.0**.
The original mean effect was `+0.0005120513746349793`; v2's fresh Champion 1
effect is about `-1.8e-05` with a 95% interval crossing zero. The measured
paths are equivalent on the historical cells. This is diagnostic
characterization, never fresh confirmation evidence.

An additional direct crossed resampling of the retained v2 capability cells
(8 initializations x 16 streams, 10,000 draws, diagnostic seed 20260923)
gave adaptation mean `-1.8139925e-05`, interval approximately
`[-4.03e-04, +4.30e-04]`, and retention forgetting mean `-2.0236225e-04`,
interval approximately `[-4.08e-04, -1.50e-05]`. This independently checks
the signs and statistical resolution; it does not replace the frozen
bootstrap identities or the stored decision.

The stability margin was present at protocol commit `621e2a9`, before the
development artifact at `2fc01a5`. The later change to `plan.py` added
diagnostic-block enumeration and left the margin unchanged. The development
ledger shows the unclipped divergence boundary at lr 0.1 for five candidate
families, capping eligibility at 0.01; Elman's boundary is 0.03, capping it
at 0.003. Every selected configuration follows that rule. Excluded clipped
development observations remain excluded.

The exploratory capacity ledger has five of the 26 declared internal families
meeting its >5% and interval-below-zero rule at ~4K, below the one-third
`NOT_CAPACITY_LIMITED` boundary. The four coarse-observation family point
changes at ~4K are about -1.2%, +0.5%, -0.1% and -1.8%, each with an interval
crossing zero. An external Mackey-Glass comparison also improves, but external
tasks are outside the declared 26-family capacity verdict. The present dot
weakness therefore supplies no evidence for a 105M jump.

## Changes and source boundaries

This review changed current guidance in `README.md`, `docs/hardware.md`,
`CHANGELOG.md`, `docs/limitations.md` and the issue ledger; added
`docs/research_direction.md` and `docs/scaling_readiness.md`; and added the
reproducible diagnostic `tools/check_adaptation_parity.py` to CPU CI. The
current v2 report generator's interpretive sentences were narrowed and its
generated report refreshed. The generator is outside the frozen v2 source.

The v1 charter and phase source, v2 protocol and phase source, dependency
locks, Champion 0/1 implementation, and every historical or current raw
scientific artifact were intentionally left unchanged. The v1 fingerprint
remains `5ce6e019bd5389c5d8d466cc2be809ff9777e3bd6acba10917bc6a7ce98f771e`;
v2 remains `e5b11bdba08c6db77a1b2fb494bbea26b3a6c5352b54523c166c9788b59026eb`.
The repository-wide observation-noise source fingerprint necessarily moves
with new current documentation and CI files. Historical observation-noise
identity and unexecuted formal status are not relabelled as current evidence.

## Verification and limitations

The local Python 3.12.3 environment matched `requirements-lock.txt` SHA-256
`2b5a23094b0cd99df66fa534b2e979a7a3e8c8b393df71cd53f1345281ab45b6`:
NumPy 2.5.3, scipy-OpenBLAS 0.3.34.106.0, Ryzen 7 5700G (Zen 3).
Ruff lint and format, mypy, the full unittest suite with branch coverage,
lock and exit-code checks, benchmark and observation-noise protocol identities,
AAA-1K parameter audit and full gradient check, both phase fingerprints,
Champion 0/1 integrity, loop recomputation, v2 identity-disjointness proof,
v2 independent recomputation, report regeneration, and the Q2 diagnostic
passed locally. The v2 recompute agrees with the stored `NO_CHALLENGER`.
The full gradient check covered 4,918 individual parameters across five
invocations with no violation. V2 accounting reports 994 trainable and 1,414
total adaptive-state scalars for Champion 1.

A wheel was built and installed into a separate Python environment on the
dedicated HDD. From outside the checkout, the installed CLI found the
benchmark and observation-noise specifications, and v2 package data existed.
The optional CUDA lock SHA-256 was
`aa12fceae9b2ae72e1a4e6215b2f4f1d3981b2f61dfa591ec66dd43342ceb9f6`;
its separate environment installed all 30 pins and resolved the RTX 2060 via
CuPy 14.2. Fourteen compute tests passed there. A live rerun of the
qualification parity fixture found Champion 1 CPU/CUDA relative MAE
differences at most `1.22e-15` across four families, with 16/16 deterministic
cells on each backend and matching failure status. Elman and LRU unstable
conditions again diverged materially across backends (up to about 0.32 and
1.05 relative MAE respectively), while repeat runs on each backend were
bitwise stable. This does not turn the CPU confirmation into GPU evidence.
An explicit request for nonexistent `cuda:999` exited 2 with a device error;
it did not fall back to CPU.
GitHub's exact-head PR check state and the full FLOWBOX confirmation replay
are recorded in the final PR, because their identity is the final Git commit
and the replay completion may follow document authoring. The replay compares
all retained primitives; its bitwise result is separate from the verdict.

The current evidence does not demonstrate Python coding, justify 105M, or
remove `AAA-180`. The proper readiness verdict is
**`READY_FOR_NEXT_PHASE_DESIGN`**, with `AAA-180` blocking any new promotion.
The author of these documentation/tool changes also performed their local
validation; that validation is adversarial self-review of the changes, not an
independent post-change confirmation. Exact-head CI and any separate reviewer
remain distinct sources of evidence. The final PR body and its post-merge
comment record the exact head SHA, merge SHA, required checks and verified
`main` state; a file inside the commit cannot contain its own final Git SHA.
