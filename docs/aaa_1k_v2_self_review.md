# AAA-1K v2 adversarial self-review

This is the implementer's own attempt to break the `aaa.1k.v2` results. It is
**not independent review**. The same agent designed the protocol, wrote the
code, ran every stage and wrote this document. Read it as a list of places to
push, not as approval.

The result under review: no challenger. Champion 1 remains the final 1K system.
The confirmation was characterization-only (`NO_CHALLENGER`, recomputed), and
the capacity verdict is `NOT_CAPACITY_LIMITED`
([report](aaa_1k_v2_report.md), [decisions](aaa_1k_v2_decisions.md)).

---

## Defects found

### SR-1. The confirmation's K5 omitted Monash, and its two implementations disagree (`AAA-180`)

Found by the report tool, which applies the frozen K1-K5 criteria to every arm
with both `confirmation.decide` and the independent `recompute`. The
confirmation passed no Monash primitives to `decide`. With them included,
`decide` says `INCONCLUSIVE` and `recompute` says `FAIL` for every Monash arm:
`saugeen` is a single series, which the shared bootstrap cannot resample.

**Effect here:** none, because no challenger existed. **Effect in a
promotion:** K5 would have been decided without Monash, and the recompute's
point check would then have refused to agree, which fails safe. Not repaired,
because the v2 source is frozen. A later phase must fix it before relying on
K5.

### SR-2. Provenance recorded after the run (`AAA-179`)

The first development run captured its commit and fingerprint after running,
and spawned workers re-imported edited code mid-run. It was stopped before it
produced evidence. Every formal stage since then ran from a pinned worktree
with provenance captured first. The rerun reproduced the aborted run's one
completed archive byte for byte.

### SR-3. A commit that claimed a change which never took effect

`0eb0f2a` ("representative size grid for non-online qualification workloads")
added a `WORKLOAD_SIZES` table that nothing read. The qualification it pinned
ran the full size grid for every workload. `47cec71` removed the dead constant
and says so. No number was affected, but the commit message is false as
written, and history is not rewritten.

### SR-4. Two hand-written report sentences contradicted the data in their first draft

In its first draft the report generator said that "neither run resolves
forgetting", but the v2 retention interval is entirely below zero. It also
called the error head "a calibrated uncertainty signal", while the head's
mean rank correlation with realized error is 0.30. Both passages are now
computed or rewritten to match. Other interpretive sentences in
`tools/write_aaa_1k_v2_report.py` are still static text: the M1 reading of the
family table, the Monash interpretation, the capacity interpretation and the
error-head interpretation. **A reviewer should check each of them against the
tables they sit under.**

## Threats to the conclusion

### SR-5. Is "no challenger" an artifact of the stability rule?

Partly, and the documents say so. The preregistered margin was measured on
unclipped references and capped every candidate's learning rate below Champion
1's frozen 0.03. Under the same rule Champion 1's own configuration would have
been ineligible. Clipped configurations at lr 0.03-0.1 scored 0.84-0.97 on
development. The conclusion is therefore "none of the preregistered candidates
under the preregistered rule", not "no 1K model beats Champion 1". The
confirmation data narrow the gap only a little: the confirmed arms are the
rule-selected configurations, not the excluded ones.

### SR-6. Round 3's adaptation result did not replicate. Is the v2 capability design faithful?

`research/aaa_1k_v2/capability.py` re-implements round 3's Q1, Q2 and Q5 designs
in batched form, using the same generators and parameters. The batched
*learner* has a parity test against the historical implementation (within
1e-12, mostly bitwise). The batched *capability designs* do not: nothing
compares them cell for cell with `research/aaa_1k/measurements.py` on round 3's
identities. The non-replication (-1.8e-05 [-3.9e-04, +4.3e-04], against round
3's +5.12e-04 [+1.69e-04, +8.67e-04]) could be a real failure to replicate, or
a difference between the designs. **This is the most valuable thing an
independent reviewer could check:** run the v2 adaptation design on round 3's
paired-change identities and compare the result with round 3's primitives.

### SR-7. The attack stage and criterion A6 (CUDA re-run) never ran on real data

With no challenger, `attack.py` ran only in unit tests and smoke runs. Its
first real use will be in a later phase.

### SR-8. Cross-backend parity holds only for stable configurations

Gated arms agree across CPU and CUDA to about 1e-15. Elman and LRU arms at
lr 0.03 on long or wall-rich streams diverge by up to 105% relative MAE, while
failure classifications agree and each backend is bitwise deterministic run to
run. This is chaotic amplification, the same mechanism as `AAA-173`. Every
formal stage ran on the CPU, so no confirmed number depends on CUDA.

### SR-9. The descriptive comparisons use post-hoc bootstrap seeds

The report's per-family, held-out and capability intervals use indices 4-8 of
the frozen confirmation bootstrap block. The frozen criteria used indices 0-3.
The report tool chose these indices after the freeze. They are descriptive
intervals, not decisions. As a check, rerunning the per-family table with a
different seed from the same block changed none of the 286 significance
markers.

### SR-10. The Monash comparison is not like-for-like

AAA models learn online per series and forecast recursively. The published
methods fit offline, often as global models. `saugeen` is one series.
`aus_elec_demand` is matched to the archive's results row by name only. The
report calls this an out-of-domain check and claims no competitiveness.

## Engineering notes

- `confirmation.json` is 40 MB, the largest tracked file, below GitHub's 50 MB
  warning. It is kept because the recompute needs its primitives.
- The optional durable remote claim ref was not created: it is an outward
  push. The committed claim (`65a3bb1`) and the registry's fail-closed
  spent-block check prevent reuse in this repository, but not in a fork that
  drops the commit.
- The RTX 2060 also drives the desktop. The qualification numbers include that
  load.
- The planned `report.phase_report` was never written before the freeze. The
  phase report comes from a tool outside the fingerprint (V2-D17).

## What was checked and held

- `aaa.1k.v1` fingerprint `5ce6e019…` unchanged. Champion 0 and Champion 1
  verify. The loop's PROMOTE recompute still agrees.
- The live repository matches the v2 freeze (`freeze.verify` returns nothing;
  `tests/test_aaa_1k_v2_evidence.py`).
- All 2,162 v2 seeds are disjoint from every AAA-1K and loop seed
  (`prove-fresh`).
- Consistency: `gru_v1_retuned` at Champion 1's configuration scores exactly
  1.000 against Champion 1 in development. Champion 0 matches Champion 1 to
  0.1% on every family except the coarse, quantized and noisy ones, where the
  reach gate (their only difference) engages.
- Every confirmation arm had zero failed cells, dot, external and Monash.
