# Loop iterations 0004–0006 — independent-review handoff

For a reviewer (GPT-5.6 Sol) who should verify every conclusion without
trusting the implementer. The result is in
[`loop_report_0004_0006.md`](loop_report_0004_0006.md). The brief was the
external audit `AAA_Research_Audit_2026-09-21.md` (reviewing `main` at
`daea2ff`); its sequence is quoted in the report. Nothing here is approval,
and the protocol stays `aaa.loop.v0-pilot` until this review concludes (audit
R-13).

## 1. Identity

| Item | Value |
|---|---|
| base (`main`) | `daea2ffe838927bc6307f1652fac6b93373b164f` |
| branch | `opus/audit-tbptt-semantics` |
| Champion 0 phase fingerprint | `python -m research.aaa_1k fingerprint` → `5ce6e019bd5389c5d8d466cc2be809ff9777e3bd6acba10917bc6a7ce98f771e` (unchanged; nothing under `research/aaa_1k/`, `aaa/` or `results/` is modified) |
| freeze commit (attempt 2) | `25ff14d` — `docs/evidence/aaa1k_loop_0006/freeze_2.json` |
| confirmation commit | `96eaa31` — `confirmation_2.json`, `recomputation_2.json` |
| remote claims | `refs/heads/aaa-confirmation-claims/aaa1k-loop-0006-confirmation-{env,init}` (attempt 1, burned unobserved) and `…-confirmation-2-{env,init}` (attempt 2) |
| outcome | 0004 REJECT, 0005 REJECT, 0006 **PROMOTE** → Champion 1 |

## 2. Read in this order

1. [`loop_report_0004_0006.md`](loop_report_0004_0006.md): the result.
2. `git log --reverse --format='%h %s' main..HEAD`. Every design commit
   (hypotheses, thresholds, candidates, rules) precedes the evidence commit
   that judges it; check this rather than trusting it.
3. [`research/aaa_1k_loop/iteration4.py`](../research/aaa_1k_loop/iteration4.py)
   docstring: the prospective rules and the K1–K3 decision.
4. `docs/evidence/aaa1k_loop_000{4,5,6}/iteration.json`: machine-readable records.
5. [`issue_ledger.md`](issue_ledger.md) `AAA-169`–`AAA-172`, the new sections of
   [`errata.md`](errata.md), [`limitations.md`](limitations.md) and
   [`loop_protocol.md`](loop_protocol.md).

## 3. Files

| Path | What |
|---|---|
| `research/aaa_1k_loop/tbptt.py` | R-02: live/snapshot/replay/T=1 rules, finite-difference references, the comparing instrument |
| `gain.py`, `overshoot.py`, `unfolding.py` | M2 instruments (loop gain, same-sample amplification, unfolding trace) and candidates c7–c10 |
| `diagnosis4.py`, `diagnosis4b.py`, `diagnosis4c.py`, `diagnosis4d.py` | diagnosis rounds 1–4, H20–H34 with predeclared verdict rules |
| `stages4.py` | command line for the diagnoses, confirmation-primitive replay and `reproduce` |
| `iteration4.py`, `iteration5.py`, `iteration6.py` | the precommitted rules, candidates, cell designs, screen, attack and decision (0005/0006 import 0004's judgement functions) |
| `outer4.py`, `outer5.py`, `outer6.py` | develop/attack/freeze/confirm per iteration; confirmation path closed over its frozen source set (tested) |
| `recompute4.py` | independent recomputation; no shared decision code, rules read from the freeze |
| `champion1.py` | Champion 1 record and the full round-3 re-run in Champion 1's world |
| `records4.py` | iteration records 0004–0006 |
| `identities.py` (`require_claimed`), `freeze.py` (source-set parameter), `cli.py` (blocks, records, validate) | the only edits to pilot modules |
| `tests/test_aaa_1k_loop_tbptt.py`, `tests/test_aaa_1k_loop_iteration4.py` | 33 + 27 new tests; 694 repository tests total |
| `.github/workflows/loop-reproduction.yml`, `ci.yml` | full-stage reproduction job; Champion 1 and recomputation checks in CPU CI |

## 4. Reproduce

```bash
python -m research.aaa_1k fingerprint                          # 5ce6e019..., unchanged
python -m research.aaa_1k_loop prove-fresh                     # 31 blocks disjoint from every AAA-1K identity
python -m research.aaa_1k_loop validate                        # six records (0006 recomputes PROMOTE), champion, ledger
python -m research.aaa_1k_loop.champion1 verify                # Champion 1 record against the repository
python -m research.aaa_1k_loop.recompute4 \
    --confirmation docs/evidence/aaa1k_loop_0006/confirmation_2.json \
    --freeze docs/evidence/aaa1k_loop_0006/freeze_2.json       # independent decision
for s in diagnose gain overshoot unfold develop4 develop5 develop6 attack6 confirmation6; do
    python -m research.aaa_1k_loop.stages4 reproduce $s        # ~0.5-1.5 min each on 16 workers
done
python -m research.aaa_1k_loop.champion1 round3                # ~6 min serial; rewrites round3_champion_1.json,
git diff --stat docs/evidence/aaa1k_loop_0006/                 #   whose non-provenance content must not change
python -m unittest discover -s tests -t .
```

On the implementer's machine every `reproduce` stage is **bit-identical**
(only `git` and `compute_seconds` are exempt). Across machines it is not
always: on the first CI run, eight of nine stages were bit-identical, but
`develop6` differed in the last digit of MAEs (relative ~4×10⁻¹⁵). `reproduce`
therefore declares a float tolerance (relative 1e-9) and reports the largest
deviation. Strings, integers, booleans (divergence flags, locks, verdicts) and
structure must still match exactly (`stages4.FLOAT_RTOL`,
`TolerantComparisonTests`). To check that the confirmation
ran under its freeze, verify at the freeze commit:

```bash
git worktree add /tmp/aaa-freeze 25ff14d && cd /tmp/aaa-freeze
python -c "
import json; from pathlib import Path
from research.aaa_1k_loop.freeze import verify_freeze
from research.aaa_1k_loop.identities import load_ledger
from research.aaa_1k_loop.outer6 import CONFIRMATION_SOURCES_6, frozen_content
m = json.load(open('docs/evidence/aaa1k_loop_0006/freeze_2.json'))
print(verify_freeze(m, Path('.'), ledger=load_ledger(Path('benchmarks/aaa1k_loop_identity_ledger.json')),
      frozen_content=frozen_content('c10_reach_gated_unfold'), sources=CONFIRMATION_SOURCES_6))"   # expect []
```

On today's `HEAD` the same check reports one changed source, `freeze.py`: after
the confirmation, iteration 0003's (never-built) frozen tuple gained the modules
`cli.py` now reaches. No other frozen 0006 byte changed.

## 5. What to challenge

The implementer's own list of weak points, most important first:

1. **Multiplicity.** Four M2 candidates (c7–c10) were screened across three
   iterations. Each iteration had a precommitted budget, but the budget was
   renewed twice after rejections. The fresh attack and fresh confirmation,
   not the screen, carry the promotion. Judge whether that guard is adequate,
   and whether declaring c10 "the last candidate" in advance means anything.
2. **c10 was designed after looking at development data.** A disclosed look at
   0005's already-observed development cells (mirror-run lengths, two
   initializations) motivated the reach gate `distance(p, wall) > |v| + |y − p|`.
   The report calls this bound "derived from reflection geometry". Challenge
   whether it is a principled necessary condition or a threshold that happens
   to pass the smooth-bouncing screen. Streams where `|v|` badly
   underestimates the next step are untested.
3. **H32–H34 were chosen by a post-hoc look.** The overshoot block's
   already-observed cells and one traced cell suggested the frame lock; it was
   then tested on a fresh block. Check that the fresh block, the lock
   threshold (10 steps) and the probes were committed before that run
   (`git log` on `diagnosis4d.py` versus `diagnosis_unfold.json`).
4. **Confirmation attempt 1.** It claimed its identities remotely and crashed
   on an admission defect before any cell ran (`confirmation_attempt_1.json`,
   `AAA-171`). The challenger was refrozen unchanged on new blocks. Judge
   whether refreezing was legitimate, and whether "nothing was observed" is
   adequately evidenced: the traceback came from `confirmation_cells()`,
   before `run()`.
5. **Champion 1's round-3 world.** The re-run applied the reach gate to every
   neural learner, including the controls and the ablations (via a
   module-level class substitution, `champion1.champion_1_world`), so that
   architecture comparisons keep one agent rule. That is a choice. With
   controls on the old rule, Q3/Q4 would compare different target rules.
6. **The TBPTT docstring is not fixed in code.** `model.py` still says
   "exact". It is corrected in the errata and `AAA-169` only, to keep the
   phase fingerprint. Judge whether that is the right trade.
7. **Diagnostic verdicts that are weaker than they read.** H31 is SUPPORTED as
   declared but immaterial (~1% curvature share); H26 is INCONCLUSIVE; H28
   INSUFFICIENT_EVIDENCE. The 2/80 lock rate without input 3 is from a
   reported, non-adjudicated arm.
8. **Iteration 0003's decision code.** `decision.py`'s permissive C2/C4/C5
   (audit R-04) were not repaired. They are unreachable because 0003 cannot
   freeze. The new K1–K3 rules are uncertainty-aware, and the tests in
   `DecisionTests` include a straddling-interval case.
9. **Scope of the M2 repair.** Tested at 1120 steps on quantized coarse,
   smooth bouncing, occlusion and four nearby quantized constructions only.
   `AAA-172` (the v2.1 RLS learner uses the same own-prediction unfolding) is
   open and untested.
10. **Evidence volume.** About 27 MB of new JSON (downsampled per-cell
    traces and primitives that `reproduce` compares against), against the
    pilot's ~6 MB. It was not trimmed after the fact, because trimming would
    rewrite observed evidence.
11. **Tests use one arbitrary stream seed (424242).** Only in unit tests of
    instrument identity, never as evidence. It is not a ledger identity.
12. **`docs/final_audit.md` is not regenerated** (audit R-14): this branch
    adds 40 files and modifies 12 (`git diff --name-status main`).

Failure injections worth trying by hand:

- Set `MARGIN = -0.01` in `iteration4.py`. `recompute4` still reports PROMOTE
  (it reads the freeze's margin), while the live `iteration6.decide` now
  returns REJECT. `MARGIN = 0` does *not* discriminate: every K2/K3 interval
  is exactly [0, 0].
- Mark a `confirmation-2` block `reserved` in a copy of the ledger and call
  `iteration6.confirmation_cells(ledger, observer=…)`. It must refuse.
- Flip one champion `diverged` flag in a `long:coarse_no_switch` cell of
  `confirmation_2.json`. `validate` must fail (the artifact hash is pinned),
  and `recompute4` must report DISAGREES on K1's interval even though every
  status and the outcome are unchanged. The first version of `recompute4`
  compared statuses only and missed this; it was tightened while preparing
  this handoff (`RecomputationTamperTests`). The committed
  `recomputation_2.json` is that first version's output, kept as observed.
- Edit a byte of `unfolding.py` and run `champion1 verify`. It must fail.
