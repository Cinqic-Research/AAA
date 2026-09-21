# Loop pilot — independent-review handoff

For a reviewer (GPT-5.6 Sol) who should verify every conclusion without
trusting the implementer. The result is in
[`loop_pilot_report.md`](loop_pilot_report.md); the protocol under trial is
[`loop_protocol.md`](loop_protocol.md). Nothing here is approval.

## 1. Identity

| Item | Value |
|---|---|
| base (`main`) | `1ce716c42500003881e379f6d23dc0d807301001` |
| branch | `opus/birth-of-the-loop` |
| Champion 0 fingerprint | `python -m research.aaa_1k fingerprint` → `5ce6e019bd5389c5d8d466cc2be809ff9777e3bd6acba10917bc6a7ce98f771e` (must be unchanged on this branch) |
| protocol | `aaa.loop.v0-pilot` |
| outcome | three iterations, eight candidates, all `REJECT` before any freeze; no confirmation identity reserved or spent; Champion 0 unchanged |

## 2. Read in this order

1. [`loop_pilot_report.md`](loop_pilot_report.md): the result and the evaluation of the loop.
2. [`loop_protocol.md`](loop_protocol.md): what the loop is, what enforces it, and its known gaps.
3. `docs/evidence/aaa1k_loop_000{1,2,3}/iteration.json`: the machine-readable records.
4. `git log --reverse main..HEAD`: every design commit precedes the evidence it judges.
5. [`issue_ledger.md`](issue_ledger.md) `AAA-162`–`AAA-164`, the pilot section of
   [`limitations.md`](limitations.md), and the new section of [`errata.md`](errata.md).

## 3. Files

| Path | What |
|---|---|
| `research/aaa_1k_loop/identities.py` | per-iteration salted seeds, the ledger, the freshness proof |
| `champion.py`, `observation.py` | Champion 0 derived from the repository; the round-3 Q4 decomposition |
| `harness.py`, `arms.py`, `dynamics.py` | parallel cell runner over AAA-1K's `run_stream`, the arm registry, read-only Jacobian instrumentation |
| `diagnosis.py`, `diagnosis2.py`, `diagnosis3.py` | H1–H19 with predeclared verdict rules |
| `challengers.py`, `develop.py`, `attack.py` | iteration 0001: candidates c1–c3, the development rule, attack A1–A10 |
| `bounded.py`, `iteration2.py` | iteration 0002: bounded-error agent, candidates c4–c6 |
| `iteration3.py`, `decision.py`, `freeze.py`, `recompute.py` | iteration 0003: claims v2/v3, attacks, and the outer loop (freeze, decision, independent recomputation) |
| `iteration.py`, `records.py` | the state model and fail-closed validator; the three records |
| `capabilities.py` | Q2/Q5 for any registered arm (used by the attack) |
| `cli.py` | every command below |
| `tests/test_aaa_1k_loop.py` | 60 tests, mostly failure injection |
| `benchmarks/aaa1k_loop_identity_ledger.json` | 13 blocks, none `confirmation` |

Nothing under `research/aaa_1k/`, `aaa/`, `results/`, or any AAA-1K evidence
or report is modified. `tests/test_aaa_1k.py` is unchanged.

## 4. Reproduce

```bash
python -m research.aaa_1k fingerprint                       # 5ce6e019..., unchanged
python -m research.aaa_1k_loop champion --verify            # Champion 0 against the repository
python -m research.aaa_1k_loop prove-fresh                  # ledger disjoint from every AAA-1K identity
python -m research.aaa_1k_loop validate                     # all records, champion, ledger
python -m research.aaa_1k_loop reproduce observe            # Q4 decomposition from round-3 primitives
python -m research.aaa_1k_loop reproduce diagnose           # ~20 s each on 16 workers
python -m research.aaa_1k_loop reproduce diagnose2
python -m research.aaa_1k_loop reproduce diagnose3          # ~2 min
python -m research.aaa_1k_loop reproduce develop
python -m research.aaa_1k_loop reproduce develop-round-2
python -m research.aaa_1k_loop reproduce attack
python -m research.aaa_1k_loop reproduce develop2
python -m research.aaa_1k_loop reproduce attack3
python -m research.aaa_1k_loop reproduce attack3b
python -m unittest tests.test_aaa_1k_loop
```

`reproduce` reruns a stage into a scratch file and requires every committed
value to reappear exactly; only `git`, `compute_seconds` and wall-clock timing
fields are exempt. On the implementer's machine all ten stages reproduced with
zero mismatches (fields added by later code are reported, not ignored).
`AAA_LOOP_WORKERS=1` forces serial execution; results do not depend on the
worker count.

## 5. What to challenge

The implementer's own list of weak points:

1. **Deviation L-4.** c2 failed development screening and was advanced to
   attack anyway, as a labelled tradeoff challenger. The attack rejected it,
   so it never reached confirmation, but the decision to advance it was made
   after seeing the screen. Judge whether that was defensible.
2. **Three iterations instead of one.** The brief asked for one cycle through
   confirmation. The first iteration ended at attack; the next two were started
   from its resulting state. Judge whether iterations 0002 and 0003 were
   genuine next steps or a search for something that would reach confirmation.
3. **Iteration 0003's challenger is a claim.** Check whether treating a
   documentation correction as a challenger is sound, and whether R4 was a
   fair criterion given that c2's instability was already known.
4. **The DoD item that is not met.** No fresh confirmation was frozen or
   observed. The outer loop's code (`freeze3`, `confirm3`, `recompute3`,
   `decide`) has run only against synthetic confirmations in tests. It is the
   least-exercised part of the pilot.
5. **Diagnostic depth.** H13 is supported by four predeclared predictions on a
   fresh block, but the Jacobian statistic is a per-cell median of a per-step
   quantity. M2's refined mechanism (a closed-loop gain above one) is a
   hypothesis, not a finding.
6. **Development initializations.** Diagnosis and development used the
   champion's five `model_init` seeds; only attacks used fresh
   initializations, and that is where c2 broke.
7. **Unregistered observations.** One exploratory look at a diagnostic cell
   (before diagnosis round 3) and one sanity run on an arbitrary stream (before
   iteration 0002's development). Both are disclosed in the records.
8. **The closure audit is not regenerated.** `docs/final_audit.md` inventories
   the 533 files of the closure review; this branch adds 41 files and modifies
   9 (all listed by `git diff --name-status main`). Regenerating that inventory
   would assert a closure-style review of every file, which this pilot did not
   perform.
9. **Evidence volume.** About 6 MB of JSON, mostly per-cell primitives needed
   for recomputation. See [`evidence_policy.md`](evidence_policy.md).

Failure injections worth trying by hand: edit any committed artifact and run
`validate`; add a `confirmation` block overlapping a development block to the
ledger and run `prove-fresh`; change a threshold in `decision.py` and compare
against a manifest built by `build_freeze`; delete a candidate from an
iteration record.
