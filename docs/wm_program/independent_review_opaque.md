# Independent adversarial review: `aaa.python.opaque.v0` WM-S confirmation

Date: 2026-09-27. Reviewer: independent adversarial reviewer (did not write the work). I only read the
repository. Nothing was edited, committed or pushed. No confirmation agent was re-run and no
confirmation task was regenerated. I recomputed everything from the stored bits in
`docs/evidence/aaa_wm_opaque_v0/confirmation.json`.

My scripts and logs are in `/media/cinqic/Cinqic Storage/AAA/wm/notes/review_scratch/`:
`recompute.py` (my own crossed bootstrap, which does not import `summarize.py`), `extra.py`, and
their `.log` and `.json` outputs. Every script ran through `capped.sh 2G`, one process at a time, on
the CPU.

## Verdict

**The confirmed claim survives, but only in a narrower form than the prose states.**

- I found no invalidating defect.
- The integrity chain holds.
- My independent recomputation reproduces all 11 verdicts exactly.
- The grammar shortcut does not explain the WM-S gain.

Several statements in `opaque_confirmation.md` claim more than the evidence supports:

- **C4 (online adaptation) is fragile.** It depends on the table seed. At the level of seeds its
  upper bound does not clear 0.95.
- **The "strongest learned model-free agent" is a single small network.** It barely uses the planner.
- **Non-inferiority is vacuous.** Its bar sits far above the observed ratios, and it is checked
  against one reference only.
- **The size of the gain depends heavily on which library-table seed was drawn.** Only 5
  initializations were used. The design document pre-registered 10-20.

## 1. Integrity (no invalidating finding)

**Order of events.** The reflog order is `02796a5` (20:40:13) → **freeze `f994d91` (20:40:21)** →
evidence `14eaf5d` (04:39:59). The freeze commit only adds `freeze.json`.

- `confirmation.json` has provenance commit `f994d91` with `dirty: false`, and started at 20:41:32.
- `logs/confirmation.log` was born at 20:40:27, 6 s after the freeze commit. It shows one process:
  7:58:47 wall time, 100% of one CPU, 3.3 GB RSS, and the per-arm rates in the order the freeze
  declares.
- The embedded `freeze` equals the committed `freeze.json`.

**Source files.** All 30 fingerprinted source files hash to their frozen SHA-256 values at
`f994d91`, at HEAD and in the working tree. Between the freeze and the evidence commit, nothing
under `research/` or `aaa/` changed.

**Artifacts.** All 15 artifact hashes (5 tables and 10 checkpoints for seeds 100-104) match the
files on disk under `$AAA_DATA_ROOT/opaque/{wms,ckpt}`.

- Every artifact's mtime is before the freeze. The last one, `wm_s104`, is from 20:39:45.
- I found no evaluation output that used seeds 100-104 before the confirmation. There is no sign
  that bad seeds were dropped and redrawn.

**Thresholds and C1 reference.** The thresholds (0.75, 0.80, 0.95, 1.0) and the contract set were
committed at `272eb5c` (11:51). The C1 reference (`policy_aux:plan`) was fixed at `087774c` (16:32),
by the rule "strongest learned non-WM arm on evaluate". That rule picks the stronger of the two
policy arms, so it is the conservative choice.

Nothing depends on confirmation data. However:

- **MINOR.** The `write_freeze.py` docstring says the thresholds were fixed "after the development
  evaluate stage". In fact, only part of the evaluate stage existed at 11:51: `wms:online` s0-s2 and
  `policy_aux` s0. The thresholds are called "practical". They were not derived from an MDE, as
  `opaque_benchmark_design.md` pre-registered.
- **MINOR (anchoring).** The freeze was first pushed to GitHub at 2026-09-27T04:45Z (00:45 EDT). That
  is about 4 h into the 8 h run, not before it started. `generator.draft("confirmation", …)` is not
  guarded; only `build`/`pool` are.
  - Git history and logs cannot rule out an earlier, unrecorded confirmation run.
  - I found no evidence of one. The scratch scripts, logs and eval directory contain no confirmation
    generation or output other than this run.
  - Recommendation: in future, push the freeze and record its remote timestamp before starting.

## 2. Independent recomputation (matches)

My own code does the following:

- It builds streams as the global blocks of 40 tasks, which is the unit at which the online agent
  resets.
- It checks that all 100 blocks are slice-pure and that each slice has 20 streams.
- It checks that all 4,000 task hashes are unique.
- It applies Jeffreys smoothing, (e + 0.5)/41, per init × stream cell.
- It broadcasts seed "0" to every initialization for the deterministic references (`tells`,
  `wms:empty`, and the unused `tool_gap`/`ceiling`).
- It computes the per-slice ratio of mean errors and takes the geometric mean over slices.
- It runs a crossed bootstrap: initializations are resampled jointly across slices, and streams are
  resampled independently per slice.

**Results.**

- **Point ratios and group ratios.** All 11 are identical to the stored values to 4+ decimals:
  - C1 0.4813;
  - C2a 0.5315;
  - C2b 0.4516;
  - C3 0.5158;
  - C4 0.8732;
  - C5 0.5024;
  - N_* 0.417, 0.474, 0.370, 0.508 and 0.695.
- **Intervals.** They agree within 0.007. For example, C1 is [0.379, 0.602] against the stored
  [0.377, 0.600], and C4 is [0.838, 0.929] against [0.838, 0.932].
- **No silent drops.** No cell or group is dropped: 5 slices × 20 streams × 5 inits for every arm.
- **Smoothing has no effect on the conclusion.** Unsmoothed ratios are almost the same (C1 0.474).
- **Success table.** It reproduces exactly: WM-S online 0.568, and `policy_aux:plan` 0.111.

**Seed-level fragility.** These numbers are from my script, not from the documents.

| Contract | Per-seed ratio (100, 101, 102, 103, 104) | t(4) interval on per-seed log ratio | Threshold |
|---|---|---|---|
| C1 | 0.32, **0.71**, 0.41, 0.51, 0.45 | [0.32, 0.66] | 0.75 |
| C3 | 0.34, **0.748**, 0.44, 0.56, 0.47 | [0.35, 0.71] | 0.75 |
| C5 | 0.33, **0.751**, 0.43, 0.53, 0.47 | [0.33, 0.70] | 0.75 |
| C2a | 0.35, **0.77**, 0.46, 0.57, 0.49 | [0.36, 0.73] | 0.80 |
| **C4** | **0.997**, 0.83, 0.92, 0.85, 0.85 | **[0.81, 0.975]** | 0.95 |

- The superiority claims C1, C2a, C2b, C3 and C5 hold under any of these views. Leaving out any one
  seed gives C1 between 0.42 and 0.52.
- **C4 does not clear 0.95 under a seed-level interval.** It shows **no online gain at all for the
  most complete table** (seed 100 holds 9,984 entries; the online bits equal the frozen bits on 3,890
  of 4,000 tasks).
- The online gain tracks the holes in the table. Seed 101 has the smallest table, 7,205 entries,
  and online learning lifts it from 0.253 to 0.381.
- WM-S success by seed ranges from 0.381 to 0.697. With 5 clusters, a percentile bootstrap
  under-covers this population.

## 3. Validity threats

1. **Leakage: none found.**
   - A `View` carries only visible fields.
   - WM-S uses `gen.library("A")` for names only.
   - Each stream starts from a fresh `deepcopy` of the table, so nothing crosses arms or streams.
   - Online updates use only the agent's own failing `RUN` results on visible tests.
   - This agrees with the earlier red team and with the M1-M12 mutation harness.
2. **Online learning across tasks is legitimate.** It uses only real observations inside the
   declared 2-run budget, and it resets per stream.
   - WM-S online in fact uses *fewer* runs: 1.32 per episode against 1.90 for `policy_aux:plan`.
   - The gain is mostly within the episode. Online minus frozen by position in the stream is +0.048,
     +0.053, +0.075 and +0.073.
   - The C1 gap is not caused by online learning: frozen WM-S against `policy_aux:plan` is 0.551.
3. **Grammar shortcut: it does not explain the gain.** I restricted to the 3,316 tasks the `tells`
   arm fails.
   - On those tasks, the WM-S online error ratio is **0.488** against `policy_aux:plan`, compared
     with 0.481 overall, and 0.535 against `wms:online+empty`.
   - The shortcut helps every arm about equally in absolute terms. `policy_aux:plan` scores 0.31 when
     `tells` succeeds and 0.07 when it fails.
4. **Fairness and attribution (SCOPE).**
   - `wms:online+empty` uses the hand-written interpreter plus online abduction, with **zero
     training data**. It already beats the C1 reference (ratio 0.905) and nearly matches `tells`
     (0.97). Much of the distance to the model-free arm is prior knowledge: the interpreter and the
     exhaustive depth-2 enumeration. The training-learned part is correctly isolated by C2a (0.53).
   - The model-free reference is a single transformer of about 1M parameters (3.9 MB checkpoint),
     trained for 20k steps. It reaches 0.10-0.12.
   - Putting it inside the planner barely changes it: `policy_aux:plan` against `policy_aux` is
     0.99. `PolicyPlanner` also expands second edits only for the best 8 first edits, whereas WM-S
     enumerates every plan. The controllers are therefore close, but not identical.
   - Compute is unmatched, and the doc says so. The freeze declares a 2.0 CPU-s/episode cap for
     WM-S, but the confirmation neither measures nor enforces it (**MINOR**).
5. **What is "learned" (SCOPE).** The learned component is an exact memo of 6 fixed unary functions,
   abduced from about 5.2M observations of the *same* library A. Library A is used at test in 4 of
   the 5 slices, so the novelty at confirmation is new programs, not new dynamics.
   - On the only slice where the dynamics shift (`library_B`), WM-S is weakest: 0.409 against a
     ceiling of 0.797, and its N ratio is 0.695.
   - Forgetting was not measured at confirmation.
6. **Statistics (SCOPE / MINOR).** The confirmation deviates from the pre-registered design. It used
   5 initializations and 20 streams per slice, where the design said 10-20 initializations and at
   least 30 streams. The deviation is disclosed only as "increase from 3 to 5".
   - The per-slice non-inferiority bar of 1.0 sits far above the observed ratios (0.37-0.69), so it
     is effectively vacuous.
   - Those contracts are checked only against `policy_aux:plan`.

## 4. Claim scope: wording that overclaims in `opaque_confirmation.md`

- "the strongest learned model-free agent": say instead "the one learned model-free architecture
  trained here (a ~1M-parameter transformer, 20k steps)".
- "even inside the same planner": say instead "inside a planner with the same verifier and budgets
  (policy beam 8 on second edits)".
- "It is not worse on any slice": say "against `policy_aux:plan`, with a 1.0 bar that the observed
  ratios (≤ 0.70) clear trivially".
- "Updating the library model online improves on the frozen model (C4)": add that the gain
  compensates for gaps in the training table and is mostly within the episode. It is about 0 for the
  most complete table (seed 100). The t(4) interval over seeds reaches 0.975, above the 0.95
  threshold.
- "roughly half the errors": add that per table seed the ratio is 0.32-0.71 against C1, and that
  there are 5 initializations against the 10-20 the design pre-registered.
- "learned from real observations": add "of the same fixed six-function library that 4 of 5 test
  slices use".

The "Not licensed" list is honest. It should also say: not licensed that the model-free reference
is strong or scaled.

## Ranked issues

1. **SCOPE-NARROWING.** The strength of C4 is overstated. It fails a seed-level bound, and seed 100
   shows no gain. The online benefit is gap-filling for an incomplete table.
2. **SCOPE-NARROWING.** The gain depends on the table seed (per-seed C1 ratios 0.32-0.71). Only 5
   initializations were used, against the 10-20 pre-registered, and 20 streams against at least 30.
3. **SCOPE-NARROWING.** The model-free reference is weak and small. Interpreter plus abduction with
   no training already beats it. "Strongest learned" overclaims.
4. **SCOPE-NARROWING.** The "world model" is a memo of a fixed library shared by train and test.
   WM-S is weakest under the only dynamics shift (`library_B`).
5. **SCOPE-NARROWING.** The per-slice non-inferiority contracts are vacuous and are checked against
   one reference only.
6. **MINOR.** The freeze was not anchored remotely before the run (it was pushed about 4 h in).
   `draft("confirmation")` is not guarded. A single run cannot be proven, although every observable
   is consistent with one.
7. **MINOR.** The thresholds were set after partial evaluate results and were not derived from an
   MDE. The docstring misstates the timing.
8. **MINOR.** The declared CPU budget per episode was not verified during confirmation.

**Positive findings.**

- Source and artifact hashes match.
- The ratios reproduce exactly from my own code.
- There are no dropped cells, and the deterministic-reference broadcasting is correct.
- There is no leakage.
- The grammar shortcut does not explain the gain (ratio 0.488 on tasks where `tells` fails).
- Online learning is legitimate and within the run budget.
