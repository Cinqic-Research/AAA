# World-model and language program: handoff

Dates: 2026-09-26 to 2026-09-27. Branch `opus/world-model-program`, draft PR
[#30](https://github.com/Cinqic-Research/AAA/pull/30). Base `main` `dbee13c`. Frozen phases are
untouched, and `tools/check_protected_identities.py` passes (493 files).

## What is established, and exactly how far

1. **A world model that improves AAA's decisions (confirmed, independently reviewed, scoped).**
   - **Setting.** `aaa.python.opaque.v0`: debugging code that calls an opaque library. The only source of
     truth is two real test runs per episode.
   - **Result.** AAA planning with **WM-S** makes about half the errors of every comparator. WM-S is an exact
     interpreter of the visible Python plus a tabular library model, abduced from real observations and
     updated online. The comparators:
     - the learned model-free architecture trained here, even inside the same planner;
     - the strongest non-learned shortcut;
     - the same agent without its learned library;
     - a black-box neural world model.
   - **Evidence.** All 11 pre-declared `crossed.v1` contracts PROMOTE on 4,000 fresh tasks. The primary and
     independent implementations agree. The post-run audit and an independent reviewer's own code
     reproduce every ratio.
   - **Limits.** Seed-dependent (C1 0.32-0.71 by table seed). Online-update benefit fragile (C4). Library shared
     between training and test. Weak model-free reference. See `opaque_confirmation.md` and
     `independent_review_opaque.md`.
2. **Generic learned world models did not earn a place:**
   - WM-0 on v1 is inconclusive;
   - the black-box neural WM loses to WM-S and barely beats the shortcut;
   - the JEPA auxiliary did not help, though its anti-collapse diagnostics behaved as designed.
3. **A from-scratch American-English language model, integrated through an explicit adapter.** The
   corpus is 1.79 GB of licensed, dialect-filtered, deduplicated text. The tokenizer (BPE-8K) was chosen
   by rule, the model is 5.2M parameters, and 13.8M did not help. On development, pretraining raises exact
   extraction on unseen phrasings from 0.03 to 0.29, and the full system beats every pairwise system.
   **On fresh confirmation:** the LM channel beats hand-written rules with the world model (L2 PROMOTE,
   0.772 [0.622, 0.918]). Pretraining and full-system superiority are **inconclusive** (point estimates
   0.72-0.76, upper bounds 0.93-0.94 above the 0.90 bar), so `LANGUAGE_INTEGRATION_SUCCESS` and
   `FULL_SYSTEM_SUCCESS` are **not declared** (`language_confirmation.md`).
4. **The workstation freezes are explained and fixed.** They were memory exhaustion from my parallel
   experiments. Every job now runs under a cgroup memory cap within a global budget (`compute.md`).

## What is not established

- General Python ability, repository-scale debugging, or any real library.
- Transfer to a library whose functions differ from training (WM-S is weakest on `library_B`, and
  adaptation causes forgetting).
- Language understanding beyond templated paraphrases. Perfect extraction would roughly double the full system.
- Matched-compute superiority. The matched resource is real test runs.
- The 105M-parameter AAA-1 goal, and any size-scaling claim (R3 about 27M was not justified).

## Open issues and next steps (prospective, new identities only)

1. **Benchmark successor (`opaque.v1`).**
   - Confine mutations to the grammar (removes the `tells` shortcut).
   - Make library_B shifts part of training and test.
   - Use at least 10 initializations and at least 30 streams per slice.
   - Push the freeze before confirmation and guard `draft("confirmation")` with the admission.
2. **A versioned, context-keyed library model** to remove the forgetting under library change.
3. **A stronger model-free reference** (scaled or ensembled) to test whether WM-S's margin holds.
4. **A language adapter trained on richer paraphrase data** (the extraction bottleneck), then extraction
   into a learned planner rather than a hand-written interpreter.
5. **Replace the hand-written interpreter with a learned execution model** only if it can match it. This is
   the black-box WM's current failure (deficiency A).

## Operating notes

- Workspace on the HDD: `/media/cinqic/Cinqic Storage/AAA/wm/` (`repo/`, `venv/` locked numpy,
  `venv-torch/` with the torch lock, `data/` = `AAA_DATA_ROOT`, `logs/`, `notes/`).
- After a reboot: `udisksctl mount -b /dev/sda1`.
- Run experiments only through `research/aaa_wm/jobs.py` or `capped.sh`, never as bare parallel processes.
- The spent identities are `opaque.v0` confirmation `[0, 4000)` (world model) and `[4000, 6000)`
  (language). Neither may be reused.
