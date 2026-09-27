# Program timeline and time accounting

Reconstructed 2026-09-27 from primary records only:

- commit timestamps (`git log`);
- file times of the HDD workspace and its job logs (`/media/cinqic/Cinqic Storage/AAA/wm/logs`);
- `/usr/bin/time` wall-clock lines in the logs;
- the job-runner logs;
- the kernel boot journal (`journalctl --list-boots`);
- GitHub PR and Actions records (converted from UTC).

All times are local (EDT, UTC-4). Where a boundary is inferred rather than logged, the entry says so.

## Summary

| Quantity | Value |
|---|---|
| Start (workspace created on the HDD) | 2026-09-26 01:58 (the boot began at 01:56) |
| Last result (CI green on the final head `46dd747`) | 2026-09-27 14:43 |
| Elapsed wall clock | **36 h 45 min** |
| Unplanned downtime (two OOM-driven reboots) | about 15 min of reboot, plus freeze periods before them (not measurable) |
| Session paused, waiting for the user's input while the user slept (2026-09-27 about 05:45-09:01) | about 3 h 15 min |
| Longest single computations | world-model confirmation 7 h 59 min; language confirmation 3 h 25 min |
| Commits on the branch | 38 (seven pushed heads) |

## Timeline

### 2026-09-26

| Time | What happened | Record |
|---|---|---|
| 01:56-02:09 | Orientation: mounted the HDD, cloned `main` `dbee13c` to the HDD, made the locked venv, ran the storage preflight and protected-identity check. Baseline suite: 935 tests OK in 9 min 01 s. | boot journal, workspace times, `baseline_tests.log` |
| 02:00-02:15 | Literature reviews (world models / JEPA; language / data) by subagents, in parallel. WM-0 v1 diagnostic written. | notes, commit `b239023` 02:15 |
| 02:10-02:39 | WM-0 on v1 repair: tune grid and 10-seed evaluation, a **negative / inconclusive** verdict. The `seq.v0` draft was piloted, red-teamed and **rejected**. `opaque.v0` was designed and built. | commit `7cdc2e8` 02:39 |
| 02:31-02:48 | Development task sets; shared behavior dataset (600,529 programs, 16.2M executions), 8 min 12 s. | `opaque_dataset.log` |
| 02:48-05:42 | Neural arms trained in sequence on the GPU (world model, policy, value, policy_aux; about 16-35 min each) while the tune-stage evaluations ran on the CPU. | `train_*_s0.log` times |
| 02:57 | First WM-S library table (200K programs, 6 min 53 s). | `wms_table_s0.log` |
| 04:16 | **First out-of-memory event.** Six parallel evaluations (about 1.5 GB each) plus a 3 GB trainer on 14 GB of RAM; swap exhausted; the kernel killed the desktop app. | kernel journal |
| 05:42-09:16 | Tune-stage evaluations continued (value, policy_aux as world model). | eval log times |
| 09:16 | **Second out-of-memory event and reboot.** A short boot followed (09:16-09:29). WM-S commit `6bb98e1` was made in it, then the machine was rebooted again with the power key (09:29). | boot journal |
| 09:31-09:35 | New boot. Freeze cause diagnosed from the journal. Capped job launcher committed. | commit `903bcb7` 09:35 |
| 09:38-10:06 | Corpus downloads (about 5 GB, hashed), then corpus build (12 min 13 s). | `lm_download.log`, `corpus_build.log` |
| 09:55-10:18 | **Lost to a scheduling bug:** two queue scripts deadlocked on `pgrep` patterns that matched my own shells. | queue logs |
| 10:17 | Tune-stage selections recorded. | commit `6787cfb` |
| 10:18-10:55 | Job runner built. Caps were undersized: two table builds and one training job were killed by their own caps, then re-sized from measured peaks. Byte and BPE tokenization. Tokenizer experiment (four runs, BPE-8K selected). LM data sampling fixed (it had been I/O-bound on the HDD). | commits `ee86c67`..`864c48d`, logs |
| 10:58-11:30 | LM evaluation suite, attack role, freeze/confirmation code, program index. | commits `89de200`..`c08d349` |
| 11:20-11:50 | Pre-confirmation red team (subagent, 29 min). No leakage; seven findings, all accepted. Erratum written; controls added (grammar shortcut, planner-matched policy, online-empty). | commits `651dbba`, `a7db266`, `272eb5c` |
| 12:03-12:25 | Language pre-registration; boundary mutation harness (12 of 12 caught). | commits `14fb59b`, `39fae08` |
| 12:45-13:06 | LM rungs R1 and R2 trained; language adapter v1 (a mixed result). | `lm_r*.log`, commit `10a08c7` |
| 13:06-21:58 | Development evaluate stage, attack pool, adaptation, and training of the confirmation artifacts (seeds 100-104); about 9 h of queued compute. Two scheduler fixes along the way: memory budget units and job adoption (tables were stuck from about 17:33 to 17:52). JEPA runs 20:40-21:58. | `queue_evaluate*.log`, commits `a5201e4` 15:01, `59e264e` 16:30, `087774c` 16:32 |
| 20:40 | **World-model freeze committed** (`f994d91`), before any confirmation identity existed. | commit |
| 20:41 | World-model confirmation started (a single capped process). | `confirmation.log` |

### 2026-09-27

| Time | What happened | Record |
|---|---|---|
| 00:45 | First push; draft PR #30 opened, while the confirmation was running. | GitHub (04:45 UTC) |
| 04:39 | World-model confirmation finished: **7 h 59 min**, 11 of 11 contracts PROMOTE, audit clean. | `confirmation.log`, commit `14eaf5d` |
| 04:40-04:46 | Independent adversarial review (subagent, about 5.5 min). The claim survives, narrowed. | commit `16f8a78` |
| 04:41-05:29 | Language adapter v2: 12 runs of about 3-5 min each. | `queue_lang_v2.log`, commit `c70bb86` |
| 05:30-about 06:00 | First launch of the language factorial. **Defective:** with no parsed statement, WM-S planned against zero tests. | runner log (overwritten on relaunch; start and finish times seen live in the session) |
| about 05:45-09:01 | **Session paused, waiting for the user's input while the user slept** (confirmed by the user). The defective first launch finished at about 06:00 and was inspected only when the session resumed at 09:01. | commit gap; user's account |
| 09:01 | Defect fixed, outputs discarded, factorial relaunched. | commit `c577031` |
| 09:01-10:04 | Language factorial, 30 jobs. | `queue_lang_factorial.log` |
| 10:05-10:25 | Factorial results; language confirmation module; confirmation-seed adapters (6 runs, 3 min each); dry run on development tasks; **language freeze committed and pushed** (`a8c088d`). | commits `b932ab2`..`a8c088d` |
| 10:27-13:50 | Language confirmation: **3 h 25 min**. L2 PROMOTE; L1, F1, F2 INCONCLUSIVE. | `lang_confirmation.log`, commit `f0a7982` 13:51 |
| 13:50-14:05 | Full suite (966 tests OK, 2 skips, about 6.5 min); CI-compliance cleanup (format, lint, mypy); mutation harness re-run. | `final_tests.log`, commit `46dd747` |
| 14:05 | PR #30 marked ready for review. | GitHub (18:05 UTC) |
| 14:43 | All 12 CI checks green on `46dd747`. | GitHub Actions (18:43 UTC) |

## Where the time went (approximate)

| Category | Approx. wall clock | Notes |
|---|---:|---|
| Orientation, reproduction, literature | 0.3 h | parallel subagents |
| Benchmark design, including the rejected `seq.v0` | 0.5 h | |
| Development experiments (tune, evaluate, attack, adaptation, JEPA) | about 13 h | mostly queued compute, much of it overnight |
| World-model confirmation (observed once) | 8.0 h | |
| Language track (corpus, tokenizer, LM, adapters, factorial) | about 3 h | |
| Language confirmation (observed once) | 3.4 h | |
| Reviews (two red teams, one independent review) | 0.7 h | subagents |
| CI compliance, final verification, PR | 1.0 h | includes 38 min of CI |
| Lost: OOM freezes and reboots | at least 0.5 h | plus the interrupted jobs that had to be re-run |
| Lost: my scheduling and tooling defects | about 1.5 h | deadlocked queues, undersized caps, budget units, I/O-bound LM sampling, self-killing `pkill` |
| Waiting for user input (user asleep) | about 3.2 h | not lost to defects; it delayed inspection of the defective first factorial launch |

These categories overlap: development compute often ran while code was being written. The
column shows the dominant activity, not exclusive time.

## Process lapses recorded

- **The workstation freezes (09-26)** came from running memory-heavy jobs in parallel without caps. The fix is
  recorded in [compute.md](compute.md).
- **CI was red on every intermediate push** (`f994d91` through `f0a7982`, 09-27 00:45-13:51). The
  failing step each time was **Lint** in the locked-environment job. The test jobs on Python 3.10-3.13
  passed throughout. I had checked only the lint subset `F,E9` locally and did not look at CI until the end. The
  final head `46dd747` is green on all 12 checks. For the future: run the repository's full CI command set
  (`ruff check`, `ruff format --check`, `mypy`) before every push, and read CI after each push.
- **The first language-factorial launch was defective**, and its outputs were discarded. The fix and relaunch are in
  the commit history.
- **The world-model freeze was pushed remotely only about 4 h into its run** (finding 6 of the independent
  review). The language freeze was pushed before its run.
