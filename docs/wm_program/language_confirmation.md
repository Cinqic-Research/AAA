# Language-integration confirmation (`aaa.wm.lang.v0`)

Status: **CONFIRMATION**, fresh identities, observed once. The freeze
(`docs/evidence/aaa_wm_lang_v0/freeze.json`, commit `a8c088d`) was committed **and pushed** before any
identity in `[4000, 6000)` existed. Evidence: `docs/evidence/aaa_wm_lang_v0/confirmation.json`,
provenance `a8c088d`, clean tree, run time 3 h 25 min, 3.6 GB peak. The post-run audit recomputes
every verdict from the bits with no disagreement.

The raw-download and processed-corpus manifests are now retained as
`docs/evidence/aaa_wm_lang_v0/*_snapshot.*`. They were copied from the local
HDD **after** confirmation and are labeled as provenance snapshots, not as
pre-run remote anchors. Their raw-file hashes matched all eight local
downloads at review time. Rebuilding the corpus still requires those external
downloads; the repository does not include the corpus or trained checkpoints.
The [post-run contamination audit](../evidence/aaa_wm_lang_v0/contamination_post_run.json)
deterministically reconstructed all 2,000 held-out reports from the pinned
code path and found no shared 13-word sequence in a training-corpus document.
The confirmation did not retain report hashes for a byte-for-byte comparison.
This narrows contamination risk but does not retroactively create a
pre-confirmation gate.
The language admission fingerprint omits transitive source helpers including
`lm/train.py`'s tokenizer-path resolver and `aaa_python/rng.py`. The recorded
run used clean commit `a8c088d`, and the reviewed source files match that
commit; the omission leaves the admission API incomplete for future use.

- **Design.** 2,000 fresh `opaque.v0` tasks. Every statement used held-out phrasings. Initializations 100-102
  pair the adapter, the WM-S table and the policy seed. The language model is R1 (5.2M parameters, from
  scratch), with the v2 adapter.
- **Channel exact extraction** (all three tests): LM 0.331, rules 0.134, random-init LM 0.065, gold 1.0.

| Decision system @ language channel | Success [95%] |
|---|---|
| WM-S @ gold (perfect understanding; ceiling) | 0.590 [0.405, 0.730] |
| **WM-S @ from-scratch LM (full system)** | **0.354 [0.182, 0.504]** |
| WM-S @ hand-written rules | 0.164 [0.113, 0.213] |
| WM-S @ random-init LM | 0.148 [0.121, 0.174] |
| policy_aux:plan @ LM | 0.113 [0.090, 0.135] |
| policy_aux:plan @ rules | 0.108 [0.085, 0.133] |

| Contract (`crossed.v1`, primary and independent agree) | Error ratio [95%] | Rule | Verdict |
|---|---|---|---|
| L2 LM over hand-written rules (with WM) | 0.772 [0.622, 0.918] | superior < 0.95 | **PROMOTE** |
| L1 pretraining end to end (LM vs random-init LM, with WM) | 0.758 [0.591, 0.933] | superior < 0.90 | **INCONCLUSIVE** |
| F1 full system vs policy_aux:plan @ LM | 0.728 [0.552, 0.942] | superior < 0.90 | **INCONCLUSIVE** |
| F2 full system vs policy_aux:plan @ rules | 0.724 [0.551, 0.937] | superior < 0.90 | **INCONCLUSIVE** |

## Verdict under the pre-registered success rule

- **`LANGUAGE_INTEGRATION_SUCCESS` is NOT declared.** It required L1 and L2 to PROMOTE, and L1 is
  inconclusive.
- **`FULL_SYSTEM_SUCCESS` is NOT declared.** F1 and F2 are inconclusive.
- **Confirmed: L2.** On fresh identities, with the world model, the from-scratch American-English
  language channel yields fewer errors than the hand-written rule parser (upper bound 0.918 < 0.95).
- **The point estimates** favor the full system over both pairwise systems (24-28% fewer errors) and
  favor pretraining (24%). The upper bounds sit just above the 0.90 bar. The interval width comes from
  the world model's library-table seed variance (full system 0.511 / 0.177 / 0.374 across the three
  initializations), carried into every WM arm, with only 3 initializations.

These identities are spent. A successor needs more initializations (at least 10), and less
table-seed variance or a design that factors it out. Repeating the test on these identities with more
seeds is not permitted.
