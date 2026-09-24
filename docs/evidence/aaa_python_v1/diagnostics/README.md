# `aaa.python.v1` pre-design diagnostics

Run on 2026-09-24 at `main` `cb4170e` plus uncommitted scratch scripts (copied
here verbatim), with CPython 3.12.3 and NumPy 2.5.3 on FLOWBOX (Ryzen 7
5700G). They use **only identities `aaa.python.v0` had already observed** (v0
train, development and probe pools, cached by `cache_pools.py`) and v0's
unmodified learner. They informed the v1 design; they are diagnostic, not
development or confirmation evidence of v1.

| Script | Output | Finding |
|---|---|---|
| `hash_diag.py` | `hash_diag.json`, `hash_diag.log` | Signed hashing with a dedicated bias helps v0's learner at D = 256 (syntax +0.035, 5/5 initializations); every variant is within about 0.01 at D >= 1024; D barely matters for the linear learner. The subclass reproduces v0's recorded trained-state hash `1a916b9d...` exactly at v0's settings before any variant is measured. |
| `epochs_diag.py` | `epochs_diag.log` | v0's learner is under-trained (`AAA-197`): syntax 0.533 / 0.816 / 0.893 and outcome 0.641 / 0.677 / 0.677 at 2 / 6 / 20 epochs; output and localize do not move. Mean over 3 initializations, frozen, all 400 development tasks. |
| `probe_repair.py` | `probe_repair.log` | The v0 repair medoid shortcut (`AAA-192`): 0.923 train, 0.900 development without executing anything; the visible-test tool: 0.943 / 0.940. |

Reproduce from a checkout at the commit that added this directory, with the
locked environment: `python cache_pools.py`, then each script from this
directory. Outputs are deterministic; the rerun used for these logs matched
the first run exactly.
