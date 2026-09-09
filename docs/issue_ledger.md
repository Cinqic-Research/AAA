# AAA issue ledger

This ledger records defects confirmed against the pre-change repository and the regression evidence for each repair.

| Issue | Evidence | Expected behavior | Repair | Regression check | Status |
|---|---|---|---|---|---|
| Report conclusions were fixed prose | `aaa/reporting.py` contained unconditional claims about improvement and adaptation | Conclusions must be derived from stored metrics | Report now formats observed comparisons and preserves missing/insufficient evidence | Report generation tests and historical snapshot retained | repaired |
| Recovery used partial trailing windows and only one time | `aaa/metrics.py` used expanding `rolling_mean` and returned `recovery_time_steps` without onset/confirmation semantics | Complete windows, explicit elapsed convention, onset and confirmation, censoring | Added complete-window recovery and censored status | `test_recovery_requires_complete_windows_and_reports_confirmation` | repaired |
| Sampling ignored configured bounds | `aaa/environment.py` sampled `0.08..0.92` | Bounds-relative sampling | Sampling derives margins from interval width | arbitrary-bound environment test | repaired |
| Exact boundary contacts were not classified consistently | `_reflect` only checked strict inequalities | Outward exact contacts reverse and are labeled | Equality plus outward-velocity handling | exact-boundary test | repaired |
| Static Agg backend was selected during import | `aaa/visualization.py` called `matplotlib.use` at module import | GUI and static rendering initialization are separate | Backend selection moved to explicit static-rendering call | headless CI command and import inspection | repaired |
| Animation model was frozen despite online labeling | `aaa/animation.py` constructed `update_enabled=False` and never updated after scoring | Online mode learns after scoring; restart resets model state | Added online/frozen mode and state reset | animation code path review; GUI check remains optional | repaired |
| Non-finite outputs could enter metrics | predictor outputs and metrics had no finite-state guard | Invalid metric states fail explicitly | Added finite validation and normalized errors | invalid predictor/config tests | repaired |
| Only legacy SGD learner was available | baseline learner mixed correlated position features and performed poorly | Diagnose before scaling; retain legacy comparison | Added reproducible SVD diagnosis and zero-initialized normalized RLS candidate | `aaa diagnose`, RLS checkpoint tests, benchmark v2 | repaired |
| Historical results were not tied to a clean source tree | `results/final/metadata.json` recorded a dirty run | Historical evidence must remain visibly historical | Preserved snapshot and added separate benchmark-v2 provenance | source-tree hash and run metadata | repaired/documented |

The original `results/final` files are intentionally not overwritten. Their historical claims are not acceptance evidence for benchmark v2.
