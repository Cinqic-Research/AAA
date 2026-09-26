# `opaque.v0` evaluate stage (development)

Status: **development**. `development` indices 1000-4999: 4,000 tasks, 20 slice-pure streams of 40
per slice. Initializations 0-2 (WM-S table seeds and neural training seeds). Every arm and control
was declared at the tune stage or by the red team, before this stage ran. Summary:
`$AAA_DATA_ROOT/opaque/eval/evaluate_summary.json`. Crossed bootstrap over initializations and
streams.

| Arm | All [95%] | in-dist | library_B | novel_grammar | novel_literals | two_fault |
|---|---|---:|---:|---:|---:|---:|
| true-library planner (ceiling) | 0.787 [0.770, 0.803] | 0.831 | 0.807 | 0.768 | 0.844 | 0.686 |
| **WM-S online** | **0.537 [0.454, 0.629]** | 0.639 | 0.354 | 0.540 | 0.645 | 0.507 |
| WM-S frozen | 0.454 [0.341, 0.574] | 0.549 | 0.277 | 0.457 | 0.573 | 0.412 |
| WM-S online, empty library | 0.208 [0.194, 0.222] | 0.249 | 0.215 | 0.164 | 0.248 | 0.164 |
| tells (grammar shortcut, no WM) | 0.182 [0.166, 0.198] | 0.211 | 0.194 | 0.232 | 0.204 | 0.069 |
| black-box neural WM, depth 1 | 0.161 [0.139, 0.183] | 0.229 | 0.186 | 0.091 | 0.201 | 0.099 |
| WM-S, identity library | 0.127 [0.106, 0.147] | | | | | |
| policy_aux inside the WM-S planner | 0.115 [0.100, 0.130] | 0.145 | 0.141 | 0.080 | 0.162 | 0.045 |
| policy_aux, tool search | 0.105 [0.091, 0.120] | 0.137 | 0.130 | 0.077 | 0.150 | 0.034 |
| WM-S, empty library, frozen | 0.049 [0.041, 0.058] | | | | | |
| tool search, gap / prior; edit-type prior; submit as-is | 0.038 / 0.028; 0.019; 0.000 | | | | | |

The single-seed controls (`wms:random`, `wms:online+shuffle`, `wms:online+stale`, `wms:d1`) are in the
per-run files. They repeat the tune-stage collapse.

| WM-S online minus ... (paired) | Difference [95%] |
|---|---|
| policy_aux:plan | +0.422 [+0.333, +0.514] |
| policy_aux | +0.432 [+0.343, +0.524] |
| tells | +0.355 [+0.271, +0.448] |
| WM-S online, empty library | +0.329 [+0.247, +0.420] |
| black-box WM | +0.376 [+0.275, +0.483] |
| WM-S frozen | +0.083 [+0.056, +0.113] |

**Selection for the confirmation (rule fixed before this stage).**

- The C1 reference is the strongest learned non-world-model arm on this stage: **`policy_aux:plan`**
  (0.115, against `policy_aux` 0.105).

**Precision check (no threshold was changed).**

- The development error ratio against `policy_aux:plan` is about (1 - 0.537) / (1 - 0.115) = 0.52.
- Against the C1 threshold of 0.75, the 5-initialization, 4,000-task confirmation has ample precision.
  Development intervals, which come from 3 initializations and are dominated by variation between
  library-table seeds, stay well away from it.
- The per-slice minimum is library_B: 0.354 against 0.141.
