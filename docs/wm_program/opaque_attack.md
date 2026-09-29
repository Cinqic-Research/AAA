# `opaque.v0` attack pool (used once)

Status: **attack evidence**. The `attack` split has 1,480 tasks, with every pilot, train and
development program excluded. It was run once, with the selections and budgets fixed at the tune
stage (plus the red-team controls declared before this run). Nothing was re-tuned afterward.
Summary: `$AAA_DATA_ROOT/opaque/eval/attack_summary.json`. Crossed bootstrap over seeds and streams.

| Arm | Seeds | Success [95%] |
|---|---:|---|
| WM-S online (candidate) | 3 | **0.525 [0.428, 0.621]** |
| WM-S frozen | 1 | 0.322 |
| black-box neural WM, depth 1 | 1 | 0.193 |
| WM-S, online learning from an empty library | 1 | 0.180 |
| grammar-legality shortcut (`tells`, no world model) | - | 0.162 [0.141, 0.182] |
| policy_aux inside the WM-S planner | 1 | 0.103 |
| policy_aux, tool search | 3 | 0.100 [0.081, 0.119] |
| WM-S, empty library (interpreter only) | - | 0.039 |
| tool search, gap-reading / prior | - | 0.031 / 0.018 |
| **WM-S online with every expected value corrupted by +1** | 1 | **0.024** |

| Contrast (paired, crossed) | Difference [95%] |
|---|---|
| WM-S online - policy_aux | +0.425 [+0.330, +0.516] |
| WM-S online - tells | +0.362 [+0.266, +0.459] |

Reading:

- **No attack explains the gain.** The strongest shortcut (grammar legality) and the learned
  model-free policy, even inside the same planner, stay far below.
- **WM-S depends on the stated specification.** Corrupting the expected values collapses it to 0.024,
  so it is not solving the tasks from surface form.
- **The learned library carries the gain.** An empty library gives 0.039 frozen and 0.180 when learning
  online within episodes.
- Seed 0 alone scored 0.426; across three seeds the mean is 0.525. Variation between library-table
  seeds is large, which is why confirmation uses five fresh initializations.
