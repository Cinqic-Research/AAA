# AAA x world model x language model: factorial (development)

Status: **development**, `development` tune tasks (1,000) with statements in the **held-out** phrasing
families. Three seeds: adapter seed *s* is paired with WM-S table seed *s* and neural policy seed *s*.
Language channel: the R1 adapter v2 (5.2M parameters, from scratch, selection-family fine-tuning).
Summary: `language_factorial_dev_summary.json`. Code: `research/aaa_wm/lang/{experiment,factorial}.py`.

Every agent sees only the statement. Its tests come from the channel (`LANGUAGE_MODEL_PROPOSAL`).
Real `RUN` observations still compare against the true tests. With no parse, every decision system
falls back to prior-ordered search verified by real runs. A first launch lacked that fallback for
WM-S (it planned against zero tests); its outputs were discarded before any analysis and the run
was repeated (commit history).

| Decision system x language channel | Success [95%] |
|---|---|
| WM-S online x gold (perfect extraction; ceiling of the channel) | 0.525 [0.422, 0.629] |
| **WM-S online x from-scratch LM (full system)** | **0.276 [0.206, 0.339]** |
| WM-S online x hand-written rules | 0.159 [0.129, 0.189] |
| policy_aux:plan x gold / LM / rules / random LM | 0.126 / 0.122 / 0.113 / 0.110 |
| policy_aux x gold / LM / rules / random LM | 0.115 / 0.111 / 0.103 / 0.104 |
| WM-S online x random-init LM | 0.107 [0.076, 0.137] |
| any system x no language channel | 0.009 |

Exact extraction of all three tests by channel: gold 1.00, LM 0.285, rules 0.130, random-init LM 0.033, none 0.

| Pre-registered contrast (paired) | Difference [95%] | Sign |
|---|---|---|
| L1 pretraining (WM + LM vs WM + random-init LM) | +0.169 [+0.089, +0.248] | POSITIVE |
| L2 LM over rules (with WM) | +0.117 [+0.070, +0.162] | POSITIVE |
| language channel over none (with WM) | +0.267 [+0.197, +0.330] | POSITIVE |
| WM given the LM (WM + LM vs policy-in-planner + LM) | +0.154 [+0.077, +0.220] | POSITIVE |
| full system vs model-free policy + LM | +0.165 [+0.089, +0.232] | POSITIVE |
| full system vs policy-in-planner + rules | +0.163 [+0.085, +0.231] | POSITIVE |
| LM over rules *without* WM | +0.009 [-0.001, +0.020] | INCONCLUSIVE |
| gap from full system to perfect understanding (with WM) | +0.249 [+0.203, +0.300] | POSITIVE |

Reading (development only):

- **The full system beats every pairwise system on the development tune tasks.** That is option A of
  the program's full-system criterion.
- **The components interact.** The LM's benefit appears only when the world model can use the stated
  tests. The model-free policy barely conditions on test values: with perfect extraction it
  reaches 0.126, against 0.009 with none. The world model's benefit likewise needs the tests the LM
  supplies.
- **Language understanding is the bottleneck.** Perfect extraction would nearly double the full system
  (0.525 against 0.276). The adapter extracts every test exactly on only 28.5% of the held-out statements.
- **No regression to the non-language path.** The language code wraps agents from outside and never
  changes `opaque.v0`, WM-S or the confirmed evidence (L3 holds by construction; the confirmed numeric
  results are unaffected).

Limits:

- The statements come from templates, so the paraphrases are narrow.
- The tasks are development tune tasks: the same pool the adapter's selection family used, though
  with disjoint phrasing.
- This is **not** a confirmation. Declaring `LANGUAGE_INTEGRATION_SUCCESS` or `FULL_SYSTEM_SUCCESS`
  requires a prospective confirmation on fresh identities.
