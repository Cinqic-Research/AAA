# Selected AAA final results — historical v1 snapshot

> Historical v1 evidence, preserved unchanged. This is the original moving-dot
> evaluation, and its result was **unfavourable**: the online linear learner did
> not convincingly improve with experience, generalized poorly when frozen, and
> lost to the analytic constant-motion baseline. Continued updates did improve
> it relative to its matched frozen copy after a change, but it remained
> inferior to constant motion.
>
> It is kept precisely because it was a negative result. The same learner is
> retained in the active benchmark as the `legacy_linear_sgd` diagnostic arm and
> reproduces that behaviour under the repaired protocol.
>
> This is not acceptance evidence for the active benchmark. See
> [`docs/benchmark_protocol.md`](../../docs/benchmark_protocol.md).

---

This directory contains a compact, reviewable snapshot from the completed local full evaluation run `20260909T160623Z-full`.

The snapshot includes the resolved configuration, development-only learning-rate selection, execution metadata, trained JSON checkpoint, aggregate summary, generated report, and static plots. The complete per-step JSONL/CSV logs remain in the ignored run directory `runs/20260909T160623Z-full/` on the machine that produced them; they are excluded from version control because they are several megabytes of generated data.

The run used separate development seeds `(101, 102, 103)`, training seeds `(11, 12, 13, 14, 15)`, and final evaluation seeds `(201, ..., 210)`. The local execution recorded commit `b8586f5f54a14fc9150d00b120e08f506bfbdd45` with a dirty working tree because the implementation was not committed when the run was produced. The complete source state is represented by the implementation commit that includes this snapshot; the metadata flag is retained so the original run is not misrepresented as having executed from a clean checkout.
