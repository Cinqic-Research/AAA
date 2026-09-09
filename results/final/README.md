# Selected AAA final results

This directory contains a compact, reviewable snapshot from the completed local full evaluation run `20260909T160623Z-full`.

The snapshot includes the resolved configuration, development-only learning-rate selection, execution metadata, trained JSON checkpoint, aggregate summary, generated report, and static plots. The complete per-step JSONL/CSV logs remain available in the local ignored run directory at `/home/cinqic/Documents/AAA/runs/20260909T160623Z-full/`; they are excluded from version control because they are several megabytes of generated data.

The run used separate development seeds `(101, 102, 103)`, training seeds `(11, 12, 13, 14, 15)`, and final evaluation seeds `(201, ..., 210)`. The local execution recorded commit `b8586f5f54a14fc9150d00b120e08f506bfbdd45` with a dirty working tree because the implementation was not committed when the run was produced. The complete source state is represented by the implementation commit that includes this snapshot; the metadata flag is retained so the original run is not misrepresented as having executed from a clean checkout.
