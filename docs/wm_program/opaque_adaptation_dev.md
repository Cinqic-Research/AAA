# `opaque.v0` adaptation, retention and plasticity (development)

Status: **development**, `adapt` range (development indices 5000-5999), WM-S table seeds 0-2.
Evidence: `$AAA_DATA_ROOT/opaque/eval/adapt_s012.json` (per-task bits). Command:
`python -m research.aaa_wm.opaque.experiment adapt --seeds 0,1,2`.

Design. The adapt range yields five library_B streams, each paired with an in_distribution
(library A) probe stream of 40 tasks. For every pair and seed:

- **online:** WM-S starts from the trained library-A table and learns from its own real runs.
- **frozen:** the same trained table, with no updates.
- **fresh:** online learning that starts from an *empty* table.
- **retention:** the table after the online B stream is frozen and scored on the A probe. The
  comparison is the unadapted table on the same probe.

| Condition (15 stream pairs x 40 tasks) | Success | First 10 tasks | Last 10 tasks |
|---|---:|---:|---:|
| online, library B | 0.373 | 0.353 | 0.300 |
| frozen, library B | 0.293 | 0.280 | 0.233 |
| fresh learner, library B | 0.215 | 0.240 | 0.200 |
| A probe after adapting to B | 0.470 | 0.487 | 0.340 |
| A probe, unadapted | 0.537 | 0.547 | 0.467 |

Reading (descriptive; no intervals yet):

- **Adaptation after a switch.** Online beats frozen by +0.080 on the changed library. Prior experience
  helps: experienced online beats the fresh learner by +0.158. The unchanged library functions stay
  useful.
- **There is no clean learning curve inside a stream.** The first-10 and last-10 means are dominated by
  task difficulty at those positions (frozen shows the same drop), so the size of the benefit is
  measured, but its speed is not resolved.
- **Adaptation causes forgetting: -0.067 on library-A probes.** The library model is one table
  with no notion of library versions. Real observations under B contradict A entries, and the online
  learner forgets and overwrites them. This is catastrophic interference in its simplest form.
- **No controller adaptation exists.** The planner is fixed. The combinations of online/frozen AAA with
  online/frozen WM therefore collapse to online versus frozen WM. That is a limitation of this design,
  not a result about controllers.

Implication for the claim. Any WM-S claim must be scoped to *within-version* benefit plus
adaptation with measured forgetting. A successor should test a versioned or context-keyed library
model (entries tagged by a detected library context) against this single table.
