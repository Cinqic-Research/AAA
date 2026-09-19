# AAA observation-noise phase-closure decision ledger

Date: 2026-09-19

This ledger was created after the read-only repository, GitHub, storage, and
branch snapshot and before phase-closure remediation.  It records decisions,
not experimental results.  Historical failures and superseded designs remain
part of the repository record.

## Observation-noise replication budget

Decision:
Do not execute the v1.1 formal A/B pair during this engineering closure.

Context:
The retained pilot projects 59,043,840 scored records and about 38.5 GB of
compressed primitive records per batch.  The dedicated HDD removes the local
capacity blocker, but no quantitative precision analysis yet justifies the
10-lineage x 32-episode x 3-realization formal budget.

Options considered:

Option A:
  Pros: Executes the already frozen design and could establish the declared
  observation-noise endpoints.
  Cons: Consumes roughly 77 GB plus overhead and tens of hours per batch before
  the replication count has a demonstrated Monte Carlo precision basis.

Option B:
  Pros: Closes the engineering phase honestly, preserves v1.1 as unobserved,
  and permits a successor protocol to size replication before any confirmation
  stream is observed.
  Cons: Produces no formal observation-noise scientific outcome in this phase.

Chosen option:
Option B.

Reason:
Capacity is not scientific justification.  Running the existing large budget
would create expensive evidence without first establishing that the budget is
necessary or sufficient.

Scientific-identity impact:
No v1.1 confirmation stream will be observed.  Its planned batches will be
retired unobserved during closure so that they cannot be reused accidentally.
Any reduced or otherwise changed formal design must receive a new protocol
version, source freeze, and fresh batch identities before observation.

Evidence impact:
The scientific outcome remains `NOT EXECUTED`, not `PASS`, `FAIL`, or
`INCONCLUSIVE`.

Reversibility:
High.  A future phase can retain the v1.1 budget or create a justified
successor before observing fresh data.

Follow-up, if any:
Use the actual primary estimands and pilot variance structure to define a Monte
Carlo precision target before reserving successor batches.

## Local HDD and external archival requirement

Decision:
Treat the dedicated HDD as verified local working storage, and treat external
immutable archival as publication-grade evidence rather than an engineering
merge gate for the current internal phase.

Context:
`/media/cinqic/Cinqic Storage` is a dedicated ext4 filesystem with ample space,
but it is neither off-site nor an independent failure domain.

Options considered:

Option A:
  Pros: Keeping external immutable storage as a hard prerequisite provides the
  strongest durability and independent retrieval semantics.
  Cons: It blocks engineering integration on credentials, billing, retention,
  and provider work that does not test the predictive hypothesis.

Option B:
  Pros: Allows tested infrastructure to merge while claims distinguish local
  verification from externally durable evidence.
  Cons: A local disk failure can destroy primitive evidence.

Option C:
  Pros: Implementing a genuine versioned/WORM external backend now could meet
  the strongest evidence policy.
  Cons: No authorized destination or credentials exist, and speculative
  storage infrastructure adds substantial scope and maintenance burden.

Chosen option:
Option B.

Reason:
It matches AAA's current internal R&D purpose without fabricating archival
properties or forcing a cloud purchase.

Scientific-identity impact:
No retroactive claim changes.  Any future protocol that relaxes or changes a
formal archive requirement must be versioned before observation.

Evidence impact:
Local evidence retention may be `VERIFIED`; external durable archival remains
`NOT VERIFIED` and is not required for this internal engineering closure.

Reversibility:
High.  A real immutable backend can be added when publication or collaboration
requires it.

Follow-up, if any:
Select and validate an external failure domain before making publication-grade
durability or independent-retrieval claims.

## PR #12 disposition

Decision:
Close PR #12 without merging it into the core repository.

Context:
The PR adds an isolated Playground and a deliberately tiny neural experiment,
4,812 lines on a branch that diverged before the benchmark-v2.1 integration.
Its neural candidate loses important comparisons against existing baselines and
is not part of official selection.

Options considered:

Option A:
  Pros: A visual debugging and education surface can make AAA behavior easier
  to inspect.
  Cons: It adds a large GUI/backend maintenance and fingerprint surface during
  scientific phase closure.

Option B:
  Pros: Preserves a small visualization while discarding the neural work.
  Cons: Requires a new extraction and integration effort with no current core
  acceptance need.

Option C:
  Pros: Preserves the complete exploratory record in PR history without
  expanding the maintained core or implying neural promotion.
  Cons: The Playground is not available from `main`.

Chosen option:
Option C.

Reason:
The exploratory value does not outweigh present maintenance and scientific
identity costs.  A future visualization phase can reuse the recorded work.

Scientific-identity impact:
None; the Playground and tiny NN do not enter the final scientific tree.

Evidence impact:
The PR's evidence remains historical exploratory evidence, not official AAA
model evidence.

Reversibility:
High; the closed PR and commits remain reviewable.

Follow-up, if any:
Start a fresh, current-main visualization proposal if a maintained Playground
becomes a product requirement.

## PR #13 disposition

Decision:
Close PR #13 without merging its archive prototype or future experiment design.

Context:
The draft is stacked on PR #11 and contains a speculative external archive
adapter plus a `DESIGN_ONLY_UNFROZEN_UNEXECUTED` post-noise proposal.

Options considered:

Option A:
  Pros: Promotes reusable archive abstractions and fault-injection work.
  Cons: Adds unneeded production surface without an authorized external store.

Option B:
  Pros: Keeps current runtime small and preserves research in the PR record.
  Cons: Production adoption later will require a fresh review and integration.

Chosen option:
Option B.

Reason:
Neither component is necessary to close the present phase, and the future
experiment must remain separate from current evidence.

Scientific-identity impact:
None; neither prototype nor future design enters the final freeze.

Evidence impact:
The PR remains design history only and is not represented as executed evidence.

Reversibility:
High; selected concepts can be reintroduced from the recorded commits.

Follow-up, if any:
Reassess the minimal backend only after a real external destination and evidence
requirement exist.

## PR #14 disposition

Decision:
Integrate FontTools 4.65.0 before the final observation-noise source freeze.

Context:
The update changes one exact lock entry.  It is the latest available release,
its upstream changes include path-containment and escaping fixes, and both
hosted CPU CI executions passed all required jobs.

Options considered:

Option A:
  Pros: Retains the already tested lock and avoids identity churn.
  Cons: Leaves a legitimate current dependency/security-hardening update for
  immediately after the freeze.

Option B:
  Pros: Establishes the final dependency identity before freezing and includes
  upstream correctness/security hardening.
  Cons: Requires complete validation on the integrated tree.

Chosen option:
Option B.

Reason:
The update is small, green, current, and belongs before—not after—the final
scientific source identity is generated.

Scientific-identity impact:
The lock change participates in the final fingerprint.

Evidence impact:
Historical evidence retains its original lock identity; new validation records
the updated lock.

Reversibility:
High through a normal lock revert if integrated validation exposes a defect.

Follow-up, if any:
Run lock, package, clean-install, and complete CI validation after integration.

## Artifact filename representation

Decision:
Use a deterministic opaque storage identifier derived from the canonical trial
identity; keep the full identity in structured schedule metadata and indexes.

Context:
PR #11 writes colon-separated trial identities into filenames, and GitHub's
artifact service rejects those components.

Options considered:

Option A:
  Pros: Percent-encoding preserves recognizable identity text.
  Cons: It is easy to implement incompletely, length can grow substantially,
  and case/normalization rules complicate collision analysis.

Option B:
  Pros: A full cryptographic digest over canonical identity bytes is portable,
  fixed-length, deterministic, and collision-resistant.
  Cons: Filenames are not human-readable without the index.

Chosen option:
Option B, with a short safe prefix and the full lowercase SHA-256 digest.

Reason:
Scientific identity already belongs in structured data.  Storage names should
be portable identifiers rather than an alternate parser-dependent identity.

Scientific-identity impact:
Trial identities are unchanged; only their storage representation changes.

Evidence impact:
Schedules and indexes must bind storage IDs to complete canonical identities;
verifiers must not reconstruct identity by parsing filenames.

Reversibility:
Moderate; changing representation later requires a format version or migration.

Follow-up, if any:
Exercise representative values across every identity dimension and the actual
hosted artifact-upload path.

## Sharding versus packed containers

Decision:
Retain the existing deterministic shard format for this closure unless direct
HDD profiling demonstrates an operational failure.

Context:
Packing would reduce filesystem objects but introduces a new container/index
format immediately before phase closure.

Options considered:

Option A:
  Pros: Existing sharding is implemented, independently verified, resumable,
  and covered by mutation tests.
  Cons: Formal scale creates many files and may stress HDD metadata operations.

Option B:
  Pros: Packed containers improve sequential transfer and reduce object count.
  Cons: Adds offset/range logic and new corruption/resume failure modes.

Chosen option:
Option A, subject to measured HDD behavior.

Reason:
Format redesign is justified by measured harm, not aesthetic preference.

Scientific-identity impact:
No container-format or protocol change is introduced.

Evidence impact:
HDD creation, traversal, verification, resume-scan, and cleanup measurements
will be retained in the closure evidence.

Reversibility:
Moderate; a future format version can introduce packing.

Follow-up, if any:
Revisit packing only if measured scale projections show unacceptable metadata
or verification cost.

## Scientific fingerprint scope

Decision:
Keep the broad tracked, non-generated repository fingerprint scope.

Context:
Narrowing scope could avoid churn but creates new opportunities for relevant
code, tests, tools, workflow, or configuration to escape identity.

Options considered:

Option A:
  Pros: Broad scope is simple to reason about and fail-closed.
  Cons: Legitimate repository changes require a new fingerprint.

Option B:
  Pros: A narrow dependency closure reduces unrelated churn.
  Cons: Proving that excluded surfaces cannot influence execution adds a more
  complex security boundary and regression burden.

Chosen option:
Option A.

Reason:
AAA-121 showed that incomplete source identity creates false confidence.  The
cost of a new fingerprint is lower than a subtle bypass.

Scientific-identity impact:
All accepted repository changes are integrated before the final freeze.

Evidence impact:
Historical freezes retain their exact earlier scope and hashes.

Reversibility:
Low without a new manifest schema and enforcement proof.

Follow-up, if any:
None for this phase.

## Formal A/B execution

Decision:
Defer formal A/B and merge only after the retained implementation, verifier,
smoke, and CI paths are validated.

Context:
No formal observation-noise stream has been observed.  Engineering integration
and scientific confirmation are distinct gates.

Options considered:

Option A:
  Pros: Produces a formal scientific endpoint during closure.
  Cons: Inherits the unjustified replication budget and unresolved
  publication-grade archival choice.

Option B:
  Pros: Removes ambiguous PR state while keeping scientific claims honest.
  Cons: Leaves the primary outcome unexecuted.

Chosen option:
Option B.

Reason:
The repository can be ready for its next phase without pretending that missing
confirmation data exists.

Scientific-identity impact:
No confirmation identity is consumed.

Evidence impact:
Observation-noise scientific outcome is `NOT EXECUTED`.

Reversibility:
High before any future stream is observed.

Follow-up, if any:
Create a versioned, precision-justified formal plan in the next phase.

## Branch cleanup

Decision:
After accepted work is merged, close all four open PRs with factual
dispositions and delete obsolete ordinary remote branches; preserve PR refs,
merged history, tags, and append-only reservation refs.

Context:
Open stacked and exploratory branches currently present multiple conflicting
"current" states.  `feat/aaa-prototype` has one unique commit predating and
superseded by merged prototype history.

Options considered:

Option A:
  Pros: Leaving branches maximizes easy discoverability.
  Cons: Preserves ambiguous active-looking integration states indefinitely.

Option B:
  Pros: Leaves one coherent `main` while GitHub PR history and immutable refs
  retain provenance.
  Cons: Reusing an old branch requires recovering it from PR/commit history.

Chosen option:
Option B.

Reason:
Historical preservation does not require stale ordinary branches.

Scientific-identity impact:
None; ref cleanup does not rewrite commits.

Evidence impact:
No evidence commits are deleted or rewritten.

Reversibility:
High while commits remain reachable through PRs, merged history, or tags.

Follow-up, if any:
Verify reachability before deleting each branch and never delete reservation
refs.

## Phase tag

Decision:
Create an annotated research-milestone tag on the exact final merge commit, but
do not create a GitHub Release.

Context:
A human-readable immutable boundary is useful before the next architectural
phase, while a product release would overstate the repository's status.

Options considered:

Option A:
  Pros: No new permanent ref.
  Cons: Makes the phase boundary harder to find and communicate.

Option B:
  Pros: Gives the reviewed state a stable checkout name.
  Cons: Could be mistaken for a product release if poorly named or described.

Chosen option:
Option B, explicitly labeled as a research milestone.

Reason:
The provenance benefit outweighs the small permanent-reference cost when the
annotation states the non-release boundary.

Scientific-identity impact:
The tag does not change tree contents or scientific hashes.

Evidence impact:
The tag identifies the exact final engineering phase boundary only.

Reversibility:
Low; published tags should be treated as permanent.

Follow-up, if any:
Do not attach a GitHub Release or imply product stability.
