"""AAA-1K v2: the final research pass at roughly 1,000 trainable parameters.

This phase is new and separately versioned. It does not modify ``aaa.1k.v1``
(``research/aaa_1k/``), Champion 0, Champion 1, the loop package or any of
their evidence. Champion 1 -- the 994-parameter GRU with reach-gated target
unfolding -- is this phase's reference system and the baseline to beat.

What is new here is (1) a batched, backend-generic implementation of the online
predictive core that runs the same array program on NumPy (CPU) or CuPy (CUDA),
many independent cells in lockstep; (2) a separately versioned stress suite of
dot families; (3) independent external benchmarks; and (4) a fresh candidate
tournament under the loop's development / attack / freeze / confirmation
discipline, on identities of this phase's own namespace.

See ``docs/aaa_1k_v2_research_brief.md`` for the question this phase asks and
``docs/aaa_1k_v2_architecture.md`` for the specification.
"""

from __future__ import annotations

PHASE_VERSION = "aaa.1k.v2"
"""Scientific identity of this phase. Its evidence records this value."""

REFERENCE_SYSTEM = "aaa1k-champion-1"
"""The prior system every claim in this phase is measured against."""

PARAMETER_CAP = 1000
"""Maximum trainable parameters for any candidate of this phase, recounted from arrays."""

__all__ = ["PARAMETER_CAP", "PHASE_VERSION", "REFERENCE_SYSTEM"]
