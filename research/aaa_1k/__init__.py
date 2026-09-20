"""AAA-1K: a persistent recurrent predictive core, small enough to audit.

This phase is a research seed, not the Juniper agent. It builds and tests one
primitive: a 994-parameter gated recurrent network that learns online, keeps a
persistent hidden state, estimates its own predictive error, and is compared
against matched-capacity neural controls and the existing AAA baselines.

See `docs/aaa_1k_architecture.md` for the frozen specification and
`docs/aaa_charter.md` for what AAA is and is not.
"""

from __future__ import annotations

PHASE_VERSION = "aaa.1k.v1"
"""Scientific identity of this research phase. Bump before observing new evidence."""

__all__ = ["PHASE_VERSION"]
