"""The AAA iterative-improvement loop, first piloted and validated on AAA-1K.

The package implements a scientific process::

    Observe -> Classify -> Diagnose -> Hypothesize -> Falsify -> Intervene
    Minimally -> Attack -> Confirm Fresh -> Decide -> Preserve -> Repeat

as two nested loops: an inner development loop that may iterate on
development and diagnostic identities, and an outer promotion loop that runs
once per frozen challenger on fresh confirmation identities.

It is deliberately a sibling of :mod:`research.aaa_1k` rather than a module
inside it. The AAA-1K phase fingerprint covers ``research/aaa_1k/**``; keeping
the loop outside that prefix leaves Champion 0's scientific identity
byte-for-byte unchanged, so the champion this loop starts from can be
reproduced at exactly its recorded fingerprint on the same checkout.

Iterations 0001--0003 were the pilot.  Iteration 0006 completed the first
freeze, durable claim, fresh confirmation, independent recomputation and
promotion; independent review then established the current ``aaa.loop.v1``
governance version.  Historical evidence remains labelled
``aaa.loop.v0-pilot`` and is never rewritten by the version transition.

See ``docs/loop_protocol.md`` for the current protocol,
``docs/loop_pilot_report.md`` for the pilot, and
``docs/loop_report_0004_0006.md`` for the first completed outer cycle.
"""

from __future__ import annotations

LOOP_PROTOCOL_VERSION = "aaa.loop.v1"
"""The current governance version for future loop work."""

HISTORICAL_PILOT_PROTOCOL_VERSION = "aaa.loop.v0-pilot"
"""Version recorded by immutable iteration-0001--0006 artifacts."""

PILOT_ITERATION_ID = "aaa1k-loop-0001"
"""Stable identity of the first iteration."""

__all__ = ["HISTORICAL_PILOT_PROTOCOL_VERSION", "LOOP_PROTOCOL_VERSION", "PILOT_ITERATION_ID"]
