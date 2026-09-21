"""The AAA iterative-improvement loop, piloted on AAA-1K.

This package is a *pilot* of a scientific process, not a permanent framework.
It implements one iteration of::

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

See ``docs/loop_protocol.md`` for the protocol and
``docs/loop_pilot_1_report.md`` for the first iteration.
"""

from __future__ import annotations

LOOP_PROTOCOL_VERSION = "aaa.loop.v0-pilot"
"""The protocol version under trial. ``v0`` because it has not earned ``v1``."""

PILOT_ITERATION_ID = "aaa1k-loop-0001"
"""Stable identity of the first iteration."""

__all__ = ["LOOP_PROTOCOL_VERSION", "PILOT_ITERATION_ID"]
