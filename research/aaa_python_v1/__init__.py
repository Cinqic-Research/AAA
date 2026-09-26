"""``aaa.python.v1``: the evidence-gated pre-scale phase for AAA's Python learners.

v1 is a *successor* to ``aaa.python.v0``, not an edit of it. v0's source,
specification, golden keys and negative development evidence are unchanged
history. v1 reuses v0's frozen safe subset, sandboxed CPython oracle and
counter-mode random stream by import (their bytes are part of v1's
fingerprint), and adds:

* :mod:`.generator` -- a versioned task generator (``aaa.python.gen.v1``) that
  repairs the v0 repair-candidate medoid shortcut (``AAA-192``) and separates
  generalization into declared slices: in-distribution, novel literals, novel
  identifier names, novel program structure and novel compositions;
* :mod:`.encoders` -- an explicit encoder interface: a minimal control
  (``e0``), the v0-tied lexical encoder (``e1``) and a structural encoder
  (``e2``), all fixed and parameter-free, with leakage rules enforced in code;
* :mod:`.models` -- a linear learner and a one-hidden-layer core whose width is
  the capacity variable, with pluggable heads, counted optimizer state and
  complete serialization;
* :mod:`.experiment`, :mod:`.stats`, :mod:`.recompute` -- diagnostic,
  development and (after a committed freeze) confirmation runs through the
  causal boundary, with crossed statistics and independent recomputation.

Nothing here is a demonstrated Python capability until the evidence says so.
"""

PROTOCOL_VERSION = "aaa.python.v1"
GENERATOR_VERSION = "aaa.python.gen.v1"
