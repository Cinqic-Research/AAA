"""``aaa.python.v0``: AAA's first Python-learning research phase.

A small, deterministic, causal environment in which an adaptive system can be
measured learning Python from programs and interpreter feedback:

* :mod:`.subset` and :mod:`.oracle` -- a frozen safe subset of Python, checked
  by AST validation and executed by CPython in an isolated subprocess. The
  interpreter is the external ground truth, never a baseline;
* :mod:`.generator` -- deterministic, version-independent task generation with
  mechanically separated train / development / probe / attack / confirmation
  identities;
* :mod:`.episode` -- the causal boundary: present, act, commit, reveal, score,
  and learn only where the protocol permits;
* :mod:`.learners` -- a minimal online learner, its frozen and memory-disabled
  controls, and deterministic baselines;
* :mod:`.experiment`, :mod:`.stats`, :mod:`.recompute` -- development runs,
  crossed-design statistics and independent recomputation.

Nothing in this package is a demonstrated Python capability. Formal
confirmation is not admitted in v0.
"""

PROTOCOL_VERSION = "aaa.python.v0"
GENERATOR_VERSION = "aaa.python.gen.v0"
