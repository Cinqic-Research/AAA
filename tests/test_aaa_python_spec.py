"""Every value the aaa.python.v0 specification declares is read by the code, or declared descriptive (the AAA-040 lesson)."""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from research.aaa_python import GENERATOR_VERSION, PROTOCOL_VERSION, experiment, generator
from research.aaa_python import spec as spec_module

# Prose or identity fields: they document the protocol rather than drive it.
DESCRIPTIVE = {
    "protocol",
    "status",
    "generator_version",
    "question",
    "interpreter",
    "subset.forbidden_examples",
    "sandbox.isolation",
    "families.syntax.rung",
    "families.outcome.rung",
    "families.output.rung",
    "families.output.labels",
    "families.localize.rung",
    "families.localize.labels",
    "families.repair.rung",
    "families.repair.labels",
    "splits.order",
    "splits.confirmation",
    "splits.disjointness",
    "metrics",
    "statistics.unit",
    "statistics.method",
    "statistics.adaptation_estimand",
    "statistics.retention_estimand",
    "promotion",
}


class Recording(dict):  # type: ignore[type-arg]
    """A dict that records every key path read through it."""

    def __init__(self, data: dict[str, Any], seen: set[str], prefix: str = "") -> None:
        super().__init__(data)
        self._seen, self._prefix = seen, prefix

    def _wrap(self, key: str, value: Any) -> Any:
        path = f"{self._prefix}{key}"
        self._seen.add(path)
        return Recording(value, self._seen, path + ".") if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        return self._wrap(key, super().__getitem__(key))

    def get(self, key: str, default: Any = None) -> Any:
        return self._wrap(key, super().get(key, default)) if key in self else default

    # items() and values() deliberately do not record: json.dumps walks a dict
    # subclass through items() to hash the whole specification, and hashing a
    # value is not consuming it. Without this the test passed vacuously.
    def items(self) -> Any:
        return dict.items(self)

    def values(self) -> Any:
        return dict.values(self)


def _leaves(data: dict[str, Any], prefix: str = "") -> set[str]:
    out: set[str] = set()
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value and path not in DESCRIPTIVE:
            out |= _leaves(value, path + ".")
        else:
            out.add(path)
    return out


class SpecConsumptionTests(unittest.TestCase):
    def test_every_declared_leaf_is_read_or_explicitly_descriptive(self) -> None:
        seen: set[str] = set()
        recording = Recording(spec_module.load(), seen)
        generator._pool.cache_clear()
        try:
            with mock.patch.object(spec_module, "load", return_value=recording):
                experiment.develop(spec=recording, quick=True, log=lambda _: None)
                generator.build("attack", "syntax", [0], recording)  # reads the probe and attack pools
        finally:
            generator._pool.cache_clear()
        unread = sorted(
            leaf
            for leaf in _leaves(spec_module.load())
            if leaf not in DESCRIPTIVE and not any(p == leaf or p.startswith(leaf + ".") for p in seen)
        )
        self.assertEqual(unread, [], "declared values the code never reads")

    def test_identity_fields_match_the_code(self) -> None:
        spec = spec_module.load()
        self.assertEqual(spec["protocol"], PROTOCOL_VERSION)
        self.assertEqual(spec["generator_version"], GENERATOR_VERSION)


if __name__ == "__main__":
    unittest.main()
