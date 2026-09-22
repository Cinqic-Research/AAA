"""Repository automation must not rewrite files that carry scientific identity."""

from __future__ import annotations

import unittest
from pathlib import Path

from research.aaa_1k.identity import DEPENDENCY_LOCK

ROOT = Path(__file__).resolve().parents[1]
DEPENDABOT = ROOT / ".github" / "dependabot.yml"


def ecosystem_block(text: str, ecosystem: str) -> list[str]:
    """The lines of one ``updates`` entry, without comments or blank lines.

    PyYAML is not a dependency, and this file is small and flat enough that
    splitting on the list-item marker is exact.
    """

    blocks: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- package-ecosystem:"):
            blocks.append([])
        if blocks:
            blocks[-1].append(stripped)
    matches = [block for block in blocks if block[0] == f"- package-ecosystem: {ecosystem}"]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {ecosystem} entry, found {len(matches)}")
    return matches[0]


class DependabotTests(unittest.TestCase):
    """AAA-175: the lock is inside the aaa.1k.v1 fingerprint, so no bot may edit it."""

    def test_the_lock_is_a_fingerprinted_file(self) -> None:
        self.assertEqual(DEPENDENCY_LOCK, "requirements-lock.txt")

    def test_pip_version_updates_are_disabled(self) -> None:
        block = ecosystem_block(DEPENDABOT.read_text(encoding="utf-8"), "pip")
        self.assertIn("open-pull-requests-limit: 0", block)

    def test_every_pip_update_is_ignored_including_security_updates(self) -> None:
        block = ecosystem_block(DEPENDABOT.read_text(encoding="utf-8"), "pip")
        self.assertIn("ignore:", block)
        rules = block[block.index("ignore:") + 1 :]
        self.assertEqual(rules[:1], ['- dependency-name: "*"'])
        # An update-types or versions qualifier would narrow the rule and let
        # some updates (and security updates) through again.
        self.assertFalse([rule for rule in rules[1:] if rule.startswith(("update-types", "versions"))], rules)

    def test_the_parser_sees_both_ecosystems(self) -> None:
        text = DEPENDABOT.read_text(encoding="utf-8")
        self.assertEqual(ecosystem_block(text, "github-actions")[0], "- package-ecosystem: github-actions")
        with self.assertRaises(AssertionError):
            ecosystem_block(text, "npm")


if __name__ == "__main__":
    unittest.main()
