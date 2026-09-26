"""Corpus contamination and extraction-channel tests for the language track (numpy only)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.aaa_wm.lang import rules
from research.aaa_wm.lang.reports import number_words
from research.aaa_wm.lm.corpus import contamination, dialect_counts, verdict


class Contamination(unittest.TestCase):
    def test_injected_evaluation_text_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            clean = "The committee reviewed the annual budget and approved the new program for the county schools this year." * 2
            leak = "Heads up: f has a bug. Feeding in four is expected to produce thirty-two. In addition, the answer for ten has to be sixteen."
            Path(d, "x.train.txt").write_bytes((clean + "\x00" + "Intro text. " + leak + "\x00").encode())
            report = contamination(Path(d), [leak])
            self.assertEqual(report["contaminated_train_docs"].get("x.train.txt"), 1)


class Dialect(unittest.TestCase):
    def test_british_documents_are_dropped(self) -> None:
        uk = "The colour of the centre was a favourite of the neighbourhood, and the organisation liked its behaviour. " * 3
        us = "The color of the center was a favorite of the neighborhood, and the organization liked its behavior. " * 3
        self.assertEqual(verdict(uk), "british_spelling")
        self.assertIsNone(verdict(us))
        self.assertGreater(dialect_counts(us)[0], 0)


class Numbers(unittest.TestCase):
    def test_number_words_roundtrip_through_rules(self) -> None:
        for n in (-47, -4, 0, 7, 13, 20, 58, 103, 999, 1250):
            self.assertEqual(rules.words_to_int(number_words(n)), n)


if __name__ == "__main__":
    unittest.main()
