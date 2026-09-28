"""The post-run overlap detector recognizes simple numeric and punctuation variants."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_aaa_wm_language_contamination import extended_overlap


class ContaminationVariants(unittest.TestCase):
    def test_number_words_and_punctuation_variants_match(self) -> None:
        evaluation = "Heads up: f has a bug. If you call f on negative four, you should get back six."
        training = "Heads up, f has a bug! If you call f on -4 you should get back 6."
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "synthetic.train.txt"
            shard.write_text(training + "\0A wholly unrelated training document.")
            result = extended_overlap(Path(directory), [evaluation])
        self.assertEqual(result["sources"][shard.name]["normalized_13gram_docs"], 1)
        self.assertEqual(result["sources"][shard.name]["static_phrase_docs"], 1)


if __name__ == "__main__":
    unittest.main()
