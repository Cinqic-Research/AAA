"""Post-freeze CI gates compare live WM/language source with retained freezes."""

from __future__ import annotations

import unittest
from unittest import mock

from tools import check_aaa_wm_source_fingerprints as audit


class SourceFingerprintAudit(unittest.TestCase):
    def test_live_sources_match_the_retained_freezes(self) -> None:
        self.assertEqual(audit.verify(), [])

    def test_opaque_fingerprint_drift_fails(self) -> None:
        live = audit.opaque_freeze.fingerprint(audit.ROOT)
        drifted = {**live, "sha256": "0" * 64}
        with mock.patch.object(audit.opaque_freeze, "fingerprint", return_value=drifted):
            problems = audit.verify()
        self.assertIn("opaque source fingerprint differs from the committed freeze", problems)

    def test_opaque_source_file_hash_drift_fails(self) -> None:
        live = audit.opaque_freeze.fingerprint(audit.ROOT)
        drifted = {**live, "files": {**live["files"], "research/aaa_wm/opaque/models.py": "0" * 64}}
        with mock.patch.object(audit.opaque_freeze, "fingerprint", return_value=drifted):
            problems = audit.verify()
        self.assertIn("opaque source-file hashes differ from the committed freeze", problems)

    def test_language_fingerprint_drift_fails(self) -> None:
        with mock.patch.object(audit.language_confirm, "fingerprint", return_value="0" * 64):
            problems = audit.verify()
        self.assertIn("language source fingerprint differs from the committed freeze", problems)

    def test_language_source_list_drift_fails(self) -> None:
        with mock.patch.object(audit.language_confirm, "source_files", return_value=[]):
            problems = audit.verify()
        self.assertIn("language source-file list differs from the committed freeze", problems)


if __name__ == "__main__":
    unittest.main()
