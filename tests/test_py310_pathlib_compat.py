"""Python 3.10 compatibility for tests that patch ``os.replace``.

On Python 3.10, ``pathlib.Path.replace`` calls ``os.replace`` through
``pathlib._NormalAccessor.replace``, bound once when ``pathlib`` is imported,
so ``mock.patch("os.replace")`` never reaches it. Python 3.11 and later call
``os.replace`` directly. The frozen ``aaa.erudition.v0`` store test
``test_failed_write_leaves_no_partial_object`` depends on that patch, and
its file is inside the phase fingerprint, so it cannot be edited without a
successor identity.

Importing this module on Python 3.10 rebinds the accessor to look
``os.replace`` up at call time. Behaviour is unchanged: it still calls
``os.replace`` with the same arguments. Test discovery imports every module
before running any test, so the rebinding is in place for the whole suite.
Running ``tests.test_aaa_erudition`` alone on Python 3.10 needs this module
imported first.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _replace(src: str, dst: str) -> None:
    os.replace(src, dst)  # noqa: PTH105 - the late-bound call is the point


if sys.version_info < (3, 11):
    pathlib._NormalAccessor.replace = staticmethod(_replace)


class PathReplaceTests(unittest.TestCase):
    def test_replace_still_renames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "a", Path(tmp) / "b"
            source.write_text("x", "utf-8")
            source.replace(target)
            self.assertFalse(source.exists())
            self.assertEqual(target.read_text("utf-8"), "x")

    def test_patched_os_replace_is_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "a"
            source.write_text("x", "utf-8")
            with mock.patch("os.replace", side_effect=OSError("refused")), self.assertRaises(OSError):
                source.replace(Path(tmp) / "b")
            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
