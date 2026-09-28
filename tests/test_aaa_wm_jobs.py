"""A resumed queue must report a vanished job without its output as failed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.aaa_wm.jobs import Adopted, completed_successfully


class Jobs(unittest.TestCase):
    def test_adopted_job_without_output_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = {"output": str(Path(directory) / "result.json")}
            self.assertFalse(completed_successfully(Adopted(99999999), job))
            Path(job["output"]).write_text("ok")
            self.assertTrue(completed_successfully(Adopted(99999999), job))


if __name__ == "__main__":
    unittest.main()
