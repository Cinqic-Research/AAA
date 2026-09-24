"""Synthetic mount tests for the FLOWBOX work-storage gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.storage_preflight import find_storage, validate_paths


class StoragePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.devices = [{"children": [{"label": "Cinqic Storage", "uuid": "hdd-uuid", "fstype": "ext4"}]}]
        self.mounts = [
            {
                "target": "/",
                "uuid": "ssd-uuid",
                "fstype": "ext4",
                "children": [{"target": "/mnt/cinqic", "uuid": "hdd-uuid", "fstype": "ext4"}],
            }
        ]

    def test_live_uuid_selects_mount(self) -> None:
        self.assertEqual(find_storage(self.devices, self.mounts), Path("/mnt/cinqic"))

    def test_labelled_directory_on_other_device_is_rejected(self) -> None:
        self.mounts[0]["children"][0]["uuid"] = "ssd-uuid"
        with self.assertRaisesRegex(ValueError, "not mounted"):
            find_storage(self.devices, self.mounts)

    def test_wrong_filesystem_and_missing_device_fail(self) -> None:
        self.devices[0]["children"][0]["fstype"] = "vfat"
        with self.assertRaisesRegex(ValueError, "expected exactly one"):
            find_storage(self.devices, self.mounts)

    def test_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mount = root / "hdd"
            mount.mkdir()
            (mount / "escape").symlink_to(root)
            validate_paths(mount, [mount / "workspace"])
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_paths(mount, [mount / "escape" / "elsewhere"])


if __name__ == "__main__":
    unittest.main()
