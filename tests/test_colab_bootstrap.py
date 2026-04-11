from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_bootstrap import bootstrap  # noqa: E402


class ColabBootstrapTests(unittest.TestCase):
    def test_bootstrap_skip_mount_and_bind_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_dir = root / "repo"
            repo_dir.mkdir()
            with patch(
                "scripts.colab_bootstrap.sync_repo",
                return_value={"action": "fast_forwarded", "repo_dir": str(repo_dir), "head": "abc", "short_head": "abc"},
            ):
                payload = bootstrap(
                    repo_url="https://example.invalid/repo.git",
                    branch="main",
                    repo_dir=repo_dir,
                    drive_base_root=root / "drive" / "waymax_research",
                    drive_mountpoint=root / "drive",
                    waymo_dataset_root="gs://waymo_open_dataset_motion_v_1_1_0",
                    skip_drive_mount=True,
                    skip_bind=True,
                )
            manifest_path = Path(payload["manifest_path"])
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["repo"]["short_head"], "abc")
            self.assertTrue(manifest["drive_mount"]["skipped"])
            self.assertTrue(manifest["drive_binding"]["skipped"])


if __name__ == "__main__":
    unittest.main()
