from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pull_latest_debug import _remote_path, main  # noqa: E402


class PullLatestDebugTests(unittest.TestCase):
    def test_remote_path_formats_rclone_drive_path(self) -> None:
        self.assertEqual(
            _remote_path("gdrive_ro", "waymax_research/latentdriver_waymax_experiments", "debug_runs/latest_failure"),
            "gdrive_ro:waymax_research/latentdriver_waymax_experiments/debug_runs/latest_failure",
        )

    def test_main_dry_run_does_not_require_rclone(self) -> None:
        argv = [
            "pull_latest_debug.py",
            "--dry-run",
            "--which",
            "latest_failure",
            "--target-root",
            "/tmp/latentdriver_pull_debug_test",
        ]
        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
            rc = main()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(rc, 0)
        self.assertTrue(payload["ready"])
        self.assertEqual([item["name"] for item in payload["results"]], ["LATEST.json", "LATEST_FAILURE.json", "latest_failure"])
        self.assertTrue(all(item["dry_run"] for item in payload["results"]))


if __name__ == "__main__":
    unittest.main()
