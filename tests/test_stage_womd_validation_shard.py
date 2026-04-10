from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import stage_womd_validation_shard as stage_script


class StageWomdValidationShardTests(unittest.TestCase):
    def test_transfer_command_is_emitted_to_stderr_and_stdout_is_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "stage_womd_validation_shard.py",
                "--gcs-root",
                "gs://waymo_open_dataset_motion_v_1_1_0",
                "--staging-root",
                td,
                "--shard-index",
                "0",
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(
                    stage_script,
                    "copy_gcs_to_local",
                    return_value={
                        "cli": "gcloud-storage",
                        "command": ["gcloud", "storage", "cp", "gs://bucket/file", f"{td}/file"],
                        "stdout": "",
                        "stderr": "copied",
                        "target": f"{td}/file",
                    },
                ):
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        rc = stage_script.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["gcs_root"], "gs://waymo_open_dataset_motion_v_1_1_0")
            self.assertIn("[stage-womd] $ gcloud storage cp", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
