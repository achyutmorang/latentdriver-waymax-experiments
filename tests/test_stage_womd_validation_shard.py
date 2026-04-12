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
from scripts import stage_womd_validation_shards as stage_many_script


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

    def test_many_shard_staging_is_resumable_and_uses_tmp_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing = (
                root
                / "waymo_open_dataset_motion_v_1_1_0"
                / "uncompressed"
                / "tf_example"
                / "validation"
                / "validation_tfexample.tfrecord-00000-of-00150"
            )
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"already")

            def fake_copy(_source, target):
                Path(target).write_bytes(b"copied")
                return {
                    "cli": "gcloud-storage",
                    "command": ["gcloud", "storage", "cp", _source, str(target)],
                    "stdout": "",
                    "stderr": "",
                    "target": str(target),
                }

            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "stage_womd_validation_shards.py",
                "--gcs-root",
                "gs://waymo_open_dataset_motion_v_1_1_0",
                "--staging-root",
                td,
                "--count",
                "2",
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(stage_many_script, "copy_gcs_to_local", side_effect=fake_copy):
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        rc = stage_many_script.main()

            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["skipped_existing"], 1)
            self.assertEqual(payload["copied"], 1)
            self.assertTrue((existing.parent / "validation_tfexample.tfrecord-00001-of-00150").is_file())
            self.assertFalse(any(existing.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
