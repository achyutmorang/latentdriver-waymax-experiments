from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.womd import (
    WOMD_VERSION,
    copy_gcs_to_local,
    first_sharded_tfrecord_path,
    is_gcs_uri,
    local_dataset_uri_exists,
    probe_gcs_uri,
    resolve_dataset_uri,
    validation_shard_uri,
    waymo_dataset_root,
)


class WomdPathTests(unittest.TestCase):
    def test_resolve_dataset_uri_for_local_parent_root(self) -> None:
        root = "/tmp/womd"
        relative = f"{WOMD_VERSION}/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150"
        resolved = resolve_dataset_uri(root, relative)
        self.assertEqual(resolved, f"{root}/{relative}")

    def test_resolve_dataset_uri_for_gcs_version_root(self) -> None:
        root = f"gs://{WOMD_VERSION}"
        relative = f"{WOMD_VERSION}/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150"
        resolved = resolve_dataset_uri(root, relative)
        self.assertEqual(
            resolved,
            f"gs://{WOMD_VERSION}/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150",
        )

    def test_first_sharded_tfrecord_path_materializes_first_shard(self) -> None:
        dataset_uri = "/tmp/validation_tfexample.tfrecord@150"
        path = first_sharded_tfrecord_path(dataset_uri)
        self.assertEqual(str(path), "/tmp/validation_tfexample.tfrecord-00000-of-00150")

    def test_local_dataset_uri_exists_understands_sharded_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "validation_tfexample.tfrecord-00000-of-00150"
            shard.write_text("x", encoding="utf-8")
            self.assertTrue(local_dataset_uri_exists(str(Path(td) / "validation_tfexample.tfrecord@150")))

    def test_validation_shard_uri_uses_expected_validation_filename(self) -> None:
        uri = validation_shard_uri(f"gs://{WOMD_VERSION}", 7)
        self.assertEqual(
            uri,
            f"gs://{WOMD_VERSION}/uncompressed/tf_example/validation/validation_tfexample.tfrecord-00007-of-00150",
        )

    def test_waymo_dataset_root_rejects_gcs_root(self) -> None:
        os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = f"gs://{WOMD_VERSION}"
        with self.assertRaises(EnvironmentError):
            waymo_dataset_root()

    def test_is_gcs_uri(self) -> None:
        self.assertTrue(is_gcs_uri("gs://bucket/path"))
        self.assertFalse(is_gcs_uri("/tmp/path"))

    @patch("latentdriver_waymax_experiments.womd.subprocess.run")
    @patch("latentdriver_waymax_experiments.womd.shutil.which")
    def test_probe_gcs_uri_prefers_gcloud_storage(self, mock_which, mock_run) -> None:
        mock_which.side_effect = lambda name: f"/usr/bin/{name}" if name in {"gcloud", "gsutil"} else None
        mock_run.return_value = __import__("subprocess").CompletedProcess(
            ["gcloud", "storage", "ls", "gs://bucket/file"],
            0,
            stdout="gs://bucket/file\n",
            stderr="",
        )
        result = probe_gcs_uri("gs://bucket/file")
        self.assertEqual(result["cli"], "gcloud-storage")
        self.assertEqual(result["command"], ["gcloud", "storage", "ls", "gs://bucket/file"])
        self.assertEqual(result["stdout_lines"], ["gs://bucket/file"])

    @patch("latentdriver_waymax_experiments.womd.subprocess.run")
    @patch("latentdriver_waymax_experiments.womd.shutil.which")
    def test_copy_gcs_to_local_falls_back_to_gsutil(self, mock_which, mock_run) -> None:
        mock_which.side_effect = lambda name: f"/usr/bin/{name}" if name in {"gcloud", "gsutil"} else None
        mock_run.side_effect = [
            __import__("subprocess").CompletedProcess(
                ["gcloud", "storage", "cp", "gs://bucket/file", "/tmp/file"],
                1,
                stdout="",
                stderr="broken gcloud storage",
            ),
            __import__("subprocess").CompletedProcess(
                ["gsutil", "cp", "gs://bucket/file", "/tmp/file"],
                0,
                stdout="copied",
                stderr="",
            ),
        ]
        result = copy_gcs_to_local("gs://bucket/file", "/tmp/file")
        self.assertEqual(result["cli"], "gsutil")
        self.assertEqual(result["command"], ["gsutil", "cp", "gs://bucket/file", "/tmp/file"])


if __name__ == "__main__":
    unittest.main()
