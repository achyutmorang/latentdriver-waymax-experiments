from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.womd import (
    WOMD_VERSION,
    copy_gcs_to_local,
    dataset_uri_status,
    first_sharded_tfrecord_uri,
    first_sharded_tfrecord_path,
    is_gcs_uri,
    local_dataset_uri_complete,
    local_dataset_uri_exists,
    probe_gcs_uri,
    resolve_dataset_uri,
    sharded_tfrecord_uri,
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

    def test_first_sharded_tfrecord_uri_supports_gcs_uri(self) -> None:
        dataset_uri = f"gs://{WOMD_VERSION}/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150"
        self.assertEqual(
            first_sharded_tfrecord_uri(dataset_uri),
            f"gs://{WOMD_VERSION}/uncompressed/tf_example/validation/validation_tfexample.tfrecord-00000-of-00150",
        )
        self.assertEqual(
            sharded_tfrecord_uri(dataset_uri, 7),
            f"gs://{WOMD_VERSION}/uncompressed/tf_example/validation/validation_tfexample.tfrecord-00007-of-00150",
        )

    def test_local_dataset_uri_exists_understands_sharded_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "validation_tfexample.tfrecord-00000-of-00150"
            shard.write_text("x", encoding="utf-8")
            self.assertTrue(local_dataset_uri_exists(str(Path(td) / "validation_tfexample.tfrecord@150")))

    def test_local_dataset_uri_complete_requires_all_shards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "validation_tfexample.tfrecord-00000-of-00002").write_text("x", encoding="utf-8")
            self.assertFalse(local_dataset_uri_complete(str(root / "validation_tfexample.tfrecord@2")))
            (root / "validation_tfexample.tfrecord-00001-of-00002").write_text("x", encoding="utf-8")
            self.assertTrue(local_dataset_uri_complete(str(root / "validation_tfexample.tfrecord@2")))

    def test_dataset_uri_status_reports_local_shard_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "validation_tfexample.tfrecord-00000-of-00002").write_text("x", encoding="utf-8")
            status = dataset_uri_status(str(root / "validation_tfexample.tfrecord@2"))
        self.assertFalse(status["complete"])
        self.assertEqual(status["expected_shards"], 2)
        self.assertEqual(status["existing_shards"], 1)
        self.assertEqual(status["missing_count"], 1)

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

    def test_probe_tensorflow_dataset_uri_reads_one_record(self) -> None:
        from latentdriver_waymax_experiments.womd import probe_tensorflow_dataset_uri

        class FakeDataset:
            def __init__(self, paths):
                self.paths = paths

            def take(self, _count):
                return iter([b"record"])

        fake_tf = types.SimpleNamespace(
            io=types.SimpleNamespace(gfile=types.SimpleNamespace(exists=lambda _path: True)),
            data=types.SimpleNamespace(TFRecordDataset=FakeDataset),
        )
        with patch.dict(sys.modules, {"tensorflow": fake_tf}):
            result = probe_tensorflow_dataset_uri("gs://bucket/path/data.tfrecord@150")
        self.assertTrue(result["ok"])
        self.assertEqual(result["probe"], "tensorflow_tfrecord_read")
        self.assertEqual(result["read_records"], 1)
        self.assertEqual(result["target"], "gs://bucket/path/data.tfrecord-00000-of-00150")

    def test_probe_tensorflow_dataset_uri_reports_read_failures(self) -> None:
        from latentdriver_waymax_experiments.womd import probe_tensorflow_dataset_uri

        class FakeDataset:
            def __init__(self, paths):
                self.paths = paths

            def take(self, _count):
                raise PermissionError("anonymous caller")

        fake_tf = types.SimpleNamespace(
            io=types.SimpleNamespace(gfile=types.SimpleNamespace(exists=lambda _path: True)),
            data=types.SimpleNamespace(TFRecordDataset=FakeDataset),
        )
        with patch.dict(sys.modules, {"tensorflow": fake_tf}):
            result = probe_tensorflow_dataset_uri("gs://bucket/path/data.tfrecord@150")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_kind"], "tensorflow_tfrecord_read")
        self.assertIn("anonymous caller", result["error"])

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
