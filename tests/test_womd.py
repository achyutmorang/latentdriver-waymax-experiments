from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.womd import (
    WOMD_VERSION,
    first_sharded_tfrecord_path,
    is_gcs_uri,
    local_dataset_uri_exists,
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


if __name__ == "__main__":
    unittest.main()
