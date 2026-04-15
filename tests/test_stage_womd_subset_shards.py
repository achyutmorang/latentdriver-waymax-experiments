from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stage_womd_subset_shards import main as stage_subset_main


class StageWomdSubsetShardsTests(unittest.TestCase):
    def test_stages_sparse_local_shards_into_dense_target_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_uri = str(root / "validation_interactive_tfexample.tfrecord@4")
            target_uri = str(root / "pilot" / "validation_interactive_tfexample.tfrecord@2")
            (root / "pilot").mkdir(parents=True, exist_ok=True)

            source_shards = {
                1: b"shard-one",
                3: b"shard-three",
            }
            for index, payload in source_shards.items():
                shard = root / f"validation_interactive_tfexample.tfrecord-{index:05d}-of-00004"
                shard.write_bytes(payload)

            stdout_path = root / "stdout.json"
            argv = [
                "stage_womd_subset_shards.py",
                "--source-uri",
                source_uri,
                "--source-shards",
                "1,3",
                "--target-uri",
                target_uri,
            ]

            old_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    with patch.object(sys, "argv", argv):
                        rc = stage_subset_main()
            finally:
                sys.stdout = old_stdout

            self.assertEqual(rc, 0)
            payload = json.loads(stdout_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["copied"], 2)
            target0 = root / "pilot" / "validation_interactive_tfexample.tfrecord-00000-of-00002"
            target1 = root / "pilot" / "validation_interactive_tfexample.tfrecord-00001-of-00002"
            self.assertEqual(target0.read_bytes(), b"shard-one")
            self.assertEqual(target1.read_bytes(), b"shard-three")


if __name__ == "__main__":
    unittest.main()
