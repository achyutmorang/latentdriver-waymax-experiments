from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.evaluation import build_eval_command, flatten_metrics_payload, run_public_suite


class EvaluationTests(unittest.TestCase):
    def test_build_eval_command_contains_expected_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td)
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(raw_root)
            (raw_root / "waymo_open_dataset_motion_v_1_1_0" / "uncompressed" / "tf_example" / "validation").mkdir(parents=True)
            cmd = build_eval_command(model="latentdriver_t2_j3", tier="smoke_reactive", vis=False)
            joined = " ".join(cmd)
            self.assertIn("method=latentdriver", joined)
            self.assertIn("++ego_control_setting.npc_policy_type=idm", joined)
            self.assertIn("++method.num_of_decoder=3", joined)
            self.assertIn("++run.max_batches=1", joined)
            self.assertIn("++run.seed=0", joined)

    def test_build_eval_command_honors_seed_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td)
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(raw_root)
            (raw_root / "waymo_open_dataset_motion_v_1_1_0" / "uncompressed" / "tf_example" / "validation").mkdir(parents=True)
            cmd = build_eval_command(model="latentdriver_t2_j3", tier="smoke_reactive", seed=7, vis=False)
            self.assertIn("++run.seed=7", " ".join(cmd))

    def test_build_eval_command_supports_gcs_full_validation_root(self) -> None:
        os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = "gs://waymo_open_dataset_motion_v_1_1_0"
        cmd = build_eval_command(model="latentdriver_t2_j3", tier="full_reactive", vis=False)
        self.assertIn(
            "++waymax_conf.path=gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150",
            " ".join(cmd),
        )

    def test_flatten_metrics_maps_expected_keys(self) -> None:
        payload = {
            "average": {
                "number of episodes": 12,
                "metric/AR[75:95]": 94.31,
                "metric/offroad_rate": 2.22,
                "metric/collision_rate": 3.13,
                "metric/progress_rate": 99.64,
            },
            "average_over_class": {
                "metric/AR[75:95]": 90.14,
            },
            "per_class": {},
        }
        flat = flatten_metrics_payload(payload)
        self.assertEqual(flat["number_of_episodes"], 12)
        self.assertEqual(flat["mar_75_95"], 90.14)
        self.assertEqual(flat["ar_75_95"], 94.31)

    def test_run_public_suite_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_RESULTS_ROOT"] = td
            with patch("latentdriver_waymax_experiments.evaluation.run_eval", side_effect=lambda **kwargs: kwargs):
                suite = run_public_suite(tier="smoke_reactive", dry_run=True)
        self.assertEqual(suite["tier"], "smoke_reactive")
        self.assertGreaterEqual(len(suite["runs"]), 4)


if __name__ == "__main__":
    unittest.main()
