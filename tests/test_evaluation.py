from __future__ import annotations

import json
import subprocess
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

    def test_build_eval_command_sets_plant_control_type_to_waypoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td)
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(raw_root)
            (raw_root / "waymo_open_dataset_motion_v_1_1_0" / "uncompressed" / "tf_example" / "validation").mkdir(parents=True)
            cmd = build_eval_command(model="plant", tier="smoke_reactive", vis=False)
            self.assertIn("++method.control_type=waypoint", " ".join(cmd))

    def test_visualization_patch_persists_media_for_straight_intentions(self) -> None:
        patch_path = Path(__file__).resolve().parents[1] / "patches" / "latentdriver_eval_contract.patch"
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertIn("safe_intention = intention.replace('/', '_').replace(' ', '_') or 'unknown'", patch_text)
        self.assertIn("for state in tqdm.tqdm(self.env.states):", patch_text)
        self.assertIn("mediapy.write_video(name+'.mp4', imgs, fps=10)", patch_text)


    def test_build_eval_command_honors_seed_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td)
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(raw_root)
            (raw_root / "waymo_open_dataset_motion_v_1_1_0" / "uncompressed" / "tf_example" / "validation").mkdir(parents=True)
            cmd = build_eval_command(model="latentdriver_t2_j3", tier="smoke_reactive", seed=7, vis=False)
            self.assertIn("++run.seed=7", " ".join(cmd))

    def test_build_eval_command_supports_gcs_full_validation_root(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LATENTDRIVER_WAYMO_DATASET_ROOT": "gs://waymo_open_dataset_motion_v_1_1_0",
                "LATENTDRIVER_EVAL_DEVICE_COUNT": "7",
            },
        ):
            cmd = build_eval_command(model="latentdriver_t2_j3", tier="full_reactive", vis=False)
        self.assertIn(
            "++waymax_conf.path=gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150",
            " ".join(cmd),
        )

    def test_build_eval_command_clamps_full_batch_dims_to_available_devices(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LATENTDRIVER_WAYMO_DATASET_ROOT": "gs://waymo_open_dataset_motion_v_1_1_0",
                "LATENTDRIVER_EVAL_DEVICE_COUNT": "1",
            },
        ):
            cmd = build_eval_command(model="latentdriver_t2_j3", tier="full_reactive", vis=False)
        self.assertIn("++batch_dims=[1,125]", " ".join(cmd))

    def test_build_eval_command_keeps_full_batch_dims_on_multi_device_hosts(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LATENTDRIVER_WAYMO_DATASET_ROOT": "gs://waymo_open_dataset_motion_v_1_1_0",
                "LATENTDRIVER_EVAL_DEVICE_COUNT": "8",
            },
        ):
            cmd = build_eval_command(model="latentdriver_t2_j3", tier="full_reactive", vis=False)
        self.assertIn("++batch_dims=[7,125]", " ".join(cmd))

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

    def test_run_eval_failure_includes_stderr_tail(self) -> None:
        from latentdriver_waymax_experiments.evaluation import run_eval

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = {
                "run_id": "run-1",
                "run_dir": root,
                "metrics_path": root / "metrics.json",
                "stdout_path": root / "stdout.log",
                "stderr_path": root / "stderr.log",
                "config_snapshot": root / "config_snapshot.json",
                "run_manifest": root / "run_manifest.json",
                "vis_dir": root / "vis",
            }
            bundle["vis_dir"].mkdir()
            with patch("latentdriver_waymax_experiments.evaluation.ensure_upstream_exists", return_value=root), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_python312_compat_sitecustomize", return_value=root / "sitecustomize.py"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_lightning_compat_source_patches", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_crdp_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_jax_tree_map_compat_source_patch", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_matplotlib_canvas_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.create_run_bundle", return_value=bundle), \
                 patch("latentdriver_waymax_experiments.evaluation._verify_inputs", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.build_eval_command", return_value=["python3", "simulate.py"]), \
                 patch("latentdriver_waymax_experiments.evaluation.subprocess.run", return_value=subprocess.CompletedProcess(["python3", "simulate.py"], 1, stdout="stdout line\nmore stdout", stderr="Traceback line\nroot cause boom")):
                with self.assertRaisesRegex(RuntimeError, "root cause boom") as ctx:
                    run_eval(model="latentdriver_t2_j3", tier="smoke_reactive", seed=0, vis=False, dry_run=False)
            self.assertIn("stderr_path:", str(ctx.exception))
            self.assertIn("stdout tail:", str(ctx.exception))


    def test_run_eval_visualization_requires_media_artifact(self) -> None:
        from latentdriver_waymax_experiments.evaluation import run_eval

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = {
                "run_id": "run-1",
                "run_dir": root,
                "metrics_path": root / "metrics.json",
                "stdout_path": root / "stdout.log",
                "stderr_path": root / "stderr.log",
                "config_snapshot": root / "config_snapshot.json",
                "run_manifest": root / "run_manifest.json",
                "vis_dir": root / "vis",
            }
            bundle["vis_dir"].mkdir()

            def _completed_process(*args, **kwargs):
                bundle["metrics_path"].write_text(json.dumps({"average": {}, "average_over_class": {}, "per_class": {}}), encoding="utf-8")
                return subprocess.CompletedProcess(["python3", "simulate.py"], 0, stdout="ok", stderr="")

            with patch("latentdriver_waymax_experiments.evaluation.ensure_upstream_exists", return_value=root), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_python312_compat_sitecustomize", return_value=root / "sitecustomize.py"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_lightning_compat_source_patches", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_crdp_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_jax_tree_map_compat_source_patch", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_matplotlib_canvas_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.create_run_bundle", return_value=bundle), \
                 patch("latentdriver_waymax_experiments.evaluation._verify_inputs", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.build_eval_command", return_value=["python3", "simulate.py"]), \
                 patch("latentdriver_waymax_experiments.evaluation.subprocess.run", side_effect=_completed_process):
                with self.assertRaisesRegex(RuntimeError, "without producing media artifacts") as ctx:
                    run_eval(model="latentdriver_t2_j3", tier="smoke_reactive", seed=0, vis="video", dry_run=False)
            self.assertIn("vis_dir:", str(ctx.exception))


    def test_run_eval_visualization_returns_media_manifest(self) -> None:
        from latentdriver_waymax_experiments.evaluation import run_eval

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = {
                "run_id": "run-1",
                "run_dir": root,
                "metrics_path": root / "metrics.json",
                "stdout_path": root / "stdout.log",
                "stderr_path": root / "stderr.log",
                "config_snapshot": root / "config_snapshot.json",
                "run_manifest": root / "run_manifest.json",
                "vis_dir": root / "vis",
            }
            bundle["vis_dir"].mkdir()

            def _completed_process(*args, **kwargs):
                bundle["metrics_path"].write_text(json.dumps({"average": {}, "average_over_class": {}, "per_class": {}}), encoding="utf-8")
                media_path = bundle["vis_dir"] / "straight_" / "demo.mp4"
                media_path.parent.mkdir(parents=True, exist_ok=True)
                media_path.write_bytes(b"fake")
                return subprocess.CompletedProcess(["python3", "simulate.py"], 0, stdout="ok", stderr="")

            with patch("latentdriver_waymax_experiments.evaluation.ensure_upstream_exists", return_value=root), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_python312_compat_sitecustomize", return_value=root / "sitecustomize.py"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_lightning_compat_source_patches", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_crdp_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_jax_tree_map_compat_source_patch", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_matplotlib_canvas_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.create_run_bundle", return_value=bundle), \
                 patch("latentdriver_waymax_experiments.evaluation._verify_inputs", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.build_eval_command", return_value=["python3", "simulate.py"]), \
                 patch("latentdriver_waymax_experiments.evaluation.subprocess.run", side_effect=_completed_process):
                payload = run_eval(model="latentdriver_t2_j3", tier="smoke_reactive", seed=0, vis="video", dry_run=False)

            self.assertEqual(payload["media_file_count"], 1)
            self.assertTrue(payload["media_files"][0].endswith("demo.mp4"))



if __name__ == "__main__":
    unittest.main()
