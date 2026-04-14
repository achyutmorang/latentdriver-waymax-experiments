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

from latentdriver_waymax_experiments.evaluation import (
    aggregate_metrics_payloads,
    build_eval_command,
    flatten_metrics_payload,
    materialize_preprocess_cache,
    run_eval_resumable,
    run_public_suite,
)


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

    def test_full_eval_verify_requires_complete_local_raw_shards_and_preprocess_marker(self) -> None:
        from latentdriver_waymax_experiments.evaluation import _verify_inputs

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / "model.ckpt"
            ckpt.write_bytes(b"ckpt")
            raw = root / "raw" / "validation_tfexample.tfrecord@2"
            raw.parent.mkdir(parents=True)
            (raw.parent / "validation_tfexample.tfrecord-00000-of-00002").write_bytes(b"first")
            preprocess = root / "preprocessed" / "val_preprocessed_path"
            intention = root / "preprocessed" / "val_intention_label"
            (preprocess / "map").mkdir(parents=True)
            (preprocess / "route").mkdir(parents=True)
            intention.mkdir(parents=True)
            with patch(
                "latentdriver_waymax_experiments.evaluation.load_config",
                return_value={"evaluation": {"tiers": {"full_reactive": {"dataset_mode": "full"}}}},
            ), patch(
                "latentdriver_waymax_experiments.evaluation._validation_inputs",
                return_value={"waymo_path": str(raw), "preprocess_path": preprocess, "intention_path": intention},
            ), patch(
                "latentdriver_waymax_experiments.evaluation.checkpoint_path",
                return_value=ckpt,
            ), patch(
                "latentdriver_waymax_experiments.evaluation.ensure_upstream_exists",
                return_value=root,
            ):
                missing = _verify_inputs("latentdriver_t2_j3", "full_reactive")
        self.assertIn("waymo_path", missing)
        self.assertIn("preprocess_completion", missing)
        self.assertIn("_SUCCESS", missing["preprocess_completion"])

    def test_full_eval_verify_reports_tensorflow_gcs_read_failure(self) -> None:
        from latentdriver_waymax_experiments.evaluation import _verify_inputs

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ckpt = root / "model.ckpt"
            ckpt.write_bytes(b"ckpt")
            preprocess = root / "preprocessed" / "val_preprocessed_path"
            intention = root / "preprocessed" / "val_intention_label"
            (preprocess / "map").mkdir(parents=True)
            (preprocess / "route").mkdir(parents=True)
            intention.mkdir(parents=True)
            (preprocess / "_SUCCESS").write_text("complete\n", encoding="utf-8")
            (preprocess / "preprocess_manifest.json").write_text("{}", encoding="utf-8")
            with patch(
                "latentdriver_waymax_experiments.evaluation.load_config",
                return_value={"evaluation": {"tiers": {"full_reactive": {"dataset_mode": "full"}}}},
            ), patch(
                "latentdriver_waymax_experiments.evaluation._validation_inputs",
                return_value={
                    "waymo_path": "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150",
                    "preprocess_path": preprocess,
                    "intention_path": intention,
                },
            ), patch(
                "latentdriver_waymax_experiments.evaluation.checkpoint_path",
                return_value=ckpt,
            ), patch(
                "latentdriver_waymax_experiments.evaluation.ensure_upstream_exists",
                return_value=root,
            ), patch(
                "latentdriver_waymax_experiments.evaluation.probe_tensorflow_dataset_uri",
                return_value={"ok": False, "error": "403 anonymous"},
            ):
                missing = _verify_inputs("latentdriver_t2_j3", "full_reactive", verify_remote_reads=True)
        self.assertIn("waymo_path_gcs_read", missing)
        self.assertIn("403 anonymous", missing["waymo_path_gcs_read"])

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

    def test_aggregate_metrics_payloads_weights_by_episode_count(self) -> None:
        payload = aggregate_metrics_payloads(
            [
                {
                    "average": {
                        "number of episodes": 1,
                        "metric/collision_rate": 0.0,
                        "metric/progress_rate": 0.5,
                    },
                    "per_class": {},
                },
                {
                    "average": {
                        "number of episodes": 3,
                        "metric/collision_rate": 1.0,
                        "metric/progress_rate": 1.0,
                    },
                    "per_class": {},
                },
            ],
            shard_count=2,
        )
        self.assertEqual(payload["average"]["number of episodes"], 4)
        self.assertAlmostEqual(payload["average"]["metric/collision_rate"], 0.75)
        self.assertAlmostEqual(payload["average"]["metric/progress_rate"], 0.875)
        self.assertEqual(payload["meta"]["shards_completed"], 2)

    def test_run_public_suite_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_RESULTS_ROOT"] = td
            with patch("latentdriver_waymax_experiments.evaluation.run_eval", side_effect=lambda **kwargs: kwargs):
                suite = run_public_suite(tier="smoke_reactive", dry_run=True)
        self.assertEqual(suite["tier"], "smoke_reactive")
        self.assertGreaterEqual(len(suite["runs"]), 4)

    def test_run_eval_resumable_skips_completed_shards_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = {
                "run_id": "resumable_full_reactive_latentdriver_t2_j3_seed0",
                "run_dir": root,
                "metrics_path": root / "metrics.json",
                "stdout_path": root / "stdout.log",
                "stderr_path": root / "stderr.log",
                "config_snapshot": root / "config_snapshot.json",
                "run_manifest": root / "run_manifest.json",
                "vis_dir": root / "vis",
            }
            (root / "vis").mkdir()
            shard0 = root / "shards" / "shard-00000"
            shard0.mkdir(parents=True)
            (shard0 / "metrics.json").write_text(
                json.dumps(
                    {
                        "average": {"number of episodes": 1, "metric/progress_rate": 0.5},
                        "average_over_class": {},
                        "per_class": {},
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(cmd, cwd, text, capture_output, check, env):
                metrics_arg = next(arg for arg in cmd if arg.startswith("++run.metrics_json_path="))
                metrics_path = Path(metrics_arg.split("=", 1)[1])
                metrics_path.write_text(
                    json.dumps(
                        {
                            "average": {"number of episodes": 1, "metric/progress_rate": 1.0},
                            "average_over_class": {},
                            "per_class": {},
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch.dict(os.environ, {"LATENTDRIVER_MATERIALIZE_PREPROCESS_CACHE": "0"}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_upstream_exists", return_value=root), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_python312_compat_sitecustomize", return_value=root / "sitecustomize.py"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_lightning_compat_source_patches", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_crdp_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_jax_tree_map_compat_source_patch", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_matplotlib_canvas_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.create_named_run_bundle", return_value=bundle), \
                 patch("latentdriver_waymax_experiments.evaluation._verify_inputs", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation._validation_inputs", return_value={"waymo_path": "gs://bucket/validation.tfrecord@2"}), \
                 patch("latentdriver_waymax_experiments.evaluation.checkpoint_path", return_value=root / "model.ckpt"), \
                 patch("latentdriver_waymax_experiments.evaluation.build_eval_command", return_value=["python3", "simulate.py", "++waymax_conf.path=gs://bucket/validation.tfrecord@2"]), \
                 patch("latentdriver_waymax_experiments.evaluation.load_config", return_value={"evaluation": {"tiers": {"full_reactive": {"dataset_mode": "full", "seed": 0}}}}), \
                 patch("latentdriver_waymax_experiments.evaluation.subprocess.run", side_effect=fake_run) as run_mock:
                payload = run_eval_resumable(model="latentdriver_t2_j3", tier="full_reactive", seed=0, max_shards=2)

            self.assertEqual(run_mock.call_count, 1)
            self.assertEqual([item["status"] for item in payload["shards"]], ["skipped", "completed"])
            self.assertAlmostEqual(payload["summary"]["progress_rate"], 0.75)
            progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["shards_completed"], 2)
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(progress["shards_remaining"], 0)
            self.assertIn("eta", progress)

    def test_materialize_preprocess_cache_copies_to_local_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_preprocess = root / "drive" / "val_preprocessed_path"
            source_intention = root / "drive" / "val_intention_label"
            (source_preprocess / "map").mkdir(parents=True)
            (source_preprocess / "route").mkdir(parents=True)
            source_intention.mkdir(parents=True)
            (source_preprocess / "map" / "123.npy").write_bytes(b"map")
            (source_preprocess / "route" / "123.npy").write_bytes(b"route")
            (source_intention / "123.txt").write_text("straight_", encoding="utf-8")
            (source_preprocess / "_SUCCESS").write_text("complete\n", encoding="utf-8")
            (source_preprocess / "preprocess_manifest.json").write_text("{}", encoding="utf-8")

            local_root = root / "local"
            with patch.dict(os.environ, {"LATENTDRIVER_LOCAL_PREPROCESS_ROOT": str(local_root)}):
                payload = materialize_preprocess_cache(
                    dataset_mode="full",
                    preprocess_path=source_preprocess,
                    intention_path=source_intention,
                )

            target_preprocess = Path(payload["preprocess_path"])
            target_intention = Path(payload["intention_path"])
            self.assertEqual((target_preprocess / "map" / "123.npy").read_bytes(), b"map")
            self.assertEqual((target_preprocess / "route" / "123.npy").read_bytes(), b"route")
            self.assertEqual((target_intention / "123.txt").read_text(encoding="utf-8"), "straight_")
            self.assertTrue((target_preprocess / "_SUCCESS").is_file())
            self.assertEqual(payload["summary"]["total_files"], 3)
            self.assertIn("eta", payload["summary"])

    def test_materialize_preprocess_cache_rejects_empty_source_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_preprocess = root / "drive" / "val_preprocessed_path"
            source_intention = root / "drive" / "val_intention_label"
            (source_preprocess / "map").mkdir(parents=True)
            (source_preprocess / "route").mkdir(parents=True)
            source_intention.mkdir(parents=True)
            with patch("latentdriver_waymax_experiments.evaluation.time.sleep", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "contains no files"):
                    materialize_preprocess_cache(
                        dataset_mode="full",
                        preprocess_path=source_preprocess,
                        intention_path=source_intention,
                    )

    def test_run_eval_resumable_uses_materialized_preprocess_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_preprocess = root / "drive" / "val_preprocessed_path"
            source_intention = root / "drive" / "val_intention_label"
            (source_preprocess / "map").mkdir(parents=True)
            (source_preprocess / "route").mkdir(parents=True)
            source_intention.mkdir(parents=True)
            (source_preprocess / "map" / "123.npy").write_bytes(b"map")
            (source_preprocess / "route" / "123.npy").write_bytes(b"route")
            (source_intention / "123.txt").write_text("straight_", encoding="utf-8")
            bundle = {
                "run_id": "resumable_full_reactive_latentdriver_t2_j3_seed0",
                "run_dir": root / "run",
                "metrics_path": root / "run" / "metrics.json",
                "stdout_path": root / "run" / "stdout.log",
                "stderr_path": root / "run" / "stderr.log",
                "config_snapshot": root / "run" / "config_snapshot.json",
                "run_manifest": root / "run" / "run_manifest.json",
                "vis_dir": root / "run" / "vis",
            }
            bundle["vis_dir"].mkdir(parents=True)
            local_root = root / "local"

            def fake_run(cmd, cwd, text, capture_output, check, env):
                joined = " ".join(cmd)
                self.assertIn(str(local_root / "full" / "val_preprocessed_path"), joined)
                self.assertIn(str(local_root / "full" / "val_intention_label"), joined)
                metrics_arg = next(arg for arg in cmd if arg.startswith("++run.metrics_json_path="))
                metrics_path = Path(metrics_arg.split("=", 1)[1])
                metrics_path.write_text(json.dumps({"average": {"number of episodes": 1}, "per_class": {}}), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch.dict(
                os.environ,
                {
                    "LATENTDRIVER_LOCAL_PREPROCESS_ROOT": str(local_root),
                    "LATENTDRIVER_MATERIALIZE_PREPROCESS_CACHE": "1",
                },
            ), patch("latentdriver_waymax_experiments.evaluation.ensure_upstream_exists", return_value=root), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_python312_compat_sitecustomize", return_value=root / "sitecustomize.py"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_lightning_compat_source_patches", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_crdp_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_jax_tree_map_compat_source_patch", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation.ensure_matplotlib_canvas_compat_source_patch", return_value="already_patched"), \
                 patch("latentdriver_waymax_experiments.evaluation.create_named_run_bundle", return_value=bundle), \
                 patch("latentdriver_waymax_experiments.evaluation._verify_inputs", return_value={}), \
                 patch("latentdriver_waymax_experiments.evaluation._validation_inputs", return_value={"waymo_path": "gs://bucket/validation.tfrecord@1", "preprocess_path": source_preprocess, "intention_path": source_intention}), \
                 patch("latentdriver_waymax_experiments.evaluation.checkpoint_path", return_value=root / "model.ckpt"), \
                 patch("latentdriver_waymax_experiments.evaluation.build_eval_command", return_value=[
                     "python3",
                     "simulate.py",
                     "++waymax_conf.path=gs://bucket/validation.tfrecord@1",
                     f"++data_conf.path_to_processed_map_route={source_preprocess}",
                     f"++metric_conf.intention_label_path={source_intention}",
                 ]), \
                 patch("latentdriver_waymax_experiments.evaluation.load_config", return_value={"evaluation": {"tiers": {"full_reactive": {"dataset_mode": "full", "seed": 0}}}}), \
                 patch("latentdriver_waymax_experiments.evaluation.subprocess.run", side_effect=fake_run):
                payload = run_eval_resumable(model="latentdriver_t2_j3", tier="full_reactive", seed=0, max_shards=1)

            self.assertEqual(payload["summary"]["number_of_episodes"], 1)
            self.assertEqual(payload["materialized_preprocess"]["preprocess_path"], str(local_root / "full" / "val_preprocessed_path"))
            self.assertEqual(payload["shards"][0]["status"], "completed")
            self.assertIn("duration_seconds", payload["shards"][0])

    def test_run_waymax_eval_cli_returns_nonzero_when_dry_run_not_ready(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts import run_waymax_eval as run_script

        cfg = {
            "checkpoints": {"latentdriver_t2_j3": {}},
            "evaluation": {"tiers": {"full_reactive": {}}},
        }
        argv = [
            "run_waymax_eval.py",
            "--model",
            "latentdriver_t2_j3",
            "--tier",
            "full_reactive",
            "--dry-run",
        ]
        with patch.object(sys, "argv", argv), patch.object(run_script, "load_config", return_value=cfg), patch.object(
            run_script,
            "run_eval",
            return_value={"ready": False},
        ):
            self.assertEqual(run_script.main(), 1)

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
