from __future__ import annotations

import os
import sys
import tempfile
import unittest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.colab_runner import (  # noqa: E402
    RunnerStep,
    available_profiles,
    collect_artifact_status,
    profile_steps,
    resolve_debug_root,
    run_profile,
    should_install_runtime_by_default,
)


class ColabRunnerTests(unittest.TestCase):
    def test_profile_registry_contains_full_eval_and_status_profiles(self) -> None:
        profiles = available_profiles()
        self.assertIn("full-eval-dry-run", profiles)
        self.assertIn("full-preprocess-status", profiles)
        self.assertIn("full-preprocess-repair", profiles)
        self.assertIn("bootstrap-session", profiles)
        self.assertIn("probe-candidate-diversity", profiles)
        self.assertIn("probe-candidate-diversity-single", profiles)
        self.assertIn("smoke-eval-reactive-modulation-heuristic-single", profiles)
        self.assertIn("stage-full-womd-validation", profiles)
        self.assertIn("stage-interactive-pilot-shards", profiles)
        self.assertIn("interactive-pilot-preprocess-status", profiles)
        self.assertIn("interactive-pilot-preprocess", profiles)
        self.assertIn("interactive-pilot-preprocess-archive-status", profiles)
        self.assertIn("create-interactive-pilot-preprocess-archive", profiles)
        self.assertIn("restore-interactive-pilot-preprocess-archive", profiles)
        self.assertIn("create-full-preprocess-archive", profiles)
        self.assertIn("create-full-preprocess-shard-archives", profiles)
        self.assertIn("restore-full-preprocess-archive", profiles)
        self.assertIn("restore-full-preprocess-shard-archives", profiles)
        self.assertIn("full-preprocess-archive-status", profiles)

    def test_full_preprocess_status_has_no_heavy_steps_by_default(self) -> None:
        self.assertEqual(profile_steps("full-preprocess-status"), [])

    def test_full_preprocess_repair_is_single_lightweight_step(self) -> None:
        steps = profile_steps("full-preprocess-repair")
        self.assertEqual([step.name for step in steps], ["full_preprocess_repair"])
        self.assertIn("--repair-markers", " ".join(steps[0].command))

    def test_eval_profiles_bootstrap_upstream_before_running(self) -> None:
        steps = profile_steps("full-eval-dry-run")
        self.assertEqual([step.name for step in steps], ["bootstrap_upstream", "full_eval_dry_run"])

    def test_install_runtime_bootstraps_upstream_first(self) -> None:
        steps = profile_steps("full-eval-reactive-single", install_runtime=True)
        self.assertEqual([step.name for step in steps[:2]], ["bootstrap_upstream", "setup_colab_runtime"])
        self.assertEqual(steps[2].name, "preflight_full_reactive_latentdriver_t2_j3")

    def test_full_eval_profiles_preflight_inputs_before_model_launch(self) -> None:
        steps = profile_steps("full-eval-reactive-single")
        self.assertEqual(
            [step.name for step in steps],
            ["bootstrap_upstream", "preflight_full_reactive_latentdriver_t2_j3", "full_eval_reactive_single"],
        )
        self.assertIn("--resumable", " ".join(steps[-1].command))

    def test_full_eval_suite_preflights_all_public_checkpoints(self) -> None:
        steps = profile_steps("full-eval-reactive")
        preflights = [step.name for step in steps if step.name.startswith("preflight_full_reactive_")]
        self.assertEqual(
            preflights,
            [
                "preflight_full_reactive_latentdriver_t2_j3",
                "preflight_full_reactive_latentdriver_t2_j4",
                "preflight_full_reactive_plant",
                "preflight_full_reactive_easychauffeur_ppo",
            ],
        )
        self.assertIn("--resumable", " ".join(steps[-1].command))

    def test_stage_full_womd_validation_profile_uses_drive_bound_raw_cache(self) -> None:
        steps = profile_steps("stage-full-womd-validation")
        self.assertEqual([step.name for step in steps], ["stage_full_womd_validation"])
        command = " ".join(steps[0].command)
        self.assertIn("scripts/stage_womd_validation_shards.py", command)
        self.assertIn("--staging-root artifacts/assets/raw_womd", command)

    def test_stage_interactive_pilot_profile_uses_subset_stager_and_env_source(self) -> None:
        steps = profile_steps("stage-interactive-pilot-shards")
        self.assertEqual([step.name for step in steps], ["stage_interactive_pilot_shards"])
        command = " ".join(steps[0].command)
        self.assertIn("scripts/stage_womd_subset_shards.py", command)
        self.assertIn("--source-uri gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150", command)
        self.assertIn("--target-uri", command)
        self.assertIn("validation_pilot/validation_tfexample.tfrecord@10", command)
        self.assertIn("--force", command)
        self.assertIn("--verify-required-feature roadgraph_samples/xyz", command)

    def test_candidate_diversity_profiles_bootstrap_upstream_but_skip_runtime_setup(self) -> None:
        suite_steps = profile_steps("probe-candidate-diversity")
        single_steps = profile_steps("probe-candidate-diversity-single", model="plant")
        self.assertEqual([step.name for step in suite_steps], ["bootstrap_upstream", "probe_candidate_diversity"])
        self.assertEqual([step.name for step in single_steps], ["bootstrap_upstream", "probe_candidate_diversity_plant"])
        self.assertIn("--model latentdriver_t2_j3", " ".join(suite_steps[-1].command))
        self.assertIn("--model plant", " ".join(suite_steps[-1].command))
        self.assertIn("scripts/probe_candidate_diversity.py --model plant", " ".join(single_steps[-1].command))

    def test_smoke_modulation_profile_bootstraps_upstream_and_uses_heuristic_flags(self) -> None:
        steps = profile_steps("smoke-eval-reactive-modulation-heuristic-single", model="latentdriver_t2_j3")
        self.assertEqual(
            [step.name for step in steps],
            ["bootstrap_upstream", "smoke_eval_reactive_modulation_heuristic_latentdriver_t2_j3"],
        )
        command = " ".join(steps[-1].command)
        self.assertIn("--tier smoke_reactive", command)
        self.assertIn("--modulation heuristic", command)
        self.assertIn("--modulation-trace results/modulation_traces/latentdriver_t2_j3_smoke_reactive.jsonl", command)

    def test_preprocess_archive_profiles_use_archive_cli(self) -> None:
        pilot_status_steps = profile_steps("interactive-pilot-preprocess-archive-status")
        pilot_create_steps = profile_steps("create-interactive-pilot-preprocess-archive")
        pilot_restore_steps = profile_steps("restore-interactive-pilot-preprocess-archive")
        create_steps = profile_steps("create-full-preprocess-archive")
        create_shards_steps = profile_steps("create-full-preprocess-shard-archives")
        restore_steps = profile_steps("restore-full-preprocess-archive")
        restore_shards_steps = profile_steps("restore-full-preprocess-shard-archives")
        status_steps = profile_steps("full-preprocess-archive-status")
        self.assertIn(
            "scripts/preprocess_cache_archive.py status --mode interactive_pilot",
            " ".join(pilot_status_steps[0].command),
        )
        self.assertIn(
            "scripts/preprocess_cache_archive.py create --mode interactive_pilot --force",
            " ".join(pilot_create_steps[0].command),
        )
        self.assertIn(
            "scripts/preprocess_cache_archive.py extract --mode interactive_pilot",
            " ".join(pilot_restore_steps[0].command),
        )
        self.assertIn("scripts/preprocess_cache_archive.py create --mode full --force", " ".join(create_steps[0].command))
        self.assertIn(
            "scripts/preprocess_cache_archive.py create-shards --mode full --shards 150",
            " ".join(create_shards_steps[0].command),
        )
        self.assertIn("scripts/preprocess_cache_archive.py extract --mode full", " ".join(restore_steps[0].command))
        self.assertIn(
            "scripts/preprocess_cache_archive.py extract-shards --mode full",
            " ".join(restore_shards_steps[0].command),
        )
        self.assertIn("scripts/preprocess_cache_archive.py status --mode full", " ".join(status_steps[0].command))

    def test_bootstrap_session_runs_full_setup_sequence(self) -> None:
        steps = profile_steps("bootstrap-session")
        self.assertEqual(
            [step.name for step in steps],
            ["bootstrap_upstream", "setup_colab_runtime", "download_checkpoints", "full_eval_dry_run"],
        )

    def test_runtime_policy_keeps_dry_run_and_plots_lightweight(self) -> None:
        self.assertFalse(should_install_runtime_by_default("full-eval-dry-run"))
        self.assertFalse(should_install_runtime_by_default("plot-smoke-reactive"))
        self.assertFalse(should_install_runtime_by_default("bootstrap-session"))
        self.assertFalse(should_install_runtime_by_default("probe-candidate-diversity"))
        self.assertFalse(should_install_runtime_by_default("probe-candidate-diversity-single"))
        self.assertFalse(should_install_runtime_by_default("stage-full-womd-validation"))
        self.assertFalse(should_install_runtime_by_default("stage-interactive-pilot-shards"))
        self.assertFalse(should_install_runtime_by_default("interactive-pilot-preprocess-status"))
        self.assertFalse(should_install_runtime_by_default("interactive-pilot-preprocess-archive-status"))
        self.assertFalse(should_install_runtime_by_default("create-interactive-pilot-preprocess-archive"))
        self.assertFalse(should_install_runtime_by_default("restore-interactive-pilot-preprocess-archive"))
        self.assertFalse(should_install_runtime_by_default("full-preprocess-repair"))
        self.assertFalse(should_install_runtime_by_default("create-full-preprocess-archive"))
        self.assertFalse(should_install_runtime_by_default("create-full-preprocess-shard-archives"))
        self.assertFalse(should_install_runtime_by_default("restore-full-preprocess-archive"))
        self.assertFalse(should_install_runtime_by_default("restore-full-preprocess-shard-archives"))
        self.assertTrue(should_install_runtime_by_default("full-eval-reactive"))

    def test_resolve_debug_root_uses_drive_project_sibling_of_results_runs(self) -> None:
        with patch.dict(
            os.environ,
            {"LATENTDRIVER_RESULTS_ROOT": "/content/drive/MyDrive/waymax_research/latentdriver_waymax_experiments/results/runs"},
            clear=True,
        ):
            self.assertEqual(
                resolve_debug_root(),
                Path("/content/drive/MyDrive/waymax_research/latentdriver_waymax_experiments/debug_runs"),
            )

    def test_collect_artifact_status_is_non_recursive_and_tolerates_missing_full_dataset_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            status = collect_artifact_status()
        self.assertEqual(status["preprocess"]["full"]["scan_policy"], "non_recursive_path_state_only")
        self.assertIn("error", status["datasets"]["full"])

    def test_run_profile_dry_run_writes_debug_bundle_without_executing_steps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            payload = run_profile("full-eval-dry-run", debug_root=td, dry_run=True)
            bundle_dir = Path(payload["bundle_dir"])
            self.assertEqual(payload["status"], "dry_run")
            self.assertEqual(payload["step_results"], [])
            self.assertEqual([step["name"] for step in payload["steps"]], ["bootstrap_upstream", "full_eval_dry_run"])
            self.assertTrue((bundle_dir / "manifest.json").is_file())
            self.assertTrue((bundle_dir / "artifact_status_before.json").is_file())
            self.assertTrue((bundle_dir / "artifact_status_after.json").is_file())
            self.assertTrue((Path(td) / "latest" / "manifest.json").is_file())
            self.assertTrue((Path(td) / "latest" / "ALIAS.json").is_file())
            self.assertTrue((Path(td) / "LATEST.json").is_file())
            self.assertTrue((Path(td) / "RUN_LEDGER.jsonl").is_file())
            latest_manifest = json.loads((Path(td) / "latest" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_manifest["status"], "dry_run")
            self.assertIn("debug_aliases", latest_manifest)

    def test_env_check_profile_executes_and_writes_after_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            payload = run_profile("env-check", debug_root=td)
            bundle_dir = Path(payload["bundle_dir"])
            self.assertEqual(payload["status"], "succeeded")
            self.assertEqual(payload["step_results"], [])
            self.assertTrue((bundle_dir / "artifact_status_after.json").is_file())
            latest_manifest = json.loads((Path(td) / "latest" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_manifest["status"], "succeeded")
            self.assertIn("debug_aliases", latest_manifest)

    def test_failed_profile_updates_latest_failure_alias_and_pointer(self) -> None:
        failed_step = RunnerStep(
            name="forced_failure",
            command=(sys.executable, "-c", "import sys; print('forced failure'); sys.exit(3)"),
            description="Deliberately fail for debug alias testing.",
        )
        with tempfile.TemporaryDirectory() as td:
            with patch("latentdriver_waymax_experiments.colab_runner.profile_steps", return_value=[failed_step]):
                payload = run_profile("env-check", debug_root=td)
            debug_root = Path(td)
            self.assertEqual(payload["status"], "failed")
            self.assertTrue((debug_root / "latest_failure" / "manifest.json").is_file())
            self.assertTrue((debug_root / "latest_failure" / "ALIAS.json").is_file())
            self.assertTrue((debug_root / "LATEST_FAILURE.json").is_file())
            pointer = json.loads((debug_root / "LATEST_FAILURE.json").read_text(encoding="utf-8"))
            self.assertEqual(pointer["run_id"], payload["run_id"])
            latest_failure_manifest = json.loads((debug_root / "latest_failure" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_failure_manifest["status"], "failed")
            self.assertIn("debug_aliases", latest_failure_manifest)
            ledger_rows = (debug_root / "RUN_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(ledger_rows), 1)
            self.assertEqual(json.loads(ledger_rows[0])["failed_step"], "forced_failure")

    def test_colab_canary_cli_sets_waymo_dataset_root_for_dry_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/colab_canary.py",
                    "--profile",
                    "full-eval-dry-run",
                    "--dry-run",
                    "--debug-root",
                    td,
                    "--waymo-dataset-root",
                    "gs://waymo_open_dataset_motion_v_1_1_0",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            bundle = next(Path(td).glob("*_full_eval_dry_run"))
            status = json.loads((bundle / "artifact_status_before.json").read_text(encoding="utf-8"))
            self.assertTrue(status["datasets"]["full"]["exists_or_remote"])
            self.assertIn("gs://waymo_open_dataset_motion_v_1_1_0", status["datasets"]["full"]["uri"])


if __name__ == "__main__":
    unittest.main()
