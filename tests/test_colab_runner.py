from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.colab_runner import (  # noqa: E402
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
        self.assertIn("bootstrap-session", profiles)

    def test_full_preprocess_status_has_no_heavy_steps_by_default(self) -> None:
        self.assertEqual(profile_steps("full-preprocess-status"), [])

    def test_runtime_policy_keeps_dry_run_and_plots_lightweight(self) -> None:
        self.assertFalse(should_install_runtime_by_default("full-eval-dry-run"))
        self.assertFalse(should_install_runtime_by_default("plot-smoke-reactive"))
        self.assertFalse(should_install_runtime_by_default("bootstrap-session"))
        self.assertTrue(should_install_runtime_by_default("full-eval-reactive"))

    def test_resolve_debug_root_uses_drive_project_sibling_of_results_runs(self) -> None:
        with patch.dict(os.environ, {"LATENTDRIVER_RESULTS_ROOT": "/content/drive/MyDrive/waymax_research/latentdriver_waymax_experiments/results/runs"}):
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
            self.assertEqual(len(payload["steps"]), 1)
            self.assertTrue((bundle_dir / "manifest.json").is_file())
            self.assertTrue((bundle_dir / "artifact_status_before.json").is_file())
            self.assertTrue((bundle_dir / "artifact_status_after.json").is_file())

    def test_env_check_profile_executes_and_writes_after_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            payload = run_profile("env-check", debug_root=td)
            bundle_dir = Path(payload["bundle_dir"])
            self.assertEqual(payload["status"], "succeeded")
            self.assertEqual(payload["step_results"], [])
            self.assertTrue((bundle_dir / "artifact_status_after.json").is_file())


if __name__ == "__main__":
    unittest.main()
