from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.preprocess_validation_only import (
    build_preprocess_command,
    can_repair_preprocess_markers,
    clear_preprocess_outputs,
    main as preprocess_main,
    preprocess_cache_status,
    repair_preprocess_complete_markers,
)


class PreprocessValidationOnlyTests(unittest.TestCase):
    def test_build_preprocess_command_uses_module_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td)
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(raw_root)
            cmd = build_preprocess_command(mode="smoke")
        self.assertGreaterEqual(len(cmd), 3)
        self.assertEqual(cmd[1:3], ["-m", "src.preprocess.preprocess_data"])

    def test_main_injects_upstream_repo_into_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td)
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(raw_root)
            upstream_dir = Path(td) / "LatentDriver"
            (upstream_dir / "src" / "utils").mkdir(parents=True, exist_ok=True)
            (upstream_dir / "src" / "policy" / "latentdriver").mkdir(parents=True, exist_ok=True)
            (upstream_dir / "src" / "policy" / "baseline").mkdir(parents=True, exist_ok=True)
            (upstream_dir / "src" / "utils" / "utils.py").write_text("import pytorch_lightning as pl\n", encoding="utf-8")
            (upstream_dir / "src" / "policy" / "latentdriver" / "lantentdriver_model.py").write_text(
                "import torch.nn as nn\nimport pytorch_lightning as pl\n",
                encoding="utf-8",
            )
            (upstream_dir / "src" / "policy" / "baseline" / "bc_baseline.py").write_text(
                "from torch import nn\nimport pytorch_lightning as pl\n",
                encoding="utf-8",
            )
            argv = ["preprocess_validation_only.py", "--mode", "smoke"]
            captured: dict[str, object] = {}

            def fake_run(cmd, cwd, check, text, env):
                captured["cmd"] = cmd
                captured["cwd"] = cwd
                captured["env"] = env
                return __import__("subprocess").CompletedProcess(cmd, 0)

            with patch.object(sys, "argv", argv):
                with patch("scripts.preprocess_validation_only.ensure_upstream_exists", return_value=upstream_dir):
                    with patch(
                        "scripts.preprocess_validation_only.ensure_preprocess_multiprocessing_compat_source_patch",
                        return_value={"host_materialization": "patched", "safe_start_method": "patched"},
                    ):
                        with patch("scripts.preprocess_validation_only.subprocess.run", side_effect=fake_run):
                            rc = preprocess_main()

            self.assertEqual(rc, 0)
            self.assertEqual(captured["cwd"], upstream_dir)
            self.assertTrue(str(upstream_dir) in str(captured["env"]["PYTHONPATH"]))
            self.assertEqual(captured["env"]["LATENTDRIVER_PREPROCESS_START_METHOD"], "spawn")
            self.assertEqual(captured["env"]["LATENTDRIVER_PREPROCESS_WORKERS"], "1")
            self.assertEqual(captured["env"]["JAX_PLATFORMS"], "cpu")
            self.assertTrue((upstream_dir / "sitecustomize.py").exists())
            self.assertTrue((upstream_dir / "src" / "ops" / "crdp" / "__init__.py").exists())
            self.assertIn(
                "try:\n    import pytorch_lightning as pl",
                (upstream_dir / "src" / "utils" / "utils.py").read_text(encoding="utf-8"),
            )

    def test_preprocess_cache_status_marks_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(Path(td) / "raw")
            smoke_root = REPO_ROOT / "artifacts" / "assets" / "preprocessed" / "smoke"
            map_dir = smoke_root / "val_preprocessed_path" / "map"
            route_dir = smoke_root / "val_preprocessed_path" / "route"
            intention_dir = smoke_root / "val_intention_label"
            try:
                map_dir.mkdir(parents=True, exist_ok=True)
                route_dir.mkdir(parents=True, exist_ok=True)
                intention_dir.mkdir(parents=True, exist_ok=True)
                (map_dir / "a.npy").write_text("x", encoding="utf-8")
                (route_dir / "a.npy").write_text("x", encoding="utf-8")
                (intention_dir / "a.txt").write_text("x", encoding="utf-8")
                (smoke_root / "val_preprocessed_path" / "_SUCCESS").write_text("complete\n", encoding="utf-8")
                status = preprocess_cache_status("smoke")
                self.assertTrue(status["complete"])
                self.assertFalse(status["partial"])
            finally:
                clear_preprocess_outputs("smoke")

    def test_preprocess_cache_status_reuses_success_manifest_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(Path(td) / "raw")
            smoke_root = REPO_ROOT / "artifacts" / "assets" / "preprocessed" / "smoke"
            preprocess_root = smoke_root / "val_preprocessed_path"
            map_dir = preprocess_root / "map"
            route_dir = preprocess_root / "route"
            intention_dir = smoke_root / "val_intention_label"
            try:
                map_dir.mkdir(parents=True, exist_ok=True)
                route_dir.mkdir(parents=True, exist_ok=True)
                intention_dir.mkdir(parents=True, exist_ok=True)
                (preprocess_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
                (preprocess_root / "preprocess_manifest.json").write_text(
                    json.dumps({"counts": {"map_npy": 44097, "route_npy": 44097, "intention_txt": 44097}}),
                    encoding="utf-8",
                )
                status = preprocess_cache_status("smoke")
                self.assertTrue(status["complete"])
                self.assertEqual(status["counts_source"], "manifest")
                self.assertEqual(status["counts"]["map_npy"], 44097)
            finally:
                clear_preprocess_outputs("smoke")

    def test_can_repair_preprocess_markers_detects_consistent_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(Path(td) / "raw")
            smoke_root = REPO_ROOT / "artifacts" / "assets" / "preprocessed" / "smoke"
            map_dir = smoke_root / "val_preprocessed_path" / "map"
            route_dir = smoke_root / "val_preprocessed_path" / "route"
            intention_dir = smoke_root / "val_intention_label"
            try:
                map_dir.mkdir(parents=True, exist_ok=True)
                route_dir.mkdir(parents=True, exist_ok=True)
                intention_dir.mkdir(parents=True, exist_ok=True)
                for index in range(3):
                    (map_dir / f"{index}.npy").write_text("x", encoding="utf-8")
                    (route_dir / f"{index}.npy").write_text("x", encoding="utf-8")
                    (intention_dir / f"{index}.txt").write_text("x", encoding="utf-8")
                ok, detail = can_repair_preprocess_markers("smoke")
                self.assertTrue(ok)
                self.assertEqual(detail["counts"]["map_npy"], 3)
            finally:
                clear_preprocess_outputs("smoke")

    def test_repair_preprocess_complete_markers_writes_manifest_and_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(Path(td) / "raw")
            smoke_root = REPO_ROOT / "artifacts" / "assets" / "preprocessed" / "smoke"
            preprocess_root = smoke_root / "val_preprocessed_path"
            map_dir = preprocess_root / "map"
            route_dir = preprocess_root / "route"
            intention_dir = smoke_root / "val_intention_label"
            payload = {
                "command": ["python3", "-m", "src.preprocess.preprocess_data"],
                "waymo_path": "/tmp/raw",
                "preprocess_path": str(preprocess_root),
                "intention_path": str(intention_dir),
            }
            try:
                map_dir.mkdir(parents=True, exist_ok=True)
                route_dir.mkdir(parents=True, exist_ok=True)
                intention_dir.mkdir(parents=True, exist_ok=True)
                for index in range(2):
                    (map_dir / f"{index}.npy").write_text("x", encoding="utf-8")
                    (route_dir / f"{index}.npy").write_text("x", encoding="utf-8")
                    (intention_dir / f"{index}.txt").write_text("x", encoding="utf-8")
                manifest = repair_preprocess_complete_markers("smoke", payload)
                self.assertTrue((preprocess_root / "_SUCCESS").is_file())
                self.assertTrue((preprocess_root / "preprocess_manifest.json").is_file())
                self.assertTrue(manifest["repair"])
                self.assertEqual(manifest["counts"]["map_npy"], 2)
            finally:
                clear_preprocess_outputs("smoke")

    def test_main_repair_markers_skips_subprocess_and_recreates_completion_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(Path(td) / "raw")
            smoke_root = REPO_ROOT / "artifacts" / "assets" / "preprocessed" / "smoke"
            preprocess_root = smoke_root / "val_preprocessed_path"
            map_dir = preprocess_root / "map"
            route_dir = preprocess_root / "route"
            intention_dir = smoke_root / "val_intention_label"
            argv = ["preprocess_validation_only.py", "--mode", "smoke", "--repair-markers"]
            try:
                map_dir.mkdir(parents=True, exist_ok=True)
                route_dir.mkdir(parents=True, exist_ok=True)
                intention_dir.mkdir(parents=True, exist_ok=True)
                for index in range(4):
                    (map_dir / f"{index}.npy").write_text("x", encoding="utf-8")
                    (route_dir / f"{index}.npy").write_text("x", encoding="utf-8")
                    (intention_dir / f"{index}.txt").write_text("x", encoding="utf-8")
                with patch.object(sys, "argv", argv):
                    with patch("scripts.preprocess_validation_only.subprocess.run") as run_mock:
                        rc = preprocess_main()
                self.assertEqual(rc, 0)
                run_mock.assert_not_called()
                self.assertTrue((preprocess_root / "_SUCCESS").is_file())
                self.assertTrue((preprocess_root / "preprocess_manifest.json").is_file())
            finally:
                clear_preprocess_outputs("smoke")

    def test_main_repair_markers_rejects_mismatched_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(Path(td) / "raw")
            smoke_root = REPO_ROOT / "artifacts" / "assets" / "preprocessed" / "smoke"
            preprocess_root = smoke_root / "val_preprocessed_path"
            map_dir = preprocess_root / "map"
            route_dir = preprocess_root / "route"
            intention_dir = smoke_root / "val_intention_label"
            argv = ["preprocess_validation_only.py", "--mode", "smoke", "--repair-markers"]
            try:
                map_dir.mkdir(parents=True, exist_ok=True)
                route_dir.mkdir(parents=True, exist_ok=True)
                intention_dir.mkdir(parents=True, exist_ok=True)
                (map_dir / "0.npy").write_text("x", encoding="utf-8")
                (route_dir / "0.npy").write_text("x", encoding="utf-8")
                (route_dir / "1.npy").write_text("x", encoding="utf-8")
                (intention_dir / "0.txt").write_text("x", encoding="utf-8")
                with patch.object(sys, "argv", argv):
                    with self.assertRaisesRegex(RuntimeError, "count_mismatch"):
                        preprocess_main()
            finally:
                clear_preprocess_outputs("smoke")

    def test_main_reuses_existing_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_root = Path(td) / "raw"
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(raw_root)
            upstream_dir = Path(td) / "LatentDriver"
            (upstream_dir / "src" / "utils").mkdir(parents=True, exist_ok=True)
            (upstream_dir / "src" / "policy" / "latentdriver").mkdir(parents=True, exist_ok=True)
            (upstream_dir / "src" / "policy" / "baseline").mkdir(parents=True, exist_ok=True)
            (upstream_dir / "src" / "utils" / "utils.py").write_text("import pytorch_lightning as pl\n", encoding="utf-8")
            (upstream_dir / "src" / "policy" / "latentdriver" / "lantentdriver_model.py").write_text(
                "import torch.nn as nn\nimport pytorch_lightning as pl\n",
                encoding="utf-8",
            )
            (upstream_dir / "src" / "policy" / "baseline" / "bc_baseline.py").write_text(
                "from torch import nn\nimport pytorch_lightning as pl\n",
                encoding="utf-8",
            )
            smoke_root = REPO_ROOT / "artifacts" / "assets" / "preprocessed" / "smoke"
            map_dir = smoke_root / "val_preprocessed_path" / "map"
            route_dir = smoke_root / "val_preprocessed_path" / "route"
            intention_dir = smoke_root / "val_intention_label"
            try:
                map_dir.mkdir(parents=True, exist_ok=True)
                route_dir.mkdir(parents=True, exist_ok=True)
                intention_dir.mkdir(parents=True, exist_ok=True)
                (map_dir / "a.npy").write_text("x", encoding="utf-8")
                (route_dir / "a.npy").write_text("x", encoding="utf-8")
                (intention_dir / "a.txt").write_text("x", encoding="utf-8")
                (smoke_root / "val_preprocessed_path" / "_SUCCESS").write_text("complete\n", encoding="utf-8")
                argv = ["preprocess_validation_only.py", "--mode", "smoke"]
                with patch.object(sys, "argv", argv):
                    with patch("scripts.preprocess_validation_only.ensure_upstream_exists", return_value=upstream_dir):
                        with patch(
                            "scripts.preprocess_validation_only.ensure_preprocess_multiprocessing_compat_source_patch",
                            return_value={"host_materialization": "patched", "safe_start_method": "patched"},
                        ):
                            with patch("scripts.preprocess_validation_only.subprocess.run") as run_mock:
                                rc = preprocess_main()
                self.assertEqual(rc, 0)
                run_mock.assert_not_called()
            finally:
                clear_preprocess_outputs("smoke")

    def test_clear_preprocess_outputs_removes_generated_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.environ["LATENTDRIVER_WAYMO_DATASET_ROOT"] = str(Path(td) / "raw")
            smoke_root = REPO_ROOT / "artifacts" / "assets" / "preprocessed" / "smoke"
            map_dir = smoke_root / "val_preprocessed_path" / "map"
            route_dir = smoke_root / "val_preprocessed_path" / "route"
            intention_dir = smoke_root / "val_intention_label"
            map_dir.mkdir(parents=True, exist_ok=True)
            route_dir.mkdir(parents=True, exist_ok=True)
            intention_dir.mkdir(parents=True, exist_ok=True)
            (map_dir / "a.npy").write_text("x", encoding="utf-8")
            (route_dir / "a.npy").write_text("x", encoding="utf-8")
            (intention_dir / "a.txt").write_text("x", encoding="utf-8")
            payload = clear_preprocess_outputs("smoke")
            self.assertFalse(map_dir.exists())
            self.assertFalse(route_dir.exists())
            self.assertFalse(intention_dir.exists())
            self.assertEqual(len(payload["removed"]), 3)


if __name__ == "__main__":
    unittest.main()
