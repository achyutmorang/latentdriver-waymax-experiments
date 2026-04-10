from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.preprocess_validation_only import build_preprocess_command, main as preprocess_main


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
            upstream_dir.mkdir(parents=True, exist_ok=True)
            argv = ["preprocess_validation_only.py", "--mode", "smoke"]
            captured: dict[str, object] = {}

            def fake_run(cmd, cwd, check, text, env):
                captured["cmd"] = cmd
                captured["cwd"] = cwd
                captured["env"] = env
                return __import__("subprocess").CompletedProcess(cmd, 0)

            with patch.object(sys, "argv", argv):
                with patch("scripts.preprocess_validation_only.ensure_upstream_exists", return_value=upstream_dir):
                    with patch("scripts.preprocess_validation_only.subprocess.run", side_effect=fake_run):
                        rc = preprocess_main()

            self.assertEqual(rc, 0)
            self.assertEqual(captured["cwd"], upstream_dir)
            self.assertTrue(str(upstream_dir) in str(captured["env"]["PYTHONPATH"]))
            self.assertTrue((upstream_dir / "sitecustomize.py").exists())


if __name__ == "__main__":
    unittest.main()
