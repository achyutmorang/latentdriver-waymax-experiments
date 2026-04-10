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
                    with patch("scripts.preprocess_validation_only.subprocess.run", side_effect=fake_run):
                        rc = preprocess_main()

            self.assertEqual(rc, 0)
            self.assertEqual(captured["cwd"], upstream_dir)
            self.assertTrue(str(upstream_dir) in str(captured["env"]["PYTHONPATH"]))
            self.assertTrue((upstream_dir / "sitecustomize.py").exists())
            self.assertTrue((upstream_dir / "src" / "ops" / "crdp" / "__init__.py").exists())
            self.assertIn(
                "try:\n    import pytorch_lightning as pl",
                (upstream_dir / "src" / "utils" / "utils.py").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
