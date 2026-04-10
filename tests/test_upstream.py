from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.upstream import (
    MODEL_LIGHTNING_NEW_IMPORT,
    PYTHON312_SITE_CUSTOMIZE_BLOCK,
    UTILS_LIGHTNING_NEW_IMPORT,
    ensure_lightning_compat_source_patches,
    ensure_python312_compat_sitecustomize,
)


class UpstreamCompatTests(unittest.TestCase):
    def test_ensure_python312_compat_sitecustomize_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            sitecustomize_path = ensure_python312_compat_sitecustomize(upstream_dir)
            self.assertEqual(sitecustomize_path, upstream_dir / "sitecustomize.py")
            self.assertIn(PYTHON312_SITE_CUSTOMIZE_BLOCK, sitecustomize_path.read_text(encoding="utf-8"))

    def test_ensure_python312_compat_sitecustomize_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            first = ensure_python312_compat_sitecustomize(upstream_dir)
            first_text = first.read_text(encoding="utf-8")
            second = ensure_python312_compat_sitecustomize(upstream_dir)
            self.assertEqual(first_text, second.read_text(encoding="utf-8"))

    def test_ensure_lightning_compat_source_patches_rewrites_eval_only_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            utils_path = upstream_dir / "src" / "utils"
            model_path = upstream_dir / "src" / "policy" / "latentdriver"
            baseline_path = upstream_dir / "src" / "policy" / "baseline"
            utils_path.mkdir(parents=True, exist_ok=True)
            model_path.mkdir(parents=True, exist_ok=True)
            baseline_path.mkdir(parents=True, exist_ok=True)
            (utils_path / "utils.py").write_text("import pytorch_lightning as pl\n", encoding="utf-8")
            (model_path / "lantentdriver_model.py").write_text("import torch.nn as nn\nimport pytorch_lightning as pl\n", encoding="utf-8")
            (baseline_path / "bc_baseline.py").write_text("from torch import nn\nimport pytorch_lightning as pl\n", encoding="utf-8")

            result = ensure_lightning_compat_source_patches(upstream_dir)

            self.assertEqual(result["utils"], "patched")
            self.assertEqual(result["latentdriver_model"], "patched")
            self.assertEqual(result["bc_baseline"], "patched")
            self.assertIn(UTILS_LIGHTNING_NEW_IMPORT, (utils_path / "utils.py").read_text(encoding="utf-8"))
            self.assertIn(MODEL_LIGHTNING_NEW_IMPORT, (model_path / "lantentdriver_model.py").read_text(encoding="utf-8"))
            self.assertIn(MODEL_LIGHTNING_NEW_IMPORT, (baseline_path / "bc_baseline.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
