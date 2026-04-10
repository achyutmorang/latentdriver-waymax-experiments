from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.upstream import (
    CRDP_FALLBACK_INIT,
    MODEL_LIGHTNING_NEW_IMPORT,
    PYTHON312_SITE_CUSTOMIZE_BLOCK,
    UTILS_LIGHTNING_NEW_IMPORT,
    ensure_crdp_compat_source_patch,
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
            (model_path / "lantentdriver_model.py").write_text(
                "import torch.nn as nn\nimport pytorch_lightning as pl\n",
                encoding="utf-8",
            )
            (baseline_path / "bc_baseline.py").write_text(
                "from torch import nn\nimport pytorch_lightning as pl\n",
                encoding="utf-8",
            )

            result = ensure_lightning_compat_source_patches(upstream_dir)

            self.assertEqual(result["utils"], "patched")
            self.assertEqual(result["latentdriver_model"], "patched")
            self.assertEqual(result["bc_baseline"], "patched")
            self.assertIn(UTILS_LIGHTNING_NEW_IMPORT, (utils_path / "utils.py").read_text(encoding="utf-8"))
            self.assertIn(MODEL_LIGHTNING_NEW_IMPORT, (model_path / "lantentdriver_model.py").read_text(encoding="utf-8"))
            self.assertIn(MODEL_LIGHTNING_NEW_IMPORT, (baseline_path / "bc_baseline.py").read_text(encoding="utf-8"))

    def test_ensure_crdp_compat_source_patch_writes_fallback_package_init(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            result = ensure_crdp_compat_source_patch(upstream_dir)
            init_path = upstream_dir / "src" / "ops" / "crdp" / "__init__.py"
            self.assertEqual(result, "patched")
            self.assertTrue(init_path.exists())
            self.assertEqual(init_path.read_text(encoding="utf-8"), CRDP_FALLBACK_INIT)

    def test_crdp_fallback_module_downsamples_collinear_points(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            ensure_crdp_compat_source_patch(upstream_dir)
            sys.path.insert(0, str(upstream_dir))
            try:
                sys.modules.pop("src.ops.crdp", None)
                module = importlib.import_module("src.ops.crdp")
                result = module.crdp.rdp([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], 0.1)
                self.assertEqual(result, [[0.0, 0.0], [2.0, 0.0]])
            finally:
                sys.modules.pop("src.ops.crdp", None)
                sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
