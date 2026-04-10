from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.setup_colab_runtime import (
    GPT2_NEW_IMPORT_BLOCK,
    GPT2_OLD_IMPORT_BLOCK,
    JAX_GPU_PIN,
    SORT_VERT_FALLBACK_CODE,
    patch_gpt2_model,
    patch_sort_vertices,
    runtime_install_commands,
    verify_jax_gpu_backend,
)


class SetupColabRuntimeTests(unittest.TestCase):
    def test_runtime_install_commands_use_modern_waymax_and_jax_flow(self) -> None:
        commands = runtime_install_commands()
        joined = [" ".join(cmd) for cmd in commands]
        self.assertTrue(any("waymo-research/waymax.git@main#egg=waymo-waymax" in cmd for cmd in joined))
        self.assertTrue(any(JAX_GPU_PIN in cmd for cmd in joined))
        self.assertFalse(any("tensorflow==2.15.0" in cmd for cmd in joined))
        self.assertFalse(any("setup.py install" in cmd for cmd in joined))
        self.assertFalse(any("pytorch-lightning" in cmd for cmd in joined))
        self.assertFalse(any(" lightning==" in cmd or cmd.endswith(" lightning") for cmd in joined))

    def test_patch_gpt2_model_rewrites_old_transformers_import_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "src" / "policy" / "latentdriver"
            path.mkdir(parents=True, exist_ok=True)
            file = path / "gpt2_model.py"
            file.write_text(f"prefix\n{GPT2_OLD_IMPORT_BLOCK}\nsuffix\n", encoding="utf-8")
            result = patch_gpt2_model(root)
            self.assertEqual(result, "patched")
            text = file.read_text(encoding="utf-8")
            self.assertIn(GPT2_NEW_IMPORT_BLOCK, text)
            self.assertNotIn(GPT2_OLD_IMPORT_BLOCK, text)

    def test_patch_sort_vertices_overwrites_module_with_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "src" / "ops" / "sort_vertices"
            path.mkdir(parents=True, exist_ok=True)
            file = path / "sort_vert.py"
            file.write_text("old", encoding="utf-8")
            result = patch_sort_vertices(root)
            self.assertEqual(result, "patched")
            self.assertEqual(file.read_text(encoding="utf-8"), SORT_VERT_FALLBACK_CODE)

    def test_verify_jax_gpu_backend_raises_when_gpu_visible_but_jax_is_cpu(self) -> None:
        import subprocess
        from unittest.mock import patch

        failing_probe = subprocess.CompletedProcess(
            args=["python3", "-c", "probe"],
            returncode=1,
            stdout='{"gpu_visible": true, "jax_has_gpu": false}',
            stderr="",
        )
        with patch("scripts.setup_colab_runtime.subprocess.run", return_value=failing_probe):
            with self.assertRaisesRegex(RuntimeError, "JAX is not using the visible NVIDIA GPU"):
                verify_jax_gpu_backend()


if __name__ == "__main__":
    unittest.main()
