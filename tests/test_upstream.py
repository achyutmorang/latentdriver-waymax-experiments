from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.upstream import (
    CRDP_FALLBACK_INIT,
    JAX_TREE_MAP_COMPAT_FILES,
    MATPLOTLIB_IMG_FROM_FIG_NEW_BLOCK,
    MODEL_LIGHTNING_NEW_IMPORT,
    PREPROCESS_DIR_CHECK_NEW_BLOCK,
    PREPROCESS_POOL_NEW_BLOCK,
    PREPROCESS_SCENARIO_NEW_BLOCK,
    PREPROCESS_TASKS_NEW_BLOCK,
    PYTHON312_SITE_CUSTOMIZE_BLOCK,
    UTILS_LIGHTNING_NEW_IMPORT,
    ensure_crdp_compat_source_patch,
    ensure_jax_tree_map_compat_source_patch,
    ensure_lightning_compat_source_patches,
    ensure_matplotlib_canvas_compat_source_patch,
    ensure_preprocess_multiprocessing_compat_source_patch,
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

    def test_ensure_preprocess_multiprocessing_compat_source_patch_rewrites_preprocess_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            preprocess_dir = upstream_dir / "src" / "preprocess"
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            preprocess_path = preprocess_dir / "preprocess_data.py"
            preprocess_path.write_text(
                """import numpy as np\nimport multiprocessing as mp\nimport os\nimport time\n\nclass Preprocessor:\n    def _check_and_create_dirs(self):\n        if os.path.exists(self.path_to_map):\n            raise ValueError(f'The map has been dumped in {self.path_to_map}, please delete the map first')\n        if os.path.exists(self.path_to_route):\n            raise ValueError(f'The route has been dumped in {self.path_to_route}, please delete the route first')\n        if os.path.exists(self.intention_label_path):\n            raise ValueError(f'The intention label has been dumped in {self.intention_label_path}, please delete the intention label first')\n        \n        os.makedirs(self.path_to_map, exist_ok=True)\n        os.makedirs(self.path_to_route, exist_ok=True)\n        os.makedirs(self.intention_label_path, exist_ok=True)\n\n    def _process_scenario(self, scen):\n"""
                + """        cur_id = scen._scenario_id.reshape(-1)\n        \n        # Extract map data\n        road_obs, ids = get_whole_map(scen)\n        \n        # Extract route data\n        routes, ego_car_width = get_route_global(scen)\n        routes = np.array(routes)\n        ego_car_width = float(ego_car_width)\n        \n        # Extract intention label data\n        mask = scen.object_metadata.is_sdc\n        sdc_xy = np.array(scen.log_trajectory.xy[mask, ...])\n        yaw = np.array(scen.log_trajectory.yaw[mask, ...])\n\n"""
                + """        tasks = []\n        for bs in range(len(cur_id)):\n            tasks.append((\n                # Map data\n                road_obs[bs],\n                ids[bs],\n                self.data_conf.max_map_segments,\n                os.path.join(self.path_to_map, \x27{}\x27.format(cur_id[bs])),\n                # Route data\n                routes[bs:bs+1],\n                self.data_conf.max_route_segments,\n                ego_car_width,\n                os.path.join(self.path_to_route, \x27{}\x27.format(cur_id[bs])),\n                # Intention label data\n                sdc_xy[bs],\n                yaw[bs],\n                cur_id[bs],\n                self.intention_label_path\n            ))\n        \n        return tasks\n\n"""
                + """    def run(self):\n        with mp.Pool(processes=mp.cpu_count()) as pool:\n            for batch_id, scen in enumerate(self.data_iter):\n                t_start = time.time()\n                tasks = self._process_scenario(scen)\n                pool.starmap(workers, tasks)\n                \n                print(f"Processed; current batch is: {batch_id}; Using time is: {time.time() - t_start}")\n""",
                encoding="utf-8",
            )

            result = ensure_preprocess_multiprocessing_compat_source_patch(upstream_dir)

            self.assertEqual(result["resume_dirs"], "patched")
            self.assertEqual(result["host_materialization"], "patched")
            self.assertEqual(result["resume_tasks"], "patched")
            self.assertEqual(result["safe_start_method"], "patched")
            rewritten = preprocess_path.read_text(encoding="utf-8")
            self.assertIn(PREPROCESS_DIR_CHECK_NEW_BLOCK, rewritten)
            self.assertIn(PREPROCESS_SCENARIO_NEW_BLOCK, rewritten)
            self.assertIn(PREPROCESS_TASKS_NEW_BLOCK, rewritten)
            self.assertIn(PREPROCESS_POOL_NEW_BLOCK, rewritten)

    def test_ensure_jax_tree_map_compat_source_patch_rewrites_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            for relative_path in JAX_TREE_MAP_COMPAT_FILES:
                path = upstream_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("import jax\nvalue = jax.tree_map(lambda x: x, tree)\n", encoding="utf-8")

            result = ensure_jax_tree_map_compat_source_patch(upstream_dir)

            for relative_path in JAX_TREE_MAP_COMPAT_FILES:
                key = str(relative_path)
                self.assertEqual(result[key], "patched")
                rewritten = (upstream_dir / relative_path).read_text(encoding="utf-8")
                self.assertIn("jax.tree_util.tree_map(", rewritten)
                self.assertNotIn("jax.tree_map(", rewritten)

    def test_ensure_matplotlib_canvas_compat_source_patch_rewrites_img_from_fig(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            upstream_dir = Path(td)
            utils_path = upstream_dir / "waymax" / "visualization" / "utils.py"
            utils_path.parent.mkdir(parents=True, exist_ok=True)
            utils_path.write_text(
                """def img_from_fig(fig):
  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return img
""",
                encoding="utf-8",
            )

            result = ensure_matplotlib_canvas_compat_source_patch(upstream_dir)

            self.assertEqual(result, "patched")
            rewritten = utils_path.read_text(encoding="utf-8")
            self.assertIn(MATPLOTLIB_IMG_FROM_FIG_NEW_BLOCK, rewritten)
            self.assertIn("canvas.buffer_rgba()", rewritten)

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
