from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from .config import load_config, resolve_repo_relative


PYTHON312_SITE_CUSTOMIZE_BLOCK = """# latentdriver-waymax-experiments python3.12 compatibility
import pkgutil

if not hasattr(pkgutil, \"ImpImporter\"):
    class _CodexCompatImpImporter:
        pass

    pkgutil.ImpImporter = _CodexCompatImpImporter

if not hasattr(pkgutil, \"ImpLoader\"):
    class _CodexCompatImpLoader:
        pass

    pkgutil.ImpLoader = _CodexCompatImpLoader
"""

UTILS_LIGHTNING_OLD_IMPORT = "import pytorch_lightning as pl\n"
UTILS_LIGHTNING_NEW_IMPORT = """try:
    import pytorch_lightning as pl
except Exception:
    class _PLCompat:
        @staticmethod
        def seed_everything(*args, **kwargs):
            return None

    pl = _PLCompat()
"""

MODEL_LIGHTNING_OLD_IMPORT = "import pytorch_lightning as pl\n"
MODEL_LIGHTNING_NEW_IMPORT = """try:
    import pytorch_lightning as pl
except Exception:
    class _LightningModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            return None

        def log_dict(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    class _PLCompat:
        LightningModule = _LightningModule

    pl = _PLCompat()
"""

CRDP_FALLBACK_INIT = """from __future__ import annotations

import importlib
import numpy as np

try:
    _crdp = importlib.import_module(f"{__name__}.crdp")
except Exception:
    _crdp = None


def _rdp_mask(points: np.ndarray, epsilon: float) -> np.ndarray:
    n = int(points.shape[0])
    if n <= 2:
        return np.ones(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    stack = [(0, n - 1)]
    while stack:
        st, ed = stack.pop()
        if ed - st <= 1:
            continue
        start = points[st]
        end = points[ed]
        segment = end - start
        segment_norm = float(np.linalg.norm(segment))
        dmax = 0.0
        index = st
        for i in range(st + 1, ed):
            if segment_norm:
                distance = abs(segment[1] * points[i, 0] - segment[0] * points[i, 1] + end[0] * start[1] - end[1] * start[0]) / segment_norm
            else:
                distance = float(np.linalg.norm(points[i] - start))
            if distance > dmax:
                dmax = distance
                index = i
        if dmax > epsilon:
            stack.append((st, index))
            stack.append((index, ed))
        else:
            mask[st + 1:ed] = False
    return mask


class _CRDPCompatModule:
    @staticmethod
    def rdp(points, epsilon=0.0, return_mask=False):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected points with shape [N, 2], got {pts.shape!r}")
        mask = _rdp_mask(pts, float(epsilon))
        if return_mask:
            return mask.tolist()
        return [points[i] for i, keep in enumerate(mask.tolist()) if keep]


crdp = _crdp if _crdp is not None else _CRDPCompatModule()
"""

PREPROCESS_DIR_CHECK_OLD_BLOCK = """        if os.path.exists(self.path_to_map):
            raise ValueError(f'The map has been dumped in {self.path_to_map}, please delete the map first')
        if os.path.exists(self.path_to_route):
            raise ValueError(f'The route has been dumped in {self.path_to_route}, please delete the route first')
        if os.path.exists(self.intention_label_path):
            raise ValueError(f'The intention label has been dumped in {self.intention_label_path}, please delete the intention label first')

        os.makedirs(self.path_to_map, exist_ok=True)
        os.makedirs(self.path_to_route, exist_ok=True)
        os.makedirs(self.intention_label_path, exist_ok=True)
""".replace("\n\n", "\n        \n")

PREPROCESS_DIR_CHECK_NEW_BLOCK = """        os.makedirs(self.path_to_map, exist_ok=True)
        os.makedirs(self.path_to_route, exist_ok=True)
        os.makedirs(self.intention_label_path, exist_ok=True)
"""

PREPROCESS_CHECK_AND_CREATE_DIRS_BODY = '''        """
        Create output directories while preserving partial preprocessing caches.
        """
''' + PREPROCESS_DIR_CHECK_NEW_BLOCK + "\n"

PREPROCESS_SCENARIO_OLD_BLOCK = """        cur_id = scen._scenario_id.reshape(-1)

        # Extract map data
        road_obs, ids = get_whole_map(scen)

        # Extract route data
        routes, ego_car_width = get_route_global(scen)
        routes = np.array(routes)
        ego_car_width = float(ego_car_width)

        # Extract intention label data
        mask = scen.object_metadata.is_sdc
        sdc_xy = np.array(scen.log_trajectory.xy[mask, ...])
        yaw = np.array(scen.log_trajectory.yaw[mask, ...])
""".replace("\n\n", "\n        \n")

PREPROCESS_SCENARIO_HOST_MATERIALIZED_BLOCK = """        cur_id = np.asarray(scen._scenario_id).reshape(-1)

        # Materialize JAX outputs on host before dispatching multiprocessing work.
        # Worker processes must only receive NumPy / Python values, never JAX device arrays.
        road_obs, ids = get_whole_map(scen)
        road_obs = np.asarray(road_obs)
        ids = np.asarray(ids)

        # Extract route data
        routes, ego_car_width = get_route_global(scen)
        routes = np.asarray(routes)
        ego_car_width = float(np.asarray(ego_car_width))

        # Extract intention label data
        mask = scen.object_metadata.is_sdc
        sdc_xy = np.asarray(scen.log_trajectory.xy[mask, ...])
        yaw = np.asarray(scen.log_trajectory.yaw[mask, ...])
""".replace("\n\n", "\n        \n")

PREPROCESS_SCENARIO_NEW_BLOCK = """        cur_id = np.asarray(scen._scenario_id).reshape(-1)

        # Materialize JAX outputs on host before dispatching multiprocessing work.
        # Worker processes must only receive NumPy / Python values, never JAX device arrays.
        road_obs, ids = get_whole_map(scen)
        road_obs = np.asarray(road_obs)
        ids = np.asarray(ids)

        # Extract route data
        routes, ego_car_width = get_route_global(scen)
        routes = np.asarray(routes)
        ego_car_width = float(np.asarray(ego_car_width))

        # Extract intention label data
        mask = scen.object_metadata.is_sdc
        sdc_xy = np.asarray(scen.log_trajectory.xy[mask, ...])
        yaw = np.asarray(scen.log_trajectory.yaw[mask, ...])
""".replace("\n\n", "\n        \n")

PREPROCESS_TASKS_OLD_BLOCK = """        tasks = []
        for bs in range(len(cur_id)):
            tasks.append((
                # Map data
                road_obs[bs],
                ids[bs],
                self.data_conf.max_map_segments,
                os.path.join(self.path_to_map, '{}'.format(cur_id[bs])),
                # Route data
                routes[bs:bs+1],
                self.data_conf.max_route_segments,
                ego_car_width,
                os.path.join(self.path_to_route, '{}'.format(cur_id[bs])),
                # Intention label data
                sdc_xy[bs],
                yaw[bs],
                cur_id[bs],
                self.intention_label_path
            ))

        return tasks
""".replace("\n\n", "\n        \n")

PREPROCESS_TASKS_NEW_BLOCK = """        tasks = []
        skipped = 0
        for bs in range(len(cur_id)):
            scenario_id = cur_id[bs]
            map_path = os.path.join(self.path_to_map, '{}'.format(scenario_id))
            route_path = os.path.join(self.path_to_route, '{}'.format(scenario_id))
            intention_path = os.path.join(self.intention_label_path, f'{scenario_id}.txt')
            if all(os.path.exists(path) and os.path.getsize(path) > 0 for path in (map_path + '.npy', route_path + '.npy', intention_path)):
                skipped += 1
                continue
            tasks.append((
                # Map data
                road_obs[bs],
                ids[bs],
                self.data_conf.max_map_segments,
                map_path,
                # Route data
                routes[bs:bs+1],
                self.data_conf.max_route_segments,
                ego_car_width,
                route_path,
                # Intention label data
                sdc_xy[bs],
                yaw[bs],
                scenario_id,
                self.intention_label_path
            ))
        if skipped:
            print(f\"Skipped existing preprocessed scenarios: {skipped}\")

        return tasks
""".replace("\n\n", "\n        \n")

PREPROCESS_PROCESS_SCENARIO_BODY = '''        """
        Process a single scenario to extract map, route, and intention label data.

        Existing per-scenario outputs are skipped so interrupted full validation preprocessing can resume.
        """
''' + PREPROCESS_SCENARIO_NEW_BLOCK + "\n" + PREPROCESS_TASKS_NEW_BLOCK + "\n"

PREPROCESS_POOL_OLD_BLOCK = """        with mp.Pool(processes=mp.cpu_count()) as pool:
            for batch_id, scen in enumerate(self.data_iter):
                t_start = time.time()
                tasks = self._process_scenario(scen)
                pool.starmap(workers, tasks)

                print(f\"Processed; current batch is: {batch_id}; Using time is: {time.time() - t_start}\")
""".replace("\n\n", "\n                \n")

PREPROCESS_POOL_START_METHOD_BLOCK = """        start_method = os.environ.get('LATENTDRIVER_PREPROCESS_START_METHOD', 'spawn')
        worker_count = max(1, int(os.environ.get('LATENTDRIVER_PREPROCESS_WORKERS', mp.cpu_count())))
        mp_ctx = mp.get_context(start_method)
        with mp_ctx.Pool(processes=worker_count) as pool:
            for batch_id, scen in enumerate(self.data_iter):
                t_start = time.time()
                tasks = self._process_scenario(scen)
                pool.starmap(workers, tasks)

                print(f\"Processed; current batch is: {batch_id}; Using time is: {time.time() - t_start}\")
""".replace("\n\n", "\n                \n")

PREPROCESS_POOL_NEW_BLOCK = """        start_method = os.environ.get('LATENTDRIVER_PREPROCESS_START_METHOD', 'spawn')
        worker_count = max(1, int(os.environ.get('LATENTDRIVER_PREPROCESS_WORKERS', '1')))
        if worker_count <= 1:
            for batch_id, scen in enumerate(self.data_iter):
                t_start = time.time()
                tasks = self._process_scenario(scen)
                for task in tasks:
                    workers(*task)

                print(f\"Processed; current batch is: {batch_id}; Tasks: {len(tasks)}; Using time is: {time.time() - t_start}\")
            return
        mp_ctx = mp.get_context(start_method)
        with mp_ctx.Pool(processes=worker_count) as pool:
            for batch_id, scen in enumerate(self.data_iter):
                t_start = time.time()
                tasks = self._process_scenario(scen)
                pool.starmap(workers, tasks)

                print(f\"Processed; current batch is: {batch_id}; Tasks: {len(tasks)}; Using time is: {time.time() - t_start}\")
""".replace("\n\n", "\n                \n")

PREPROCESS_RUN_BODY = '''        """
        Run the preprocessing pipeline.
        """
        self._check_and_create_dirs()

        print(f'Start dumping whole map, the map will be saved in {self.path_to_map}')
        print(f'Start dumping route, the route will be saved in {self.path_to_route}')
        print(f'Start dumping intention label, the intention label will be saved in {self.intention_label_path}')

''' + PREPROCESS_POOL_NEW_BLOCK


MATPLOTLIB_IMG_FROM_FIG_OLD_BLOCK = """  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
"""

MATPLOTLIB_IMG_FROM_FIG_NEW_BLOCK = """  fig.canvas.draw()
  canvas = fig.canvas
  if hasattr(canvas, 'tostring_rgb'):
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(canvas.get_width_height()[::-1] + (3,))
  else:
    img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
"""

JAX_TREE_MAP_COMPAT_FILES = (
    Path("simulator") / "utils.py",
    Path("waymax") / "agents" / "expert.py",
    Path("waymax") / "agents" / "waypoint_following_agent.py",
    Path("waymax") / "visualization" / "viz.py",
)


def upstream_paths() -> Dict[str, Path]:
    cfg = load_config()
    return {
        "repo_dir": resolve_repo_relative(cfg["upstream"]["repo_dir"]),
        "patch_path": resolve_repo_relative(cfg["upstream"]["patch_path"]),
        "lock_root": resolve_repo_relative(cfg["assets"]["lock_root"]),
    }


def clone_and_patch_upstream() -> Dict[str, Any]:
    cfg = load_config()
    paths = upstream_paths()
    repo_dir = paths["repo_dir"]
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", cfg["upstream"]["fork_repo_url"], str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", cfg["upstream"]["pinned_commit"]], check=True)
    patch_path = paths["patch_path"]
    check = subprocess.run(["git", "-C", str(repo_dir), "apply", "--check", str(patch_path)], capture_output=True, text=True)
    reverse = subprocess.run(["git", "-C", str(repo_dir), "apply", "--reverse", "--check", str(patch_path)], capture_output=True, text=True)
    if check.returncode == 0:
        subprocess.run(["git", "-C", str(repo_dir), "apply", str(patch_path)], check=True)
        patch_state = "applied"
    elif reverse.returncode == 0:
        patch_state = "already_applied"
    else:
        raise RuntimeError(f"Unable to apply patch {patch_path}: {check.stderr or reverse.stderr}")

    lock_root = paths["lock_root"]
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / "upstream_bootstrap.json"
    payload = {
        "repo_dir": str(repo_dir),
        "fork_repo_url": cfg["upstream"]["fork_repo_url"],
        "upstream_repo_url": cfg["upstream"]["upstream_repo_url"],
        "pinned_commit": cfg["upstream"]["pinned_commit"],
        "patch_path": str(patch_path),
        "patch_state": patch_state,
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def ensure_upstream_exists() -> Path:
    repo_dir = upstream_paths()["repo_dir"]
    if not repo_dir.exists():
        raise FileNotFoundError(f"Upstream repo missing: {repo_dir}. Run scripts/bootstrap_upstream.py first.")
    return repo_dir


def ensure_python312_compat_sitecustomize(upstream_dir: Path) -> Path:
    sitecustomize_path = upstream_dir / "sitecustomize.py"
    existing = sitecustomize_path.read_text(encoding="utf-8") if sitecustomize_path.exists() else ""
    if PYTHON312_SITE_CUSTOMIZE_BLOCK in existing:
        return sitecustomize_path
    updated = PYTHON312_SITE_CUSTOMIZE_BLOCK if not existing else f"{PYTHON312_SITE_CUSTOMIZE_BLOCK}\n{existing}"
    sitecustomize_path.write_text(updated, encoding="utf-8")
    return sitecustomize_path


def _replace_source_block(path: Path, old: str, new: str) -> str:
    return _replace_source_block_candidates(path, (old,), new)


def _replace_source_block_candidates(path: Path, old_candidates: tuple[str, ...], new: str) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return "already_patched"
    for old in old_candidates:
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")
            return "patched"
    return "not_found"


def _replace_method_body(path: Path, method_signature: str, end_marker: str, new_body: str) -> str:
    text = path.read_text(encoding="utf-8")
    method_start = text.find(method_signature)
    if method_start == -1:
        return "not_found"
    body_start = text.find("\n", method_start)
    if body_start == -1:
        return "not_found"
    body_start += 1
    body_end = text.find(end_marker, body_start)
    if body_end == -1:
        return "not_found"
    current_body = text[body_start:body_end]
    if current_body == new_body:
        return "already_patched"
    path.write_text(text[:body_start] + new_body + text[body_end:], encoding="utf-8")
    return "patched"


def ensure_lightning_compat_source_patches(upstream_dir: Path) -> Dict[str, str]:
    return {
        "utils": _replace_source_block(
            upstream_dir / "src" / "utils" / "utils.py",
            UTILS_LIGHTNING_OLD_IMPORT,
            UTILS_LIGHTNING_NEW_IMPORT,
        ),
        "latentdriver_model": _replace_source_block(
            upstream_dir / "src" / "policy" / "latentdriver" / "lantentdriver_model.py",
            MODEL_LIGHTNING_OLD_IMPORT,
            MODEL_LIGHTNING_NEW_IMPORT,
        ),
        "bc_baseline": _replace_source_block(
            upstream_dir / "src" / "policy" / "baseline" / "bc_baseline.py",
            MODEL_LIGHTNING_OLD_IMPORT,
            MODEL_LIGHTNING_NEW_IMPORT,
        ),
    }


def ensure_crdp_compat_source_patch(upstream_dir: Path) -> str:
    init_path = upstream_dir / "src" / "ops" / "crdp" / "__init__.py"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    existing = init_path.read_text(encoding="utf-8") if init_path.exists() else ""
    if existing == CRDP_FALLBACK_INIT:
        return "already_patched"
    init_path.write_text(CRDP_FALLBACK_INIT, encoding="utf-8")
    return "patched"


def ensure_preprocess_multiprocessing_compat_source_patch(upstream_dir: Path) -> Dict[str, str]:
    preprocess_path = upstream_dir / "src" / "preprocess" / "preprocess_data.py"
    resume_dirs = _replace_method_body(
        preprocess_path,
        "    def _check_and_create_dirs(self):",
        "    def _process_scenario(self, scen):",
        PREPROCESS_CHECK_AND_CREATE_DIRS_BODY,
    )
    process_scenario = _replace_method_body(
        preprocess_path,
        "    def _process_scenario(self, scen):",
        "    def run(self):",
        PREPROCESS_PROCESS_SCENARIO_BODY,
    )
    safe_start_method = _replace_method_body(
        preprocess_path,
        "    def run(self):",
        "\n@hydra.main",
        PREPROCESS_RUN_BODY,
    )
    return {
        "resume_dirs": resume_dirs,
        "host_materialization": process_scenario,
        "resume_tasks": process_scenario,
        "safe_start_method": safe_start_method,
    }

def ensure_matplotlib_canvas_compat_source_patch(upstream_dir: Path) -> str:
    return _replace_source_block(
        upstream_dir / "waymax" / "visualization" / "utils.py",
        MATPLOTLIB_IMG_FROM_FIG_OLD_BLOCK,
        MATPLOTLIB_IMG_FROM_FIG_NEW_BLOCK,
    )


def ensure_jax_tree_map_compat_source_patch(upstream_dir: Path) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for relative_path in JAX_TREE_MAP_COMPAT_FILES:
        path = upstream_dir / relative_path
        key = str(relative_path)
        if not path.exists():
            statuses[key] = "missing"
            continue
        source = path.read_text(encoding="utf-8")
        if "jax.tree_map(" not in source:
            statuses[key] = "already_patched"
            continue
        path.write_text(source.replace("jax.tree_map(", "jax.tree_util.tree_map("), encoding="utf-8")
        statuses[key] = "patched"
    return statuses
