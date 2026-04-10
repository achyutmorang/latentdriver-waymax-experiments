from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from .config import load_config, resolve_repo_relative


PYTHON312_SITE_CUSTOMIZE_BLOCK = """# latentdriver-waymax-experiments python3.12 compatibility
import pkgutil

if not hasattr(pkgutil, "ImpImporter"):
    class _CodexCompatImpImporter:
        pass

    pkgutil.ImpImporter = _CodexCompatImpImporter

if not hasattr(pkgutil, "ImpLoader"):
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


def _replace_import_block(path: Path, old: str, new: str) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return "already_patched"
    if old not in text:
        return "not_found"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return "patched"


def ensure_lightning_compat_source_patches(upstream_dir: Path) -> Dict[str, str]:
    return {
        "utils": _replace_import_block(
            upstream_dir / "src" / "utils" / "utils.py",
            UTILS_LIGHTNING_OLD_IMPORT,
            UTILS_LIGHTNING_NEW_IMPORT,
        ),
        "latentdriver_model": _replace_import_block(
            upstream_dir / "src" / "policy" / "latentdriver" / "lantentdriver_model.py",
            MODEL_LIGHTNING_OLD_IMPORT,
            MODEL_LIGHTNING_NEW_IMPORT,
        ),
        "bc_baseline": _replace_import_block(
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
