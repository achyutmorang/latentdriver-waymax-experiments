#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config
from latentdriver_waymax_experiments.evaluation import _validation_inputs
from latentdriver_waymax_experiments.upstream import (
    ensure_crdp_compat_source_patch,
    ensure_lightning_compat_source_patches,
    ensure_preprocess_multiprocessing_compat_source_patch,
    ensure_python312_compat_sitecustomize,
    ensure_upstream_exists,
)


def build_preprocess_command(*, mode: str, batch_size: int | None = None) -> list[str]:
    cfg = load_config()
    inputs = _validation_inputs(mode)
    batch_dims = list(cfg["validation"][mode]["preprocess_batch_dims"])
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch_dims[-1] = batch_size
    return [
        sys.executable,
        "-m",
        "src.preprocess.preprocess_data",
        f"++batch_dims=[{batch_dims[0]},{batch_dims[1]}]",
        f"++waymax_conf.path={inputs['waymo_path']}",
        "++waymax_conf.drop_remainder=False",
        f"++data_conf.path_to_processed_map_route={inputs['preprocess_path']}",
        f"++metric_conf.intention_label_path={inputs['intention_path']}",
    ]


def _preprocess_output_paths(mode: str) -> dict[str, Path]:
    inputs = _validation_inputs(mode)
    preprocess_root = Path(inputs["preprocess_path"])
    return {
        "preprocess_root": preprocess_root,
        "map_dir": preprocess_root / "map",
        "route_dir": preprocess_root / "route",
        "intention_dir": Path(inputs["intention_path"]),
        "success_marker": preprocess_root / "_SUCCESS",
        "manifest": preprocess_root / "preprocess_manifest.json",
    }


def _dir_has_entries(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _count_files(path: Path, suffix: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix == suffix and item.stat().st_size > 0)


def preprocess_cache_status(mode: str) -> dict[str, object]:
    paths = _preprocess_output_paths(mode)
    map_ready = _dir_has_entries(paths["map_dir"])
    route_ready = _dir_has_entries(paths["route_dir"])
    intention_ready = _dir_has_entries(paths["intention_dir"])
    success_ready = paths["success_marker"].is_file()
    any_present = any(
        path.exists()
        for path in [paths["map_dir"], paths["route_dir"], paths["intention_dir"], paths["success_marker"], paths["manifest"]]
    )
    complete = map_ready and route_ready and intention_ready and success_ready
    partial = any_present and not complete
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "map_ready": map_ready,
        "route_ready": route_ready,
        "intention_ready": intention_ready,
        "success_ready": success_ready,
        "any_present": any_present,
        "complete": complete,
        "partial": partial,
        "counts": {
            "map_npy": _count_files(paths["map_dir"], ".npy"),
            "route_npy": _count_files(paths["route_dir"], ".npy"),
            "intention_txt": _count_files(paths["intention_dir"], ".txt"),
        },
    }


def clear_preprocess_outputs(mode: str) -> dict[str, object]:
    paths = _preprocess_output_paths(mode)
    removed: list[str] = []
    for key in ("map_dir", "route_dir", "intention_dir"):
        path = paths[key]
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    for key in ("success_marker", "manifest"):
        path = paths[key]
        if path.exists():
            path.unlink()
            removed.append(str(path))
    preprocess_root = paths["preprocess_root"]
    if preprocess_root.exists() and not any(preprocess_root.iterdir()):
        preprocess_root.rmdir()
    return {"removed": removed}


def mark_preprocess_complete(mode: str, payload: dict[str, Any]) -> dict[str, object]:
    paths = _preprocess_output_paths(mode)
    paths["preprocess_root"].mkdir(parents=True, exist_ok=True)
    status = preprocess_cache_status(mode)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": "complete",
        "command": payload["command"],
        "waymo_path": payload["waymo_path"],
        "preprocess_path": payload["preprocess_path"],
        "intention_path": payload["intention_path"],
        "counts": status["counts"],
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["success_marker"].write_text("complete\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only preprocessing for smoke or full validation.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Delete existing generated preprocess outputs and rebuild them.")
    parser.add_argument(
        "--auto-force-partial",
        action="store_true",
        help="Deprecated compatibility flag. Partial preprocess outputs are resumed by default; use --force to rebuild.",
    )
    parser.add_argument("--batch-size", type=int, help="Override the inner preprocessing batch dimension.")
    parser.add_argument("--workers", type=int, default=1, help="Worker count for CPU post-processing. Default 1 avoids Colab multiprocessing OOMs.")
    parser.add_argument(
        "--jax-platform",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="JAX platform for preprocessing. Default cpu avoids GPU memory contention during full preprocessing.",
    )
    args = parser.parse_args()

    upstream_dir = ensure_upstream_exists()
    compat_sitecustomize = ensure_python312_compat_sitecustomize(upstream_dir)
    lightning_compat = ensure_lightning_compat_source_patches(upstream_dir)
    crdp_compat = ensure_crdp_compat_source_patch(upstream_dir)
    preprocess_multiprocessing_compat = ensure_preprocess_multiprocessing_compat_source_patch(upstream_dir)
    missing_preprocess_patches = {
        name: status for name, status in preprocess_multiprocessing_compat.items() if status == "not_found"
    }
    if missing_preprocess_patches:
        raise RuntimeError(f"Unable to patch upstream preprocessing safely: {missing_preprocess_patches}")
    inputs = _validation_inputs(args.mode)
    cmd = build_preprocess_command(mode=args.mode, batch_size=args.batch_size)
    cache_status = preprocess_cache_status(args.mode)
    payload: dict[str, Any] = {
        "mode": args.mode,
        "command": cmd,
        "waymo_path": str(inputs["waymo_path"]),
        "preprocess_path": str(inputs["preprocess_path"]),
        "intention_path": str(inputs["intention_path"]),
        "compat_sitecustomize": str(compat_sitecustomize),
        "lightning_compat": lightning_compat,
        "crdp_compat": crdp_compat,
        "preprocess_multiprocessing_compat": preprocess_multiprocessing_compat,
        "cache_status": cache_status,
        "force": args.force,
        "auto_force_partial": args.auto_force_partial,
        "batch_size_override": args.batch_size,
        "workers": args.workers,
        "jax_platform": args.jax_platform,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if cache_status["complete"] and not args.force:
        payload["cache_action"] = "reused_existing_outputs"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.force and cache_status["any_present"]:
        payload["cache_action"] = "cleared_existing_outputs"
        payload["cache_clear"] = clear_preprocess_outputs(args.mode)
    elif cache_status["partial"]:
        payload["cache_action"] = "resumed_partial_outputs"

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(upstream_dir) if not existing_pythonpath else f"{upstream_dir}{os.pathsep}{existing_pythonpath}"
    env.setdefault("LATENTDRIVER_PREPROCESS_START_METHOD", "spawn")
    env["LATENTDRIVER_PREPROCESS_WORKERS"] = str(args.workers)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.jax_platform != "auto":
        env["JAX_PLATFORMS"] = args.jax_platform

    proc = subprocess.run(cmd, cwd=upstream_dir, check=False, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    payload["completion"] = mark_preprocess_complete(args.mode, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
