#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


def build_preprocess_command(*, mode: str) -> list[str]:
    cfg = load_config()
    inputs = _validation_inputs(mode)
    batch_dims = cfg["validation"][mode]["preprocess_batch_dims"]
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
    }


def _dir_has_entries(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def preprocess_cache_status(mode: str) -> dict[str, object]:
    paths = _preprocess_output_paths(mode)
    map_ready = _dir_has_entries(paths["map_dir"])
    route_ready = _dir_has_entries(paths["route_dir"])
    intention_ready = _dir_has_entries(paths["intention_dir"])
    any_present = any(path.exists() for path in [paths["map_dir"], paths["route_dir"], paths["intention_dir"]])
    complete = map_ready and route_ready and intention_ready
    partial = any_present and not complete
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "map_ready": map_ready,
        "route_ready": route_ready,
        "intention_ready": intention_ready,
        "any_present": any_present,
        "complete": complete,
        "partial": partial,
    }


def clear_preprocess_outputs(mode: str) -> dict[str, object]:
    paths = _preprocess_output_paths(mode)
    removed: list[str] = []
    for key in ("map_dir", "route_dir", "intention_dir"):
        path = paths[key]
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    preprocess_root = paths["preprocess_root"]
    if preprocess_root.exists() and not any(preprocess_root.iterdir()):
        preprocess_root.rmdir()
    return {"removed": removed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only preprocessing for smoke or full validation.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Delete existing generated preprocess outputs and rebuild them.")
    args = parser.parse_args()

    upstream_dir = ensure_upstream_exists()
    compat_sitecustomize = ensure_python312_compat_sitecustomize(upstream_dir)
    lightning_compat = ensure_lightning_compat_source_patches(upstream_dir)
    crdp_compat = ensure_crdp_compat_source_patch(upstream_dir)
    preprocess_multiprocessing_compat = ensure_preprocess_multiprocessing_compat_source_patch(upstream_dir)
    inputs = _validation_inputs(args.mode)
    cmd = build_preprocess_command(mode=args.mode)
    cache_status = preprocess_cache_status(args.mode)
    payload = {
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
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if cache_status["complete"] and not args.force:
        payload["cache_action"] = "reused_existing_outputs"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if cache_status["partial"] and not args.force:
        raise SystemExit(
            "Existing preprocess outputs are partial. "
            "Delete them or rerun with --force to rebuild the cache."
        )
    if args.force and cache_status["any_present"]:
        payload["cache_clear"] = clear_preprocess_outputs(args.mode)
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(upstream_dir) if not existing_pythonpath else f"{upstream_dir}{os.pathsep}{existing_pythonpath}"
    env.setdefault("LATENTDRIVER_PREPROCESS_START_METHOD", "spawn")
    proc = subprocess.run(cmd, cwd=upstream_dir, check=False, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
