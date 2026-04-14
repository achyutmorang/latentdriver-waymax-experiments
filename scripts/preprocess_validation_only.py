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
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _load_manifest_counts(path: Path) -> dict[str, int]:
    try:
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        return {}
    return {key: value for key, value in counts.items() if isinstance(value, int)}


def _count_files(path: Path, suffix: str) -> int | None:
    try:
        if not path.is_dir():
            return 0
        return sum(1 for item in path.iterdir() if item.is_file() and item.suffix == suffix and item.stat().st_size > 0)
    except OSError:
        return None


def _positive_count(counts: dict[str, int], key: str) -> bool:
    return int(counts.get(key, 0)) > 0


def _has_positive_manifest_counts(counts: dict[str, int]) -> bool:
    return all(_positive_count(counts, key) for key in ("map_npy", "route_npy", "intention_txt"))


def filesystem_preprocess_counts(mode: str) -> dict[str, int | None]:
    paths = _preprocess_output_paths(mode)
    return {
        "map_npy": _count_files(paths["map_dir"], ".npy"),
        "route_npy": _count_files(paths["route_dir"], ".npy"),
        "intention_txt": _count_files(paths["intention_dir"], ".txt"),
    }


def preprocess_cache_status(mode: str) -> dict[str, object]:
    paths = _preprocess_output_paths(mode)
    success_ready = paths["success_marker"].is_file()
    manifest_counts = _load_manifest_counts(paths["manifest"]) if success_ready else {}
    use_manifest_counts = _has_positive_manifest_counts(manifest_counts)
    if use_manifest_counts:
        counts = {
            "map_npy": manifest_counts.get("map_npy", 0),
            "route_npy": manifest_counts.get("route_npy", 0),
            "intention_txt": manifest_counts.get("intention_txt", 0),
        }
        map_ready = _path_is_dir(paths["map_dir"]) and _positive_count(counts, "map_npy")
        route_ready = _path_is_dir(paths["route_dir"]) and _positive_count(counts, "route_npy")
        intention_ready = _path_is_dir(paths["intention_dir"]) and _positive_count(counts, "intention_txt")
    else:
        map_ready = _dir_has_entries(paths["map_dir"])
        route_ready = _dir_has_entries(paths["route_dir"])
        intention_ready = _dir_has_entries(paths["intention_dir"])
        counts = filesystem_preprocess_counts(mode)
    any_present = any(
        _path_exists(path)
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
        "counts": counts,
        "counts_source": "manifest" if use_manifest_counts else "filesystem",
    }


def can_repair_preprocess_markers(mode: str) -> tuple[bool, dict[str, object]]:
    counts = filesystem_preprocess_counts(mode)
    if any(value is None for value in counts.values()):
        return False, {"reason": "filesystem_scan_error", "counts": counts}
    normalized = {key: int(value or 0) for key, value in counts.items()}
    if not all(value > 0 for value in normalized.values()):
        return False, {"reason": "non_positive_counts", "counts": normalized}
    distinct = set(normalized.values())
    if len(distinct) != 1:
        return False, {"reason": "count_mismatch", "counts": normalized}
    return True, {"reason": "repairable", "counts": normalized}


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


def repair_preprocess_complete_markers(mode: str, payload: dict[str, Any]) -> dict[str, object]:
    paths = _preprocess_output_paths(mode)
    ok, detail = can_repair_preprocess_markers(mode)
    if not ok:
        raise RuntimeError(
            f"Cannot repair preprocess markers for mode={mode}: {detail['reason']}. counts={detail['counts']}"
        )
    counts = detail["counts"]
    assert isinstance(counts, dict)
    paths["preprocess_root"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": "complete",
        "repair": True,
        "command": payload["command"],
        "waymo_path": payload["waymo_path"],
        "preprocess_path": payload["preprocess_path"],
        "intention_path": payload["intention_path"],
        "counts": counts,
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
    parser.add_argument(
        "--repair-markers",
        action="store_true",
        help="Recreate `_SUCCESS` and `preprocess_manifest.json` from existing filesystem counts when outputs already exist.",
    )
    args = parser.parse_args()

    inputs = _validation_inputs(args.mode)
    cmd = build_preprocess_command(mode=args.mode, batch_size=args.batch_size)
    cache_status = preprocess_cache_status(args.mode)
    payload: dict[str, Any] = {
        "mode": args.mode,
        "command": cmd,
        "waymo_path": str(inputs["waymo_path"]),
        "preprocess_path": str(inputs["preprocess_path"]),
        "intention_path": str(inputs["intention_path"]),
        "cache_status": cache_status,
        "force": args.force,
        "auto_force_partial": args.auto_force_partial,
        "batch_size_override": args.batch_size,
        "workers": args.workers,
        "jax_platform": args.jax_platform,
        "repair_markers": args.repair_markers,
    }
    if args.repair_markers:
        payload["repair_probe"] = can_repair_preprocess_markers(args.mode)[1]
        if args.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        payload["repair"] = repair_preprocess_complete_markers(args.mode, payload)
        payload["cache_action"] = "repaired_markers"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

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
    payload.update(
        {
            "compat_sitecustomize": str(compat_sitecustomize),
            "lightning_compat": lightning_compat,
            "crdp_compat": crdp_compat,
            "preprocess_multiprocessing_compat": preprocess_multiprocessing_compat,
        }
    )
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
