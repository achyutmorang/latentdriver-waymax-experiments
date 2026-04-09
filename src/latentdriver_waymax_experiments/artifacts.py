from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import load_config, resolve_repo_relative


def _ensure_directory(path: Path) -> Path:
    if path.is_symlink():
        path.resolve(strict=False).mkdir(parents=True, exist_ok=True)
        return path
    if path.exists():
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_root() -> Path:
    override = Path(__import__("os").environ.get("LATENTDRIVER_RESULTS_ROOT", "")).expanduser()
    if str(override) and str(override) != ".":
        return _ensure_directory(override)
    cfg = load_config()
    root = resolve_repo_relative(cfg["assets"]["results_root"])
    return _ensure_directory(root)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_run_bundle(*, tag: str | None = None, tier: str) -> Dict[str, Path | str]:
    if tag is None:
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{tag}_{tier}"
    run_dir = results_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = run_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "stdout_path": run_dir / "stdout.log",
        "stderr_path": run_dir / "stderr.log",
        "config_snapshot": run_dir / "config_snapshot.json",
        "metrics_path": run_dir / "metrics.json",
        "run_manifest": run_dir / "run_manifest.json",
        "vis_dir": vis_dir,
    }
