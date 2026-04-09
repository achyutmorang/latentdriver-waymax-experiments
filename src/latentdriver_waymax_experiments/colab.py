from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict

from .config import resolve_repo_relative


def _bind_symlink(target: Path, source: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            entries = [entry for entry in target.iterdir() if entry.name != ".gitkeep"]
            if entries:
                raise RuntimeError(f"Refusing to replace non-empty directory with symlink: {target}")
            shutil.rmtree(target)
    target.symlink_to(source)


def bind_drive_layout(drive_root: str) -> Dict[str, str]:
    root = Path(drive_root).expanduser() / "latentdriver_waymax_experiments"
    checkpoints = root / "assets" / "checkpoints"
    preprocessed = root / "assets" / "preprocessed"
    smoke = root / "assets" / "smoke"
    results = root / "results" / "runs"
    for path in (checkpoints, preprocessed, smoke, results):
        path.mkdir(parents=True, exist_ok=True)

    _bind_symlink(resolve_repo_relative("artifacts/assets/checkpoints"), checkpoints)
    _bind_symlink(resolve_repo_relative("artifacts/assets/preprocessed"), preprocessed)
    _bind_symlink(resolve_repo_relative("artifacts/assets/smoke"), smoke)
    _bind_symlink(resolve_repo_relative("results/runs"), results)
    os.environ["LATENTDRIVER_RESULTS_ROOT"] = str(results)
    return {
        "drive_root": str(root),
        "checkpoints": str(checkpoints),
        "preprocessed": str(preprocessed),
        "smoke": str(smoke),
        "results": str(results),
    }
