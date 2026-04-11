from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict

from .config import resolve_repo_relative


def _migrate_directory_contents(target: Path, source: Path) -> None:
    entries = [entry for entry in target.iterdir() if entry.name != ".gitkeep"]
    if not entries:
        shutil.rmtree(target)
        return
    source.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        destination = source / entry.name
        if destination.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing Drive-bound content while migrating {entry} -> {destination}"
            )
        shutil.move(str(entry), str(destination))
    gitkeep = target / ".gitkeep"
    if gitkeep.exists():
        gitkeep.unlink()
    shutil.rmtree(target)


def _bind_symlink(target: Path, source: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            _migrate_directory_contents(target, source)
    target.symlink_to(source)


def bind_drive_layout(drive_root: str) -> Dict[str, str]:
    root = Path(drive_root).expanduser() / "latentdriver_waymax_experiments"
    checkpoints = root / "assets" / "checkpoints"
    preprocessed = root / "assets" / "preprocessed"
    smoke = root / "assets" / "smoke"
    results = root / "results" / "runs"
    debug_runs = root / "debug_runs"
    for path in (checkpoints, preprocessed, smoke, results, debug_runs):
        path.mkdir(parents=True, exist_ok=True)

    _bind_symlink(resolve_repo_relative("artifacts/assets/checkpoints"), checkpoints)
    _bind_symlink(resolve_repo_relative("artifacts/assets/preprocessed"), preprocessed)
    _bind_symlink(resolve_repo_relative("artifacts/assets/smoke"), smoke)
    _bind_symlink(resolve_repo_relative("results/runs"), results)
    _bind_symlink(resolve_repo_relative("results/debug_runs"), debug_runs)
    os.environ["LATENTDRIVER_RESULTS_ROOT"] = str(results)
    os.environ["LATENTDRIVER_DEBUG_ROOT"] = str(debug_runs)
    return {
        "drive_root": str(root),
        "checkpoints": str(checkpoints),
        "preprocessed": str(preprocessed),
        "smoke": str(smoke),
        "results": str(results),
        "debug_runs": str(debug_runs),
    }
