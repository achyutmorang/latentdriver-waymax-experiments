#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REPO_URL = "https://github.com/achyutmorang/latentdriver-waymax-experiments.git"
DEFAULT_REPO_BRANCH = "main"
DEFAULT_REPO_DIR = Path("/content/latentdriver-waymax-experiments")
DEFAULT_DRIVE_BASE_ROOT = Path("/content/drive/MyDrive/waymax_research")
DEFAULT_WAYMO_DATASET_ROOT = "gs://waymo_open_dataset_motion_v_1_1_0"
DRIVE_MOUNT_HELP = (
    "Google Drive is not mounted at {my_drive}. The shell-only Colab runner cannot call "
    "google.colab.drive.mount() because that OAuth flow requires the live notebook IPython "
    "kernel, not a python3 subprocess from %%bash. In Colab, click the Files sidebar folder "
    "icon, choose 'Mount Drive', finish the prompt, then rerun this bootstrap cell."
)


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    print("[colab-bootstrap] $", " ".join(command), flush=True)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    result = {
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")
    return result


def sync_repo(*, repo_url: str, branch: str, repo_dir: Path) -> dict[str, Any]:
    if not repo_dir.exists():
        clone = _run(["git", "clone", "--branch", branch, repo_url, str(repo_dir)])
        action = "cloned"
        commands = [clone]
    elif not (repo_dir / ".git").exists():
        raise RuntimeError(f"repo_dir exists but is not a git checkout: {repo_dir}")
    else:
        commands = [
            _run(["git", "fetch", "origin", branch], cwd=repo_dir),
            _run(["git", "checkout", branch], cwd=repo_dir),
            _run(["git", "pull", "--ff-only", "origin", branch], cwd=repo_dir),
        ]
        action = "fast_forwarded"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    short_head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, text=True).strip()
    return {
        "action": action,
        "repo_url": repo_url,
        "branch": branch,
        "repo_dir": str(repo_dir),
        "head": head,
        "short_head": short_head,
        "commands": commands,
    }


def require_drive_mounted(mountpoint: Path) -> dict[str, Any]:
    my_drive = mountpoint / "MyDrive"
    if not my_drive.is_dir():
        raise RuntimeError(DRIVE_MOUNT_HELP.format(my_drive=my_drive))
    return {"mountpoint": str(mountpoint), "my_drive": str(my_drive), "mounted": True, "mode": "already_mounted"}


def bind_drive(*, repo_dir: Path, drive_base_root: Path) -> dict[str, str]:
    sys.path.insert(0, str(repo_dir / "src"))
    from latentdriver_waymax_experiments.colab import bind_drive_layout

    return bind_drive_layout(str(drive_base_root))


def write_bootstrap_manifest(*, repo_dir: Path, payload: dict[str, Any]) -> Path:
    manifest_path = repo_dir / "results" / "debug_runs" / "BOOTSTRAP.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def bootstrap(
    *,
    repo_url: str,
    branch: str,
    repo_dir: Path,
    drive_base_root: Path,
    drive_mountpoint: Path,
    waymo_dataset_root: str,
    skip_drive_mount: bool = False,
    skip_bind: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "waymo_dataset_root": waymo_dataset_root,
    }
    if skip_drive_mount:
        payload["drive_mount"] = {"skipped": True, "mountpoint": str(drive_mountpoint)}
    else:
        payload["drive_mount"] = require_drive_mounted(drive_mountpoint)
    payload["repo"] = sync_repo(repo_url=repo_url, branch=branch, repo_dir=repo_dir)
    if skip_bind:
        payload["drive_binding"] = {"skipped": True, "drive_base_root": str(drive_base_root)}
    else:
        payload["drive_binding"] = bind_drive(repo_dir=repo_dir, drive_base_root=drive_base_root)
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    payload["manifest_path"] = str(write_bootstrap_manifest(repo_dir=repo_dir, payload=payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the LatentDriver Waymax Colab runtime without putting Python logic in notebooks.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_REPO_BRANCH)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--drive-base-root", type=Path, default=DEFAULT_DRIVE_BASE_ROOT)
    parser.add_argument("--drive-mountpoint", type=Path, default=Path("/content/drive"))
    parser.add_argument("--waymo-dataset-root", default=DEFAULT_WAYMO_DATASET_ROOT)
    parser.add_argument("--skip-drive-mount", action="store_true")
    parser.add_argument("--skip-bind", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Re-raise bootstrap exceptions with full tracebacks.")
    args = parser.parse_args()

    try:
        payload = bootstrap(
            repo_url=args.repo_url,
            branch=args.branch,
            repo_dir=args.repo_dir,
            drive_base_root=args.drive_base_root,
            drive_mountpoint=args.drive_mountpoint,
            waymo_dataset_root=args.waymo_dataset_root,
            skip_drive_mount=args.skip_drive_mount,
            skip_bind=args.skip_bind,
        )
    except Exception as exc:
        if args.debug:
            raise
        print(f"[colab-bootstrap] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
