#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_REMOTE = "gdrive_ro"
DEFAULT_PROJECT_PATH = "waymax_research/latentdriver_waymax_experiments"
DEFAULT_TARGET_ROOT = Path.home() / "Downloads" / "waymax_rclone_cache" / "debug_runs"


def _remote_path(remote: str, project_path: str, suffix: str) -> str:
    return f"{remote}:{project_path.strip('/')}/{suffix.lstrip('/')}"


def _run(cmd: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {"command": cmd, "dry_run": True, "returncode": 0, "stdout": "", "stderr": ""}
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _ensure_rclone() -> None:
    if shutil.which("rclone") is None:
        raise EnvironmentError("rclone is not installed or not on PATH. Install it with `brew install rclone`.")


def _copy_target(*, remote: str, project_path: str, name: str, target_root: Path, dry_run: bool) -> dict[str, Any]:
    source = _remote_path(remote, project_path, f"debug_runs/{name}")
    target = target_root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["rclone", "copy", source, str(target), "-P"]
    result = _run(command, dry_run=dry_run)
    result["source"] = source
    result["target"] = str(target)
    result["name"] = name
    return result


def _copy_pointer(*, remote: str, project_path: str, pointer_name: str, target_root: Path, dry_run: bool) -> dict[str, Any]:
    source = _remote_path(remote, project_path, f"debug_runs/{pointer_name}")
    target = target_root / pointer_name
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["rclone", "copyto", source, str(target), "-P"]
    result = _run(command, dry_run=dry_run)
    result["source"] = source
    result["target"] = str(target)
    result["name"] = pointer_name
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull stable Colab debug aliases from Google Drive via rclone.")
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="rclone remote name. Default: gdrive_ro")
    parser.add_argument("--project-path", default=DEFAULT_PROJECT_PATH, help="Project folder path inside the rclone remote.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT, help="Local debug cache root.")
    parser.add_argument(
        "--which",
        choices=["latest_failure", "latest", "all"],
        default="latest_failure",
        help="Which stable debug alias to pull. Default: latest_failure.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        _ensure_rclone()

    target_root = args.target_root.expanduser()
    aliases = ["latest", "latest_failure"] if args.which == "all" else [args.which]
    pointers = ["LATEST.json"]
    if args.which in {"latest_failure", "all"}:
        pointers.append("LATEST_FAILURE.json")

    results: list[dict[str, Any]] = []
    for pointer in pointers:
        results.append(
            _copy_pointer(
                remote=args.remote,
                project_path=args.project_path,
                pointer_name=pointer,
                target_root=target_root,
                dry_run=args.dry_run,
            )
        )
    for alias in aliases:
        results.append(
            _copy_target(
                remote=args.remote,
                project_path=args.project_path,
                name=alias,
                target_root=target_root,
                dry_run=args.dry_run,
            )
        )

    payload = {
        "remote": args.remote,
        "project_path": args.project_path,
        "target_root": str(target_root),
        "which": args.which,
        "results": results,
        "ready": all(item["returncode"] == 0 for item in results),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
