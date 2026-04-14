from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .config import load_config, resolve_repo_relative


def preprocessed_root() -> Path:
    return resolve_repo_relative(load_config()["assets"]["preprocessed_root"])


def default_archive_path(mode: str = "full") -> Path:
    return preprocessed_root() / f"{mode}_preprocess_cache.tar"


def local_preprocess_root() -> Path:
    if Path("/content").exists():
        return Path("/content/latentdriver_preprocess_cache")
    return resolve_repo_relative("artifacts/local_preprocess_cache")


def _mode_members(mode: str) -> list[str]:
    return [
        f"{mode}/val_preprocessed_path",
        f"{mode}/val_intention_label",
    ]


def _run_streamed(command: list[str]) -> float:
    started = time.monotonic()
    print(f"[preprocess-archive] $ {' '.join(command)}", flush=True)
    with subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        returncode = proc.wait()
    elapsed = time.monotonic() - started
    if returncode != 0:
        raise RuntimeError(f"Command failed with code {returncode}: {' '.join(command)}")
    return elapsed


def archive_status(*, mode: str = "full", archive_path: Path | None = None, target_root: Path | None = None) -> dict[str, Any]:
    archive = archive_path or default_archive_path(mode)
    target = target_root or local_preprocess_root()
    extracted_preprocess = target / mode / "val_preprocessed_path"
    extracted_intention = target / mode / "val_intention_label"
    return {
        "mode": mode,
        "archive_path": str(archive),
        "archive_exists": archive.is_file(),
        "archive_size_bytes": archive.stat().st_size if archive.is_file() else None,
        "source_root": str(preprocessed_root()),
        "target_root": str(target),
        "target_preprocess_path": str(extracted_preprocess),
        "target_intention_path": str(extracted_intention),
        "target_preprocess_exists": extracted_preprocess.is_dir(),
        "target_intention_exists": extracted_intention.is_dir(),
    }


def create_archive(*, mode: str = "full", archive_path: Path | None = None, force: bool = False) -> dict[str, Any]:
    archive = archive_path or default_archive_path(mode)
    source_root = preprocessed_root()
    for member in _mode_members(mode):
        source = source_root / member
        if not source.exists():
            raise FileNotFoundError(f"Cannot archive missing preprocess member: {source}")
    if archive.exists() and not force:
        raise FileExistsError(f"Archive already exists: {archive}. Pass --force to rebuild it.")
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_name(f".{archive.name}.tmp")
    tmp.unlink(missing_ok=True)
    elapsed = _run_streamed(["tar", "-C", str(source_root), "-cf", str(tmp), *_mode_members(mode)])
    tmp.replace(archive)
    payload = {
        "action": "create",
        "mode": mode,
        "archive_path": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "source_root": str(source_root),
        "members": _mode_members(mode),
        "elapsed_seconds": round(elapsed, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(archive.with_suffix(archive.suffix + ".manifest.json"), payload)
    return payload


def extract_archive(
    *,
    mode: str = "full",
    archive_path: Path | None = None,
    target_root: Path | None = None,
) -> dict[str, Any]:
    archive = archive_path or default_archive_path(mode)
    target = target_root or local_preprocess_root()
    if not archive.is_file():
        raise FileNotFoundError(f"Preprocess archive not found: {archive}")
    target.mkdir(parents=True, exist_ok=True)
    elapsed = _run_streamed(["tar", "-C", str(target), "-xf", str(archive)])
    payload = {
        "action": "extract",
        "mode": mode,
        "archive_path": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "target_root": str(target),
        "target_preprocess_path": str(target / mode / "val_preprocessed_path"),
        "target_intention_path": str(target / mode / "val_intention_label"),
        "elapsed_seconds": round(elapsed, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(target / mode / "archive_restore_manifest.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Create, inspect, or extract preprocessed cache archives.")
    parser.add_argument("action", choices=["create", "extract", "status"])
    parser.add_argument("--mode", default="full", choices=["full", "smoke"])
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.action == "create":
        payload = create_archive(mode=args.mode, archive_path=args.archive_path, force=args.force)
    elif args.action == "extract":
        payload = extract_archive(mode=args.mode, archive_path=args.archive_path, target_root=args.target_root)
    else:
        payload = archive_status(mode=args.mode, archive_path=args.archive_path, target_root=args.target_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
