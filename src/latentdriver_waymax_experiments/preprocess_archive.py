from __future__ import annotations

import json
import math
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


def default_shard_archive_dir(mode: str = "full") -> Path:
    return preprocessed_root() / f"{mode}_shard_archives"


def local_preprocess_root() -> Path:
    if Path("/content").exists():
        return Path("/content/latentdriver_preprocess_cache")
    return resolve_repo_relative("artifacts/local_preprocess_cache")


def _mode_members(mode: str) -> list[str]:
    return [
        f"{mode}/val_preprocessed_path",
        f"{mode}/val_intention_label",
    ]


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_payload(*, completed: int, total: int, started_at: float) -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - started_at)
    rate = completed / elapsed if elapsed > 0 and completed > 0 else None
    remaining = max(total - completed, 0)
    eta_seconds = remaining / rate if rate else None
    return {
        "completed": completed,
        "total": total,
        "remaining": remaining,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed": _format_duration(elapsed),
        "rate_per_second": round(rate, 6) if rate else None,
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "eta": _format_duration(eta_seconds),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _progress_line(*, label: str, completed: int, total: int, started_at: float, extra: str = "") -> str:
    payload = _progress_payload(completed=completed, total=total, started_at=started_at)
    percent = (completed / total * 100.0) if total else 100.0
    suffix = f" {extra}" if extra else ""
    return (
        f"[{label}] {completed}/{total} ({percent:.1f}%) "
        f"elapsed={payload['elapsed']} eta={payload['eta']}{suffix}"
    )


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


def _scenario_stems(mode: str) -> list[str]:
    root = preprocessed_root() / mode
    map_dir = root / "val_preprocessed_path" / "map"
    route_dir = root / "val_preprocessed_path" / "route"
    intention_dir = root / "val_intention_label"
    for label, path in {"map": map_dir, "route": route_dir, "intention": intention_dir}.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Cannot build shard archives because {label} directory is missing: {path}")
    map_stems = {item.stem for item in map_dir.glob("*.npy") if item.is_file()}
    route_stems = {item.stem for item in route_dir.glob("*.npy") if item.is_file()}
    intention_stems = {item.stem for item in intention_dir.glob("*.txt") if item.is_file()}
    common = map_stems & route_stems & intention_stems
    counts = {"map": len(map_stems), "route": len(route_stems), "intention": len(intention_stems), "common": len(common)}
    if not common:
        raise RuntimeError(f"No complete preprocess triples found for mode={mode}: {counts}")
    if len(common) != len(map_stems) or len(common) != len(route_stems) or len(common) != len(intention_stems):
        raise RuntimeError(f"Cannot shard incomplete preprocess cache for mode={mode}: {counts}")
    return sorted(common)


def _split_evenly(items: list[str], parts: int) -> list[list[str]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if not items:
        return []
    chunk_count = min(parts, len(items))
    chunk_size = math.ceil(len(items) / chunk_count)
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _shard_archive_path(archive_dir: Path, index: int) -> Path:
    return archive_dir / f"shard-{index:05d}.tar"


def _shard_members(mode: str, stems: list[str]) -> list[str]:
    members: list[str] = []
    for stem in stems:
        members.extend(
            [
                f"{mode}/val_preprocessed_path/map/{stem}.npy",
                f"{mode}/val_preprocessed_path/route/{stem}.npy",
                f"{mode}/val_intention_label/{stem}.txt",
            ]
        )
    return members


def _manifest_path_for_archive(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _completed_shard_manifest(path: Path, *, mode: str, stems: list[str]) -> dict[str, Any] | None:
    manifest = _load_json(_manifest_path_for_archive(path))
    if not path.is_file() or not manifest:
        return None
    if manifest.get("mode") != mode or manifest.get("scenario_count") != len(stems):
        return None
    if manifest.get("first_scenario_id") != stems[0] or manifest.get("last_scenario_id") != stems[-1]:
        return None
    if path.stat().st_size <= 0:
        return None
    return manifest


def archive_status(
    *,
    mode: str = "full",
    archive_path: Path | None = None,
    archive_dir: Path | None = None,
    target_root: Path | None = None,
) -> dict[str, Any]:
    archive = archive_path or default_archive_path(mode)
    shard_dir = archive_dir or default_shard_archive_dir(mode)
    shard_archives = sorted(shard_dir.glob("shard-*.tar")) if shard_dir.is_dir() else []
    shard_manifests = sorted(shard_dir.glob("shard-*.tar.manifest.json")) if shard_dir.is_dir() else []
    target = target_root or local_preprocess_root()
    extracted_preprocess = target / mode / "val_preprocessed_path"
    extracted_intention = target / mode / "val_intention_label"
    return {
        "mode": mode,
        "archive_path": str(archive),
        "archive_exists": archive.is_file(),
        "archive_size_bytes": archive.stat().st_size if archive.is_file() else None,
        "shard_archive_dir": str(shard_dir),
        "shard_archive_dir_exists": shard_dir.is_dir(),
        "shard_archive_count": len(shard_archives),
        "shard_manifest_count": len(shard_manifests),
        "shard_archive_size_bytes": sum(path.stat().st_size for path in shard_archives),
        "shard_manifest_path": str(shard_dir / "manifest.json"),
        "shard_manifest_exists": (shard_dir / "manifest.json").is_file(),
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


def create_shard_archives(
    *,
    mode: str = "full",
    archive_dir: Path | None = None,
    shards: int = 150,
    force: bool = False,
) -> dict[str, Any]:
    if shards <= 0:
        raise ValueError("shards must be positive")
    source_root = preprocessed_root()
    target_dir = archive_dir or default_shard_archive_dir(mode)
    target_dir.mkdir(parents=True, exist_ok=True)
    stems = _scenario_stems(mode)
    chunks = _split_evenly(stems, shards)
    started_at = time.monotonic()
    created = 0
    skipped = 0
    shard_payloads: list[dict[str, Any]] = []
    print(
        _progress_line(
            label="archive-shards",
            completed=0,
            total=len(chunks),
            started_at=started_at,
            extra=f"scenarios={len(stems)} target_dir={target_dir}",
        ),
        flush=True,
    )
    for index, chunk in enumerate(chunks):
        archive = _shard_archive_path(target_dir, index)
        existing = None if force else _completed_shard_manifest(archive, mode=mode, stems=chunk)
        if existing is not None:
            skipped += 1
            payload = {
                "index": index,
                "archive_path": str(archive),
                "status": "skipped",
                "scenario_count": len(chunk),
                "file_count": len(chunk) * 3,
                "archive_size_bytes": archive.stat().st_size,
                "first_scenario_id": chunk[0],
                "last_scenario_id": chunk[-1],
            }
        else:
            tmp = archive.with_name(f".{archive.name}.tmp")
            list_path = archive.with_name(f".{archive.name}.members.txt")
            tmp.unlink(missing_ok=True)
            list_path.write_text("\n".join(_shard_members(mode, chunk)) + "\n", encoding="utf-8")
            shard_started = time.monotonic()
            print(
                _progress_line(
                    label="archive-shards",
                    completed=index,
                    total=len(chunks),
                    started_at=started_at,
                    extra=f"running=shard-{index:05d} scenarios={len(chunk)}",
                ),
                flush=True,
            )
            try:
                elapsed = _run_streamed(["tar", "-C", str(source_root), "-cf", str(tmp), "-T", str(list_path)])
                tmp.replace(archive)
            finally:
                list_path.unlink(missing_ok=True)
            created += 1
            payload = {
                "index": index,
                "archive_path": str(archive),
                "status": "created",
                "scenario_count": len(chunk),
                "file_count": len(chunk) * 3,
                "archive_size_bytes": archive.stat().st_size,
                "first_scenario_id": chunk[0],
                "last_scenario_id": chunk[-1],
                "elapsed_seconds": round(elapsed, 3),
                "wall_seconds": round(time.monotonic() - shard_started, 3),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(_manifest_path_for_archive(archive), {"mode": mode, **payload})
        shard_payloads.append(payload)
        print(
            _progress_line(
                label="archive-shards",
                completed=index + 1,
                total=len(chunks),
                started_at=started_at,
                extra=f"created={created} skipped={skipped} latest={payload['status']}:shard-{index:05d}",
            ),
            flush=True,
        )
    manifest = {
        "action": "create-shards",
        "mode": mode,
        "archive_dir": str(target_dir),
        "source_root": str(source_root),
        "requested_shards": shards,
        "shards_total": len(chunks),
        "shards_created": created,
        "shards_skipped": skipped,
        "scenario_count": len(stems),
        "file_count": len(stems) * 3,
        "archive_size_bytes": sum(_shard_archive_path(target_dir, item["index"]).stat().st_size for item in shard_payloads),
        "shards": shard_payloads,
        **_progress_payload(completed=len(chunks), total=len(chunks), started_at=started_at),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(target_dir / "manifest.json", manifest)
    return manifest


def extract_shard_archives(
    *,
    mode: str = "full",
    archive_dir: Path | None = None,
    target_root: Path | None = None,
) -> dict[str, Any]:
    source_dir = archive_dir or default_shard_archive_dir(mode)
    target = target_root or local_preprocess_root()
    manifest_path = source_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Shard archive directory not found: {source_dir}")
    archives = sorted(source_dir.glob("shard-*.tar"))
    if not archives:
        raise FileNotFoundError(f"No shard archives found in: {source_dir}")
    expected = int(manifest.get("shards_total", len(archives))) if manifest else len(archives)
    if len(archives) != expected:
        raise RuntimeError(f"Shard archive count mismatch: found={len(archives)} expected={expected} dir={source_dir}")
    target.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    extracted: list[dict[str, Any]] = []
    print(
        _progress_line(label="extract-shards", completed=0, total=len(archives), started_at=started_at, extra=f"target={target}"),
        flush=True,
    )
    for index, archive in enumerate(archives, start=1):
        elapsed = _run_streamed(["tar", "-C", str(target), "-xf", str(archive)])
        extracted.append(
            {
                "archive_path": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        print(
            _progress_line(
                label="extract-shards",
                completed=index,
                total=len(archives),
                started_at=started_at,
                extra=f"latest={archive.name}",
            ),
            flush=True,
        )
    payload = {
        "action": "extract-shards",
        "mode": mode,
        "archive_dir": str(source_dir),
        "target_root": str(target),
        "target_preprocess_path": str(target / mode / "val_preprocessed_path"),
        "target_intention_path": str(target / mode / "val_intention_label"),
        "shards_total": len(archives),
        "archive_size_bytes": sum(path.stat().st_size for path in archives),
        "extracted": extracted,
        **_progress_payload(completed=len(archives), total=len(archives), started_at=started_at),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(target / mode / "shard_archive_restore_manifest.json", payload)
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
    parser.add_argument("action", choices=["create", "extract", "status", "create-shards", "extract-shards"])
    parser.add_argument("--mode", default="full", choices=["full", "smoke"])
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--shards", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.action == "create":
        payload = create_archive(mode=args.mode, archive_path=args.archive_path, force=args.force)
    elif args.action == "extract":
        payload = extract_archive(mode=args.mode, archive_path=args.archive_path, target_root=args.target_root)
    elif args.action == "create-shards":
        payload = create_shard_archives(mode=args.mode, archive_dir=args.archive_dir, shards=args.shards, force=args.force)
    elif args.action == "extract-shards":
        payload = extract_shard_archives(mode=args.mode, archive_dir=args.archive_dir, target_root=args.target_root)
    else:
        payload = archive_status(
            mode=args.mode,
            archive_path=args.archive_path,
            archive_dir=args.archive_dir,
            target_root=args.target_root,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
