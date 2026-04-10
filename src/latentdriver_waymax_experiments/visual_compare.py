from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .artifacts import results_root
from .wayboard.data import RunRecord, discover_runs

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}


@dataclass(frozen=True)
class SelectedVisualization:
    model: str
    run_id: str
    run_dir: Path
    media_path: Path


def _video_artifacts(record: RunRecord) -> tuple[Path, ...]:
    manifest_paths = tuple(
        path
        for path in (Path(path) for path in record.manifest.get("media_files", []))
        if path.suffix.lower() in VIDEO_SUFFIXES and path.exists()
    )
    if manifest_paths:
        return manifest_paths
    return tuple(
        artifact.path
        for artifact in record.media_artifacts
        if artifact.path.suffix.lower() in VIDEO_SUFFIXES and artifact.path.exists()
    )


def find_latest_visualization(*, records: Iterable[RunRecord], model: str, tier: str, seed: int) -> SelectedVisualization:
    for record in records:
        if record.model != model or record.tier != tier or record.seed != seed:
            continue
        videos = _video_artifacts(record)
        if not videos:
            continue
        return SelectedVisualization(
            model=model,
            run_id=record.run_id,
            run_dir=record.run_dir,
            media_path=videos[0],
        )
    raise FileNotFoundError(f"No video artifact found for model={model!r}, tier={tier!r}, seed={seed!r}")


def default_comparison_path(*, root: Path, tier: str, seed: int, left_model: str, right_model: str) -> Path:
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = f"{tag}_{tier}_{left_model}_vs_{right_model}_seed{seed}.mp4".replace("/", "_")
    return root / "comparisons" / safe


def build_ffmpeg_hstack_command(*, left: Path, right: Path, output: Path, height: int = 720) -> list[str]:
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        f"[0:v]scale=-2:{height},setsar=1[left];[1:v]scale=-2:{height},setsar=1[right];[left][right]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]


def create_side_by_side_video(*, left: Path, right: Path, output: Path, height: int = 720, dry_run: bool = False) -> dict[str, object]:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    output = output.expanduser().resolve()
    if not left.exists():
        raise FileNotFoundError(f"Left video does not exist: {left}")
    if not right.exists():
        raise FileNotFoundError(f"Right video does not exist: {right}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to create side-by-side comparison videos")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_hstack_command(left=left, right=right, output=output, height=height)
    if dry_run:
        return {"command": cmd, "output": str(output), "ready": True}
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed while creating side-by-side video.\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg finished but did not create a non-empty output: {output}")
    return {"command": cmd, "output": str(output), "bytes": output.stat().st_size}


def compare_latest_visualizations(
    *,
    root: Path | None,
    left_model: str,
    right_model: str,
    tier: str,
    seed: int,
    output: Path | None = None,
    height: int = 720,
    dry_run: bool = False,
) -> dict[str, object]:
    resolved_root = (root or results_root()).expanduser().resolve()
    records = discover_runs(resolved_root)
    left = find_latest_visualization(records=records, model=left_model, tier=tier, seed=seed)
    right = find_latest_visualization(records=records, model=right_model, tier=tier, seed=seed)
    output_path = output or default_comparison_path(root=resolved_root, tier=tier, seed=seed, left_model=left_model, right_model=right_model)
    comparison = create_side_by_side_video(left=left.media_path, right=right.media_path, output=output_path, height=height, dry_run=dry_run)
    return {
        "left": {
            "model": left.model,
            "run_id": left.run_id,
            "media_path": str(left.media_path),
        },
        "right": {
            "model": right.model,
            "run_id": right.run_id,
            "media_path": str(right.media_path),
        },
        "comparison": comparison,
        "results_root": str(resolved_root),
    }
