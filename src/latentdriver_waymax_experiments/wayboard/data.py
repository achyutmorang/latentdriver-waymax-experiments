from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import quote

from ..evaluation import flatten_metrics_payload

MEDIA_EXTENSIONS = {".mp4", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webm"}


@dataclass(frozen=True)
class MediaArtifact:
    path: Path
    relative_path: str
    media_type: str

    def artifact_url(self, route_prefix: str) -> str:
        encoded = quote(self.relative_path.replace("\\", "/"))
        return f"{route_prefix.rstrip('/')}/{encoded}"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    model: str | None
    tier: str | None
    seed: int | None
    vis: str | bool | None
    summary: Dict[str, Any]
    manifest: Dict[str, Any]
    metrics: Dict[str, Any]
    manifest_path: Path | None
    metrics_path: Path | None
    stdout_path: Path | None
    stderr_path: Path | None
    config_snapshot_path: Path | None
    media_artifacts: tuple[MediaArtifact, ...]


@dataclass(frozen=True)
class SuiteRecord:
    run_id: str
    run_dir: Path
    tier: str | None
    seed: int | None
    models: tuple[str, ...]
    runs: tuple[Dict[str, Any], ...]
    suite_summary_path: Path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".webm"}:
        return "video"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".png", ".jpg", ".jpeg", ".gif"}:
        return "image"
    return "other"


def _media_artifacts(run_dir: Path, results_root: Path) -> tuple[MediaArtifact, ...]:
    vis_dir = run_dir / "vis"
    if not vis_dir.exists():
        return ()
    artifacts: List[MediaArtifact] = []
    for path in sorted(p for p in vis_dir.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS):
        artifacts.append(
            MediaArtifact(
                path=path,
                relative_path=str(path.relative_to(results_root)),
                media_type=_infer_media_type(path),
            )
        )
    return tuple(artifacts)


def _maybe_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    return Path(path_value)


def discover_runs(results_root: Path) -> List[RunRecord]:
    root = results_root.expanduser().resolve()
    records: List[RunRecord] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        metrics_path = _maybe_path(manifest.get("metrics_path"))
        metrics_payload = _read_json(metrics_path) if metrics_path and metrics_path.exists() else {}
        records.append(
            RunRecord(
                run_id=manifest.get("run_id", run_dir.name),
                run_dir=run_dir,
                model=manifest.get("model"),
                tier=manifest.get("tier"),
                seed=manifest.get("seed"),
                vis=manifest.get("vis"),
                summary=flatten_metrics_payload(metrics_payload) if metrics_payload else {},
                manifest=manifest,
                metrics=metrics_payload,
                manifest_path=manifest_path,
                metrics_path=metrics_path if metrics_path and metrics_path.exists() else None,
                stdout_path=_maybe_path(manifest.get("stdout_path")),
                stderr_path=_maybe_path(manifest.get("stderr_path")),
                config_snapshot_path=run_dir / "config_snapshot.json" if (run_dir / "config_snapshot.json").exists() else None,
                media_artifacts=_media_artifacts(run_dir, root),
            )
        )
    return sorted(records, key=lambda record: record.run_id, reverse=True)


def discover_suites(results_root: Path) -> List[SuiteRecord]:
    root = results_root.expanduser().resolve()
    suites: List[SuiteRecord] = []
    for suite_path in sorted(root.rglob("suite_summary.json")):
        payload = _read_json(suite_path)
        suites.append(
            SuiteRecord(
                run_id=suite_path.parent.name,
                run_dir=suite_path.parent,
                tier=payload.get("tier"),
                seed=payload.get("seed"),
                models=tuple(payload.get("models", [])),
                runs=tuple(payload.get("runs", [])),
                suite_summary_path=suite_path,
            )
        )
    return sorted(suites, key=lambda record: record.run_id, reverse=True)


def models(records: Iterable[RunRecord]) -> List[str]:
    return sorted({record.model for record in records if record.model})


def tiers(records: Iterable[RunRecord]) -> List[str]:
    return sorted({record.tier for record in records if record.tier})
