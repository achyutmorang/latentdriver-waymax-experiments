from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .artifacts import create_named_run_bundle, create_run_bundle, write_json
from .config import load_config, resolve_repo_relative
from .modulation.config import collect_modulation_environment
from .preprocess_archive import default_archive_path, default_shard_archive_dir, extract_archive, extract_shard_archives
from .upstream import (
    ensure_action_modulation_source_patch,
    ensure_crdp_compat_source_patch,
    ensure_jax_tree_map_compat_source_patch,
    ensure_lightning_compat_source_patches,
    ensure_matplotlib_canvas_compat_source_patch,
    ensure_python312_compat_sitecustomize,
    ensure_upstream_exists,
)
from .womd import (
    is_gcs_uri,
    local_dataset_uri_complete,
    probe_tensorflow_dataset_uri,
    resolve_dataset_uri,
    sharded_tfrecord_parts,
    sharded_tfrecord_uri,
    waymo_dataset_root_value,
)


@dataclass(frozen=True)
class EvalRequest:
    model: str
    tier: str
    seed: int | None = None
    vis: str | bool = False
    dry_run: bool = False


def _checkpoints_root() -> Path:
    cfg = load_config()
    return resolve_repo_relative(cfg["assets"]["checkpoints_root"])


def checkpoint_path(model: str) -> Path:
    cfg = load_config()
    spec = cfg["checkpoints"][model]
    return _checkpoints_root() / spec["filename"]

def _validation_inputs(dataset_mode: str) -> Dict[str, str | Path]:
    cfg = load_config()
    preprocessed_root = resolve_repo_relative(cfg["assets"]["preprocessed_root"])
    smoke_root = resolve_repo_relative(cfg["assets"]["smoke_root"])
    if dataset_mode == "full":
        dataset_root = waymo_dataset_root_value()
        return {
            "waymo_path": resolve_dataset_uri(dataset_root, cfg["validation"]["full"]["dataset_pattern"]),
            "preprocess_path": preprocessed_root / "full" / "val_preprocessed_path",
            "intention_path": preprocessed_root / "full" / "val_intention_label",
        }
    if dataset_mode == "smoke":
        return {
            "waymo_path": str(smoke_root / cfg["validation"]["smoke"]["dataset_pattern"]),
            "preprocess_path": preprocessed_root / "smoke" / "val_preprocessed_path",
            "intention_path": preprocessed_root / "smoke" / "val_intention_label",
        }
    raise ValueError(f"Unsupported dataset_mode={dataset_mode!r}")


def _parse_batch_dims(batch_dims: Iterable[int]) -> str:
    values = [int(v) for v in batch_dims]
    return f"[{','.join(str(v) for v in values)}]"


def _available_eval_device_count() -> int | None:
    override = os.environ.get("LATENTDRIVER_EVAL_DEVICE_COUNT", "").strip()
    if override:
        value = int(override)
        if value <= 0:
            raise ValueError("LATENTDRIVER_EVAL_DEVICE_COUNT must be positive")
        return value

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices and cuda_visible_devices != "-1":
        visible = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
        if visible:
            return len(visible)

    try:
        proc = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        proc = None
    if proc is not None and proc.returncode == 0:
        gpu_count = sum(1 for line in proc.stdout.splitlines() if line.strip().startswith("GPU "))
        if gpu_count > 0:
            return gpu_count

    try:
        import jax  # type: ignore
    except Exception:
        return None
    return max(1, int(jax.local_device_count()))


def _effective_batch_dims(batch_dims: Iterable[int]) -> list[int]:
    values = [int(v) for v in batch_dims]
    if not values:
        raise ValueError("batch_dims must not be empty")
    if values[0] <= 0:
        raise ValueError("batch_dims[0] must be positive")
    device_count = _available_eval_device_count()
    if device_count is not None:
        values[0] = min(values[0], device_count)
    return values


def _tail_text(text: str, *, max_lines: int = 80, max_chars: int = 8000) -> str:
    stripped = text.strip()
    if not stripped:
        return "<empty>"
    lines = stripped.splitlines()[-max_lines:]
    tail = "\n".join(lines)
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


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


def _media_paths(vis_dir: Path) -> List[Path]:
    return sorted(path for path in vis_dir.rglob('*') if path.is_file() and path.suffix.lower() in {'.mp4', '.pdf'})


def _vis_requested(vis: str | bool) -> bool:
    return vis not in (False, None, '', 'False', 'false', '0')


def flatten_metrics_payload(metrics_payload: Dict[str, Any]) -> Dict[str, Any]:
    avg = metrics_payload.get("average", {})
    avg_cls = metrics_payload.get("average_over_class", {})
    return {
        "number_of_episodes": avg.get("number of episodes"),
        "mar_75_95": avg_cls.get("metric/AR[75:95]"),
        "ar_75_95": avg.get("metric/AR[75:95]"),
        "offroad_rate": avg.get("metric/offroad_rate"),
        "collision_rate": avg.get("metric/collision_rate"),
        "progress_rate": avg.get("metric/progress_rate"),
        "average": avg,
        "average_over_class": avg_cls,
        "per_class": metrics_payload.get("per_class", {}),
    }


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isnan(float(value))
    return False


def _episode_count(metrics_payload: Dict[str, Any]) -> int:
    avg = metrics_payload.get("average", {})
    if not isinstance(avg, dict):
        return 0
    value = avg.get("number of episodes")
    return int(value) if _is_number(value) else 0


def _weighted_mean(values: List[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _, weight in values if weight > 0)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values if weight > 0) / total_weight


def _aggregate_average_metrics(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, List[tuple[float, int]]] = {}
    total_episodes = 0
    for payload in payloads:
        avg = payload.get("average", {})
        if not isinstance(avg, dict):
            continue
        episodes = _episode_count(payload)
        if episodes <= 0:
            continue
        total_episodes += episodes
        for key, value in avg.items():
            if key == "number of episodes" or not _is_number(value):
                continue
            totals.setdefault(key, []).append((float(value), episodes))
    aggregated: Dict[str, Any] = {"number of episodes": total_episodes}
    for key, values in totals.items():
        mean = _weighted_mean(values)
        if mean is not None:
            aggregated[key] = mean
    return aggregated


def _aggregate_per_class_metrics(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        per_class = payload.get("per_class", {})
        if not isinstance(per_class, dict):
            continue
        for label, stats in per_class.items():
            if not isinstance(stats, dict):
                continue
            episodes = int(stats["number of episodes"]) if _is_number(stats.get("number of episodes")) else 0
            bucket = buckets.setdefault(label, {"number of episodes": 0, "_metrics": {}})
            bucket["number of episodes"] += episodes
            metrics = bucket["_metrics"]
            for key, value in stats.items():
                if key == "number of episodes" or not _is_number(value):
                    continue
                metrics.setdefault(key, []).append((float(value), episodes))

    aggregated: Dict[str, Any] = {}
    for label, bucket in buckets.items():
        metrics = bucket.pop("_metrics")
        summary = {"number of episodes": bucket["number of episodes"]}
        for key, values in metrics.items():
            mean = _weighted_mean(values)
            if mean is not None:
                summary[key] = mean
        aggregated[label] = summary
    return aggregated


def _aggregate_average_over_class(per_class: Dict[str, Any]) -> Dict[str, Any]:
    totals: Dict[str, List[float]] = {}
    for stats in per_class.values():
        if not isinstance(stats, dict):
            continue
        episodes = stats.get("number of episodes")
        if not _is_number(episodes) or int(episodes) <= 0:
            continue
        for key, value in stats.items():
            if key == "number of episodes" or not _is_number(value):
                continue
            totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in totals.items() if values}


def aggregate_metrics_payloads(payloads: List[Dict[str, Any]], *, shard_count: int) -> Dict[str, Any]:
    per_class = _aggregate_per_class_metrics(payloads)
    return {
        "average": _aggregate_average_metrics(payloads),
        "average_over_class": _aggregate_average_over_class(per_class),
        "per_class": per_class,
        "meta": {"shards_completed": len(payloads), "shards_total": shard_count},
    }


def build_eval_command(*, model: str, tier: str, seed: int | None = None, vis: str | bool = False, metrics_path: Path | None = None, vis_output_dir: Path | None = None) -> List[str]:
    cfg = load_config()
    tier_cfg = cfg["evaluation"]["tiers"][tier]
    model_spec = cfg["checkpoints"][model]
    inputs = _validation_inputs(tier_cfg["dataset_mode"])
    ckpt = checkpoint_path(model)
    resolved_seed = int(tier_cfg.get("seed", 0) if seed is None else seed)
    cmd = [
        sys.executable,
        "simulate.py",
        f"method={model_spec['method']}",
        f"++waymax_conf.path={inputs['waymo_path']}",
        f"++data_conf.path_to_processed_map_route={inputs['preprocess_path']}",
        f"++metric_conf.intention_label_path={inputs['intention_path']}",
        f"++batch_dims={_parse_batch_dims(_effective_batch_dims(tier_cfg['batch_dims']))}",
        f"++ego_control_setting.npc_policy_type={tier_cfg['npc_policy_type']}",
        f"++method.ckpt_path={ckpt}",
        f"++vis={vis}",
        f"++run.seed={resolved_seed}",
    ]
    if tier_cfg.get("max_batches") is not None:
        cmd.append(f"++run.max_batches={int(tier_cfg['max_batches'])}")
    if metrics_path is not None:
        cmd.append(f"++run.metrics_json_path={metrics_path}")
    if vis_output_dir is not None:
        cmd.append(f"++run.vis_output_dir={vis_output_dir}")
    cmd.extend(model_spec.get("hydra_overrides", []))
    return cmd


def _replace_waymo_path(cmd: List[str], waymo_path: str) -> List[str]:
    return _replace_hydra_override(cmd, "++waymax_conf.path=", waymo_path)


def _replace_hydra_override(cmd: List[str], prefix: str, value: str | Path) -> List[str]:
    updated = []
    replaced = False
    for item in cmd:
        if item.startswith(prefix):
            updated.append(f"{prefix}{value}")
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        raise RuntimeError(f"Unable to replace Hydra override with prefix={prefix!r}")
    return updated


def _replace_preprocess_paths(cmd: List[str], *, preprocess_path: Path, intention_path: Path) -> List[str]:
    updated = _replace_hydra_override(
        cmd,
        "++data_conf.path_to_processed_map_route=",
        preprocess_path,
    )
    return _replace_hydra_override(
        updated,
        "++metric_conf.intention_label_path=",
        intention_path,
    )


def _sharded_eval_targets(dataset_uri: str, *, max_shards: int | None = None) -> List[str]:
    parts = sharded_tfrecord_parts(dataset_uri)
    if parts is None:
        return [dataset_uri]
    _, count = parts
    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError("max_shards must be positive")
        count = min(count, max_shards)
    return [sharded_tfrecord_uri(dataset_uri, index) for index in range(count)]


def _full_preprocess_completion_errors(preprocess_path: Path, intention_path: Path) -> list[str]:
    required_dirs = {
        "map_dir": preprocess_path / "map",
        "route_dir": preprocess_path / "route",
        "intention_dir": intention_path,
    }
    errors = [f"{name} missing: {path}" for name, path in required_dirs.items() if not path.is_dir()]
    success_marker = preprocess_path / "_SUCCESS"
    manifest = preprocess_path / "preprocess_manifest.json"
    if not success_marker.is_file():
        errors.append(f"success marker missing: {success_marker}")
    if not manifest.is_file():
        errors.append(f"preprocess manifest missing: {manifest}")
    return errors


def _local_preprocess_root() -> Path:
    override = os.environ.get("LATENTDRIVER_LOCAL_PREPROCESS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    if Path("/content").exists():
        return Path("/content/latentdriver_preprocess_cache")
    return resolve_repo_relative("artifacts/local_preprocess_cache")


def _materialize_preprocess_enabled(dataset_mode: str) -> bool:
    raw = os.environ.get("LATENTDRIVER_MATERIALIZE_PREPROCESS_CACHE", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return dataset_mode == "full"


def _copy_file_with_retries(src: Path, dst: Path, *, attempts: int = 5) -> bool:
    src_stat = src.stat()
    if dst.exists() and dst.stat().st_size == src_stat.st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.copy2(src, tmp)
            tmp.replace(dst)
            return True
        except OSError as exc:
            last_error = exc
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(min(2 ** attempt, 10))
    raise OSError(f"Failed to copy {src} -> {dst} after {attempts} attempts: {last_error}")


def _list_files_with_retries(src: Path, *, label: str, attempts: int = 6) -> list[Path]:
    if not src.is_dir():
        raise FileNotFoundError(f"Preprocess cache source directory missing: {src}")
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            files = [item for item in src.rglob("*") if item.is_file()]
        except OSError as exc:
            last_error = exc
            files = []
        if files:
            return files
        sleep_seconds = min(2 ** attempt, 10)
        print(
            f"[materialize:{label}] source scan returned 0 files; "
            f"retrying in {sleep_seconds}s ({attempt + 1}/{attempts}) src={src}",
            flush=True,
        )
        time.sleep(sleep_seconds)
    if last_error is not None:
        raise OSError(f"Failed to scan preprocess cache source {src}: {last_error}")
    raise RuntimeError(f"Preprocess cache source {src} contains no files after {attempts} scan attempts")


def _copy_tree_incremental(src: Path, dst: Path, *, label: str, progress_every: int = 1000) -> dict[str, Any]:
    files = _list_files_with_retries(src, label=label)
    total = len(files)
    started_at = time.monotonic()
    stats: dict[str, Any] = {"copied": 0, "skipped": 0, "total": total}
    print(_progress_line(label=f"materialize:{label}", completed=0, total=total, started_at=started_at), flush=True)
    for index, item in enumerate(files, start=1):
        rel = item.relative_to(src)
        copied = _copy_file_with_retries(item, dst / rel)
        stats["copied" if copied else "skipped"] += 1
        if index == total or index % progress_every == 0:
            print(
                _progress_line(
                    label=f"materialize:{label}",
                    completed=index,
                    total=total,
                    started_at=started_at,
                    extra=f"copied={stats['copied']} skipped={stats['skipped']}",
                ),
                flush=True,
            )
    stats.update(_progress_payload(completed=total, total=total, started_at=started_at))
    return stats


def materialize_preprocess_cache(
    *,
    dataset_mode: str,
    preprocess_path: Path,
    intention_path: Path,
) -> dict[str, Any]:
    target_root = _local_preprocess_root() / dataset_mode
    target_preprocess = target_root / "val_preprocessed_path"
    target_intention = target_root / "val_intention_label"
    started_at = time.monotonic()
    shard_archive_dir = default_shard_archive_dir(dataset_mode)
    shard_manifest = shard_archive_dir / "manifest.json"
    if shard_manifest.is_file():
        print(f"[materialize] restoring preprocessed shard archives: {shard_archive_dir}", flush=True)
        archive_payload = extract_shard_archives(
            mode=dataset_mode,
            archive_dir=shard_archive_dir,
            target_root=_local_preprocess_root(),
        )
        _list_files_with_retries(target_preprocess / "map", label="shard-archive-check:map")
        _list_files_with_retries(target_preprocess / "route", label="shard-archive-check:route")
        _list_files_with_retries(target_intention, label="shard-archive-check:intention")
        summary = _progress_payload(completed=1, total=1, started_at=started_at)
        return {
            "enabled": True,
            "strategy": "shard_archives",
            "dataset_mode": dataset_mode,
            "archive": archive_payload,
            "source_preprocess_path": str(preprocess_path),
            "source_intention_path": str(intention_path),
            "preprocess_path": str(target_preprocess),
            "intention_path": str(target_intention),
            "summary": summary,
        }

    archive_path = default_archive_path(dataset_mode)
    if archive_path.is_file():
        print(f"[materialize] restoring preprocessed cache archive: {archive_path}", flush=True)
        archive_payload = extract_archive(
            mode=dataset_mode,
            archive_path=archive_path,
            target_root=_local_preprocess_root(),
        )
        _list_files_with_retries(target_preprocess / "map", label="archive-check:map")
        _list_files_with_retries(target_preprocess / "route", label="archive-check:route")
        _list_files_with_retries(target_intention, label="archive-check:intention")
        summary = _progress_payload(completed=1, total=1, started_at=started_at)
        return {
            "enabled": True,
            "strategy": "archive",
            "dataset_mode": dataset_mode,
            "archive": archive_payload,
            "source_preprocess_path": str(preprocess_path),
            "source_intention_path": str(intention_path),
            "preprocess_path": str(target_preprocess),
            "intention_path": str(target_intention),
            "summary": summary,
        }

    print(f"[materialize] staging preprocessed cache to local disk: {target_root}", flush=True)
    copy_stats = {
        "map": _copy_tree_incremental(preprocess_path / "map", target_preprocess / "map", label="map"),
        "route": _copy_tree_incremental(preprocess_path / "route", target_preprocess / "route", label="route"),
        "intention": _copy_tree_incremental(intention_path, target_intention, label="intention"),
    }
    for marker in ("_SUCCESS", "preprocess_manifest.json"):
        source_marker = preprocess_path / marker
        if source_marker.is_file():
            _copy_file_with_retries(source_marker, target_preprocess / marker)
    total_files = sum(int(stats["total"]) for stats in copy_stats.values())
    total_copied = sum(int(stats["copied"]) for stats in copy_stats.values())
    total_skipped = sum(int(stats["skipped"]) for stats in copy_stats.values())
    print(
        _progress_line(
            label="materialize:total",
            completed=total_files,
            total=total_files,
            started_at=started_at,
            extra=f"copied={total_copied} skipped={total_skipped}",
        ),
        flush=True,
    )
    return {
        "enabled": True,
        "strategy": "incremental_copy",
        "dataset_mode": dataset_mode,
        "source_preprocess_path": str(preprocess_path),
        "source_intention_path": str(intention_path),
        "preprocess_path": str(target_preprocess),
        "intention_path": str(target_intention),
        "copy_stats": copy_stats,
        "summary": {
            "total_files": total_files,
            "copied": total_copied,
            "skipped": total_skipped,
            **_progress_payload(completed=total_files, total=total_files, started_at=started_at),
        },
    }


def _verify_inputs(model: str, tier: str, *, verify_remote_reads: bool = False) -> Dict[str, str]:
    tier_cfg = load_config()["evaluation"]["tiers"][tier]
    dataset_mode = tier_cfg["dataset_mode"]
    inputs = _validation_inputs(dataset_mode)
    missing = {}
    ckpt = checkpoint_path(model)
    if not ckpt.exists():
        missing["checkpoint"] = str(ckpt)
    for key, path in inputs.items():
        if key == "waymo_path":
            waymo_path = str(path)
            if not local_dataset_uri_complete(waymo_path):
                missing[key] = str(path)
            elif verify_remote_reads and is_gcs_uri(waymo_path):
                probe = probe_tensorflow_dataset_uri(waymo_path)
                if not probe.get("ok"):
                    missing["waymo_path_gcs_read"] = json.dumps(probe, sort_keys=True)
            continue
        if not Path(path).exists():
            missing[key] = str(path)
    if dataset_mode == "full":
        preprocess_errors = _full_preprocess_completion_errors(
            Path(inputs["preprocess_path"]),
            Path(inputs["intention_path"]),
        )
        if preprocess_errors:
            missing["preprocess_completion"] = "; ".join(preprocess_errors)
    ensure_upstream_exists()
    return missing


def inspect_eval_inputs(*, model: str, tier: str, verify_remote_reads: bool = False) -> Dict[str, Any]:
    cmd = build_eval_command(model=model, tier=tier, vis=False)
    missing = _verify_inputs(model, tier, verify_remote_reads=verify_remote_reads)
    return {
        "model": model,
        "tier": tier,
        "verify_remote_reads": verify_remote_reads,
        "command": cmd,
        "missing_inputs": missing,
        "ready": not bool(missing),
    }


def _missing_inputs_message(*, model: str, tier: str, missing: Dict[str, str]) -> str:
    lines = [f"Missing required inputs for model={model} tier={tier}:"]
    for key, value in missing.items():
        lines.append(f"- {key}: {value}")
    if "checkpoint" in missing:
        lines.append("")
        lines.append("Checkpoint assets are missing.")
        lines.append(
            "Run `python3 scripts/download_checkpoints.py --evaluation-only` "
            "or execute the assets notebook `notebooks/latentdriver_assets_colab.ipynb` first."
        )
    if any(key in missing for key in ("preprocess_path", "intention_path", "waymo_path", "preprocess_completion")):
        lines.append("")
        lines.append(
            "Dataset or preprocess artifacts are missing. "
            "Run the Colab runner `full-preprocess` profile until it writes `_SUCCESS` and `preprocess_manifest.json`."
        )
    if "waymo_path_gcs_read" in missing:
        lines.append("")
        lines.append("TensorFlow cannot read the WOMD GCS URI with the current runtime credentials.")
        lines.append(
            "Preferred fix: stage the full validation TFRecords into the Drive-backed `assets/raw_womd` cache, "
            "then rerun eval with `--waymo-dataset-root /content/drive/MyDrive/waymax_research/"
            "latentdriver_waymax_experiments/assets/raw_womd`."
        )
        lines.append(
            "Alternative: authenticate Application Default Credentials for GCS in the Colab runtime and rerun with the `gs://` root."
        )
    return "\n".join(lines)


def run_eval(*, model: str, tier: str, seed: int | None = None, vis: str | bool = False, dry_run: bool = False) -> Dict[str, Any]:
    upstream_dir = ensure_upstream_exists()
    compat_sitecustomize = ensure_python312_compat_sitecustomize(upstream_dir)
    lightning_compat = ensure_lightning_compat_source_patches(upstream_dir)
    crdp_compat = ensure_crdp_compat_source_patch(upstream_dir)
    jax_tree_map_compat = ensure_jax_tree_map_compat_source_patch(upstream_dir)
    action_modulation_compat = ensure_action_modulation_source_patch(upstream_dir)
    matplotlib_canvas_compat = ensure_matplotlib_canvas_compat_source_patch(upstream_dir)
    modulation_environment = collect_modulation_environment()
    resolved_seed = int(load_config()["evaluation"]["tiers"][tier].get("seed", 0) if seed is None else seed)
    bundle = create_run_bundle(tier=f"{tier}_{model}_seed{resolved_seed}")
    cmd = build_eval_command(model=model, tier=tier, seed=resolved_seed, vis=vis, metrics_path=bundle["metrics_path"], vis_output_dir=bundle["vis_dir"])
    missing = _verify_inputs(model, tier, verify_remote_reads=not dry_run)
    snapshot = {
        "model": model,
        "tier": tier,
        "seed": resolved_seed,
        "vis": vis,
        "command": cmd,
        "missing_inputs": missing,
        "compat_sitecustomize": str(compat_sitecustomize),
        "lightning_compat": lightning_compat,
        "crdp_compat": crdp_compat,
        "jax_tree_map_compat": jax_tree_map_compat,
        "action_modulation_compat": action_modulation_compat,
        "matplotlib_canvas_compat": matplotlib_canvas_compat,
        "modulation_environment": modulation_environment,
    }
    write_json(bundle["config_snapshot"], snapshot)
    if dry_run:
        return {
            "run_id": bundle["run_id"],
            "run_dir": str(bundle["run_dir"]),
            "seed": resolved_seed,
            "command": cmd,
            "missing_inputs": missing,
            "ready": not bool(missing),
        }
    if missing:
        raise FileNotFoundError(_missing_inputs_message(model=model, tier=tier, missing=missing))
    proc = subprocess.run(
        cmd,
        cwd=upstream_dir,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    bundle["stdout_path"].write_text(proc.stdout, encoding="utf-8")
    bundle["stderr_path"].write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        stderr_tail = _tail_text(proc.stderr)
        stdout_tail = _tail_text(proc.stdout)
        raise RuntimeError(
            f"Evaluation failed with code {proc.returncode}.\n"
            f"stderr_path: {bundle['stderr_path']}\n"
            f"stdout_path: {bundle['stdout_path']}\n\n"
            f"stderr tail:\n{stderr_tail}\n\n"
            f"stdout tail:\n{stdout_tail}"
        )
    metrics_path = Path(bundle["metrics_path"])
    if not metrics_path.exists():
        raise RuntimeError(
            f"Evaluation finished without metrics output.\n"
            f"metrics_path: {metrics_path}\n"
            f"stderr_path: {bundle['stderr_path']}\n"
            f"stdout_path: {bundle['stdout_path']}\n\n"
            f"stderr tail:\n{_tail_text(proc.stderr)}\n\n"
            f"stdout tail:\n{_tail_text(proc.stdout)}"
        )
    media_files = _media_paths(Path(bundle["vis_dir"]))
    if _vis_requested(vis) and not media_files:
        raise RuntimeError(
            f"Visualization run finished without producing media artifacts.\n"
            f"vis_dir: {bundle['vis_dir']}\n"
            f"stderr_path: {bundle['stderr_path']}\n"
            f"stdout_path: {bundle['stdout_path']}\n\n"
            f"stderr tail:\n{_tail_text(proc.stderr)}\n\n"
            f"stdout tail:\n{_tail_text(proc.stdout)}"
        )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = flatten_metrics_payload(metrics_payload)
    manifest = {
        "run_id": bundle["run_id"],
        "run_dir": str(bundle["run_dir"]),
        "model": model,
        "tier": tier,
        "seed": resolved_seed,
        "vis": vis,
        "checkpoint_path": str(checkpoint_path(model)),
        "upstream_dir": str(upstream_dir),
        "command": cmd,
        "modulation_environment": modulation_environment,
        "metrics_path": str(bundle["metrics_path"]),
        "stdout_path": str(bundle["stdout_path"]),
        "stderr_path": str(bundle["stderr_path"]),
        "vis_dir": str(bundle["vis_dir"]),
        "media_files": [str(path) for path in media_files],
        "media_file_count": len(media_files),
    }
    write_json(bundle["run_manifest"], manifest)
    return {**manifest, "summary": summary}


def _resumable_run_id(*, tier: str, model: str, seed: int) -> str:
    return f"resumable_{tier}_{model}_seed{seed}"


def run_eval_resumable(
    *,
    model: str,
    tier: str,
    seed: int | None = None,
    vis: str | bool = False,
    dry_run: bool = False,
    resume: bool = True,
    max_shards: int | None = None,
) -> Dict[str, Any]:
    upstream_dir = ensure_upstream_exists()
    compat_sitecustomize = ensure_python312_compat_sitecustomize(upstream_dir)
    lightning_compat = ensure_lightning_compat_source_patches(upstream_dir)
    crdp_compat = ensure_crdp_compat_source_patch(upstream_dir)
    jax_tree_map_compat = ensure_jax_tree_map_compat_source_patch(upstream_dir)
    action_modulation_compat = ensure_action_modulation_source_patch(upstream_dir)
    matplotlib_canvas_compat = ensure_matplotlib_canvas_compat_source_patch(upstream_dir)
    modulation_environment = collect_modulation_environment()
    tier_cfg = load_config()["evaluation"]["tiers"][tier]
    resolved_seed = int(tier_cfg.get("seed", 0) if seed is None else seed)
    bundle = create_named_run_bundle(run_id=_resumable_run_id(tier=tier, model=model, seed=resolved_seed))
    cmd = build_eval_command(
        model=model,
        tier=tier,
        seed=resolved_seed,
        vis=vis,
        metrics_path=Path(bundle["metrics_path"]),
        vis_output_dir=Path(bundle["vis_dir"]),
    )
    missing = _verify_inputs(model, tier, verify_remote_reads=not dry_run)
    snapshot = {
        "model": model,
        "tier": tier,
        "seed": resolved_seed,
        "vis": vis,
        "command": cmd,
        "missing_inputs": missing,
        "compat_sitecustomize": str(compat_sitecustomize),
        "lightning_compat": lightning_compat,
        "crdp_compat": crdp_compat,
        "jax_tree_map_compat": jax_tree_map_compat,
        "action_modulation_compat": action_modulation_compat,
        "matplotlib_canvas_compat": matplotlib_canvas_compat,
        "modulation_environment": modulation_environment,
        "resume": resume,
        "max_shards": max_shards,
    }
    write_json(Path(bundle["config_snapshot"]), snapshot)
    if dry_run:
        return {
            "run_id": bundle["run_id"],
            "run_dir": str(bundle["run_dir"]),
            "seed": resolved_seed,
        "command": cmd,
        "modulation_environment": modulation_environment,
        "missing_inputs": missing,
            "ready": not bool(missing),
            "resume": resume,
            "max_shards": max_shards,
        }
    if missing:
        raise FileNotFoundError(_missing_inputs_message(model=model, tier=tier, missing=missing))

    inputs = _validation_inputs(tier_cfg["dataset_mode"])
    shard_uris = _sharded_eval_targets(str(inputs["waymo_path"]), max_shards=max_shards)
    shard_root = Path(bundle["run_dir"]) / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    progress_path = Path(bundle["run_dir"]) / "progress.json"
    materialized_preprocess: dict[str, Any] | None = None

    completed_payloads: List[Dict[str, Any]] = []
    shard_status: List[Dict[str, Any]] = []
    run_started_at = time.monotonic()

    def write_progress(*, status: str = "running") -> None:
        progress = _progress_payload(
            completed=len(completed_payloads),
            total=len(shard_uris),
            started_at=run_started_at,
        )
        write_json(
            progress_path,
            {
                "run_id": bundle["run_id"],
                "model": model,
                "tier": tier,
                "seed": resolved_seed,
                "status": status,
                "shards_total": len(shard_uris),
                "shards_completed": len(completed_payloads),
                "shards_remaining": progress["remaining"],
                "elapsed": progress["elapsed"],
                "elapsed_seconds": progress["elapsed_seconds"],
                "eta": progress["eta"],
                "eta_seconds": progress["eta_seconds"],
                "updated_at": progress["updated_at"],
                "shards": shard_status,
                "materialized_preprocess": materialized_preprocess,
            },
        )

    dataset_mode = str(tier_cfg["dataset_mode"])
    if _materialize_preprocess_enabled(dataset_mode):
        write_progress(status="materializing_preprocess")
        materialized_preprocess = materialize_preprocess_cache(
            dataset_mode=dataset_mode,
            preprocess_path=Path(inputs["preprocess_path"]),
            intention_path=Path(inputs["intention_path"]),
        )
        cmd = _replace_preprocess_paths(
            cmd,
            preprocess_path=Path(materialized_preprocess["preprocess_path"]),
            intention_path=Path(materialized_preprocess["intention_path"]),
        )
        write_progress(status="running")

    print(
        f"[resumable-eval] starting {len(shard_uris)} shards "
        f"model={model} tier={tier} seed={resolved_seed}",
        flush=True,
    )
    for shard_index, shard_uri in enumerate(shard_uris):
        shard_id = f"shard-{shard_index:05d}"
        shard_dir = shard_root / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = shard_dir / "metrics.json"
        stdout_path = shard_dir / "stdout.log"
        stderr_path = shard_dir / "stderr.log"
        vis_dir = shard_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        if resume and metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                completed_payloads.append(payload)
                shard_status.append({"shard": shard_id, "uri": shard_uri, "status": "skipped"})
                write_progress()
                print(
                    _progress_line(
                        label="resumable-eval",
                        completed=len(completed_payloads),
                        total=len(shard_uris),
                        started_at=run_started_at,
                        extra=f"last={shard_id} status=skipped",
                    ),
                    flush=True,
                )
                continue
            except json.JSONDecodeError:
                pass

        shard_started_at = time.monotonic()
        print(
            _progress_line(
                label="resumable-eval",
                completed=len(completed_payloads),
                total=len(shard_uris),
                started_at=run_started_at,
                extra=f"running={shard_id}",
            ),
            flush=True,
        )
        shard_cmd = _replace_waymo_path(cmd, shard_uri)
        shard_cmd = [
            arg
            for arg in shard_cmd
            if not arg.startswith("++run.metrics_json_path=") and not arg.startswith("++run.vis_output_dir=")
        ]
        shard_cmd.append(f"++run.metrics_json_path={metrics_path}")
        shard_cmd.append(f"++run.vis_output_dir={vis_dir}")
        proc = subprocess.run(
            shard_cmd,
            cwd=upstream_dir,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            shard_status.append(
                {
                    "shard": shard_id,
                    "uri": shard_uri,
                    "status": "failed",
                    "returncode": proc.returncode,
                    "duration_seconds": round(time.monotonic() - shard_started_at, 3),
                }
            )
            write_progress(status="failed")
            raise RuntimeError(
                f"Shard {shard_id} failed with code {proc.returncode}.\n"
                f"stderr_path: {stderr_path}\n"
                f"stdout_path: {stdout_path}\n\n"
                f"stderr tail:\n{_tail_text(proc.stderr)}\n\n"
                f"stdout tail:\n{_tail_text(proc.stdout)}"
            )
        if not metrics_path.exists():
            shard_status.append(
                {
                    "shard": shard_id,
                    "uri": shard_uri,
                    "status": "no_metrics",
                    "duration_seconds": round(time.monotonic() - shard_started_at, 3),
                }
            )
            write_progress(status="failed")
            raise RuntimeError(f"Shard {shard_id} finished without metrics.json: {metrics_path}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        completed_payloads.append(payload)
        shard_duration = time.monotonic() - shard_started_at
        shard_status.append(
            {
                "shard": shard_id,
                "uri": shard_uri,
                "status": "completed",
                "duration_seconds": round(shard_duration, 3),
            }
        )
        write_progress()
        print(
            _progress_line(
                label="resumable-eval",
                completed=len(completed_payloads),
                total=len(shard_uris),
                started_at=run_started_at,
                extra=f"last={shard_id} shard_time={_format_duration(shard_duration)}",
            ),
            flush=True,
        )

    metrics_payload = aggregate_metrics_payloads(completed_payloads, shard_count=len(shard_uris))
    write_json(Path(bundle["metrics_path"]), metrics_payload)
    summary = flatten_metrics_payload(metrics_payload)
    manifest = {
        "run_id": bundle["run_id"],
        "run_dir": str(bundle["run_dir"]),
        "model": model,
        "tier": tier,
        "seed": resolved_seed,
        "vis": vis,
        "checkpoint_path": str(checkpoint_path(model)),
        "upstream_dir": str(upstream_dir),
        "command": cmd,
        "modulation_environment": modulation_environment,
        "metrics_path": str(bundle["metrics_path"]),
        "stdout_path": str(bundle["stdout_path"]),
        "stderr_path": str(bundle["stderr_path"]),
        "vis_dir": str(bundle["vis_dir"]),
        "shards": shard_status,
        "shard_root": str(shard_root),
        "progress_path": str(progress_path),
        "materialized_preprocess": materialized_preprocess,
    }
    write_json(Path(bundle["run_manifest"]), manifest)
    write_progress(status="completed")
    return {**manifest, "summary": summary}


def run_public_suite(
    *,
    tier: str,
    seed: int | None = None,
    models: Iterable[str] | None = None,
    dry_run: bool = False,
    resumable: bool = False,
    resume: bool = True,
    max_shards: int | None = None,
) -> Dict[str, Any]:
    cfg = load_config()
    selected = list(models or [m for m, spec in cfg["checkpoints"].items() if spec["method"]])
    suite = []
    for model in selected:
        if resumable:
            payload = run_eval_resumable(
                model=model,
                tier=tier,
                seed=seed,
                vis=False,
                dry_run=dry_run,
                resume=resume,
                max_shards=max_shards,
            )
        else:
            payload = run_eval(model=model, tier=tier, seed=seed, vis=False, dry_run=dry_run)
        suite.append(payload)
    tag_bundle = create_run_bundle(tier=f"suite_{tier}")
    summary = {
        "tier": tier,
        "seed": int(cfg["evaluation"]["tiers"][tier].get("seed", 0) if seed is None else seed),
        "models": selected,
        "runs": suite,
        "resumable": resumable,
        "resume": resume,
        "max_shards": max_shards,
    }
    write_json(tag_bundle["run_dir"] / "suite_summary.json", summary)
    return summary
