from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .artifacts import create_run_bundle, write_json
from .config import load_config, resolve_repo_relative
from .upstream import (
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
    matplotlib_canvas_compat = ensure_matplotlib_canvas_compat_source_patch(upstream_dir)
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
        "matplotlib_canvas_compat": matplotlib_canvas_compat,
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
        "metrics_path": str(bundle["metrics_path"]),
        "stdout_path": str(bundle["stdout_path"]),
        "stderr_path": str(bundle["stderr_path"]),
        "vis_dir": str(bundle["vis_dir"]),
        "media_files": [str(path) for path in media_files],
        "media_file_count": len(media_files),
    }
    write_json(bundle["run_manifest"], manifest)
    return {**manifest, "summary": summary}


def run_public_suite(*, tier: str, seed: int | None = None, models: Iterable[str] | None = None, dry_run: bool = False) -> Dict[str, Any]:
    cfg = load_config()
    selected = list(models or [m for m, spec in cfg["checkpoints"].items() if spec["method"]])
    suite = []
    for model in selected:
        payload = run_eval(model=model, tier=tier, seed=seed, vis=False, dry_run=dry_run)
        suite.append(payload)
    tag_bundle = create_run_bundle(tier=f"suite_{tier}")
    summary = {
        "tier": tier,
        "seed": int(cfg["evaluation"]["tiers"][tier].get("seed", 0) if seed is None else seed),
        "models": selected,
        "runs": suite,
    }
    write_json(tag_bundle["run_dir"] / "suite_summary.json", summary)
    return summary
