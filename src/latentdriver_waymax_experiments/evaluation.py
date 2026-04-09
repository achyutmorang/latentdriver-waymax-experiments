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
from .upstream import ensure_upstream_exists
from .womd import local_dataset_uri_exists, resolve_dataset_uri, waymo_dataset_root_value


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
        f"++batch_dims={_parse_batch_dims(tier_cfg['batch_dims'])}",
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


def _verify_inputs(model: str, tier: str) -> Dict[str, str]:
    inputs = _validation_inputs(load_config()["evaluation"]["tiers"][tier]["dataset_mode"])
    missing = {}
    ckpt = checkpoint_path(model)
    if not ckpt.exists():
        missing["checkpoint"] = str(ckpt)
    for key, path in inputs.items():
        if key == "waymo_path":
            if not local_dataset_uri_exists(str(path)):
                missing[key] = str(path)
            continue
        if not Path(path).exists():
            missing[key] = str(path)
    ensure_upstream_exists()
    return missing


def run_eval(*, model: str, tier: str, seed: int | None = None, vis: str | bool = False, dry_run: bool = False) -> Dict[str, Any]:
    upstream_dir = ensure_upstream_exists()
    resolved_seed = int(load_config()["evaluation"]["tiers"][tier].get("seed", 0) if seed is None else seed)
    bundle = create_run_bundle(tier=f"{tier}_{model}_seed{resolved_seed}")
    cmd = build_eval_command(model=model, tier=tier, seed=resolved_seed, vis=vis, metrics_path=bundle["metrics_path"], vis_output_dir=bundle["vis_dir"])
    missing = _verify_inputs(model, tier)
    snapshot = {
        "model": model,
        "tier": tier,
        "seed": resolved_seed,
        "vis": vis,
        "command": cmd,
        "missing_inputs": missing,
    }
    write_json(bundle["config_snapshot"], snapshot)
    if dry_run:
        return {
            "run_id": bundle["run_id"],
            "run_dir": str(bundle["run_dir"]),
            "seed": resolved_seed,
            "command": cmd,
            "missing_inputs": missing,
        }
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
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
        raise RuntimeError(f"Evaluation failed with code {proc.returncode}. See {bundle['stderr_path']}")
    metrics_payload = json.loads(Path(bundle["metrics_path"]).read_text(encoding="utf-8"))
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
