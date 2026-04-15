from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import load_config, resolve_repo_relative

_COMMENT_RE = re.compile(r"\s+#.*$")
_TOP_LEVEL_YAML_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
_MPAD_BLOCK_RE = re.compile(r"mpad_blocks\.(\d+)\.")


def _strip_inline_comment(value: str) -> str:
    return _COMMENT_RE.sub("", value).strip()


def _parse_simple_yaml_value(raw: str) -> Any:
    value = _strip_inline_comment(raw)
    if not value:
        return None
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if value.startswith(("[", "{", '"', "'")):
            return ast.literal_eval(value)
        return int(value)
    except (SyntaxError, ValueError):
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_top_level_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line or raw_line[0].isspace():
            continue
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _TOP_LEVEL_YAML_RE.match(raw_line)
        if not match:
            continue
        key, raw_value = match.groups()
        values[key] = {
            "value": _parse_simple_yaml_value(raw_value),
            "line": line_number,
            "raw": _strip_inline_comment(raw_value),
        }
    return values


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _find_first_line(path: Path, needle: str, *, kind: str, description: str) -> dict[str, Any] | None:
    for line_number, line in enumerate(_read_lines(path), start=1):
        if needle in line:
            return {
                "kind": kind,
                "description": description,
                "path": str(path),
                "line": line_number,
                "snippet": line.strip(),
            }
    return None


def _checkpoint_path_for_model(model: str) -> Path:
    cfg = load_config()
    spec = cfg["checkpoints"][model]
    return resolve_repo_relative(cfg["assets"]["checkpoints_root"]) / spec["filename"]


def _load_state_dict(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when checkpoints exist and torch is absent
        return None, f"{type(exc).__name__}: {exc}"
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - depends on external checkpoints
        return None, f"{type(exc).__name__}: {exc}"
    if isinstance(payload, dict):
        if isinstance(payload.get("state_dict"), dict):
            return payload["state_dict"], None
        if all(isinstance(key, str) for key in payload.keys()):
            return payload, None
    return None, f"Unsupported checkpoint payload type: {type(payload).__name__}"


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except Exception:
        return None


def _checkpoint_probe_latentdriver(model: str) -> dict[str, Any]:
    checkpoint_path = _checkpoint_path_for_model(model)
    payload: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
    }
    if not checkpoint_path.exists():
        payload["available"] = False
        payload["reason"] = "checkpoint_missing"
        return payload
    state_dict, error = _load_state_dict(checkpoint_path)
    if state_dict is None:
        payload["available"] = False
        payload["error"] = error
        return payload
    payload["available"] = True
    query_shapes: dict[str, list[int]] = {}
    for key, value in state_dict.items():
        if "query_pe" in key or "action_distribution_queries" in key:
            shape = _shape_of(value)
            if shape is not None:
                query_shapes[key] = shape
    payload["query_shapes"] = query_shapes
    query_counts = sorted({shape[0] for shape in query_shapes.values() if shape})
    payload["candidate_counts_from_checkpoint"] = query_counts
    mpad_indices = sorted(
        {
            int(match.group(1))
            for key in state_dict
            for match in [_MPAD_BLOCK_RE.search(key)]
            if match is not None
        }
    )
    payload["decoder_stage_count_from_checkpoint"] = (mpad_indices[-1] + 1) if mpad_indices else None
    return payload


def _checkpoint_probe_baseline(model: str) -> dict[str, Any]:
    checkpoint_path = _checkpoint_path_for_model(model)
    payload: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
    }
    if not checkpoint_path.exists():
        payload["available"] = False
        payload["reason"] = "checkpoint_missing"
        return payload
    state_dict, error = _load_state_dict(checkpoint_path)
    if state_dict is None:
        payload["available"] = False
        payload["error"] = error
        return payload
    payload["available"] = True
    linear_shapes = {
        key: shape
        for key, value in state_dict.items()
        for shape in [_shape_of(value)]
        if shape is not None and key.startswith("fc_head")
    }
    payload["fc_head_shapes"] = linear_shapes
    return payload


def _config_snapshot(path: Path, keys: Sequence[str]) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "error": "missing"}
    top_level = _parse_top_level_yaml(path)
    snapshot: dict[str, Any] = {"path": str(path)}
    for key in keys:
        item = top_level.get(key)
        if item is None:
            snapshot[key] = None
        else:
            snapshot[key] = {"value": item["value"], "line": item["line"]}
    return snapshot


def _missing_upstream_report(*, model: str, method: str, missing_paths: Sequence[Path]) -> dict[str, Any]:
    return {
        "model": model,
        "configured_method": method,
        "policy_family": "unavailable",
        "config_evidence": {},
        "source_evidence": [],
        "checkpoint_evidence": {
            "path": str(_checkpoint_path_for_model(model)),
            "exists": _checkpoint_path_for_model(model).exists(),
            "available": False,
            "reason": "upstream_not_bootstrapped",
        },
        "verdict": {
            "probe_scope": "repository_wiring",
            "supports_exposed_candidate_diversity": False,
            "candidate_interface": "unknown",
            "exposed_candidate_count": None,
            "refinement_stage_count": None,
            "native_selector": "unknown",
            "reranking_ready": False,
            "confidence": "low",
            "notes": [
                "Required upstream files are missing. Run scripts/bootstrap_upstream.py before probing candidate diversity.",
                f"Missing paths: {[str(path) for path in missing_paths]}",
            ],
        },
        "instrumentation_notes": ["Bootstrap the upstream LatentDriver repo before running this probe."],
    }


def _latentdriver_probe(model: str) -> dict[str, Any]:
    cfg = load_config()
    upstream_root = resolve_repo_relative(cfg["upstream"]["repo_dir"])
    method_config_path = upstream_root / "configs" / "method" / "latentdriver.yaml"
    source_path = upstream_root / "src" / "policy" / "latentdriver" / "lantentdriver_model.py"
    missing_paths = [path for path in (method_config_path, source_path) if not path.is_file()]
    if missing_paths:
        return _missing_upstream_report(model=model, method=cfg["checkpoints"][model]["method"], missing_paths=missing_paths)
    config_evidence = _config_snapshot(method_config_path, ["model_name", "mode", "num_of_decoder", "function"])
    source_evidence = [
        _find_first_line(
            source_path,
            "self.query_pe = TrainableQueryProvider(num_queries=mode",
            kind="candidate_queries",
            description="Learned candidate queries are parameterized by `mode`.",
        ),
        _find_first_line(
            source_path,
            "self.action_distribution_queries = TrainableQueryProvider(num_queries=mode",
            kind="candidate_action_queries",
            description="Action-distribution queries repeat once per candidate mode.",
        ),
        _find_first_line(
            source_path,
            "self.mpad_blocks = nn.ModuleList([MPA_blocks",
            kind="decoder_refinement_stages",
            description="Decoder stages are repeated `num_of_decoder` times.",
        ),
        _find_first_line(
            source_path,
            "actions_layers.append(action_dis.reshape(B,T,-1,7).clone())",
            kind="candidate_tensor_axis",
            description="Forward path preserves a candidate axis before final selection.",
        ),
        _find_first_line(
            source_path,
            "mode = prob.reshape(prob.shape[0],-1).argmax(dim=-1)",
            kind="native_selector",
            description="Native inference collapses candidates with argmax over mode probability.",
        ),
        _find_first_line(
            source_path,
            "return action[:,-1,:]",
            kind="final_selected_action",
            description="Public inference API returns only the selected final action.",
        ),
    ]
    source_evidence = [item for item in source_evidence if item is not None]
    mode = config_evidence["mode"]["value"] if config_evidence["mode"] else None
    num_of_decoder = config_evidence["num_of_decoder"]["value"] if config_evidence["num_of_decoder"] else None
    checkpoint_evidence = _checkpoint_probe_latentdriver(model)
    checkpoint_candidate_counts = checkpoint_evidence.get("candidate_counts_from_checkpoint") or []
    exposed_candidate_count = checkpoint_candidate_counts[0] if checkpoint_candidate_counts else mode
    decoder_stage_count = checkpoint_evidence.get("decoder_stage_count_from_checkpoint") or num_of_decoder
    verdict = {
        "probe_scope": "repository_wiring",
        "supports_exposed_candidate_diversity": True,
        "candidate_interface": "explicit_mode_distribution",
        "exposed_candidate_count": exposed_candidate_count,
        "refinement_stage_count": decoder_stage_count,
        "native_selector": "final_layer_argmax_over_mode_probability",
        "reranking_ready": True,
        "confidence": "high" if exposed_candidate_count and decoder_stage_count else "medium",
        "notes": [
            "The exposed candidate count is the mode axis, not mode x decoder stages.",
            "Decoder stages refine the same candidate set across layers before inference collapses to one action.",
        ],
    }
    return {
        "model": model,
        "configured_method": cfg["checkpoints"][model]["method"],
        "policy_family": "latentdriver",
        "config_evidence": config_evidence,
        "source_evidence": source_evidence,
        "checkpoint_evidence": checkpoint_evidence,
        "verdict": verdict,
        "instrumentation_notes": [
            "Patch `LantentDriver.forward()` or `get_predictions()` to persist `actions_layers[-1]` before `unpack_action()` collapses the mode axis.",
            "Candidate reranking can operate on the final-layer `(B, T, K, 7)` tensor without retraining the generator.",
        ],
    }


def _plant_probe(model: str) -> dict[str, Any]:
    cfg = load_config()
    upstream_root = resolve_repo_relative(cfg["upstream"]["repo_dir"])
    method_config_path = upstream_root / "configs" / "method" / "planT.yaml"
    source_path = upstream_root / "src" / "policy" / "baseline" / "bc_baseline.py"
    policy_init_path = upstream_root / "src" / "policy" / "__init__.py"
    missing_paths = [path for path in (method_config_path, source_path, policy_init_path) if not path.is_file()]
    if missing_paths:
        return _missing_upstream_report(model=model, method=cfg["checkpoints"][model]["method"], missing_paths=missing_paths)
    config_evidence = _config_snapshot(method_config_path, ["model_name", "hidden_channels", "max_len"])
    source_evidence = [
        _find_first_line(
            policy_init_path,
            "'baseline': Simple_driver",
            kind="policy_binding",
            description="The `planT` config resolves to the shared baseline `Simple_driver` policy class.",
        ),
        _find_first_line(
            source_path,
            "self.fc_head =MLP(",
            kind="single_head",
            description="Baseline planner uses one direct regression head.",
        ),
        _find_first_line(
            source_path,
            "out = self.fc_head(fea)",
            kind="single_forward_head",
            description="Forward path produces one output tensor from the head without a candidate axis.",
        ),
        _find_first_line(
            source_path,
            "out = out.reshape(batch_size,seq_length,self.out_dim)",
            kind="single_plan_shape",
            description="Prediction tensor is shaped `(B, T, action_dim)` rather than `(B, T, K, ...)`.",
        ),
        _find_first_line(
            source_path,
            "return out[:,-1]",
            kind="single_prediction_api",
            description="Public inference API returns one final action vector.",
        ),
    ]
    source_evidence = [item for item in source_evidence if item is not None]
    checkpoint_evidence = _checkpoint_probe_baseline(model)
    verdict = {
        "probe_scope": "repository_wiring",
        "supports_exposed_candidate_diversity": False,
        "candidate_interface": "single_plan_regression",
        "exposed_candidate_count": 1,
        "refinement_stage_count": 1,
        "native_selector": "direct_regression_head",
        "reranking_ready": False,
        "confidence": "high",
        "notes": [
            "This verdict is about the PLANT checkpoint/configuration as wired in this repository.",
            "The current implementation does not surface a rerankable candidate set analogous to LatentDriver's mode axis.",
        ],
    }
    return {
        "model": model,
        "configured_method": cfg["checkpoints"][model]["method"],
        "policy_family": "baseline_planT",
        "config_evidence": config_evidence,
        "source_evidence": source_evidence,
        "checkpoint_evidence": checkpoint_evidence,
        "verdict": verdict,
        "instrumentation_notes": [
            "A diversity probe for this planner would require architectural changes or a new multi-candidate head.",
            "Reranking is not possible with the current single-output interface alone.",
        ],
    }


def probe_candidate_diversity(model: str) -> dict[str, Any]:
    cfg = load_config()
    if model not in cfg["checkpoints"]:
        raise ValueError(f"Unknown model={model!r}")
    method = str(cfg["checkpoints"][model].get("method") or "")
    if method == "latentdriver":
        report = _latentdriver_probe(model)
    elif method.lower() == "plant":
        report = _plant_probe(model)
    else:
        report = {
            "model": model,
            "configured_method": method,
            "policy_family": "unsupported",
            "config_evidence": {},
            "source_evidence": [],
            "checkpoint_evidence": {
                "path": str(_checkpoint_path_for_model(model)),
                "exists": _checkpoint_path_for_model(model).exists(),
                "available": False,
                "reason": "unsupported_method_for_probe",
            },
            "verdict": {
                "probe_scope": "repository_wiring",
                "supports_exposed_candidate_diversity": False,
                "candidate_interface": "unknown",
                "exposed_candidate_count": None,
                "refinement_stage_count": None,
                "native_selector": "unknown",
                "reranking_ready": False,
                "confidence": "low",
                "notes": [f"No candidate-diversity probe is implemented for configured method={method!r}."],
            },
            "instrumentation_notes": ["Add a method-specific probe before using this model in candidate-diversity experiments."],
        }
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


def probe_candidate_diversity_suite(models: Iterable[str]) -> dict[str, Any]:
    reports = [probe_candidate_diversity(model) for model in models]
    by_model = {report["model"]: report for report in reports}
    comparisons: list[dict[str, Any]] = []
    if "latentdriver_t2_j3" in by_model and "plant" in by_model:
        left = by_model["latentdriver_t2_j3"]["verdict"]
        right = by_model["plant"]["verdict"]
        comparisons.append(
            {
                "left_model": "latentdriver_t2_j3",
                "right_model": "plant",
                "summary": (
                    "LatentDriver exposes an explicit mode axis that can be reranked before native argmax selection, "
                    "while the PLANT wiring in this repository exposes only a single final waypoint output."
                ),
                "left_exposed_candidate_count": left["exposed_candidate_count"],
                "right_exposed_candidate_count": right["exposed_candidate_count"],
                "left_reranking_ready": left["reranking_ready"],
                "right_reranking_ready": right["reranking_ready"],
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": reports,
        "comparisons": comparisons,
    }


def suite_json(models: Iterable[str]) -> str:
    return json.dumps(probe_candidate_diversity_suite(models), indent=2, sort_keys=True)
