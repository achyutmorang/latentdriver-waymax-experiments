from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from .artifacts import write_json
from .config import load_config, resolve_repo_relative
from .womd import local_dataset_uri_exists, resolve_dataset_uri, waymo_dataset_root_value

DEFAULT_MODEL = "latentdriver_t2_j3"
DEFAULT_SEED = 0
DEFAULT_VIS = "video"
NO_RUNTIME_SETUP_PROFILES = {
    "bootstrap-session",
    "env-check",
    "download-checkpoints",
    "full-preprocess-status",
    "full-eval-dry-run",
    "plot-smoke-reactive",
    "plot-smoke-non-reactive",
    "plot-full-reactive",
    "plot-full-non-reactive",
}


@dataclass(frozen=True)
class RunnerStep:
    name: str
    command: tuple[str, ...]
    description: str
    continue_on_failure: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "command_text": shlex.join(self.command),
            "description": self.description,
            "continue_on_failure": self.continue_on_failure,
        }


PROFILE_DESCRIPTIONS: Mapping[str, str] = {
    "env-check": "Capture environment, git, GPU, and artifact status without running heavy commands.",
    "install-runtime": "Install/patch the Colab runtime and editable project package.",
    "download-checkpoints": "Download or verify public evaluation checkpoints into the Drive-bound checkpoint cache.",
    "smoke-preprocess": "Run the smoke validation preprocessing command.",
    "smoke-eval-reactive": "Run all public checkpoints on smoke_reactive.",
    "smoke-eval-non-reactive": "Run all public checkpoints on smoke_non_reactive.",
    "full-preprocess-status": "Check full preprocess artifact paths without scanning the large cache directories.",
    "full-preprocess": "Run full validation preprocessing; use only when you intentionally want to rebuild/resume preprocessing.",
    "full-eval-dry-run": "Dry-run full_reactive evaluation for one model without launching simulation.",
    "full-eval-reactive-single": "Run one model on full_reactive.",
    "full-eval-non-reactive-single": "Run one model on full_non_reactive.",
    "full-eval-reactive": "Run all public checkpoints on full_reactive.",
    "full-eval-non-reactive": "Run all public checkpoints on full_non_reactive.",
    "visualize-smoke": "Run a smoke visualization job and capture MP4/PDF outputs.",
    "plot-smoke-reactive": "Generate comparison plots for smoke_reactive.",
    "plot-smoke-non-reactive": "Generate comparison plots for smoke_non_reactive.",
    "plot-full-reactive": "Generate comparison plots for full_reactive.",
    "plot-full-non-reactive": "Generate comparison plots for full_non_reactive.",
    "bootstrap-session": "Install runtime, verify checkpoints, and run the full_reactive dry-run preflight.",
}


def available_profiles() -> dict[str, str]:
    return dict(sorted(PROFILE_DESCRIPTIONS.items()))


def _script_command(script: str, *args: object) -> tuple[str, ...]:
    return tuple([sys.executable, script, *[str(arg) for arg in args]])


def should_install_runtime_by_default(profile: str) -> bool:
    _validate_profile(profile)
    return profile not in NO_RUNTIME_SETUP_PROFILES


def _validate_profile(profile: str) -> None:
    if profile not in PROFILE_DESCRIPTIONS:
        valid = ", ".join(sorted(PROFILE_DESCRIPTIONS))
        raise ValueError(f"Unknown profile={profile!r}. Valid profiles: {valid}")


def profile_steps(
    profile: str,
    *,
    model: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    vis: str = DEFAULT_VIS,
    install_runtime: bool = False,
    download_checkpoints: bool = False,
) -> list[RunnerStep]:
    _validate_profile(profile)
    if model not in load_config()["checkpoints"]:
        raise ValueError(f"Unknown model={model!r}")

    steps: list[RunnerStep] = []
    if install_runtime and profile not in {"install-runtime", "bootstrap-session"}:
        steps.append(
            RunnerStep(
                name="setup_colab_runtime",
                command=_script_command("scripts/setup_colab_runtime.py", "--editable-project"),
                description="Install and patch the Colab runtime before running the selected profile.",
            )
        )
    if download_checkpoints and profile not in {"download-checkpoints", "bootstrap-session"}:
        steps.append(
            RunnerStep(
                name="download_checkpoints",
                command=_script_command("scripts/download_checkpoints.py", "--evaluation-only"),
                description="Ensure released evaluation checkpoints are present in the Drive-bound cache.",
            )
        )

    if profile == "env-check":
        return steps
    if profile == "install-runtime":
        return [
            *steps,
            RunnerStep(
                name="setup_colab_runtime",
                command=_script_command("scripts/setup_colab_runtime.py", "--editable-project"),
                description="Install and patch the Colab runtime.",
            ),
        ]
    if profile == "download-checkpoints":
        return [
            *steps,
            RunnerStep(
                name="download_checkpoints",
                command=_script_command("scripts/download_checkpoints.py", "--evaluation-only"),
                description="Download released evaluation checkpoints.",
            ),
        ]
    if profile == "smoke-preprocess":
        return [
            *steps,
            RunnerStep(
                name="smoke_preprocess",
                command=_script_command("scripts/preprocess_validation_only.py", "--mode", "smoke"),
                description="Build or reuse smoke validation preprocessing artifacts.",
            ),
        ]
    if profile == "smoke-eval-reactive":
        return [
            *steps,
            RunnerStep(
                name="smoke_eval_reactive_suite",
                command=_script_command("scripts/run_public_suite.py", "--tier", "smoke_reactive", "--seed", seed),
                description="Evaluate all public checkpoints on smoke_reactive.",
            ),
        ]
    if profile == "smoke-eval-non-reactive":
        return [
            *steps,
            RunnerStep(
                name="smoke_eval_non_reactive_suite",
                command=_script_command("scripts/run_public_suite.py", "--tier", "smoke_non_reactive", "--seed", seed),
                description="Evaluate all public checkpoints on smoke_non_reactive.",
            ),
        ]
    if profile == "full-preprocess-status":
        return steps
    if profile == "full-preprocess":
        return [
            *steps,
            RunnerStep(
                name="full_preprocess",
                command=_script_command(
                    "scripts/preprocess_validation_only.py",
                    "--mode",
                    "full",
                    "--workers",
                    1,
                    "--jax-platform",
                    "cpu",
                ),
                description="Resume or build full validation preprocessing artifacts on CPU-safe settings.",
            ),
        ]
    if profile == "full-eval-dry-run":
        return [
            *steps,
            RunnerStep(
                name="full_eval_dry_run",
                command=_script_command(
                    "scripts/run_waymax_eval.py",
                    "--model",
                    model,
                    "--tier",
                    "full_reactive",
                    "--seed",
                    seed,
                    "--dry-run",
                ),
                description="Verify full_reactive command construction and required input paths for one model.",
            ),
        ]
    if profile == "full-eval-reactive-single":
        return [
            *steps,
            RunnerStep(
                name="full_eval_reactive_single",
                command=_script_command("scripts/run_waymax_eval.py", "--model", model, "--tier", "full_reactive", "--seed", seed),
                description="Evaluate one model on full_reactive.",
            ),
        ]
    if profile == "full-eval-non-reactive-single":
        return [
            *steps,
            RunnerStep(
                name="full_eval_non_reactive_single",
                command=_script_command("scripts/run_waymax_eval.py", "--model", model, "--tier", "full_non_reactive", "--seed", seed),
                description="Evaluate one model on full_non_reactive.",
            ),
        ]
    if profile == "full-eval-reactive":
        return [
            *steps,
            RunnerStep(
                name="full_eval_reactive_suite",
                command=_script_command("scripts/run_public_suite.py", "--tier", "full_reactive", "--seed", seed),
                description="Evaluate all public checkpoints on full_reactive.",
            ),
        ]
    if profile == "full-eval-non-reactive":
        return [
            *steps,
            RunnerStep(
                name="full_eval_non_reactive_suite",
                command=_script_command("scripts/run_public_suite.py", "--tier", "full_non_reactive", "--seed", seed),
                description="Evaluate all public checkpoints on full_non_reactive.",
            ),
        ]
    if profile == "visualize-smoke":
        return [
            *steps,
            RunnerStep(
                name="visualize_smoke",
                command=_script_command("scripts/run_visualization.py", "--model", model, "--tier", "smoke_reactive", "--seed", seed, "--vis", vis),
                description="Generate a smoke visualization artifact.",
            ),
        ]
    if profile == "plot-smoke-reactive":
        return [
            *steps,
            RunnerStep(
                name="plot_smoke_reactive",
                command=_script_command("scripts/plot_model_metrics.py", "--tier", "smoke_reactive", "--seed", seed),
                description="Plot latest smoke_reactive metrics across models.",
            ),
        ]
    if profile == "plot-smoke-non-reactive":
        return [
            *steps,
            RunnerStep(
                name="plot_smoke_non_reactive",
                command=_script_command("scripts/plot_model_metrics.py", "--tier", "smoke_non_reactive", "--seed", seed),
                description="Plot latest smoke_non_reactive metrics across models.",
            ),
        ]
    if profile == "plot-full-reactive":
        return [
            *steps,
            RunnerStep(
                name="plot_full_reactive",
                command=_script_command("scripts/plot_model_metrics.py", "--tier", "full_reactive", "--seed", seed),
                description="Plot latest full_reactive metrics across models.",
            ),
        ]
    if profile == "plot-full-non-reactive":
        return [
            *steps,
            RunnerStep(
                name="plot_full_non_reactive",
                command=_script_command("scripts/plot_model_metrics.py", "--tier", "full_non_reactive", "--seed", seed),
                description="Plot latest full_non_reactive metrics across models.",
            ),
        ]
    if profile == "bootstrap-session":
        return [
            *steps,
            RunnerStep(
                name="setup_colab_runtime",
                command=_script_command("scripts/setup_colab_runtime.py", "--editable-project"),
                description="Install and patch the Colab runtime.",
            ),
            RunnerStep(
                name="download_checkpoints",
                command=_script_command("scripts/download_checkpoints.py", "--evaluation-only"),
                description="Ensure released evaluation checkpoints are present.",
            ),
            RunnerStep(
                name="full_eval_dry_run",
                command=_script_command(
                    "scripts/run_waymax_eval.py",
                    "--model",
                    model,
                    "--tier",
                    "full_reactive",
                    "--seed",
                    seed,
                    "--dry-run",
                ),
                description="Verify full_reactive command construction and input paths.",
            ),
        ]
    raise AssertionError(f"Unhandled profile={profile!r}")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "item"


def _safe_path_state(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"path": str(path)}
    try:
        state["exists"] = path.exists()
        state["is_dir"] = path.is_dir()
        state["is_file"] = path.is_file()
        state["is_symlink"] = path.is_symlink()
        if path.is_symlink():
            state["symlink_target"] = os.readlink(path)
        if path.exists() and path.is_file():
            state["size_bytes"] = path.stat().st_size
    except OSError as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    return state


def _safe_json_file(path: Path) -> dict[str, Any]:
    state = _safe_path_state(path)
    if state.get("is_file"):
        try:
            state["json"] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            state["json_error"] = f"{type(exc).__name__}: {exc}"
    return state


def _dataset_status(mode: str) -> dict[str, Any]:
    cfg = load_config()
    if mode == "full":
        try:
            dataset_root = waymo_dataset_root_value()
            uri = resolve_dataset_uri(dataset_root, cfg["validation"]["full"]["dataset_pattern"])
        except Exception as exc:
            return {"mode": mode, "error": f"{type(exc).__name__}: {exc}"}
    elif mode == "smoke":
        smoke_root = resolve_repo_relative(cfg["assets"]["smoke_root"])
        uri = str(smoke_root / cfg["validation"]["smoke"]["dataset_pattern"])
    else:
        raise ValueError(f"Unsupported mode={mode!r}")
    try:
        exists_or_remote = local_dataset_uri_exists(str(uri))
    except Exception as exc:
        return {"mode": mode, "uri": str(uri), "error": f"{type(exc).__name__}: {exc}"}
    return {"mode": mode, "uri": str(uri), "exists_or_remote": exists_or_remote}


def _preprocess_status(mode: str) -> dict[str, Any]:
    cfg = load_config()
    root = resolve_repo_relative(cfg["assets"]["preprocessed_root"]) / mode
    preprocess_root = root / "val_preprocessed_path"
    status = {
        "mode": mode,
        "root": _safe_path_state(root),
        "preprocess_root": _safe_path_state(preprocess_root),
        "map_dir": _safe_path_state(preprocess_root / "map"),
        "route_dir": _safe_path_state(preprocess_root / "route"),
        "intention_dir": _safe_path_state(root / "val_intention_label"),
        "success_marker": _safe_path_state(preprocess_root / "_SUCCESS"),
        "manifest": _safe_json_file(preprocess_root / "preprocess_manifest.json"),
        "scan_policy": "non_recursive_path_state_only",
    }
    status["path_ready_without_counts"] = all(
        bool(status[key].get("is_dir")) for key in ("map_dir", "route_dir", "intention_dir")
    )
    status["marker_ready"] = bool(status["success_marker"].get("is_file")) and bool(status["manifest"].get("is_file"))
    return status


def collect_artifact_status() -> dict[str, Any]:
    cfg = load_config()
    checkpoints_root = resolve_repo_relative(cfg["assets"]["checkpoints_root"])
    checkpoints: dict[str, Any] = {}
    for name, spec in cfg["checkpoints"].items():
        if not spec.get("method"):
            continue
        path = checkpoints_root / spec["filename"]
        state = _safe_path_state(path)
        expected_size = int(spec.get("size_bytes", 0))
        state["expected_size_bytes"] = expected_size
        state["matches_expected_size"] = state.get("size_bytes") == expected_size
        checkpoints[name] = state
    return {
        "checkpoints_root": _safe_path_state(checkpoints_root),
        "checkpoints": checkpoints,
        "datasets": {
            "smoke": _dataset_status("smoke"),
            "full": _dataset_status("full"),
        },
        "preprocess": {
            "smoke": _preprocess_status("smoke"),
            "full": _preprocess_status("full"),
        },
    }


def _run_probe(command: list[str], *, timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _package_versions(names: Iterable[str]) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_runtime_context() -> dict[str, Any]:
    env_allowlist = [
        "COLAB_GPU",
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "LATENTDRIVER_DEBUG_ROOT",
        "LATENTDRIVER_RESULTS_ROOT",
        "LATENTDRIVER_WAYMO_DATASET_ROOT",
        "PYTHONPATH",
        "TF_CPP_MIN_LOG_LEVEL",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "env": {key: os.environ.get(key) for key in env_allowlist if key in os.environ},
        "packages": _package_versions(["jax", "jaxlib", "torch", "waymo-waymax", "transformers", "pytorch-lightning"]),
        "git": {
            "status": _run_probe(["git", "status", "--short", "--branch"]),
            "head": _run_probe(["git", "rev-parse", "HEAD"]),
            "branch": _run_probe(["git", "branch", "--show-current"]),
            "remote": _run_probe(["git", "remote", "-v"]),
        },
        "gpu": {
            "nvidia_smi_L": _run_probe(["nvidia-smi", "-L"]),
        },
    }


def resolve_debug_root(debug_root: str | Path | None = None) -> Path:
    if debug_root:
        return Path(debug_root).expanduser()
    env_root = os.environ.get("LATENTDRIVER_DEBUG_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser()
    results_root = os.environ.get("LATENTDRIVER_RESULTS_ROOT", "").strip()
    if results_root:
        results = Path(results_root).expanduser()
        if results.name == "runs" and results.parent.name == "results":
            return results.parent.parent / "debug_runs"
        return results.parent / "debug_runs"
    return resolve_repo_relative("results/debug_runs")


def _create_debug_bundle(profile: str, debug_root: Path) -> dict[str, Path | str]:
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{tag}_{_slug(profile)}"
    bundle_dir = debug_root / run_id
    steps_dir = bundle_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "bundle_dir": bundle_dir,
        "steps_dir": steps_dir,
        "manifest_path": bundle_dir / "manifest.json",
        "runtime_context_path": bundle_dir / "runtime_context.json",
        "artifact_status_before_path": bundle_dir / "artifact_status_before.json",
        "artifact_status_after_path": bundle_dir / "artifact_status_after.json",
        "failure_summary_path": bundle_dir / "failure_summary.json",
    }


def _tail_text(value: str, *, max_lines: int = 80, max_chars: int = 8000) -> str:
    stripped = value.strip()
    if not stripped:
        return "<empty>"
    lines = stripped.splitlines()[-max_lines:]
    tail = "\n".join(lines)
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _tail_file(path: Path) -> str:
    try:
        return _tail_text(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return f"<{type(exc).__name__}: {exc}>"


def _pipe_to_log(pipe: TextIO, target: TextIO, stream: TextIO) -> None:
    try:
        for line in iter(pipe.readline, ""):
            target.write(line)
            target.flush()
            stream.write(line)
            stream.flush()
    finally:
        pipe.close()


def run_step(step: RunnerStep, *, index: int, cwd: Path, steps_dir: Path, env: Mapping[str, str]) -> dict[str, Any]:
    safe_name = f"{index:02d}_{_slug(step.name)}"
    stdout_path = steps_dir / f"{safe_name}.stdout.log"
    stderr_path = steps_dir / f"{safe_name}.stderr.log"
    started = time.monotonic()
    print(f"\n[canary] step {index}: {step.name}")
    print(f"[canary] $ {shlex.join(step.command)}")
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.Popen(
            list(step.command),
            cwd=cwd,
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        stdout_thread = threading.Thread(target=_pipe_to_log, args=(proc.stdout, stdout_handle, sys.stdout), daemon=True)
        stderr_thread = threading.Thread(target=_pipe_to_log, args=(proc.stderr, stderr_handle, sys.stderr), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()
    duration = time.monotonic() - started
    result = {
        **step.as_dict(),
        "index": index,
        "cwd": str(cwd),
        "returncode": returncode,
        "duration_seconds": round(duration, 3),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": _tail_file(stdout_path),
        "stderr_tail": _tail_file(stderr_path),
    }
    print(f"[canary] step {index} returncode={returncode} duration={duration:.1f}s")
    return result


def run_profile(
    profile: str,
    *,
    model: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    vis: str = DEFAULT_VIS,
    install_runtime: bool = False,
    download_checkpoints: bool = False,
    debug_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    _validate_profile(profile)
    cwd = resolve_repo_relative("")
    bundle = _create_debug_bundle(profile, resolve_debug_root(debug_root))
    bundle_dir = Path(bundle["bundle_dir"])
    steps_dir = Path(bundle["steps_dir"])
    steps = profile_steps(
        profile,
        model=model,
        seed=seed,
        vis=vis,
        install_runtime=install_runtime,
        download_checkpoints=download_checkpoints,
    )
    request = {
        "profile": profile,
        "profile_description": PROFILE_DESCRIPTIONS[profile],
        "model": model,
        "seed": seed,
        "vis": vis,
        "install_runtime": install_runtime,
        "download_checkpoints": download_checkpoints,
        "dry_run": dry_run,
        "run_id": bundle["run_id"],
        "bundle_dir": str(bundle_dir),
        "steps": [step.as_dict() for step in steps],
    }
    write_json(bundle_dir / "request.json", request)
    runtime_context = collect_runtime_context()
    artifact_status_before = collect_artifact_status()
    write_json(Path(bundle["runtime_context_path"]), runtime_context)
    write_json(Path(bundle["artifact_status_before_path"]), artifact_status_before)

    manifest: dict[str, Any] = {
        **request,
        "status": "dry_run" if dry_run else "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime_context_path": str(bundle["runtime_context_path"]),
        "artifact_status_before_path": str(bundle["artifact_status_before_path"]),
        "artifact_status_after_path": str(bundle["artifact_status_after_path"]),
        "step_results": [],
    }
    write_json(Path(bundle["manifest_path"]), manifest)
    if dry_run:
        write_json(Path(bundle["artifact_status_after_path"]), artifact_status_before)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(Path(bundle["manifest_path"]), manifest)
        return manifest

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    failed_step: dict[str, Any] | None = None
    for index, step in enumerate(steps, start=1):
        result = run_step(step, index=index, cwd=cwd, steps_dir=steps_dir, env=env)
        manifest["step_results"].append(result)
        write_json(Path(bundle["manifest_path"]), manifest)
        if result["returncode"] != 0 and not step.continue_on_failure:
            failed_step = result
            break

    artifact_status_after = collect_artifact_status()
    write_json(Path(bundle["artifact_status_after_path"]), artifact_status_after)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["artifact_status_after_path"] = str(bundle["artifact_status_after_path"])
    if failed_step:
        manifest["status"] = "failed"
        failure_summary = {
            "profile": profile,
            "bundle_dir": str(bundle_dir),
            "failed_step": failed_step,
            "hint": "Pull this debug bundle with rclone and inspect manifest.json plus the step stdout/stderr logs.",
        }
        write_json(Path(bundle["failure_summary_path"]), failure_summary)
        manifest["failure_summary_path"] = str(bundle["failure_summary_path"])
    else:
        manifest["status"] = "succeeded"
    write_json(Path(bundle["manifest_path"]), manifest)
    return manifest
