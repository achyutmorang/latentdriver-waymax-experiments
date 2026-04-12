from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Final

from .config import load_config

WOMD_VERSION: Final[str] = "waymo_open_dataset_motion_v_1_1_0"
_SHARDED_PATTERN_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<prefix>.+)@(?P<count>\d+)$")


def dataset_root_env_name() -> str:
    cfg = load_config()
    return str(cfg["assets"]["waymo_dataset_root_env"])


def is_gcs_uri(value: str) -> bool:
    return value.startswith("gs://")


def waymo_dataset_root_value() -> str:
    env_name = dataset_root_env_name()
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {env_name}")
    return value.rstrip("/")


def _strip_redundant_version_prefix(root: str, relative_path: str) -> str:
    normalized = relative_path.strip().lstrip("/")
    root_name = root.rstrip("/").split("/")[-1]
    prefix = f"{WOMD_VERSION}/"
    if normalized.startswith(prefix) and root_name == WOMD_VERSION:
        return normalized[len(prefix):]
    return normalized


def resolve_dataset_uri(root: str, relative_path: str) -> str:
    normalized_root = root.strip().rstrip("/")
    if not normalized_root:
        raise ValueError("Dataset root must be non-empty")
    normalized_relative = _strip_redundant_version_prefix(normalized_root, relative_path)
    if is_gcs_uri(normalized_root):
        return f"{normalized_root}/{normalized_relative}"
    return str(Path(normalized_root).expanduser() / normalized_relative)


def waymo_dataset_root() -> Path:
    root = waymo_dataset_root_value()
    if is_gcs_uri(root):
        raise EnvironmentError(
            "LATENTDRIVER_WAYMO_DATASET_ROOT points to a GCS URI; this command requires a local directory root."
        )
    return Path(root).expanduser()


def sharded_tfrecord_parts(dataset_uri: str) -> tuple[str, int] | None:
    match = _SHARDED_PATTERN_RE.match(dataset_uri)
    if not match:
        return None
    return match.group("prefix"), int(match.group("count"))


def sharded_tfrecord_uri(dataset_uri: str, shard_index: int) -> str | None:
    parts = sharded_tfrecord_parts(dataset_uri)
    if parts is None:
        return None
    prefix, count = parts
    if shard_index < 0 or shard_index >= count:
        raise ValueError(f"shard_index must be in [0, {count})")
    return f"{prefix}-{shard_index:05d}-of-{count:05d}"


def first_sharded_tfrecord_uri(dataset_uri: str) -> str | None:
    return sharded_tfrecord_uri(dataset_uri, 0)


def first_sharded_tfrecord_path(dataset_uri: str) -> Path | None:
    if is_gcs_uri(dataset_uri):
        return None
    first = first_sharded_tfrecord_uri(dataset_uri)
    if first is None:
        return None
    return Path(first)


def local_sharded_tfrecord_status(dataset_uri: str, *, max_missing: int = 10) -> dict[str, Any]:
    parts = sharded_tfrecord_parts(dataset_uri)
    if parts is None:
        path = Path(dataset_uri).expanduser()
        exists = path.exists()
        return {
            "uri": dataset_uri,
            "is_sharded": False,
            "is_gcs": False,
            "complete": exists,
            "exists": exists,
            "path": str(path),
        }
    if is_gcs_uri(dataset_uri):
        prefix, count = parts
        return {
            "uri": dataset_uri,
            "is_sharded": True,
            "is_gcs": True,
            "complete": None,
            "expected_shards": count,
            "first_shard_uri": f"{prefix}-00000-of-{count:05d}",
        }

    _, count = parts
    existing = 0
    missing: list[str] = []
    zero_byte: list[str] = []
    for index in range(count):
        shard = Path(sharded_tfrecord_uri(dataset_uri, index) or "").expanduser()
        try:
            stat = shard.stat()
        except FileNotFoundError:
            if len(missing) < max_missing:
                missing.append(str(shard))
            continue
        if stat.st_size <= 0:
            if len(zero_byte) < max_missing:
                zero_byte.append(str(shard))
            continue
        existing += 1
    return {
        "uri": dataset_uri,
        "is_sharded": True,
        "is_gcs": False,
        "complete": existing == count,
        "expected_shards": count,
        "existing_shards": existing,
        "missing_count": count - existing,
        "missing_examples": missing,
        "zero_byte_examples": zero_byte,
        "first_shard_uri": sharded_tfrecord_uri(dataset_uri, 0),
    }


def local_dataset_uri_exists(dataset_uri: str) -> bool:
    if is_gcs_uri(dataset_uri):
        return True
    sharded = first_sharded_tfrecord_path(dataset_uri)
    if sharded is not None:
        return sharded.exists()
    return Path(dataset_uri).expanduser().exists()


def local_dataset_uri_complete(dataset_uri: str) -> bool:
    if is_gcs_uri(dataset_uri):
        return True
    return bool(local_sharded_tfrecord_status(dataset_uri)["complete"])


def dataset_uri_status(dataset_uri: str) -> dict[str, Any]:
    if is_gcs_uri(dataset_uri):
        parts = sharded_tfrecord_parts(dataset_uri)
        status: dict[str, Any] = {
            "uri": dataset_uri,
            "is_gcs": True,
            "exists_or_remote": True,
            "complete": None,
        }
        if parts is not None:
            _, count = parts
            status["is_sharded"] = True
            status["expected_shards"] = count
            status["first_shard_uri"] = sharded_tfrecord_uri(dataset_uri, 0)
        else:
            status["is_sharded"] = False
        return status
    status = local_sharded_tfrecord_status(dataset_uri)
    status["exists_or_remote"] = bool(status["complete"])
    return status


def probe_tensorflow_dataset_uri(dataset_uri: str) -> dict[str, Any]:
    target = first_sharded_tfrecord_uri(dataset_uri) or dataset_uri
    result: dict[str, Any] = {
        "uri": dataset_uri,
        "target": target,
        "probe": "tensorflow.io.gfile.exists",
    }
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the runtime image
        result.update(
            {
                "ok": False,
                "error_kind": "tensorflow_import",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result
    try:
        exists = bool(tf.io.gfile.exists(target))
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error_kind": "tensorflow_gfile",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result
    result.update({"ok": exists, "exists": exists})
    if not exists:
        result["error_kind"] = "not_found_or_not_authorized"
        result["error"] = "TensorFlow could not see the target WOMD shard."
    return result


def validation_shard_uri(root: str, shard_index: int) -> str:
    filename = f"validation_tfexample.tfrecord-{shard_index:05d}-of-00150"
    relative = f"{WOMD_VERSION}/uncompressed/tf_example/validation/{filename}"
    return resolve_dataset_uri(root, relative)


def _gcs_command_candidates(operation: str, *args: str) -> list[tuple[str, list[str]]]:
    candidates: list[tuple[str, list[str]]] = []
    if shutil.which("gcloud"):
        candidates.append(("gcloud-storage", ["gcloud", "storage", operation, *args]))
    if shutil.which("gsutil"):
        candidates.append(("gsutil", ["gsutil", operation, *args]))
    if not candidates:
        raise EnvironmentError("Neither `gcloud` nor `gsutil` is available for GCS access.")
    return candidates


def _run_gcs_command(operation: str, *args: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for cli_name, command in _gcs_command_candidates(operation, *args):
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return {
                "cli": cli_name,
                "command": command,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        errors.append(
            {
                "cli": cli_name,
                "command": command,
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip(),
                "stdout": proc.stdout.strip(),
            }
        )
    raise RuntimeError(f"GCS {operation} failed across all supported CLIs: {errors}")


def probe_gcs_uri(uri: str) -> dict[str, Any]:
    result = _run_gcs_command("ls", uri)
    result["stdout_lines"] = [line for line in result["stdout"].splitlines() if line.strip()]
    return result


def copy_gcs_to_local(source: str, target: str | Path) -> dict[str, Any]:
    target_path = str(target)
    result = _run_gcs_command("cp", source, target_path)
    result["target"] = target_path
    return result
