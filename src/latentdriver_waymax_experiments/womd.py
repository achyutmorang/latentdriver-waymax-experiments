from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

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


def first_sharded_tfrecord_path(dataset_uri: str) -> Path | None:
    if is_gcs_uri(dataset_uri):
        return None
    match = _SHARDED_PATTERN_RE.match(dataset_uri)
    if not match:
        return None
    prefix = match.group("prefix")
    count = int(match.group("count"))
    return Path(f"{prefix}-00000-of-{count:05d}")


def local_dataset_uri_exists(dataset_uri: str) -> bool:
    if is_gcs_uri(dataset_uri):
        return True
    sharded = first_sharded_tfrecord_path(dataset_uri)
    if sharded is not None:
        return sharded.exists()
    return Path(dataset_uri).expanduser().exists()


def validation_shard_uri(root: str, shard_index: int) -> str:
    filename = f"validation_tfexample.tfrecord-{shard_index:05d}-of-00150"
    relative = f"{WOMD_VERSION}/uncompressed/tf_example/validation/{filename}"
    return resolve_dataset_uri(root, relative)
