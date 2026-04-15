#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.womd import (  # noqa: E402
    copy_gcs_to_local,
    is_gcs_uri,
    sharded_tfrecord_parts,
    sharded_tfrecord_uri,
)


def _parse_shard_list(raw: str) -> list[int]:
    if not raw.strip():
        raise ValueError("--source-shards must be non-empty")
    values: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("--source-shards must contain at least one shard index")
    if len(set(values)) != len(values):
        raise ValueError("--source-shards must not contain duplicates")
    if any(index < 0 for index in values):
        raise ValueError("--source-shards must be non-negative")
    return values


def _resolve_source_uri(args: argparse.Namespace) -> str:
    if args.source_uri:
        return args.source_uri.strip()
    if args.source_uri_env:
        import os

        value = os.environ.get(args.source_uri_env, "").strip()
        if value:
            return value
        raise ValueError(f"Environment variable {args.source_uri_env} is required but not set")
    raise ValueError("Either --source-uri or --source-uri-env is required")


def _copy_uri_to_local(source_uri: str, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_gcs_uri(source_uri):
        transfer = copy_gcs_to_local(source_uri, target)
    else:
        source_path = Path(source_uri).expanduser()
        shutil.copy2(source_path, target)
        transfer = {
            "cli": "local-copy",
            "command": ["cp", str(source_path), str(target)],
            "stdout": "",
            "stderr": "",
            "target": str(target),
        }
    if not target.exists() or target.stat().st_size <= 0:
        raise RuntimeError(f"Subset staging did not produce a non-empty shard: {target}")
    return transfer


def _manifest_path(target_uri: str) -> Path:
    target_parts = sharded_tfrecord_parts(target_uri)
    if target_parts is None:
        raise ValueError("--target-uri must be a sharded dataset URI ending with @N")
    target_prefix, _ = target_parts
    return Path(f"{target_prefix}.stage_manifest.json").expanduser()


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _existing_target_shards(target_uri: str) -> list[Path]:
    target_parts = sharded_tfrecord_parts(target_uri)
    if target_parts is None:
        raise ValueError("--target-uri must be a sharded dataset URI ending with @N")
    _, target_count = target_parts
    existing: list[Path] = []
    for target_index in range(target_count):
        target = Path(sharded_tfrecord_uri(target_uri, target_index) or "").expanduser()
        if target.exists() and target.stat().st_size > 0:
            existing.append(target)
    return existing


def _validate_existing_manifest(
    *,
    source_uri: str,
    source_indices: list[int],
    target_uri: str,
    force: bool,
) -> dict[str, Any] | None:
    if force:
        return None
    manifest_path = _manifest_path(target_uri)
    manifest = _load_manifest(manifest_path)
    existing = _existing_target_shards(target_uri)
    if not existing:
        return None
    if manifest is None:
        raise RuntimeError(
            "Existing staged pilot shards were found without a staging manifest. "
            "This can silently reuse data from the wrong source split. "
            f"Remove {Path(target_uri).expanduser().parent} or rerun with --force after confirming the source URI."
        )
    expected = {
        "source_uri": source_uri,
        "target_uri": target_uri,
        "source_shards": source_indices,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Existing staged pilot shards do not match the requested source/subset. "
            f"manifest_path={manifest_path} mismatches={json.dumps(mismatches, sort_keys=True)}. "
            "Rerun with --force only if you intend to overwrite the staged pilot dataset."
        )
    return manifest


def _verify_tfrecord_feature(*, target_uri: str, required_feature: str) -> dict[str, Any]:
    first_uri = sharded_tfrecord_uri(target_uri, 0)
    if first_uri is None or is_gcs_uri(first_uri):
        raise ValueError("Feature verification requires a local sharded target URI.")
    first_path = Path(first_uri).expanduser()
    if not first_path.is_file() or first_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Cannot verify missing or empty first target shard: {first_path}")
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "--verify-required-feature requires TensorFlow to be installed in the runtime."
        ) from exc

    first_record = next(iter(tf.data.TFRecordDataset([str(first_path)]).take(1).as_numpy_iterator()), None)
    if first_record is None:
        raise RuntimeError(f"Cannot verify empty first target shard: {first_path}")
    example = tf.train.Example()
    example.ParseFromString(bytes(first_record))
    feature_keys = set(example.features.feature.keys())
    if required_feature not in feature_keys:
        sample_keys = sorted(feature_keys)[:20]
        raise RuntimeError(
            f"Staged data does not look like the expected WOMD tf.Example schema: "
            f"missing required feature {required_feature!r} in first shard {first_path}. "
            f"Sample parsed feature keys: {sample_keys}. "
            "This usually means the source URI points to scenario protos or another incompatible split."
        )
    return {
        "target_first_shard": str(first_path),
        "required_feature": required_feature,
        "feature_count": len(feature_keys),
        "ok": True,
    }


def _stage_one(*, source_uri: str, target_uri: str, force: bool) -> dict[str, Any]:
    target_path = Path(target_uri).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0 and not force:
        return {
            "source": source_uri,
            "target": str(target_path),
            "status": "skipped_existing",
            "size_bytes": target_path.stat().st_size,
        }

    tmp = target_path.with_name(f".{target_path.name}.tmp")
    tmp.unlink(missing_ok=True)
    if force and target_path.exists():
        target_path.unlink()
    try:
        transfer = _copy_uri_to_local(source_uri, tmp)
        tmp.replace(target_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"[stage-womd-subset] staged {source_uri} -> {target_path}", file=sys.stderr)
    return {
        "source": source_uri,
        "target": str(target_path),
        "status": "copied",
        "size_bytes": target_path.stat().st_size,
        "transfer": transfer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage a sparse subset of sharded WOMD TFRecord files into a dense local pilot dataset."
    )
    parser.add_argument("--source-uri", help="Sharded source dataset URI, for example gs://.../validation_interactive_tfexample.tfrecord@150")
    parser.add_argument("--source-uri-env", help="Environment variable containing the sharded source dataset URI.")
    parser.add_argument("--source-shards", required=True, help="Comma-separated sparse source shard indices, for example 0,15,30,...")
    parser.add_argument("--target-uri", required=True, help="Dense local sharded dataset URI, for example /.../validation_interactive_tfexample.tfrecord@10")
    parser.add_argument("--force", action="store_true", help="Re-copy existing non-empty target shards.")
    parser.add_argument(
        "--verify-required-feature",
        help=(
            "After staging, parse the first local TFRecord entry as tf.train.Example "
            "and require this feature key to exist. Use roadgraph_samples/xyz for the current Waymax parser."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_uri = _resolve_source_uri(args)
    source_parts = sharded_tfrecord_parts(source_uri)
    target_parts = sharded_tfrecord_parts(args.target_uri)
    if source_parts is None:
        raise ValueError("--source-uri must be a sharded dataset URI ending with @N")
    if target_parts is None:
        raise ValueError("--target-uri must be a sharded dataset URI ending with @N")

    source_indices = _parse_shard_list(args.source_shards)
    source_prefix, source_count = source_parts
    target_prefix, target_count = target_parts
    if len(source_indices) != target_count:
        raise ValueError(
            f"Target shard count {target_count} must equal the number of requested source shards {len(source_indices)}"
        )
    if any(index >= source_count for index in source_indices):
        raise ValueError(f"All source shard indices must be < {source_count}")

    existing_manifest = _validate_existing_manifest(
        source_uri=source_uri,
        source_indices=source_indices,
        target_uri=args.target_uri,
        force=args.force,
    )

    mappings = []
    for target_index, source_index in enumerate(source_indices):
        mappings.append(
            {
                "source_index": source_index,
                "target_index": target_index,
                "source_uri": sharded_tfrecord_uri(source_uri, source_index),
                "target_uri": sharded_tfrecord_uri(args.target_uri, target_index),
            }
        )

    payload: dict[str, Any] = {
        "source_uri": source_uri,
        "target_uri": args.target_uri,
        "source_shards": source_indices,
        "target_shard_count": target_count,
        "force": args.force,
        "dry_run": args.dry_run,
        "manifest_path": str(_manifest_path(args.target_uri)),
        "existing_manifest": existing_manifest,
        "mappings": mappings,
        "staged": [],
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for mapping in mappings:
        payload["staged"].append(
            _stage_one(
                source_uri=str(mapping["source_uri"]),
                target_uri=str(mapping["target_uri"]),
                force=args.force,
            )
        )
    payload["copied"] = sum(1 for item in payload["staged"] if item["status"] == "copied")
    payload["skipped_existing"] = sum(1 for item in payload["staged"] if item["status"] == "skipped_existing")
    payload["complete"] = len(payload["staged"]) == len(mappings)
    if args.verify_required_feature:
        payload["schema_probe"] = _verify_tfrecord_feature(
            target_uri=args.target_uri,
            required_feature=args.verify_required_feature,
        )
    payload["staged_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = _manifest_path(args.target_uri)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
