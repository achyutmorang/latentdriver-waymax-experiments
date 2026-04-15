#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
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
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
