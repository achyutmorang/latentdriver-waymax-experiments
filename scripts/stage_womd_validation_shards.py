#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.womd import (  # noqa: E402
    WOMD_VERSION,
    copy_gcs_to_local,
    is_gcs_uri,
    validation_shard_uri,
)

VALIDATION_SHARD_COUNT = 150


def _target_for(staging_root: Path, source_uri: str) -> Path:
    return (
        staging_root
        / WOMD_VERSION
        / "uncompressed"
        / "tf_example"
        / "validation"
        / Path(source_uri).name
    )


def _stage_one(*, gcs_root: str, staging_root: Path, shard_index: int, force: bool) -> dict[str, Any]:
    source = validation_shard_uri(gcs_root, shard_index)
    target = _target_for(staging_root, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not force:
        return {
            "shard_index": shard_index,
            "source": source,
            "target": str(target),
            "status": "skipped_existing",
            "size_bytes": target.stat().st_size,
        }

    tmp = target.with_name(f".{target.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    if force and target.exists():
        target.unlink()

    try:
        transfer = copy_gcs_to_local(source, tmp)
        if not tmp.exists() or tmp.stat().st_size <= 0:
            raise RuntimeError(f"GCS copy did not produce a non-empty shard: {tmp}")
        tmp.replace(target)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    print(f"[stage-womd-full] staged shard {shard_index:05d} -> {target}", file=sys.stderr)
    return {
        "shard_index": shard_index,
        "source": source,
        "target": str(target),
        "status": "copied",
        "size_bytes": target.stat().st_size,
        "transfer": transfer,
    }


def _shard_range(start_index: int, count: int | None, stop_index: int | None) -> range:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    stop = VALIDATION_SHARD_COUNT if stop_index is None else stop_index
    if stop <= start_index:
        raise ValueError("--stop-index must be greater than --start-index")
    if stop > VALIDATION_SHARD_COUNT:
        raise ValueError(f"--stop-index must be <= {VALIDATION_SHARD_COUNT}")
    if count is None:
        return range(start_index, stop)
    if count <= 0:
        raise ValueError("--count must be positive")
    if count > stop - start_index:
        raise ValueError("--count exceeds the selected shard interval")
    return range(start_index, start_index + count)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumably stage WOMD validation TFRecord shards from GCS into a local Drive-backed tree."
    )
    parser.add_argument("--gcs-root", required=True, help="GCS root, for example gs://waymo_open_dataset_motion_v_1_1_0")
    parser.add_argument("--staging-root", required=True, help="Local root that becomes LATENTDRIVER_WAYMO_DATASET_ROOT")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, help="Number of shards to stage. Defaults to the selected interval.")
    parser.add_argument("--stop-index", type=int, help="Exclusive upper shard bound; default is 150.")
    parser.add_argument("--force", action="store_true", help="Re-copy existing non-empty shards.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gcs_root = args.gcs_root.rstrip("/")
    if not is_gcs_uri(gcs_root):
        raise ValueError("--gcs-root must be a gs:// URI")
    staging_root = Path(args.staging_root).expanduser()
    shard_range = _shard_range(args.start_index, args.count, args.stop_index)

    payload: dict[str, Any] = {
        "gcs_root": gcs_root,
        "staging_root": str(staging_root),
        "start_index": args.start_index,
        "count": len(shard_range),
        "stop_index": args.stop_index or VALIDATION_SHARD_COUNT,
        "force": args.force,
        "dry_run": args.dry_run,
        "expected_dataset_root": str(staging_root),
        "expected_dataset_pattern": str(
            staging_root
            / WOMD_VERSION
            / "uncompressed"
            / "tf_example"
            / "validation"
            / "validation_tfexample.tfrecord@150"
        ),
        "shards": [],
        "failures": [],
    }

    if args.dry_run:
        payload["planned_shard_indices"] = list(shard_range)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for shard_index in shard_range:
        try:
            payload["shards"].append(
                _stage_one(gcs_root=gcs_root, staging_root=staging_root, shard_index=shard_index, force=args.force)
            )
        except Exception as exc:
            failure = {
                "shard_index": shard_index,
                "source": validation_shard_uri(gcs_root, shard_index),
                "error": f"{type(exc).__name__}: {exc}",
            }
            payload["failures"].append(failure)
            break

    payload["copied"] = sum(1 for shard in payload["shards"] if shard["status"] == "copied")
    payload["skipped_existing"] = sum(1 for shard in payload["shards"] if shard["status"] == "skipped_existing")
    payload["complete"] = not payload["failures"] and len(payload["shards"]) == len(shard_range)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
