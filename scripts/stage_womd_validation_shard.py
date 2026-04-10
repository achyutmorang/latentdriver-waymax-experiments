#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.womd import WOMD_VERSION, copy_gcs_to_local, is_gcs_uri, validation_shard_uri


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage one WOMD validation shard from GCS into a local directory tree.")
    parser.add_argument("--gcs-root", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not is_gcs_uri(args.gcs_root):
        raise ValueError("--gcs-root must be a gs:// URI")

    source = validation_shard_uri(args.gcs_root, args.shard_index)
    target_root = Path(args.staging_root).expanduser() / WOMD_VERSION / "uncompressed" / "tf_example" / "validation"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / Path(source).name

    payload = {
        "gcs_root": args.gcs_root.rstrip("/"),
        "shard_index": args.shard_index,
        "source": source,
        "staging_root": str(Path(args.staging_root).expanduser()),
        "target": str(target),
        "skipped_existing": target.exists() and not args.force,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if target.exists() and args.force:
        target.unlink()
    transfer = None
    if not target.exists():
        transfer = copy_gcs_to_local(source, target)
        print("[stage-womd] $", " ".join(transfer["command"]))

    payload["transfer"] = transfer

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
