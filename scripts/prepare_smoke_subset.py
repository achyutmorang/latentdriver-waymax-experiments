#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config, resolve_repo_relative
from latentdriver_waymax_experiments.womd import validation_shard_uri, waymo_dataset_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a one-shard validation smoke subset compatible with LatentDriver/Waymax path conventions.")
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    raw_root = waymo_dataset_root()
    source = Path(validation_shard_uri(str(raw_root), args.shard_index))
    if not source.exists():
        raise FileNotFoundError(f"Validation shard not found: {source}")

    cfg = load_config()
    smoke_root = resolve_repo_relative(cfg["assets"]["smoke_root"])
    smoke_root.mkdir(parents=True, exist_ok=True)
    target = smoke_root / "validation_smoke.tfrecord-00000-of-00001"
    shutil.copy2(source, target)
    payload = {
        "source": str(source),
        "target": str(target),
        "dataset_pattern": str(smoke_root / cfg["validation"]["smoke"]["dataset_pattern"]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
