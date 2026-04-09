#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config, resolve_repo_relative


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download released LatentDriver-family checkpoints.")
    parser.add_argument("--model", choices=list(load_config()["checkpoints"].keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    root = resolve_repo_relative(cfg["assets"]["checkpoints_root"])
    if args.all:
        selected = list(cfg["checkpoints"].keys())
    elif args.evaluation_only:
        selected = [name for name, spec in cfg["checkpoints"].items() if spec["method"]]
    elif args.model:
        selected = [args.model]
    else:
        selected = ["latentdriver_t2_j3"]
    payload = {}
    for name in selected:
        spec = cfg["checkpoints"][name]
        dest = root / spec["filename"]
        if not dest.exists() or dest.stat().st_size != spec["size_bytes"]:
            _download(spec["url"], dest)
        payload[name] = {
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "expected_size_bytes": spec["size_bytes"],
            "matches_expected_size": dest.stat().st_size == spec["size_bytes"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
