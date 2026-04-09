#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config
from latentdriver_waymax_experiments.evaluation import _validation_inputs
from latentdriver_waymax_experiments.upstream import ensure_upstream_exists


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only preprocessing for smoke or full validation.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    upstream_dir = ensure_upstream_exists()
    inputs = _validation_inputs(args.mode)
    batch_dims = cfg["validation"][args.mode]["preprocess_batch_dims"]
    cmd = [
        sys.executable,
        "src/preprocess/preprocess_data.py",
        f"++batch_dims=[{batch_dims[0]},{batch_dims[1]}]",
        f"++waymax_conf.path={inputs['waymo_path']}",
        "++waymax_conf.drop_remainder=False",
        f"++data_conf.path_to_processed_map_route={inputs['preprocess_path']}",
        f"++metric_conf.intention_label_path={inputs['intention_path']}",
    ]
    payload = {
        "mode": args.mode,
        "command": cmd,
        "waymo_path": str(inputs["waymo_path"]),
        "preprocess_path": str(inputs["preprocess_path"]),
        "intention_path": str(inputs["intention_path"]),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    proc = subprocess.run(cmd, cwd=upstream_dir, check=False, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
