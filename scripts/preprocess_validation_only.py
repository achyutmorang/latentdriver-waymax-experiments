#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config
from latentdriver_waymax_experiments.evaluation import _validation_inputs
from latentdriver_waymax_experiments.upstream import (
    ensure_lightning_compat_source_patches,
    ensure_python312_compat_sitecustomize,
    ensure_upstream_exists,
)


def build_preprocess_command(*, mode: str) -> list[str]:
    cfg = load_config()
    inputs = _validation_inputs(mode)
    batch_dims = cfg["validation"][mode]["preprocess_batch_dims"]
    return [
        sys.executable,
        "-m",
        "src.preprocess.preprocess_data",
        f"++batch_dims=[{batch_dims[0]},{batch_dims[1]}]",
        f"++waymax_conf.path={inputs['waymo_path']}",
        "++waymax_conf.drop_remainder=False",
        f"++data_conf.path_to_processed_map_route={inputs['preprocess_path']}",
        f"++metric_conf.intention_label_path={inputs['intention_path']}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only preprocessing for smoke or full validation.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    upstream_dir = ensure_upstream_exists()
    compat_sitecustomize = ensure_python312_compat_sitecustomize(upstream_dir)
    lightning_compat = ensure_lightning_compat_source_patches(upstream_dir)
    inputs = _validation_inputs(args.mode)
    cmd = build_preprocess_command(mode=args.mode)
    payload = {
        "mode": args.mode,
        "command": cmd,
        "waymo_path": str(inputs["waymo_path"]),
        "preprocess_path": str(inputs["preprocess_path"]),
        "intention_path": str(inputs["intention_path"]),
        "compat_sitecustomize": str(compat_sitecustomize),
        "lightning_compat": lightning_compat,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(upstream_dir) if not existing_pythonpath else f"{upstream_dir}{os.pathsep}{existing_pythonpath}"
    proc = subprocess.run(cmd, cwd=upstream_dir, check=False, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
