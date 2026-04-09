#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config
from latentdriver_waymax_experiments.evaluation import run_eval


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Run one visualization job for a released checkpoint.")
    parser.add_argument("--model", choices=list(cfg["checkpoints"].keys()), default="latentdriver_t2_j3")
    parser.add_argument("--tier", choices=list(cfg["evaluation"]["tiers"].keys()), default="smoke_reactive")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--vis", choices=["image", "video"], default="video")
    args = parser.parse_args()
    payload = run_eval(model=args.model, tier=args.tier, seed=args.seed, vis=args.vis, dry_run=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
