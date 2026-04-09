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
    parser = argparse.ArgumentParser(description="Run one standardized Waymax evaluation for a released LatentDriver-family checkpoint.")
    parser.add_argument("--model", choices=list(cfg["checkpoints"].keys()), required=True)
    parser.add_argument("--tier", choices=list(cfg["evaluation"]["tiers"].keys()), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--vis", choices=["false", "image", "video"], default="false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    vis = False if args.vis == "false" else args.vis
    payload = run_eval(model=args.model, tier=args.tier, seed=args.seed, vis=vis, dry_run=args.dry_run)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
