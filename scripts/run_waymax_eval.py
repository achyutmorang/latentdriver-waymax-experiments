#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config
from latentdriver_waymax_experiments.evaluation import run_eval, run_eval_resumable


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Run one standardized Waymax evaluation for a released LatentDriver-family checkpoint.")
    parser.add_argument("--model", choices=list(cfg["checkpoints"].keys()), required=True)
    parser.add_argument("--tier", choices=list(cfg["evaluation"]["tiers"].keys()), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--vis", choices=["false", "image", "video"], default="false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resumable", action="store_true", help="Run evaluation shard-by-shard and resume completed shards.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior for resumable evaluation runs (rerun all shards).",
    )
    parser.add_argument("--max-shards", type=int, help="Limit resumable evaluation to the first N shards.")
    args = parser.parse_args()
    vis = False if args.vis == "false" else args.vis
    if args.resumable:
        payload = run_eval_resumable(
            model=args.model,
            tier=args.tier,
            seed=args.seed,
            vis=vis,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            max_shards=args.max_shards,
        )
    else:
        payload = run_eval(model=args.model, tier=args.tier, seed=args.seed, vis=vis, dry_run=args.dry_run)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run and not payload.get("ready", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
