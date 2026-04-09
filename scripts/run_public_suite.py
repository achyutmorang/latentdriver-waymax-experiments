#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.config import load_config
from latentdriver_waymax_experiments.evaluation import run_public_suite


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Run all released public checkpoints under one standardized tier.")
    parser.add_argument("--tier", choices=list(cfg["evaluation"]["tiers"].keys()), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = run_public_suite(tier=args.tier, seed=args.seed, dry_run=args.dry_run)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
