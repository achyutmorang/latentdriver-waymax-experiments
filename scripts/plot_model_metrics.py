#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.metric_plots import DEFAULT_METRICS, generate_metric_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static plots comparing completed Waymax eval metrics across models.")
    parser.add_argument("--results-root", type=Path, help="Results runs root. Defaults to the configured project results root.")
    parser.add_argument("--tier", default="smoke_reactive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-seeds", action="store_true", help="Do not filter by seed.")
    parser.add_argument("--models", nargs="+", help="Model names to include. Defaults to all evaluation checkpoint models.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS), help="Metric keys to plot.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = generate_metric_comparison(
        root=args.results_root,
        tier=args.tier,
        seed=None if args.all_seeds else args.seed,
        models=args.models,
        metrics=args.metrics,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
