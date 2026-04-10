#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.visual_compare import compare_latest_visualizations, create_side_by_side_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a side-by-side MP4 from two Waymax visualization artifacts.")
    parser.add_argument("--results-root", type=Path, help="Results runs root. Defaults to the project configured results root.")
    parser.add_argument("--left-model", default="latentdriver_t2_j3")
    parser.add_argument("--right-model", default="plant")
    parser.add_argument("--tier", default="smoke_reactive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--left-video", type=Path, help="Explicit left video path. Skips run discovery for the left side.")
    parser.add_argument("--right-video", type=Path, help="Explicit right video path. Skips run discovery for the right side.")
    parser.add_argument("--output", type=Path, help="Output MP4 path. Defaults to <results-root>/comparisons/...")
    parser.add_argument("--height", type=int, default=720, help="Output panel height before horizontal stacking.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if bool(args.left_video) != bool(args.right_video):
        parser.error("--left-video and --right-video must be provided together")

    if args.left_video and args.right_video:
        if args.output is None:
            parser.error("--output is required when using explicit video paths")
        payload = {
            "left": {"media_path": str(args.left_video)},
            "right": {"media_path": str(args.right_video)},
            "comparison": create_side_by_side_video(
                left=args.left_video,
                right=args.right_video,
                output=args.output,
                height=args.height,
                dry_run=args.dry_run,
            ),
        }
    else:
        payload = compare_latest_visualizations(
            root=args.results_root,
            left_model=args.left_model,
            right_model=args.right_model,
            tier=args.tier,
            seed=args.seed,
            output=args.output,
            height=args.height,
            dry_run=args.dry_run,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
