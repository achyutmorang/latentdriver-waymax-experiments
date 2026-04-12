#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.evaluation import inspect_eval_inputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight LatentDriver Waymax evaluation inputs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument(
        "--verify-remote-read",
        action="store_true",
        help="For gs:// WOMD paths, verify TensorFlow can read the first shard with runtime credentials.",
    )
    args = parser.parse_args()

    payload = inspect_eval_inputs(
        model=args.model,
        tier=args.tier,
        verify_remote_reads=args.verify_remote_read,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
