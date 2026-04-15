#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.metadata_join import (  # noqa: E402
    check_reasoning_causal_join,
    load_scenario_subset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether WOMD-Reasoning and CausalAgents can be joined on a scenario subset and whether their agent IDs overlap."
        )
    )
    parser.add_argument("--reasoning-path", type=Path, required=True, help="Path to WOMD-Reasoning JSON/JSONL file or directory.")
    parser.add_argument("--causal-path", type=Path, required=True, help="Path to CausalAgents JSON/JSONL file or directory.")
    parser.add_argument("--scenario-ids-file", type=Path, help="Optional newline-delimited scenario ID subset file.")
    parser.add_argument(
        "--preprocess-map-dir",
        type=Path,
        help="Optional map directory containing `<scenario_id>.npy` files; stems are used as the subset.",
    )
    parser.add_argument("--limit", type=int, help="Optional limit applied after loading the subset IDs.")
    parser.add_argument(
        "--causal-policy",
        choices=("union", "majority"),
        default="union",
        help="How to merge multiple CausalAgents labeler sets for the same scenario.",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional directory for summary JSON and joined JSONL outputs.")
    args = parser.parse_args()

    scenario_ids = load_scenario_subset(
        scenario_ids_file=args.scenario_ids_file,
        preprocess_map_dir=args.preprocess_map_dir,
        limit=args.limit,
    )
    payload = check_reasoning_causal_join(
        reasoning_path=args.reasoning_path,
        causal_path=args.causal_path,
        scenario_ids=scenario_ids,
        causal_policy=args.causal_policy,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
