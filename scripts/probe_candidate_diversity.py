#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.artifacts import write_json  # noqa: E402
from latentdriver_waymax_experiments.candidate_diversity import probe_candidate_diversity_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect whether configured planners expose rerankable candidate diversity in the current repository wiring."
        )
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model name to probe. Repeat to inspect multiple models. Defaults to latentdriver_t2_j3 and plant.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the probe payload as JSON in addition to stdout.",
    )
    args = parser.parse_args()

    models = args.models or ["latentdriver_t2_j3", "plant"]
    payload = probe_candidate_diversity_suite(models)
    if args.json_output:
        write_json(args.json_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
