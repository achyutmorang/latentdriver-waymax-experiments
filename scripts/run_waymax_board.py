#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.artifacts import results_root
from latentdriver_waymax_experiments.wayboard import WaymaxBoard


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a NuBoard-inspired viewer for Waymax experiment runs.")
    parser.add_argument("--results-root", type=Path, default=results_root())
    parser.add_argument("--port", type=int, default=5007)
    args = parser.parse_args()
    WaymaxBoard(results_dir=args.results_root, port=args.port).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
