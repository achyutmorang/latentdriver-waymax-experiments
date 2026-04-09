#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.evaluation import run_eval


if __name__ == "__main__":
    payload = run_eval(model="latentdriver_t2_j3", tier="smoke_reactive", vis=False, dry_run=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
