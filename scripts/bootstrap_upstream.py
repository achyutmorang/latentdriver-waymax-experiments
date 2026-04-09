#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.upstream import clone_and_patch_upstream


if __name__ == "__main__":
    print(json.dumps(clone_and_patch_upstream(), indent=2, sort_keys=True))
