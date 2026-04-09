from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_relative(path: str) -> Path:
    return project_root() / path


def config_path() -> Path:
    return project_root() / "configs" / "baselines" / "latentdriver_waymax_eval.json"


def load_config() -> Dict[str, Any]:
    return json.loads(config_path().read_text(encoding="utf-8"))
