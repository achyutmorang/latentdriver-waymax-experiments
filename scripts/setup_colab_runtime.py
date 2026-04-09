#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.upstream import ensure_upstream_exists


def _run(cmd: list[str], **kwargs) -> None:
    print("[latentdriver-setup] $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a Colab runtime compatible with LatentDriver evaluation.")
    parser.add_argument("--editable-project", action="store_true")
    args = parser.parse_args()
    upstream_dir = ensure_upstream_exists()

    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run([sys.executable, "-m", "pip", "install", "tensorflow==2.15.0"])
    _run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "jax==0.4.10",
        "jaxlib==0.4.10+cuda12.cudnn88",
        "-f",
        "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ])
    _run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ])
    _run([sys.executable, "-m", "pip", "install", "-r", str(upstream_dir / "requirements.txt")])
    build_env = dict(os.environ)
    build_env["CUDA_VISIBLE_DEVICES"] = ""
    _run([sys.executable, "setup.py", "install"], cwd=upstream_dir, env=build_env)
    if args.editable_project:
        _run([sys.executable, "-m", "pip", "install", "-e", "."])
    print("[latentdriver-setup] runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
