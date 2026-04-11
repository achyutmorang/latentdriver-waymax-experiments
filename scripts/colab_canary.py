#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.colab_runner import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_VIS,
    available_profiles,
    run_profile,
    should_install_runtime_by_default,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Colab-first LatentDriver/Waymax profiles and write debug bundles.")
    parser.add_argument("--profile", choices=list(available_profiles()), help="Runner profile to execute.")
    parser.add_argument("--list-profiles", action="store_true", help="Print available profiles and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model for single-model profiles.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--vis", choices=["image", "video"], default=DEFAULT_VIS)
    parser.add_argument("--debug-root", type=Path, help="Override debug bundle root. Defaults to Drive sibling of results/runs when bound.")
    parser.add_argument("--install-runtime", action="store_true", help="Run scripts/setup_colab_runtime.py before the selected profile.")
    parser.add_argument("--auto-install-runtime", action="store_true", help="Install runtime for profiles that need it by default.")
    parser.add_argument("--download-checkpoints", action="store_true", help="Run checkpoint download/verification before the selected profile.")
    parser.add_argument("--dry-run", action="store_true", help="Create the debug bundle and planned steps without executing profile commands.")
    args = parser.parse_args()

    if args.list_profiles:
        print(json.dumps(available_profiles(), indent=2, sort_keys=True))
        return 0
    if not args.profile:
        parser.error("--profile is required unless --list-profiles is used")

    install_runtime = args.install_runtime or (
        args.auto_install_runtime and should_install_runtime_by_default(args.profile)
    )
    payload = run_profile(
        args.profile,
        model=args.model,
        seed=args.seed,
        vis=args.vis,
        install_runtime=install_runtime,
        download_checkpoints=args.download_checkpoints,
        debug_root=args.debug_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"succeeded", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
