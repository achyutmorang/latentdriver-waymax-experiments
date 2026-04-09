from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from .config import load_config, resolve_repo_relative


def upstream_paths() -> Dict[str, Path]:
    cfg = load_config()
    return {
        "repo_dir": resolve_repo_relative(cfg["upstream"]["repo_dir"]),
        "patch_path": resolve_repo_relative(cfg["upstream"]["patch_path"]),
        "lock_root": resolve_repo_relative(cfg["assets"]["lock_root"]),
    }


def clone_and_patch_upstream() -> Dict[str, Any]:
    cfg = load_config()
    paths = upstream_paths()
    repo_dir = paths["repo_dir"]
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", cfg["upstream"]["fork_repo_url"], str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", cfg["upstream"]["pinned_commit"]], check=True)
    patch_path = paths["patch_path"]
    check = subprocess.run(["git", "-C", str(repo_dir), "apply", "--check", str(patch_path)], capture_output=True, text=True)
    reverse = subprocess.run(["git", "-C", str(repo_dir), "apply", "--reverse", "--check", str(patch_path)], capture_output=True, text=True)
    if check.returncode == 0:
        subprocess.run(["git", "-C", str(repo_dir), "apply", str(patch_path)], check=True)
        patch_state = "applied"
    elif reverse.returncode == 0:
        patch_state = "already_applied"
    else:
        raise RuntimeError(f"Unable to apply patch {patch_path}: {check.stderr or reverse.stderr}")

    lock_root = paths["lock_root"]
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / "upstream_bootstrap.json"
    payload = {
        "repo_dir": str(repo_dir),
        "fork_repo_url": cfg["upstream"]["fork_repo_url"],
        "upstream_repo_url": cfg["upstream"]["upstream_repo_url"],
        "pinned_commit": cfg["upstream"]["pinned_commit"],
        "patch_path": str(patch_path),
        "patch_state": patch_state,
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def ensure_upstream_exists() -> Path:
    repo_dir = upstream_paths()["repo_dir"]
    if not repo_dir.exists():
        raise FileNotFoundError(f"Upstream repo missing: {repo_dir}. Run scripts/bootstrap_upstream.py first.")
    return repo_dir
