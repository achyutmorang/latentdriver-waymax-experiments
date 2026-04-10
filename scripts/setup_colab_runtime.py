#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentdriver_waymax_experiments.upstream import (
    ensure_crdp_compat_source_patch,
    ensure_jax_tree_map_compat_source_patch,
    ensure_lightning_compat_source_patches,
    ensure_preprocess_multiprocessing_compat_source_patch,
    ensure_python312_compat_sitecustomize,
    ensure_upstream_exists,
)


WAYMAX_GIT_SPEC = "git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax"
JAX_GPU_PIN = "jax[cuda12]==0.6.0"
GPT2_OLD_IMPORT_BLOCK = """from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)"""
GPT2_NEW_IMPORT_BLOCK = """from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.modeling_utils import SequenceSummary
except Exception:
    class SequenceSummary(nn.Module):
        def __init__(self, config=None):
            super().__init__()
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states[:, -1]

try:
    from transformers.modeling_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
except Exception:
    from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
"""
SORT_VERT_FALLBACK_CODE = """import torch
from torch.autograd import Function

try:
    from src.ops.sort_vertices import sort_vertices as _sort_vertices_ext
except Exception:
    _sort_vertices_ext = None


def _sort_vertices_fallback(vertices, mask, num_valid):
    B, N, M, _ = vertices.shape
    out = torch.zeros((B, N, 9), dtype=torch.long, device=vertices.device)
    for b in range(B):
        for n in range(N):
            nv = int(num_valid[b, n].item())
            if nv <= 0:
                continue
            valid_idx = torch.nonzero(mask[b, n], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            valid_idx = valid_idx[:nv]
            pts = vertices[b, n, valid_idx]
            ang = torch.atan2(pts[:, 1], pts[:, 0])
            order = valid_idx[torch.argsort(ang)]
            k = min(int(order.numel()), 8)
            out[b, n, :k] = order[:k]
            out[b, n, k] = order[0]
            if k + 1 < 9:
                out[b, n, k + 1:] = order[k - 1]
    return out


class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        if _sort_vertices_ext is not None:
            idx = _sort_vertices_ext.sort_vertices_forward(vertices, mask, num_valid)
        else:
            idx = _sort_vertices_fallback(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout):
        return ()


sort_v = SortVertices.apply
"""


def _run(cmd: list[str], **kwargs) -> None:
    print("[latentdriver-setup] $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def _pip_install(*packages: str, extra_args: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def runtime_install_commands() -> list[list[str]]:
    return [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        _pip_install(WAYMAX_GIT_SPEC),
        _pip_install(JAX_GPU_PIN),
        _pip_install(
            "mediapy",
            "seaborn",
            "scikit-learn",
            "tqdm",
            "hydra-core==1.3.2",
            "omegaconf==2.3.0",
            "einops==0.8.0",
            "transformers==4.46.3",
            "huggingface_hub",
            "shapely==2.0.5",
            "gymnasium==0.29.1",
            "prettytable==3.10.2",
            "tensorboard",
            "absl-py>=1.4.0",
            "dm-tree>=0.1.8",
            "immutabledict>=2.2.3",
            "Pillow>=9.4.0",
        ),
    ]


def verify_jax_gpu_backend() -> dict[str, object]:
    probe = """import json, subprocess
report = {}
try:
    import jax
    import jaxlib
    devices = jax.devices()
    report["jax"] = jax.__version__
    report["jaxlib"] = jaxlib.__version__
    report["default_backend"] = jax.default_backend()
    report["devices"] = [
        {
            "repr": str(device),
            "platform": getattr(device, "platform", None),
            "device_kind": getattr(device, "device_kind", None),
        }
        for device in devices
    ]
except Exception as exc:
    report["probe_error"] = repr(exc)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise

try:
    completed = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
    report["nvidia_smi"] = {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
except FileNotFoundError:
    completed = None
    report["nvidia_smi"] = {"missing": True}

gpu_visible = bool(completed and completed.returncode == 0 and "GPU " in completed.stdout)
jax_has_gpu = report.get("default_backend") in {"gpu", "cuda"} or any(
    device.get("platform") in {"gpu", "cuda"} for device in report.get("devices", [])
)
report["gpu_visible"] = gpu_visible
report["jax_has_gpu"] = jax_has_gpu
print(json.dumps(report, indent=2, sort_keys=True))
if gpu_visible and not jax_has_gpu:
    raise SystemExit(1)
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print("[latentdriver-setup] jax backend probe:")
        print(completed.stdout.strip())
    if completed.returncode != 0:
        raise RuntimeError(
            "JAX is not using the visible NVIDIA GPU. "
            "The Colab runtime likely installed an incompatible CUDA JAX stack."
        )
    return {"stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def patch_gpt2_model(upstream_dir: Path) -> str:
    gpt2_model_path = upstream_dir / "src" / "policy" / "latentdriver" / "gpt2_model.py"
    text = gpt2_model_path.read_text(encoding="utf-8")
    if GPT2_NEW_IMPORT_BLOCK in text:
        return "already_patched"
    if GPT2_OLD_IMPORT_BLOCK not in text:
        return "not_found"
    gpt2_model_path.write_text(text.replace(GPT2_OLD_IMPORT_BLOCK, GPT2_NEW_IMPORT_BLOCK), encoding="utf-8")
    return "patched"


def patch_sort_vertices(upstream_dir: Path) -> str:
    sort_vert_path = upstream_dir / "src" / "ops" / "sort_vertices" / "sort_vert.py"
    text = sort_vert_path.read_text(encoding="utf-8")
    if text == SORT_VERT_FALLBACK_CODE:
        return "already_patched"
    sort_vert_path.write_text(SORT_VERT_FALLBACK_CODE, encoding="utf-8")
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a Colab runtime compatible with LatentDriver evaluation.")
    parser.add_argument("--editable-project", action="store_true")
    args = parser.parse_args()
    upstream_dir = ensure_upstream_exists()

    for cmd in runtime_install_commands():
        _run(cmd)

    gpt2_state = patch_gpt2_model(upstream_dir)
    print(f"[latentdriver-setup] gpt2_model patch: {gpt2_state}")
    sort_state = patch_sort_vertices(upstream_dir)
    print(f"[latentdriver-setup] sort_vertices patch: {sort_state}")
    compat_sitecustomize = ensure_python312_compat_sitecustomize(upstream_dir)
    print(f"[latentdriver-setup] sitecustomize patch: {compat_sitecustomize}")
    lightning_compat = ensure_lightning_compat_source_patches(upstream_dir)
    print(f"[latentdriver-setup] lightning compat patch: {lightning_compat}")
    crdp_compat = ensure_crdp_compat_source_patch(upstream_dir)
    print(f"[latentdriver-setup] crdp compat patch: {crdp_compat}")
    preprocess_multiprocessing_compat = ensure_preprocess_multiprocessing_compat_source_patch(upstream_dir)
    print(f"[latentdriver-setup] preprocess multiprocessing compat patch: {preprocess_multiprocessing_compat}")
    jax_tree_map_compat = ensure_jax_tree_map_compat_source_patch(upstream_dir)
    print(f"[latentdriver-setup] jax tree_map compat patch: {jax_tree_map_compat}")

    if args.editable_project:
        _run([sys.executable, "-m", "pip", "install", "-e", "."])

    verify_jax_gpu_backend()
    print("[latentdriver-setup] runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
