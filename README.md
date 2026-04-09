# LatentDriver Waymax Experiments

Reproducible evaluation-first repository for **LatentDriver-family public checkpoints** on **Waymax**, with thin Colab notebooks, Drive-backed persistence, and standardized metric/visualization outputs.

## Goal

This repo is built for one narrow contract first:

- use only **publicly released checkpoints**,
- keep **Waymax settings identical across models**,
- evaluate under both **reactive** (`idm`) and **non-reactive** (`expert`) NPC settings,
- reproduce the paper’s reported metrics as closely as possible,
- add a small but meaningful extension: **seed/scenario consistency diagnostics** and visualization under the same evaluation stack.

## Public Checkpoints Covered

The upstream LatentDriver release publicly exposes:

- `latentdriver_t2_j3`
- `latentdriver_t2_j4`
- `plant`
- `easychauffeur_ppo`
- `pretrained_bert` (for training, not needed for evaluation-only runs)

Source: [Sephirex-x/LatentDriver on Hugging Face](https://huggingface.co/Sephirex-x/LatentDriver/tree/main)

## Repo Structure

- [`configs/baselines/latentdriver_waymax_eval.json`](./configs/baselines/latentdriver_waymax_eval.json): single source of truth for upstream pinning, public checkpoints, and evaluation tiers.
- [`docs/latentdriver_reading_note.md`](./docs/latentdriver_reading_note.md): paper summary, metrics contract, and released checkpoint inventory.
- [`docs/reproduction_plan.md`](./docs/reproduction_plan.md): exact scope for no-training replication and extension.
- [`patches/latentdriver_eval_contract.patch`](./patches/latentdriver_eval_contract.patch): deterministic patch layer applied to the upstream fork to enable bounded smoke runs, machine-readable metrics, and controlled vis output.
- [`scripts/bootstrap_upstream.py`](./scripts/bootstrap_upstream.py): clone your LatentDriver fork at a pinned commit and apply the local patch layer.
- [`scripts/download_checkpoints.py`](./scripts/download_checkpoints.py): fetch released checkpoints from Hugging Face.
- [`scripts/prepare_smoke_subset.py`](./scripts/prepare_smoke_subset.py): build a one-shard validation smoke subset from raw WOMD validation TFRecords.
- [`scripts/preprocess_validation_only.py`](./scripts/preprocess_validation_only.py): run validation-only preprocessing for smoke or full validation.
- [`scripts/run_waymax_eval.py`](./scripts/run_waymax_eval.py): run a standardized Waymax evaluation for one released checkpoint.
- [`scripts/run_smoke_eval.py`](./scripts/run_smoke_eval.py): quick smoke evaluation on the one-shard subset.
- [`scripts/run_public_suite.py`](./scripts/run_public_suite.py): evaluate all released checkpoints under one standardized tier and write a suite summary.
- [`scripts/run_visualization.py`](./scripts/run_visualization.py): run one visualization job and capture generated MP4/PDF artifacts.
- [`notebooks/`](./notebooks): Colab notebooks for assets, preprocessing, public-eval suite, and visualization.

## Evaluation Contract

We standardize the following across models:

- same validation split or smoke subset,
- same preprocessed map/route cache,
- same intention-label cache,
- same `npc_policy_type` (`idm` for reactive, `expert` for non-reactive),
- same `batch_dims`,
- same `run.seed`,
- same patched upstream commit,
- same output metric schema.

Primary metrics:

- `mAR[75:95]`
- `AR[75:95]`
- `OR`
- `CR`
- `PR`

These are recovered from the upstream metric object and written as machine-readable JSON by the local patch layer.

## Quickstart

### 1. Bootstrap the upstream fork
```bash
python3 scripts/bootstrap_upstream.py
```

### 2. Download released checkpoints
```bash
python3 scripts/download_checkpoints.py --evaluation-only
```

### 3. Prepare a one-shard validation smoke subset
```bash
export LATENTDRIVER_WAYMO_DATASET_ROOT=/path/to/waymo_dataset_root
python3 scripts/prepare_smoke_subset.py
```

### 4. Preprocess validation artifacts
```bash
python3 scripts/preprocess_validation_only.py --mode smoke
# later, for full reproduction:
python3 scripts/preprocess_validation_only.py --mode full
```

### 5. Dry-run one evaluation
```bash
python3 scripts/run_waymax_eval.py --model latentdriver_t2_j3 --tier smoke_reactive --dry-run
```

### 6. Run a smoke evaluation
```bash
python3 scripts/run_smoke_eval.py
```

### 7. Run the public evaluation suite
```bash
python3 scripts/run_public_suite.py --tier full_reactive
python3 scripts/run_public_suite.py --tier full_non_reactive
```

### 8. Generate visualization artifacts
```bash
python3 scripts/run_visualization.py --model latentdriver_t2_j3 --tier smoke_reactive --vis video
```

## Colab Path

- assets + checkpoints: [`notebooks/latentdriver_assets_colab.ipynb`](./notebooks/latentdriver_assets_colab.ipynb)
- validation preprocessing: [`notebooks/latentdriver_preprocess_val_colab.ipynb`](./notebooks/latentdriver_preprocess_val_colab.ipynb)
- public checkpoint evaluation: [`notebooks/latentdriver_public_eval_colab.ipynb`](./notebooks/latentdriver_public_eval_colab.ipynb)
- visualization: [`notebooks/latentdriver_visualize_colab.ipynb`](./notebooks/latentdriver_visualize_colab.ipynb)

## Important Boundary

This repo is **evaluation-only first**. It does **not** train LatentDriver or PlanT. The first milestone is:

- reproduce runnable evaluation for the released checkpoints,
- capture metrics and visualization under one standardized Waymax contract,
- then add consistency diagnostics on top.

## Patch Boundary

The local patch at [`patches/latentdriver_eval_contract.patch`](./patches/latentdriver_eval_contract.patch) is intentionally narrow. It adds only:

- `run.max_batches` for bounded smoke runs,
- `run.metrics_json_path` for machine-readable metrics,
- `run.vis_output_dir` for controlled visualization output,
- `run.seed` for explicit evaluation seeding,
- CPU-safe checkpoint loading and device fallback.
