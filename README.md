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
- [`docs/inspiration_implementation_plan.md`](./docs/inspiration_implementation_plan.md): phased plan for extending this repo using ideas from Diffusion-Planner, Plan-R1, and RIFT without breaking the evaluation contract.
- [`docs/waymax_board.md`](./docs/waymax_board.md): NuBoard-inspired viewer design for Waymax run bundles and how it maps onto the repo's artifact schema.
- [`references/marl_finetuning_seed_references.md`](./references/marl_finetuning_seed_references.md): local paper bundle and curated reading list for MARL fine-tuning, reranking, and lightweight adaptation ideas.
- [`patches/latentdriver_eval_contract.patch`](./patches/latentdriver_eval_contract.patch): deterministic patch layer applied to the upstream fork to enable bounded smoke runs, machine-readable metrics, and controlled vis output.
- [`scripts/bootstrap_upstream.py`](./scripts/bootstrap_upstream.py): clone your LatentDriver fork at a pinned commit and apply the local patch layer.
- [`scripts/download_checkpoints.py`](./scripts/download_checkpoints.py): fetch released checkpoints from Hugging Face.
- [`scripts/prepare_smoke_subset.py`](./scripts/prepare_smoke_subset.py): build a one-shard validation smoke subset from raw WOMD validation TFRecords.
- [`scripts/stage_womd_validation_shard.py`](./scripts/stage_womd_validation_shard.py): copy one validation shard from authenticated WOMD GCS storage into a local Drive-backed staging root for smoke preprocessing.
- [`scripts/preprocess_validation_only.py`](./scripts/preprocess_validation_only.py): run validation-only preprocessing for smoke or full validation.
- [`scripts/colab_canary.py`](./scripts/colab_canary.py): CLI-first Colab runner that executes named profiles and writes Drive-backed debug bundles.
- [`scripts/run_waymax_eval.py`](./scripts/run_waymax_eval.py): run a standardized Waymax evaluation for one released checkpoint.
- [`scripts/run_smoke_eval.py`](./scripts/run_smoke_eval.py): quick smoke evaluation on the one-shard subset.
- [`scripts/run_public_suite.py`](./scripts/run_public_suite.py): evaluate all released checkpoints under one standardized tier and write a suite summary.
- [`scripts/plot_model_metrics.py`](./scripts/plot_model_metrics.py): generate static PNG/CSV/JSON plots comparing completed model metrics.
- [`scripts/run_visualization.py`](./scripts/run_visualization.py): run one visualization job and capture generated MP4/PDF artifacts.
- [`scripts/run_waymax_board.py`](./scripts/run_waymax_board.py): launch a NuBoard-inspired local Bokeh app for browsing Waymax run bundles, metrics, and visualization artifacts.
- [`notebooks/latentdriver_colab_runner.ipynb`](./notebooks/latentdriver_colab_runner.ipynb): recommended single Colab notebook for bootstrap, Drive binding, profile selection, and CLI execution.
- [`notebooks/`](./notebooks): legacy task-specific Colab notebooks are retained for reference, but the runner notebook should be the default workflow.

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

If your WOMD access is via authenticated GCS rather than a Drive-local copy, stage a single shard first:
```bash
python3 scripts/stage_womd_validation_shard.py \
  --gcs-root gs://waymo_open_dataset_motion_v_1_1_0 \
  --staging-root "$PWD/artifacts/assets/raw_womd" \
  --shard-index 0
export LATENTDRIVER_WAYMO_DATASET_ROOT="$PWD/artifacts/assets/raw_womd"
python3 scripts/prepare_smoke_subset.py --shard-index 0
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

### 9. Generate static metric comparison plots
```bash
python3 scripts/plot_model_metrics.py --tier smoke_reactive --seed 0
```

This writes:

- `model_metrics.png`: side-by-side metric bar plots across the latest completed run for each public evaluation checkpoint
- `model_metrics.csv`: tabular metric values for experiment notes
- `model_metrics.json`: machine-readable plot source data

Pass `--results-root /content/drive/MyDrive/waymax_research/latentdriver_waymax_experiments/results/runs` in Colab if you want to read directly from the Drive-backed run store.

### 10. Launch the local Waymax viewer
```bash
python3 scripts/run_waymax_board.py --results-root "$PWD/results/runs" --port 5007
```

This launches a small NuBoard-inspired Bokeh app with three tabs:

- `Overview`: completed runs and suite summaries
- `Metrics`: run-level metric plots and distributions
- `Artifacts`: run manifest, log tails, and embedded MP4/PDF/image outputs from `vis/`

## Colab Path

Use one notebook as the Colab terminal launcher:

- recommended runner: [`notebooks/latentdriver_colab_runner.ipynb`](./notebooks/latentdriver_colab_runner.ipynb)

The runner clones or fast-forwards `main`, mounts Drive, binds the persistent artifact layout, sets the WOMD GCS root, and delegates execution to:

```bash
python3 scripts/colab_canary.py --profile full-eval-dry-run --auto-install-runtime
```

Useful profiles:

- `full-preprocess-status`: verify full preprocessing paths without scanning the large Drive cache directories.
- `full-eval-dry-run`: validate full evaluation command construction and required inputs without launching simulation.
- `full-eval-reactive-single`: run one model on full reactive evaluation.
- `full-eval-reactive`: run all public checkpoints on full reactive evaluation.
- `full-eval-non-reactive`: run all public checkpoints on full non-reactive evaluation.
- `plot-full-reactive`: generate comparison plots after full reactive runs exist.

Debug bundles are written under the Drive-bound project root:

```text
/content/drive/MyDrive/waymax_research/latentdriver_waymax_experiments/debug_runs/<timestamp>_<profile>/
```

Each bundle contains `manifest.json`, runtime context, artifact status snapshots, and per-step stdout/stderr logs. Pull them locally with `rclone` instead of pasting tracebacks manually.

Legacy task-specific notebooks remain available for reference:

- assets + checkpoints: [`notebooks/latentdriver_assets_colab.ipynb`](./notebooks/latentdriver_assets_colab.ipynb)
- validation preprocessing: [`notebooks/latentdriver_preprocess_val_colab.ipynb`](./notebooks/latentdriver_preprocess_val_colab.ipynb)
- full validation evaluation dry-run: [`notebooks/latentdriver_full_eval_colab.ipynb`](./notebooks/latentdriver_full_eval_colab.ipynb)
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

## WaymaxBoard Boundary

[`scripts/run_waymax_board.py`](./scripts/run_waymax_board.py) is inspired by nuPlan's NuBoard, but it is intentionally narrower:

- it reads this repo's existing `run_manifest.json`, `metrics.json`, `suite_summary.json`, and `vis/` artifacts,
- it does **not** attempt to replay Waymax simulator state frame-by-frame from a nuPlan-style simulation log,
- it is designed to browse the evaluation contract already produced by this repo, not replace the underlying Waymax renderer.
