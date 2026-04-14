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

## Colab Runner

Use the single runner notebook as the default execution surface:

- [`notebooks/latentdriver_colab_runner.ipynb`](./notebooks/latentdriver_colab_runner.ipynb)

The notebook is intentionally thin. It handles Colab-specific handshakes such as Drive mount and GCS auth, then delegates the actual workflow to CLI profiles.

## Milestones

- [x] Smoke preprocessing produces reusable validation artifacts.
- [x] Smoke reactive evaluation runs across the public evaluation checkpoints.
- [x] Smoke non-reactive evaluation runs across the public evaluation checkpoints.
- [x] Smoke metric comparison plots are generated from completed run bundles.
- [x] Full validation preprocessing completes with aligned map, route, and intention-label outputs.
- [x] Full preprocessing writes durable `_SUCCESS` and `preprocess_manifest.json` markers.
- [x] Full eval dry-run passes with no missing inputs.
- [x] Full evaluation profiles are resumable at shard granularity.
- [ ] One full reactive model run completes end-to-end.
- [ ] One full non-reactive model run completes end-to-end.
- [ ] Full reactive suite completes for all public evaluation checkpoints.
- [ ] Full non-reactive suite completes for all public evaluation checkpoints.
- [ ] Full-dataset comparison plots are generated for reactive and non-reactive tiers.
- [ ] Scenario-bucket analysis is added for intention classes and failure modes.
- [ ] First research intervention is selected after baseline reproduction is stable.

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
