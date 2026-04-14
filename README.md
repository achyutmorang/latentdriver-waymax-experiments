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

## Run Matrix

| Run | Scope | NPC setting | Models | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| `smoke_reactive` | One-shard validation subset | Reactive IDM agents | Public evaluation checkpoints | Fast end-to-end simulator, checkpoint, metrics, and plotting validation. | Done |
| `smoke_non_reactive` | One-shard validation subset | Expert replay agents | Public evaluation checkpoints | Fast comparison against non-reactive replay-style traffic. | Done |
| `full_preprocess` | Full WOMD validation split | Not applicable | Not applicable | Build durable map, route, and intention-label caches used by all full evaluations. | Done |
| `create-full-preprocess-shard-archives` | Completed full preprocess cache | Not applicable | Not applicable | Pack the many small Drive-backed preprocess files into 150 resumable tar parts for faster and safer Colab restores. | Next |
| `full_eval_dry_run` | Full validation config only | Reactive by default | One selected checkpoint | Verify all paths, markers, checkpoint bindings, GCS auth, and command construction before expensive simulation. | Done |
| `full_reactive_single` | Full WOMD validation split | Reactive IDM agents | One selected checkpoint | First scientifically meaningful full-scale simulation run; validates runtime stability before launching suites. | Next |
| `full_non_reactive_single` | Full WOMD validation split | Expert replay agents | One selected checkpoint | Paired baseline for isolating model behavior without closed-loop NPC reactions. | Planned |
| `full_reactive` | Full WOMD validation split | Reactive IDM agents | All public evaluation checkpoints | Main closed-loop benchmark for model comparison under interactive traffic. | Planned |
| `full_non_reactive` | Full WOMD validation split | Expert replay agents | All public evaluation checkpoints | Replay-style benchmark for measuring model behavior under fixed surrounding traffic. | Planned |
| `plot_full_reactive` | Completed full reactive runs | Not applicable | All completed models | Generate comparable CSV, JSON, and PNG summaries from saved run bundles. | Planned |
| `plot_full_non_reactive` | Completed full non-reactive runs | Not applicable | All completed models | Generate paired non-reactive comparison artifacts. | Planned |

Conceptually, a **smoke run** is an engineering correctness check, not a research result. A **full run** is the reproducible validation benchmark. A **reactive run** lets surrounding agents respond through IDM, so it is closer to closed-loop interactive autonomy evaluation. A **non-reactive run** keeps surrounding traffic closer to replay/expert behavior, which is useful for isolating ego-policy behavior from feedback effects. A **single-model full run** is the operational gate before spending compute on all checkpoints, while a **suite run** is the actual comparison layer. The full preprocess shard archive is an operational accelerator: it keeps the authoritative expanded artifacts on Drive but restores them into local Colab SSD from 150 resumable tar parts instead of many small random Drive reads.

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
