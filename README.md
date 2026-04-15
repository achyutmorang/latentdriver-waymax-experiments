# LatentDriver Waymax Experiments

Research repository for **closed-loop planner evaluation and method development** on **Waymax**, starting from LatentDriver-family public checkpoints and moving toward causal-semantic planner diagnostics, risk-aware reranking, and reproducible baseline-vs-method comparisons.

## Goal

This repo has two connected goals.

First, establish a reproducible evaluation contract:

- use only **publicly released checkpoints**,
- keep **Waymax settings identical across models**,
- evaluate under both **reactive** (`idm`) and **non-reactive** (`expert`) NPC settings,
- reproduce the paper's reported metrics as closely as possible,
- persist metrics, debug bundles, and visualization artifacts under one runner workflow.

Second, extend the reproduction into a research contribution:

- build a **causal-semantic closed-loop planner evaluation** protocol,
- compare `IDM -> LatentDriver -> YourMethod` on paired scenario IDs,
- join WOMD-Reasoning and CausalAgents **after rollout** as an evaluation metadata layer while planner inputs remain plain WOMD,
- define and test a **Causal-Semantic Safety Progress** (`CS-SP`) metric,
- add modular planner improvements such as frozen-generator **risk-aware candidate reranking** before any full fine-tuning.

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

## Research Strategy Docs

- [Causal-semantic planner evaluation strategy](./docs/causal_semantic_planner_evaluation_strategy.md)
- [Modular evaluator, reranking, and lightweight adaptation plan](./docs/inspiration_implementation_plan.md)

## Milestones

- [x] Smoke preprocessing produces reusable validation artifacts.
- [x] Smoke reactive evaluation runs across the public evaluation checkpoints.
- [x] Smoke non-reactive evaluation runs across the public evaluation checkpoints.
- [x] Smoke metric comparison plots are generated from completed run bundles.
- [x] Full validation preprocessing completes with aligned map, route, and intention-label outputs.
- [x] Full preprocessing writes durable `_SUCCESS` and `preprocess_manifest.json` markers.
- [x] Full eval dry-run passes with no missing inputs.
- [x] Full evaluation profiles are resumable at shard granularity.
- [x] Full preprocess shard archive workflow is available for faster and safer Colab restores.
- [x] Causal-semantic planner evaluation strategy is documented.
- [ ] Verify WOMD-Reasoning `sid` overlap with CausalAgents `scenario_id`.
- [ ] Add `validation_interactive` subset support to the evaluation contract.
- [ ] Export per-scenario rollout metrics, not only aggregate metrics.
- [ ] Run IDM vs LatentDriver on a fixed 10-shard plain WOMD `validation_interactive` rapid-prototyping subset.
- [ ] Join completed 10-shard rollout outputs with WOMD-Reasoning and CausalAgents for post-rollout diagnostic insight.
- [ ] Implement bucket assignment for causal-semantic diagnostics.
- [ ] Implement `CS-SP` base score and balanced bucket aggregation.
- [x] Inspect LatentDriver/PlanT candidate output availability.
- [ ] Add candidate dump support for at least one planner.
- [ ] Run native selector vs no-training risk-aware reranker.
- [ ] Choose first method intervention from pilot diagnostics.
- [ ] Run LatentDriver vs YourMethod on the same paired subset.
- [ ] Expand beyond the fixed 10-shard prototyping subset once the method and metric stabilize.
- [ ] Run full reactive and non-reactive suites after the method and metric are frozen.

## Run Matrix

| Run | Scope | NPC setting | Models | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| `smoke_reactive` | One-shard validation subset | Reactive IDM agents | Public evaluation checkpoints | Fast end-to-end simulator, checkpoint, metrics, and plotting validation. | Done |
| `smoke_non_reactive` | One-shard validation subset | Expert replay agents | Public evaluation checkpoints | Fast comparison against non-reactive replay-style traffic. | Done |
| `full_preprocess` | Full WOMD validation split | Not applicable | Not applicable | Build durable map, route, and intention-label caches used by all full evaluations. | Done |
| `create-full-preprocess-shard-archives` | Completed full preprocess cache | Not applicable | Not applicable | Pack the many small Drive-backed preprocess files into 150 resumable tar parts for faster and safer Colab restores. | Done |
| `full_eval_dry_run` | Full validation config only | Reactive by default | One selected checkpoint | Verify all paths, markers, checkpoint bindings, GCS auth, and command construction before expensive simulation. | Done |
| `validation_interactive_pilot` | Fixed 10 interaction shards | Reactive IDM agents | IDM + LatentDriver | Run the planners on plain WOMD `validation_interactive`; keep the model input contract unchanged. | Next |
| `metadata_join_check` | WOMD-Reasoning + CausalAgents over completed pilot outputs | Not applicable | Not applicable | Verify scenario and agent ID compatibility, then attach the causal-semantic overlay after rollout. | Next |
| `candidate_dump` | Pilot subset | Reactive IDM agents | LatentDriver / PlanT | Determine whether frozen planners expose candidate trajectories for reranking. | Planned |
| `risk_aware_rerank` | Pilot subset | Reactive IDM agents | LatentDriver baseline vs reranked variant | Test no-training method improvement on `CS-SP`. | Planned |
| `full_reactive_single` | Full WOMD validation split | Reactive IDM agents | One selected checkpoint | Full-scale closed-loop validation after pilot signal and runtime stability. | Planned |
| `full_non_reactive_single` | Full WOMD validation split | Expert replay agents | One selected checkpoint | Paired baseline for isolating model behavior without closed-loop NPC reactions. | Planned |
| `full_reactive` | Full WOMD validation split | Reactive IDM agents | All public evaluation checkpoints | Main closed-loop benchmark for model comparison under interactive traffic. | Planned |
| `full_non_reactive` | Full WOMD validation split | Expert replay agents | All public evaluation checkpoints | Replay-style benchmark for measuring model behavior under fixed surrounding traffic. | Planned |
| `plot_full_reactive` | Completed full reactive runs | Not applicable | All completed models | Generate comparable CSV, JSON, and PNG summaries from saved run bundles. | Planned |
| `plot_full_non_reactive` | Completed full non-reactive runs | Not applicable | All completed models | Generate paired non-reactive comparison artifacts. | Planned |

Conceptually, a **smoke run** is an engineering correctness check, not a research result. A **validation-interactive pilot** is the first research diagnostic because it targets interaction-heavy scenarios. For now, that pilot is a fixed **10-shard rapid-prototyping subset** run on plain WOMD `validation_interactive`, with the causal-semantic layer attached only **after rollout** for analysis. A **reactive run** lets surrounding agents respond through IDM, so it is closer to closed-loop interactive autonomy evaluation. A **non-reactive run** keeps surrounding traffic closer to replay/expert behavior, which helps separate ego-policy quality from feedback effects. A **candidate reranking run** tests whether a frozen pretrained planner can be improved by a better selector before training any new backbone. The full preprocess shard archive is an operational accelerator: it keeps the authoritative expanded artifacts on Drive but restores them into local Colab SSD from 150 resumable tar parts instead of many small random Drive reads.

## Evaluation Contract

We standardize the following across models:

- same validation split or smoke subset,
- same raw planner input schema, with no WOMD-Reasoning or CausalAgents labels injected into the planner during the rapid-prototyping phase,
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

Research metrics to add:

- per-scenario collision, offroad, progress, and route completion,
- optional TTC, clearance, and comfort diagnostics,
- `CS-SP` base score: safety gate times progress utility,
- balanced bucket-level `CS-SP` across causal-semantic regimes,
- paired deltas for `IDM -> LatentDriver` and `LatentDriver -> YourMethod`.

## Important Boundary

This repo is **frozen-planner first**. It does **not** start by training LatentDriver or PlanT. The first research milestone is:

- reproduce runnable evaluation for released checkpoints,
- capture per-scenario metrics and visualization under one standardized Waymax contract on plain WOMD `validation_interactive`,
- join completed rollout outputs with causal-semantic metadata after validation for useful diagnostic insight,
- test no-training reranking on frozen planner candidates,
- only then consider lightweight scorer or adapter fine-tuning.

Full backbone fine-tuning is intentionally not the first method step. The preferred ladder is:

```text
frozen planner + native selector
-> frozen planner + explicit risk-aware selector
-> frozen planner + small learned scorer or adapter
-> full fine-tuning only if earlier stages show signal
```

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
