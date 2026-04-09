# LatentDriver Reading Note

Paper: [Learning Multiple Probabilistic Decisions from Latent World Model in Autonomous Driving](https://arxiv.org/pdf/2409.15730)

Repo: [Sephirex-X/LatentDriver](https://github.com/Sephirex-X/LatentDriver)

## What the paper claims

LatentDriver frames ego driving as **multiple probabilistic decisions** generated from a **latent world model**, then derives a deterministic control signal from those hypotheses.

The official repo claims:

- expert-level performance on Waymax,
- a full Waymax close-loop pipeline,
- released code for `LatentDriver`, `PlanT`, and `EasyChauffeur-PPO`,
- released public weights for those methods.

## Public checkpoints verified

The Hugging Face model tree exposes:

- `checkpoints/lantentdriver_t2_J3.ckpt`
- `checkpoints/lantentdriver_t2_J4.ckpt`
- `checkpoints/planT.ckpt`
- `checkpoints/easychauffeur_policy_best.pth.tar`
- `checkpoints/pretrained_bert.pth.tar`

## Official evaluation settings from the upstream README

Reactive agents:

- `ego_control_setting.npc_policy_type=idm`

Non-reactive agents:

- `ego_control_setting.npc_policy_type=expert`

Documented default evaluation commands:

- `LatentDriver(T=2, J=3)`: `batch_dims=[7,125]`
- `LatentDriver(T=2, J=4)`: `batch_dims=[7,150]` in ablation
- `PlanT`: `batch_dims=[7,125]`
- `EasyChauffeur-PPO`: `batch_dims=[7,125]`

This repo standardizes around `batch_dims=[7,125]` first so all released baselines are directly comparable under one contract.

## Metrics we will standardize

The upstream metric code exposes:

- `metric/AR[75:95]`
- `metric/arrival_rate75` ... `metric/arrival_rate95`
- `metric/offroad_rate`
- `metric/collision_rate`
- `metric/progress_rate`

The printed table contains both:

- `Average_over_class` -> we treat this as `mAR`-style class-balanced reporting
- `Average` -> global reporting

The local upstream patch adds machine-readable JSON for both.

## Why a local patch layer exists

The released upstream repo is good enough to run evaluation, but not ideal for reproducible experiment orchestration:

- no bounded `max_batches` control for smoke runs,
- no machine-readable metrics artifact,
- visualization writes into a fixed `vis_results/` tree.

The local patch layer fixes exactly those issues and no more.
