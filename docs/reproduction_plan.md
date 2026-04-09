# Reproduction Plan

## Scope

This repo is designed to reproduce **evaluation** from the released LatentDriver stack without training.

The immediate target is:

1. bootstrap the pinned upstream fork,
2. download the released checkpoints,
3. preprocess validation-only artifacts,
4. run standardized Waymax evaluation for all public checkpoints,
5. generate at least one visualization artifact,
6. record a small extension: consistency across seeds/scenarios where feasible.

## Public baselines in scope

- `latentdriver_t2_j3`
- `latentdriver_t2_j4`
- `plant`
- `easychauffeur_ppo`

## Milestones

1. **Smoke path working**
   - one-shard smoke subset
   - support both Drive-local WOMD roots and authenticated WOMD GCS access for smoke staging
   - one checkpoint runs end to end
   - machine-readable metrics JSON exists
   - one visualization artifact exists

2. **Standardized full validation path**
   - all public checkpoints under `full_reactive`
   - all public checkpoints under `full_non_reactive`
   - suite summary table written to artifacts

3. **Replication note**
   - compare observed metrics against the upstream README table
   - record any mismatch caused by batch size, environment drift, or preprocessing details

4. **Small extension**
   - add one extra diagnostic dimension such as repeated-run consistency or simple risk spread on a fixed smoke slice

## What we will not do first

- no training,
- no architecture modification,
- no latentdriver decision-layer method yet,
- no unofficial external baselines unless their public checkpoints are verified.
