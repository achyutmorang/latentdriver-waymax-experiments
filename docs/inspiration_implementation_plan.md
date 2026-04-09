# Implementation Plan: Modular Evaluation, Reranking, and Lightweight Adaptation for LatentDriver on Waymax

## Problem Restatement

This repo currently targets **evaluation-only replication** of public LatentDriver-family checkpoints on Waymax.

The next step is to turn it into a **research substrate** without losing the evaluation contract. We want to borrow the strongest portable ideas from:

- **Diffusion-Planner**: modular guidance and reward composition,
- **Plan-R1**: explicit verifiable reward decomposition and grouped multi-sample reasoning,
- **RIFT**: freeze the pretrained generator, evaluate multiple rollout candidates, and improve the scoring/reranking layer instead of retraining the whole model first.

The right implementation target is therefore:

> keep the pretrained LatentDriver-family generators fixed, add a planner-agnostic evaluator/reranker layer, and only later consider lightweight scorer adaptation if the no-training reranking baselines show signal.

## Known vs Unknown

### Known

- We already have a pinned upstream LatentDriver fork and an evaluation-first repo.
- The repo already supports:
  - public checkpoint download,
  - smoke/full validation preprocessing,
  - standardized Waymax evaluation,
  - Drive-backed Colab execution,
  - visualization artifacts.
- Public baselines in scope today:
  - `latentdriver_t2_j3`
  - `latentdriver_t2_j4`
  - `plant`
  - `easychauffeur_ppo`

### Unknown

- Whether the current upstream evaluation path exposes enough **candidate-level outputs** to support reranking without extra patching.
- Whether LatentDriver exposes:
  - multiple candidate trajectories,
  - mode logits,
  - or only the final selected action/trajectory.
- Whether seed-to-seed variation and candidate-to-candidate variation are large enough to justify group-relative scoring.

### Most fragile assumption

The most fragile assumption is that the upstream LatentDriver evaluation path already exposes enough candidate diversity to support meaningful reranking. If it does not, we need a narrow upstream patch to dump candidate hypotheses before any reranking study is valid.

## First-Principles Model

The portable abstraction across LatentDriver, PlanT, and any future planner in this repo is:

1. **Generator**
   - pretrained model that proposes one or more candidate ego futures.
2. **Evaluator**
   - explicit metrics over a completed candidate rollout.
3. **Selector**
   - rule or learned scorer that chooses among candidates.
4. **Simulation contract**
   - fixed Waymax setup, fixed data split, fixed NPC control mode, fixed metrics schema.

This means the cleanest experimentation path is:

- do **not** modify the generator first,
- standardize the evaluator,
- compare selectors,
- only then consider lightweight scorer fine-tuning.

## What We Borrow From Each Project

### 1. Diffusion-Planner

Borrow:

- a **registry/config/composer** pattern for score modules,
- dual-use modules that work as:
  - evaluation metrics,
  - reranking scores,
  - and later lightweight training rewards.

Do not borrow first:

- the diffusion backbone,
- ROS/autoware pipeline,
- full RLVR training stack.

### 2. Plan-R1

Borrow:

- **verifiable reward decomposition** into interpretable components,
- **K-sample grouped evaluation**,
- hybrid thinking: learned generator + explicit evaluator.

Do not borrow first:

- full tokenized planner architecture,
- nuPlan-specific data pipeline.

### 3. RIFT

Borrow:

- **freeze generator, improve scorer**,
- **group-relative advantage / relative ranking over candidate rollouts**,
- critical-actor-focused interaction evaluation.

Do not borrow first:

- CARLA environment stack,
- controllable background-vehicle training,
- full CBV fine-tuning loop.

## Chosen Design

We should extend this repo in **three layers**, in order:

1. **Standardized evaluator layer**
   - explicit reusable metric modules,
   - no training,
   - no backbone changes.
2. **No-training reranking baselines**
   - weighted-score reranking,
   - best-of-K upper bound,
   - group-relative score normalization.
3. **Optional lightweight scorer adaptation**
   - only if the reranking baselines show clear signal,
   - freeze backbone,
   - train a tiny scorer head on candidate features.

This is the lowest-risk route that still leaves room for publishable extensions.

## Repository-Level Plan

## Phase 0: Finish Replication Contract

### Goal

Get one clean end-to-end evaluation and visualization path for the public checkpoints under fixed Waymax settings.

### Already in repo

- assets notebook,
- preprocessing notebook,
- evaluation notebook,
- visualization notebook.

### Exit criteria

- smoke preprocessing succeeds,
- smoke evaluation succeeds,
- at least one MP4/PDF visualization artifact exists,
- metrics are persisted to Drive,
- public checkpoint paths are reproducible.

## Phase 1: Add a Modular Evaluator Library

### Goal

Create a portable, explicit evaluator that can score any completed rollout or candidate rollout using reusable components.

### New modules to add

- `src/latentdriver_waymax_experiments/evaluator/base.py`
  - base interface for all score modules
- `src/latentdriver_waymax_experiments/evaluator/registry.py`
  - module registration
- `src/latentdriver_waymax_experiments/evaluator/config.py`
  - JSON-serializable evaluator config
- `src/latentdriver_waymax_experiments/evaluator/composer.py`
  - combine active modules into a total score
- `src/latentdriver_waymax_experiments/evaluator/modules/`
  - `collision.py`
  - `offroad.py`
  - `progress.py`
  - `min_ttc.py`
  - `min_clearance.py`
  - `comfort.py`
  - `consistency.py`

### Score semantics

Each module should expose:

- `metric(...)`
  - report-time scalar for analysis,
- `score(...)`
  - normalized value suitable for reranking,
- optional `weight`
  - used by the composer.

### Why this comes first

Without a stable evaluator, every later reranking or adaptation result is under-specified.

### Exit criteria

- evaluator config serializes to JSON,
- a single rollout can be scored with per-module outputs plus total score,
- outputs are written into the run artifact tree.

## Phase 2: Add Candidate-Level Collection

### Goal

Patch the upstream evaluation path, if necessary, so the repo can capture **candidate-level outputs**, not just final metrics.

### Key question

Do the public LatentDriver and PlanT evaluation paths already expose:

- multiple trajectory hypotheses,
- mode logits,
- or only the final selected trajectory?

### If candidate outputs already exist

- normalize them into a local schema.

### If candidate outputs do not exist

Add a narrow patch layer to save:

- candidate trajectories,
- candidate scores/logits if available,
- selected index,
- rollout-level metadata.

### Proposed schema

- `candidate_rollouts.npz`
  - `trajectories`
  - `scores_raw`
  - `selected_index`
  - `scene_id`
  - `seed`
  - `npc_policy_type`

### Exit criteria

- for at least one model, one scene, and one seed, the repo can save K candidate rollouts plus the final selected rollout.

## Phase 3: Add No-Training Reranking Baselines

### Goal

Test whether explicit reranking can improve behavior without retraining the backbone.

### Baselines

1. **Native selector**
   - whatever the upstream model already chooses.
2. **Best-of-K oracle**
   - choose the best candidate using future knowledge under the evaluator.
   - upper bound only, not deployable.
3. **Weighted explicit reranker**
   - choose candidate with highest weighted evaluator score.
4. **Group-relative reranker**
   - normalize candidate scores within the same scene:
     - z-score,
     - centered score,
     - or percentile rank.
5. **Critical-actor-focused reranker**
   - emphasize interaction metrics only for the most relevant nearby agents.

### New scripts

- `scripts/run_candidate_dump.py`
- `scripts/run_rerank_eval.py`
- `scripts/run_group_relative_eval.py`

### New notebook

- `notebooks/latentdriver_rerank_colab.ipynb`

### Metrics to compare

- existing Waymax metrics:
  - `mAR[75:95]`
  - `AR[75:95]`
  - `collision_rate`
  - `offroad_rate`
  - `progress_rate`
- new diagnostics:
  - minimum TTC distribution,
  - minimum clearance distribution,
  - score variance across candidates,
  - seed-to-seed consistency.

### Exit criteria

- one no-training reranker shows a visible difference relative to native selection,
- the result is reproducible on a fixed smoke/dev slice.

## Phase 4: Add Consistency and Variance Diagnostics

### Goal

Measure whether public baselines are stable across seeds, scenes, and candidate sets.

### New scripts

- `scripts/run_seed_consistency_study.py`
- `scripts/run_candidate_variance_study.py`

### New notebook

- `notebooks/latentdriver_consistency_colab.ipynb`

### Diagnostics to store

- per-scene metric spread,
- per-scene minimum TTC spread,
- per-scene minimum clearance spread,
- selected-candidate instability,
- candidate-score disagreement.

### Why this matters

This is the first small extension that can already be publishable as an empirical reliability finding, even before any adaptation.

## Phase 5: Optional Lightweight Scorer Adaptation

### Goal

If reranking works, train a tiny scorer on top of frozen candidate features.

### Constraint

No full LatentDriver retraining first.

### Candidate approaches

1. **Linear scorer**
   - input: explicit evaluator metrics,
   - output: total score.
2. **Small MLP scorer**
   - input: evaluator metrics + candidate metadata + native logits,
   - output: reranking score.
3. **Critical-actor scorer**
   - input: features focused on closest / most interacting neighbors.

### Training target options

- pairwise ranking loss,
- listwise ranking loss,
- group-relative normalized advantage target.

### New modules

- `src/latentdriver_waymax_experiments/scorer/`
- `scripts/train_lightweight_scorer.py`
- `scripts/run_scorer_eval.py`

### New notebook

- `notebooks/latentdriver_scorer_train_colab.ipynb`

### Exit criteria

- scorer improves reranking over fixed weighted baselines on the dev slice,
- backbone remains frozen,
- training is Colab-feasible.

## Recommended Evaluation Ladder

We should evaluate in this order:

1. `native`
2. `best_of_k_oracle`
3. `weighted_reranker`
4. `group_relative_reranker`
5. `lightweight_scorer`

This order matters because it separates:

- whether better selection is possible at all,
- whether hand-designed scores are enough,
- whether learning actually adds value.

## Colab Notebook Roadmap

### Existing

- `latentdriver_assets_colab.ipynb`
- `latentdriver_preprocess_val_colab.ipynb`
- `latentdriver_public_eval_colab.ipynb`
- `latentdriver_visualize_colab.ipynb`

### To add

- `latentdriver_candidate_dump_colab.ipynb`
- `latentdriver_rerank_colab.ipynb`
- `latentdriver_consistency_colab.ipynb`
- `latentdriver_scorer_train_colab.ipynb`

## Artifact Contract

All new experiment outputs should remain Drive-backed and machine-readable.

### Proposed additions to each run

- `metrics.json`
- `run_manifest.json`
- `config_snapshot.json`
- `candidate_rollouts.npz`
- `evaluator_breakdown.json`
- `rerank_summary.json`
- `consistency_summary.json`

## Validation Plan

### Unit tests

- evaluator module math,
- registry/config serialization,
- candidate artifact schema validation,
- reranking correctness on synthetic candidate sets.

### Integration tests

- one smoke run produces candidate dump,
- reranker reads dump and emits stable outputs,
- Drive-backed artifact paths remain correct.

### Edge cases

- only one candidate available,
- missing TTC / clearance information,
- no nearby actors,
- all candidates invalid,
- native logits missing.

## Complexity and Scaling

### Cheap

- evaluator module implementation,
- no-training reranking,
- consistency studies on smoke/dev slices.

### Moderate

- candidate dump patching,
- full public-suite rerank evaluation.

### Expensive

- lightweight scorer training,
- large K-sample studies,
- any backbone fine-tuning.

## Failure Modes

1. **No candidate access**
   - reranking path blocked until upstream patch exists.
2. **No useful candidate diversity**
   - oracle best-of-K improvement is flat,
   - learned scorer will not help.
3. **Evaluator too brittle**
   - hand-designed score improves proxies but hurts official metrics.
4. **Selection-compute confound**
   - gains come only from much larger K, not better scoring.

## What Would Invalidate This Design

This plan would need redesign if:

- LatentDriver only exposes one deterministic final action with no recoverable candidate set,
- or candidate-level diversity is so weak that best-of-K is flat,
- or official metrics are inaccessible under the standardized contract.

## Immediate Next Steps

1. Finish the current smoke preprocess + smoke eval path.
2. Inspect upstream LatentDriver inference to determine candidate availability.
3. Implement the evaluator registry/composer first.
4. Add candidate dump support.
5. Run a first best-of-K oracle study on a smoke slice before any learned scorer work.

## Practical Recommendation

Do **not** start with scorer training.

Start with:

- evaluator library,
- candidate dump,
- no-training reranking,
- consistency plots.

That is the fastest route to visible results and the cleanest route to a paper-worthy empirical story.
