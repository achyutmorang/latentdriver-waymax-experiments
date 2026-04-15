# Implementation Plan: Risk-Aware Action Modulation and Lightweight Adaptation for Pretrained Planners on Waymax

## Problem Restatement

This repo already supports evaluation-first reproduction of public LatentDriver-family checkpoints on Waymax.

The next step is not full retraining. It is to turn the repo into a research substrate for:

- causal-semantic closed-loop evaluation,
- planner-agnostic post-hoc safety-performance intervention,
- and later lightweight adaptation if the intervention shows signal.

The strongest current implementation target is:

> keep the pretrained planners frozen, add a runtime risk-aware action modulation layer on top of their output actions, evaluate it under the fixed Waymax contract, and only later consider planner-specific reranking or small learned adapters.

This is narrower and more portable than immediately optimizing LatentDriver-specific candidate reranking.

## Known vs Unknown

### Known

- The repo already supports public checkpoint evaluation, smoke and full preprocessing, resumable full evaluation, and Colab runner profiles.
- The current rapid-prototyping protocol is a fixed 10-shard plain WOMD `validation_interactive` subset.
- Causal-semantic metadata is attached after rollout, not injected into planner inputs.
- The current repo wiring exposes a shared waypoint-delta action interface for both LatentDriver and the PlanT baseline:
  - action shape is `[dx, dy, dyaw]`
  - action range is defined in:
    - `external/LatentDriver/configs/method/latentdriver.yaml`
    - `external/LatentDriver/configs/method/planT.yaml`
- The runtime insertion point is available in:
  - `external/LatentDriver/simulator/engines/ltd_simulator.py`
  - between `model.get_predictions(...)` and `env.step(...)`

### Unknown

- Whether simple TTC and density heuristics are already enough to improve the safety-progress tradeoff.
- Whether scalar action scaling is too blunt for turn-heavy scenarios.
- How much short-horizon ghost-rollout supervision is needed before a learned modulator beats heuristic baselines.
- Whether the learned modulator transfers cleanly across both LatentDriver and PlanT.

### Most fragile assumption

The most fragile assumption is that a lightweight action-scaling wrapper can improve collision and offroad behavior without collapsing progress in turns, merges, and dense interactions. If scalar modulation causes strong over-conservatism, the method must move to anisotropic scaling or a hybrid selector-plus-modulator design.

## First-Principles Model

The portable abstraction across LatentDriver, PlanT, and future planners in this repo is:

1. **Generator**
   - frozen pretrained planner producing an ego action.
2. **Runtime risk encoder**
   - features derived from current simulator state plus proposed ego action.
3. **Modulator**
   - maps risk features to a scale or correction for the ego action.
4. **Simulation contract**
   - fixed Waymax setup, fixed split, fixed NPC policy, fixed output schema.
5. **Post-rollout evaluator**
   - standard metrics plus causal-semantic overlay and `CS-SP`.

This leads to the cleanest sequence:

- do not change the planner first,
- export per-scenario outputs,
- add a runtime modulator around planner actions,
- compare default vs heuristic modulator vs learned modulator,
- only then consider planner-specific reranking or learned scorer heads.

## What We Borrow From Prior Work

### 1. Runtime safety layers

Borrow:

- minimal post-hoc action correction,
- state-dependent intervention intensity,
- keeping the strong base controller intact.

Most aligned sources:

- Dalal et al. safety layers,
- SafetyNet-style fallback logic,
- control-barrier-filter thinking.

Do not borrow first:

- full CBF-QP machinery,
- deployment-grade rule stacks,
- planner replacement.

### 2. Predictive runtime monitoring

Borrow:

- short-horizon future-risk estimation,
- intervene before overlap or offroad actually occurs,
- use the simulator to create labels for near-future failure.

Do not borrow first:

- formal runtime verification frameworks,
- complex intent inference models.

### 3. Diffusion and reranking papers

Borrow:

- modular reward and score composition,
- explicit decomposition of safety and progress objectives,
- freeze-generator-first philosophy,
- optional planner-specific candidate reranking later.

Do not borrow first:

- diffusion backbones,
- full RL fine-tuning,
- planner-specific architectural rewrites.

## Chosen Design

We should extend this repo in **four layers**, in order:

1. **Standardized evaluator and per-scenario logging**
   - explicit reusable metric modules,
   - no training,
   - no backbone changes.
2. **No-training heuristic action modulation**
   - TTC,
   - interaction density,
   - action-magnitude-aware scaling.
3. **Ghost-rollout label generation plus lightweight learned modulator**
   - frozen planner,
   - short-horizon simulator-consistent targets,
   - tiny risk head or direct scale predictor.
4. **Optional planner-specific branches**
   - LatentDriver candidate reranking,
   - hybrid rerank-plus-modulate experiments,
   - small learned scorer or adapter if justified.

This is the lowest-risk path that still gives a cross-planner method contribution.

## Repository-Level Plan

## Phase 0: Keep the Evaluation Contract Stable

### Goal

Preserve one clean end-to-end evaluation path for public checkpoints under fixed Waymax settings.

### Already in repo

- assets notebook,
- preprocessing notebook,
- evaluation notebook,
- visualization notebook,
- candidate-diversity probe,
- metadata join checker,
- causal-semantic evaluation strategy,
- risk-aware action modulation research note.

### Exit criteria

- smoke and full preprocessing remain reproducible,
- pilot and full evaluation paths remain stable,
- metrics and run manifests stay machine-readable,
- no planner input schema change is needed for the first method study.

## Phase 1: Add a Modular Evaluator Library

### Goal

Create a portable evaluator that can score completed rollouts and later short ghost rollouts using reusable components.

### New modules to add

- `src/latentdriver_waymax_experiments/evaluator/base.py`
- `src/latentdriver_waymax_experiments/evaluator/registry.py`
- `src/latentdriver_waymax_experiments/evaluator/config.py`
- `src/latentdriver_waymax_experiments/evaluator/composer.py`
- `src/latentdriver_waymax_experiments/evaluator/modules/`
  - `collision.py`
  - `offroad.py`
  - `progress.py`
  - `min_ttc.py`
  - `interaction_density.py`
  - `min_clearance.py`
  - `comfort.py`

### Score semantics

Each module should expose:

- `metric(...)`
  - report-time scalar for analysis,
- `score(...)`
  - normalized value suitable for ghost-rollout ranking,
- optional `weight`
  - used by the composer.

### Why this comes first

Without a stable evaluator, the modulation objective and ghost-rollout labels are under-specified.

### Exit criteria

- evaluator config serializes to JSON,
- one rollout can be scored with per-module outputs and total score,
- outputs are written into the run artifact tree.

## Phase 2: Export Per-Scenario Pilot Artifacts

### Goal

Make the 10-shard pilot useful for both causal-semantic analysis and modulator training.

### Required outputs

- per-scenario metrics,
- per-step ego actions,
- per-step simulator state summary,
- run manifests keyed by `scenario_id`,
- optional compact feature dumps for training the modulator.

### Why this matters

The learned modulator should train on data collected under the same closed-loop contract used in evaluation.

### Exit criteria

- pilot outputs can be joined by `scenario_id`,
- stepwise planner-action traces are available,
- enough runtime state is exported to compute TTC and density proxies offline if needed.

## Phase 3: Add the Runtime Modulation Hook

### Goal

Insert a planner-agnostic wrapper between the planner and Waymax environment step.

### Minimal insertion point

In:

- `external/LatentDriver/simulator/engines/ltd_simulator.py`

current flow is:

```python
action = self.model.get_predictions(...)
control_action = action
obs, obs_dict, rew, done, info = self.env.step(control_action, show_global=True)
```

The modulator should wrap `control_action` before `env.step(...)`.

### New modules to add

- `src/latentdriver_waymax_experiments/modulation/base.py`
- `src/latentdriver_waymax_experiments/modulation/features.py`
- `src/latentdriver_waymax_experiments/modulation/heuristic.py`
- `src/latentdriver_waymax_experiments/modulation/runtime.py`
- `src/latentdriver_waymax_experiments/modulation/config.py`

### Core interface

```text
current_state, planner_action -> feature_encoder -> modulator -> modulated_action
```

### Exit criteria

- modulation can be toggled from config,
- default path remains unchanged when disabled,
- both LatentDriver and PlanT can run through the same modulation wrapper.

## Phase 4: Implement No-Training Heuristic Modulation

### Goal

Test whether a simple safety-aware action scaler already improves the tradeoff.

### Baselines

1. **Native planner**
   - no intervention.
2. **Constant scale**
   - `s = 0.75`
3. **Constant conservative scale**
   - `s = 0.5`
4. **TTC-only rule**
   - scale down when approximate TTC drops below threshold.
5. **TTC plus density heuristic**
   - scale by a function of TTC, interaction density, and action magnitude.

### Inputs

- ego speed,
- action magnitude,
- nearest-neighbor distance,
- approximate minimum TTC,
- local interaction density.

### Output

Start with a scalar:

```text
s in [s_min, 1.0]
```

then:

```text
a' = s * a
```

### Why scalar first

- planner-agnostic,
- easy to interpret,
- low engineering risk,
- strong first baseline.

### Exit criteria

- heuristic modulator changes behavior on the pilot subset,
- intervention statistics are logged,
- results can be compared against progress-retention and `CS-SP`.

## Phase 5: Add Short-Horizon Ghost-Rollout Label Generation

### Goal

Generate simulator-consistent supervision for a learned modulator.

### Method

For each observed state-action pair:

1. construct a small scale set:

```text
S = {0.25, 0.5, 0.75, 1.0}
```

2. create scaled actions:

```text
a_s = s * a
```

3. run short ghost rollouts for horizon `H = 3..5` steps,
4. score each candidate with explicit safety-progress terms,
5. choose the best scale `s_star`.

### New modules to add

- `src/latentdriver_waymax_experiments/modulation/ghost_rollout.py`
- `src/latentdriver_waymax_experiments/modulation/label_generation.py`
- `scripts/generate_modulation_labels.py`

### Output schema

- `state_features`
- `planner_action`
- `scale_candidates`
- `ghost_metrics_per_scale`
- `best_scale`
- `risk_targets`
- `scenario_id`
- `step_index`

### Why this matters

This creates supervision without retraining the planner or requiring human labels.

### Exit criteria

- a label dataset is generated from the 10-shard pilot,
- label quality is inspectable and reproducible,
- oracle scale selection is measurable.

## Phase 6: Train a Lightweight Learned Modulator

### Goal

Learn a small model that predicts future interaction risk and maps it to action scaling.

### Candidate approaches

1. **Risk head plus analytic scaler**
   - input: runtime features,
   - output: risk score or short-horizon event probabilities,
   - scale derived analytically from predicted risk.
2. **Direct scale regressor**
   - input: runtime features,
   - output: `s_star`.
3. **Multi-task head**
   - collision-within-horizon,
   - offroad-within-horizon,
   - min-TTC prediction,
   - scale target.

### Recommended first model

Start with:

- tiny MLP,
- risk head plus analytic scaler,
- scalar action scale.

### New modules to add

- `src/latentdriver_waymax_experiments/modulation/model.py`
- `src/latentdriver_waymax_experiments/modulation/train.py`
- `scripts/train_action_modulator.py`
- `scripts/run_modulated_eval.py`

### Exit criteria

- learned modulator beats constant-scale and TTC-only baselines on the pilot subset,
- intervention remains lightweight,
- planner backbone remains frozen.

## Phase 7: Causal-Semantic Post-Rollout Analysis

### Goal

Test whether the modulator helps where the paper claim says it should help.

### Main comparisons

- native planner,
- heuristic modulator,
- learned modulator,
- later optional planner-specific reranker.

### Main metrics

- collision rate,
- offroad rate,
- progress rate,
- BaseScore,
- Balanced `CS-SP`,
- intervention rate,
- mean action scale,
- progress-retention ratio versus native planner.

### Main buckets

- `causal_high`
- `causal_reasoning_overlap`
- `reasoning_rule`
- `reasoning_intention`
- `dense_scene`
- `intersection_or_turn`
- `near_causal_agent`

### Exit criteria

- bucketed `CS-SP` tables exist,
- the modulator has a clear safety-progress profile,
- high-causal-pressure scenes are analyzed separately from easy scenes.

## Phase 8: Optional Planner-Specific Branches

### Goal

Use planner-specific structure only after the planner-agnostic path is stable.

### LatentDriver-specific options

- candidate dump,
- risk-aware reranking,
- hybrid rerank plus modulate,
- optional uncertainty-aware features from mode disagreement.

### Why later

This branch is valuable but narrower:

- it does not transfer cleanly to PlanT as currently wired,
- it is not needed to test the main post-hoc modulation hypothesis,
- it should not block the cross-planner method story.

## Recommended Evaluation Ladder

Evaluate in this order:

1. `native`
2. `constant_scale_0_75`
3. `constant_scale_0_5`
4. `ttc_only_modulator`
5. `ttc_plus_density_modulator`
6. `learned_modulator`
7. `latentdriver_rerank`
8. `latentdriver_rerank_plus_modulation`

This order separates:

- whether any scaling helps at all,
- whether hand-designed risk rules are enough,
- whether learning adds value,
- whether planner-specific branches add anything beyond the general method.

## Colab Notebook Roadmap

### Existing

- `latentdriver_assets_colab.ipynb`
- `latentdriver_preprocess_val_colab.ipynb`
- `latentdriver_public_eval_colab.ipynb`
- `latentdriver_visualize_colab.ipynb`

### To add

- `latentdriver_modulation_heuristic_colab.ipynb`
- `latentdriver_modulation_labels_colab.ipynb`
- `latentdriver_modulation_train_colab.ipynb`
- `latentdriver_modulation_analysis_colab.ipynb`
- later optional:
  - `latentdriver_candidate_dump_colab.ipynb`
  - `latentdriver_rerank_colab.ipynb`

## Artifact Contract

All new experiment outputs should remain machine-readable and Drive-backed.

### Proposed additions per run

- `metrics.json`
- `run_manifest.json`
- `config_snapshot.json`
- `step_trace.jsonl` or compact equivalent
- `modulation_summary.json`
- `intervention_stats.json`
- `ghost_rollout_labels.parquet` or `.jsonl`
- `causal_semantic_summary.json`

## Validation Plan

### Unit tests

- feature extraction math,
- TTC approximation behavior,
- density feature calculation,
- scale clipping and action shaping,
- modulation config serialization,
- label-generation objective selection.

### Integration tests

- one smoke scene runs with modulation enabled,
- default path remains unchanged when modulation is disabled,
- heuristic modulator emits stable intervention stats,
- learned modulator can load and run from saved weights.

### Edge cases

- no nearby agents,
- invalid neighbor trajectories,
- action norm near zero,
- high-density but non-closing traffic,
- turn scenarios where scalar scaling may under-turn,
- planner outputs containing NaNs or out-of-range values.

## Complexity and Scaling

### Cheap

- heuristic modulation,
- feature extraction,
- per-step logging,
- pilot analysis.

### Moderate

- ghost-rollout label generation,
- learned modulator training on pilot outputs.

### Expensive

- full-suite label generation,
- planner-specific reranking plus modulation hybrids,
- any backbone fine-tuning.

## Failure Modes

1. **Over-conservatism**
   - collision drops because the car barely moves.
2. **Turn degradation**
   - scalar scaling hurts route completion in curves and intersections.
3. **Proxy mismatch**
   - TTC-only features miss offroad or negotiated interaction failures.
4. **Planner transfer gap**
   - a learned modulator trained on LatentDriver traces does not transfer to PlanT.
5. **Label mismatch**
   - ghost-rollout objective does not correlate with final `CS-SP`.

## What Would Invalidate This Design

This plan would need redesign if:

- runtime state access is too limited to estimate useful short-horizon risk,
- scalar or anisotropic action scaling cannot improve safety without unacceptable progress collapse,
- or short-horizon ghost-rollout labels do not correlate with downstream closed-loop outcomes.

## Immediate Next Steps

1. Add per-scenario and per-step pilot logging needed for modulation training.
2. Implement the runtime modulation hook in `ltd_simulator.py`.
3. Build the first heuristic TTC plus density modulator.
4. Run native vs constant-scale vs heuristic modulation on the 10-shard pilot.
5. Add ghost-rollout label generation for scaled actions.
6. Train the first lightweight learned risk-aware modulator.
7. Attach the causal-semantic overlay after rollout and compute bucketed `CS-SP`.

## Practical Recommendation

Do **not** start with fine-tuning the pretrained planners.

Do **not** make LatentDriver-specific reranking the first method branch.

Start with:

- evaluator and pilot artifact export,
- runtime action modulation hook,
- heuristic risk-aware modulation,
- ghost-rollout label generation,
- lightweight learned modulator,
- causal-semantic bucket analysis.

This is the strongest route to a planner-agnostic, closed-loop, method-plus-evaluation story.
