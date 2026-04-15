# Causal-Semantic Closed-Loop Planner Evaluation Strategy

## 1. Purpose

This document defines a research evaluation strategy for planner comparison and method development on top of the current LatentDriver-Waymax reproduction pipeline.

The goal is not only to reproduce LatentDriver-style aggregate metrics, but to answer a sharper scientific question:

> Can a planner maintain route progress while reducing safety failures in scenarios where surrounding agents are causally and semantically important to ego behavior?

This supports a combined contribution:

1. A planner method improvement, such as risk-aware candidate selection.
2. A diagnostic evaluation methodology for closed-loop planners.

## 2. Research Positioning

Most planner evaluations report aggregate metrics such as collision rate, offroad rate, and progress. These are necessary but incomplete. Aggregate scores can hide whether a planner is failing in specific regimes such as dense intersections, turns, high-causal-agent scenes, or rule-induced interactions.

The proposed evaluation reframes planner assessment as a paired, causal-semantic, closed-loop diagnostic problem.

Instead of asking only:

> Which planner has the best average score?

we ask:

> Where does each planner fail, why does it fail, and does the proposed method improve the safety-progress tradeoff under causally meaningful interaction pressure?

## 3. Planner Comparison Ladder

Use three planner levels.

| Level | Planner | Purpose |
| --- | --- | --- |
| `B0` | IDM or rule-based baseline | Classical sanity anchor |
| `B1` | LatentDriver reproduced baseline | Strong learned planner baseline |
| `M1` | Proposed method | Research contribution |

The first comparison validates the benchmark:

```text
LatentDriver - IDM
```

The second comparison is the main research claim:

```text
YourMethod - LatentDriver
```

All comparisons must be paired by scenario ID.

## 4. Dataset Strategy

### 4.1 Primary Split

Use `validation_interactive` as the first research split.

Reason:

- It is interaction-heavy.
- It is aligned with the scientific question.
- It is better for diagnosing planner behavior than arbitrary regular validation shards.
- It avoids hidden test leakage.

### 4.2 Dataset Scales

| Stage | Data | Purpose |
| --- | --- | --- |
| Rapid prototyping subset | Fixed 10 `validation_interactive` shards | Default iteration loop for debugging, ablations, and early evidence |
| Expanded subset | 20 or more `validation_interactive` shards | Stronger preliminary tables after the method stabilizes |
| Full diagnostic | Full `validation_interactive`, then regular validation | Stronger final claim |

A fixed 10-shard subset is the current rapid-prototyping regime. It is useful for iteration and early evidence, but it should still not be presented as final evidence.

### 4.3 Sampling Note

Picking shard IDs such as `0, 30, 60, 90, 120` is systematic sampling, not true stratified sampling.

True stratified sampling requires first defining buckets, then sampling scenarios from each bucket.

Initial pilot can use spread-out shards. Later experiments should use bucket-aware scenario selection.

## 5. Metadata Fusion Plan

The proposed diagnostic layer combines:

1. WOMD scenario metadata.
2. WOMD-Reasoning annotations.
3. CausalAgents labels.
4. Planner rollout outputs.

This should be implemented as a metadata join table over WOMD scenario IDs, not as a duplicated raw dataset.
In the current rapid-prototyping phase, this metadata is attached only after simulation or evaluation has finished. The planners still consume plain WOMD `validation_interactive` inputs plus the existing preprocess artifacts.

### 5.1 Join Keys

Expected join key:

```text
WOMD-Reasoning.sid == CausalAgents.scenario_id == WOMD scenario_id
```

Agent-level joins should use raw WOMD IDs:

```text
WOMD-Reasoning.ego
WOMD-Reasoning.rel_id
CausalAgents.causal_agent_ids
```

Do not use `rel_qa_id` for causal-agent joins, because WOMD-Reasoning remaps Q&A IDs for language presentation.

### 5.2 Combined Metadata Schema

| Field | Source | Purpose |
| --- | --- | --- |
| `scenario_id` | WOMD / WOMD-Reasoning / CausalAgents | Scenario-level join key |
| `split` | WOMD | Dataset split |
| `shard_id` | WOMD storage path | Reproducible subset selection |
| `ego_agent_id` | WOMD-Reasoning | Ego context |
| `objects_of_interest` | WOMD | Interactive agents |
| `reasoning_categories` | WOMD-Reasoning | Semantic interaction tags |
| `interaction_qas` | WOMD-Reasoning | Interaction explanations |
| `intention_qas` | WOMD-Reasoning | Ego intention reasoning |
| `related_agent_ids` | WOMD-Reasoning `rel_id` | Agents discussed in Q&A |
| `qa_agent_ids` | WOMD-Reasoning `rel_qa_id` | Local language IDs only |
| `causal_agent_ids_by_labeler` | CausalAgents | Human-labeled causal agents |
| `causal_agent_union` | Derived | Conservative causal-agent set |
| `causal_agent_majority` | Derived | Stricter causal-agent set |
| `causal_agent_count` | Derived | Interaction pressure feature |
| `causal_agent_min_distance` | Derived | Nearest causal-agent pressure |
| `reasoning_causal_overlap` | Derived | Whether discussed agents are causal |
| `planner_name` | Evaluation | IDM, LatentDriver, or method |
| `collision` | Rollout metric | Safety |
| `offroad` | Rollout metric | Safety |
| `progress` | Rollout metric | Utility |
| `ttc_violation` | Rollout metric, if available | Near-collision risk |
| `comfort` | Rollout metric, if available | Smoothness |
| `bucket_labels` | Derived | Diagnostic analysis |
| `score` | Derived | Proposed metric |

## 6. Novel Metric: Causal-Semantic Safety Progress

Metric name:

```text
Causal-Semantic Safety Progress Score
CS-SP
```

The metric should not be only one scalar. It should have:

1. A per-scenario safety-progress base score.
2. Bucketed causal-semantic aggregation.
3. Paired planner deltas inside each bucket.

## 7. Base Safety-Progress Score

For each scenario `i`:

```text
BaseScore_i = SafetyGate_i * ProgressUtility_i
```

### 7.1 Safety Gate

Start with a binary safety gate for interpretability:

```text
SafetyGate_i = no_collision_i * no_offroad_i * no_ttc_violation_i
```

Where:

```text
no_collision_i = 0 if collision occurred, else 1
no_offroad_i   = 0 if offroad occurred, else 1
no_ttc_i       = 0 if TTC violation occurred, else 1
```

If TTC is not available initially, use:

```text
SafetyGate_i = no_collision_i * no_offroad_i
```

### 7.2 Progress Utility

Progress should be normalized against the expert or route target:

```text
ProgressUtility_i = clip(planner_progress_i / expert_progress_i, 0, 1)
```

If expert progress is unavailable, use route completion or normalized progress from the existing simulator metrics.

### 7.3 Interpretation

Unsafe progress should not receive high reward. A planner that reaches farther by colliding should not look better than a slower safe planner.

## 8. Soft Safety Variant

If the binary gate is too brittle, use a soft penalty:

```text
SafetyScore_i = 1
                - lambda_collision * collision_i
                - lambda_offroad * offroad_i
                - lambda_ttc * ttc_violation_i
```

Then:

```text
BaseScore_i = max(0, SafetyScore_i) * ProgressUtility_i
```

Recommendation:

- Use binary gate for the first implementation.
- Add soft scoring as an ablation later.

## 9. Causal-Semantic Buckets

Bucket definitions should be fixed before running the final comparison.

Initial bucket set:

| Bucket | Definition |
| --- | --- |
| `causal_none` | No causal agents |
| `causal_low` | One causal agent |
| `causal_high` | Two or more causal agents |
| `reasoning_interaction` | WOMD-Reasoning has explicit interaction Q&A |
| `reasoning_rule` | Rule-induced interaction |
| `reasoning_intention` | Intention-induced interaction |
| `causal_reasoning_overlap` | Causal agent appears in WOMD-Reasoning `rel_id` |
| `dense_scene` | High number of valid agents |
| `intersection_or_turn` | Intersection or turn context |
| `near_causal_agent` | Causal agent is within distance threshold |

For each bucket `b`:

```text
CS-SP_b(planner) = mean(BaseScore_i for i in bucket b)
```

## 10. Balanced Headline Score

Avoid letting easy scenarios dominate the headline metric.

Define:

```text
Balanced_CS-SP(planner) = mean(CS-SP_b(planner) over selected buckets)
```

This is different from raw average over scenarios. It gives each diagnostic regime equal weight.

Recommended headline buckets:

```text
causal_low
causal_high
reasoning_interaction
reasoning_rule
reasoning_intention
dense_scene
intersection_or_turn
near_causal_agent
```

## 11. Paired Deltas

For each bucket:

```text
Delta_IDM_to_LD_b = CS-SP_b(LatentDriver) - CS-SP_b(IDM)
Delta_LD_to_Method_b = CS-SP_b(YourMethod) - CS-SP_b(LatentDriver)
```

This keeps the analysis scientific. The question becomes:

> In which causal-semantic regimes does the proposed method improve or degrade relative to LatentDriver?

## 12. Required Tables

### 12.1 Aggregate Planner Table

| Planner | Collision Rate | Offroad Rate | Progress | BaseScore | Balanced CS-SP |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDM | TBD | TBD | TBD | TBD | TBD |
| LatentDriver | TBD | TBD | TBD | TBD | TBD |
| YourMethod | TBD | TBD | TBD | TBD | TBD |

### 12.2 Bucketed Method Table

| Bucket | N | LatentDriver | YourMethod | Delta | Collision Delta | Progress Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal_low | TBD | TBD | TBD | TBD | TBD | TBD |
| causal_high | TBD | TBD | TBD | TBD | TBD | TBD |
| reasoning_interaction | TBD | TBD | TBD | TBD | TBD | TBD |
| reasoning_rule | TBD | TBD | TBD | TBD | TBD | TBD |
| reasoning_intention | TBD | TBD | TBD | TBD | TBD | TBD |
| dense_scene | TBD | TBD | TBD | TBD | TBD | TBD |
| intersection_or_turn | TBD | TBD | TBD | TBD | TBD | TBD |

### 12.3 Safety-Progress Interpretation Table

| Bucket | Progress Change | Safety Change | Interpretation |
| --- | ---: | ---: | --- |
| causal_high | TBD | TBD | Safer progress / overconservative / unsafe progress |
| reasoning_rule | TBD | TBD | TBD |
| near_causal_agent | TBD | TBD | TBD |

## 13. Method Contribution Alignment

The evaluation should guide the planner method.

A natural first method is:

```text
Risk-Aware Latent Candidate Selection
```

If LatentDriver produces multiple latent decisions or candidate futures, choose the final plan using a score like:

```text
CandidateScore(k) =
  alpha * progress(k)
  - beta * collision_risk(k)
  - gamma * offroad_risk(k)
  - delta * causal_agent_proximity_risk(k)
  - eta * uncertainty(k)
```

This is modular because it can be implemented as a selection layer after candidate generation.

Pipeline:

```text
Planner candidate trajectories
-> Risk-aware selector
-> Selected trajectory
-> Closed-loop simulation
-> CS-SP evaluation
```

This method can potentially transfer to LatentDriver-like and PLANT-like planners if candidate plans or cost terms are accessible.

## 14. Hypotheses

Use these as testable claims.

### H1

Aggregate planner metrics hide interaction-specific failures.

### H2

LatentDriver improves over IDM on average, but may fail disproportionately under high causal-agent pressure.

### H3

Risk-aware candidate selection improves safety-progress tradeoff in high-causal-pressure scenarios.

### H4

The proposed method does not improve uniformly; bucketed evaluation reveals where it helps and where it hurts.

## 15. Experimental Stages

### Stage 1: Plain WOMD Pilot Evaluation

Goal:

Run IDM and LatentDriver on a fixed 10-shard plain WOMD `validation_interactive` subset.

Example fixed shard set:

```text
0, 15, 30, 45, 60, 75, 90, 105, 120, 135
```

Expected output:

- Per-scenario rollout metrics
- Run manifests keyed by `scenario_id`
- No WOMD-Reasoning or CausalAgents labels inside the planner input path

### Stage 2: Post-Rollout Metadata Join Feasibility

Goal:

Verify that WOMD-Reasoning and CausalAgents can be joined onto the completed pilot outputs.

Checks:

```text
intersection_rate = count(joined scenario IDs) / count(pilot rollout scenario IDs)
agent_overlap_rate = count(rel_id intersect causal_agent_ids) / count(joined scenarios)
```

Outputs:

- `joined_metadata.parquet` or `joined_metadata.jsonl`
- Scenario-level overlap stats
- Agent-level overlap stats
- Coverage report for `full_metadata`, `reasoning_only`, and `causal_only`

### Stage 3: Method Prototype

Goal:

Implement a minimal risk-aware selection layer.

Compare:

```text
LatentDriver vs RiskAwareLatentDriver
```

on the exact same pilot subset.

### Stage 4: Main Subset

Goal:

Stay on the fixed 10-shard subset until the metric and method stabilize, then expand to 20 or more `validation_interactive` shards.

This is the right level for preliminary paper-style evidence and GPU cluster justification.

### Stage 5: Full Evaluation

Goal:

Run full `validation_interactive`, then regular validation if needed.

Use this only after the metric, buckets, and method are frozen.

## 16. Validation Rules

The study is valid only if these invariants hold:

1. Same scenario IDs for all planners.
2. Same seed.
3. Same simulation mode.
4. Same preprocessing artifacts.
5. Same metric code.
6. Same bucket definitions.
7. Same aggregation rules.
8. No changing bucket definitions after seeing results.
9. Report both aggregate and bucketed results.
10. Report sample count `N` for every bucket.

## 17. Failure Modes

| Failure Mode | Impact | Fallback |
| --- | --- | --- |
| WOMD-Reasoning and CausalAgents IDs do not overlap enough | Cannot use combined metadata fully | Use WOMD-Reasoning only, plus heuristic causal pressure |
| Agent IDs do not join cleanly | Agent-level analysis weaker | Keep scenario-level analysis first |
| LatentDriver does not expose candidate trajectories | Risk-aware selector hard to implement | Start with evaluation methodology, then patch method later |
| Validation-interactive is not supported by current preprocessing | Requires script/config extension | Start with regular validation and interaction buckets |
| Bucket sizes too small | Unstable conclusions | Increase shard count or merge buckets |
| TTC or comfort unavailable | Metric incomplete | Start with collision, offroad, progress |

## 18. GPU Cluster Justification

The GPU request should be framed around scientific necessity, not convenience.

Suggested statement:

```text
I am building a diagnostic closed-loop planner evaluation protocol using WOMD interaction and causal labels. The experiment requires paired simulation rollouts for IDM, LatentDriver, and a proposed risk-aware planner over fixed interaction-heavy validation subsets. Colab is unreliable for this because each comparison requires persistent GPU runtime, local storage, and resumable simulation across many shards. University GPU access is needed to complete the paired causal-semantic safety-progress evaluation reproducibly.
```

## 19. Immediate Next Tasks

1. Verify overlap between WOMD-Reasoning `sid` and CausalAgents `scenario_id`.
2. Add support for `validation_interactive` subset evaluation if current configs only target regular validation.
3. Export per-scenario metrics from Waymax/LatentDriver rollouts.
4. Implement bucket assignment from metadata.
5. Implement BaseScore and Balanced CS-SP.
6. Run IDM vs LatentDriver on the fixed 10-shard interaction subset.
7. Use the result to select the first method intervention.
8. Implement a minimal risk-aware candidate selector.
9. Run LatentDriver vs YourMethod on the same subset.
10. Produce aggregate and bucketed tables.

## 20. Key Invariant

Every scientific comparison must be paired by scenario ID.

If baseline and method are evaluated on different scenarios, the result is not interpretable.

## 21. Independent Reasoning Check

Before implementation, verify this independently:

> Does the metric measure safer progress, or does it accidentally reward conservative stopping?

If a planner avoids all collisions by never moving, the metric should expose poor progress.

## 22. Conceptual Challenge

State the research claim in one sentence:

```text
Our method improves the safety-progress tradeoff over LatentDriver in high-causal-pressure interactive scenarios, as measured by paired bucketed CS-SP on WOMD validation_interactive.
```

If that sentence changes, the metric and experiment design should be reviewed.

## 23. References To Use As Methodological Inspiration

- Waymo Open Motion Dataset: interactive motion forecasting and `objects_of_interest`.
- WOMD-Reasoning: semantic interaction and intention Q&A over WOMD.
- CausalAgents: human-labeled causal agents and causal perturbation robustness.
- nuPlan: open-loop, closed-loop non-reactive, and closed-loop reactive evaluation tiers.
- NAVSIM/PDM Score: safety-progress-comfort composite scoring.
- CARLA Leaderboard: route completion with infraction penalties.
- Waymax: large-scale closed-loop simulation for planning and multi-agent behavior.

## 24. Missing Design Safeguards Added

The first version of this plan defined the scientific question, metric, and experiment ladder. The missing pieces were mostly about rigor:

1. How to avoid metric gaming.
2. How to report uncertainty.
3. How to avoid cherry-picking scenario buckets.
4. How to handle small buckets.
5. How to define valid qualitative examples.
6. How to separate method development from final evaluation.
7. How to make the result reproducible for a paper or thesis committee.

The following sections close those gaps.

## 25. Pre-Registration Protocol

Before running the main comparison, freeze an experiment manifest.

The manifest should contain:

```json
{
  "experiment_id": "validation_interactive_pilot_v1",
  "dataset_version": "waymo_open_dataset_motion_vX_Y_Z",
  "split": "validation_interactive",
  "scenario_ids_path": "artifacts/experiments/.../scenario_ids.txt",
  "planners": ["idm", "latentdriver_t2_j3", "risk_aware_latentdriver_t2_j3"],
  "simulation_mode": "reactive",
  "seed": 0,
  "metrics_version": "cs_sp_v1",
  "bucket_version": "causal_semantic_buckets_v1",
  "aggregation_rule": "balanced_bucket_mean",
  "frozen_at": "ISO_TIMESTAMP"
}
```

After this manifest is frozen, do not change:

- scenario IDs,
- bucket definitions,
- metric weights,
- safety thresholds,
- aggregation rules,
- primary comparison.

If changes are necessary, create a new experiment version.

## 26. Data Split Discipline

Use these roles:

| Split or subset | Allowed use |
| --- | --- |
| `validation_interactive_proto_10_shards` | Rapid prototyping, debugging, and early evidence |
| `validation_interactive_expanded_20_plus_shards` | Stronger preliminary method development and ablation |
| `validation_interactive_frozen_holdout` | Final validation claim after method is fixed |
| `testing_interactive` | Only for official benchmark-style final evaluation, not iterative development |

Do not tune the method repeatedly on the same subset used for the final claim.

If data is limited, split `validation_interactive` scenario IDs into:

```text
development subset: 70%
holdout subset: 30%
```

Keep the holdout untouched until the method and metric are frozen.

## 27. Subset Manifest Requirements

Every subset must be saved as a manifest file.

Required fields per scenario:

| Field | Purpose |
| --- | --- |
| `scenario_id` | Pairing key |
| `source_shard` | Reproducibility |
| `selection_reason` | Why included |
| `bucket_labels_at_selection_time` | Prevent post-hoc bucket edits |
| `has_womd_reasoning` | Metadata availability |
| `has_causal_agents` | Metadata availability |
| `objects_of_interest_count` | Interaction indicator |
| `valid_agent_count` | Density |
| `selected_for_stage` | Pilot, main, holdout |

The scenario manifest is as important as the model checkpoint. Without it, results are not reproducible.

## 28. Statistical Reporting

For each planner comparison, report more than point estimates.

Required statistical outputs:

1. Mean metric per planner.
2. Paired delta per scenario.
3. Bootstrap confidence interval for the mean paired delta.
4. Bucket sample count `N`.
5. Confidence interval per bucket when `N` is large enough.

Primary paired delta:

```text
delta_i = BaseScore_i(YourMethod) - BaseScore_i(LatentDriver)
```

Bucket-level delta:

```text
delta_b = mean(delta_i for scenario i in bucket b)
```

Bootstrap:

```text
sample scenarios with replacement within each bucket
compute bucket delta
repeat 1000 times
report 2.5 and 97.5 percentiles
```

Minimum rule:

```text
Do not make a strong claim for any bucket with N < 30.
```

For small buckets, report them as qualitative or exploratory.

## 29. Multiple Comparisons Control

Bucketed evaluation creates many comparisons. This increases the chance of finding accidental improvements.

Use this reporting discipline:

1. Declare 3 to 5 primary buckets before running the final experiment.
2. Treat all other buckets as exploratory.
3. Do not claim broad improvement if gains appear only in one small exploratory bucket.
4. Report negative buckets, not only positive buckets.

Recommended primary buckets:

```text
causal_high
reasoning_rule
reasoning_intention
dense_scene
intersection_or_turn
```

## 30. Metric Gaming Checks

The CS-SP metric can be gamed if a planner becomes too conservative.

Add these checks:

| Failure | Detection |
| --- | --- |
| Planner stops to avoid collisions | Low progress, low route completion |
| Planner drives slowly everywhere | Low progress and speed-normalized progress |
| Planner overfits to causal-agent labels | Improvement only where labels are available |
| Planner sacrifices sparse scenes for dense scenes | Negative delta in easy/sparse buckets |
| Planner improves score but increases near misses | TTC or minimum-distance degradation |

Report a method as successful only if it improves safety without unacceptable progress collapse.

A useful interpretation grid:

| Safety change | Progress change | Interpretation |
| --- | --- | --- |
| Better | Better | Strong improvement |
| Better | Same | Safety improvement |
| Better | Worse | Conservative tradeoff |
| Same | Better | Efficiency improvement |
| Worse | Better | Unsafe progress |
| Worse | Worse | Regression |

## 31. Pareto Reporting

Do not rely only on a scalar score.

For each planner, plot or tabulate:

```text
x-axis: progress
y-axis: safety failure rate
```

A method is better if it moves toward:

```text
higher progress
lower safety failure
```

The scalar CS-SP is the headline, but the Pareto view shows whether the score hides a bad tradeoff.

## 32. Baselines And Ablations

Minimum baselines:

| Baseline | Reason |
| --- | --- |
| IDM | Classical reactive anchor |
| LatentDriver T2-J3 | Main learned baseline |
| LatentDriver default selector | Required if modifying selection |
| Risk-aware selector without causal term | Tests whether causal term matters |
| Risk-aware selector without reasoning term | Tests whether semantic reasoning matters |

Recommended ablations:

| Ablation | Question |
| --- | --- |
| `progress_only` | Does progress alone cause unsafe behavior? |
| `safety_only` | Does safety alone become overconservative? |
| `safety_plus_progress` | Is causal-semantic information needed? |
| `causal_union` vs `causal_majority` | Does causal-label strictness matter? |
| `reasoning_only` | Can semantic labels help without causal labels? |
| `causal_only` | Can causal labels help without language reasoning? |

The strongest paper claim comes from showing that the combined causal-semantic version beats these simpler variants.

## 33. Per-Scenario Logging Contract

The current evaluation pipeline must eventually export per-scenario outputs.

Required per-scenario output:

| Field | Required |
| --- | --- |
| `run_id` | yes |
| `planner_name` | yes |
| `scenario_id` | yes |
| `source_shard` | yes |
| `seed` | yes |
| `collision` | yes |
| `offroad` | yes |
| `progress` | yes |
| `route_completion` | preferred |
| `min_ttc` | preferred |
| `min_distance_to_agent` | preferred |
| `comfort_acceleration` | optional |
| `comfort_jerk` | optional |
| `selected_candidate_id` | required for candidate selector studies |
| `candidate_scores` | required for selector ablations |
| `failure_reason` | derived |

Aggregate-only metrics are insufficient for this research direction.

## 34. Qualitative Case Study Protocol

Qualitative examples are useful, but they must not be cherry-picked.

Select case studies using fixed rules:

1. Largest positive delta in `causal_high`.
2. Largest negative delta in `causal_high`.
3. Collision fixed by method.
4. Progress regression caused by method.
5. Example where causal agent appears in WOMD-Reasoning `rel_id`.

For each case:

- show baseline rollout,
- show method rollout,
- show causal agents,
- show reasoning text summary,
- show metric delta,
- explain failure or improvement in one paragraph.

## 35. Validation-Interactive Compatibility Check

Before committing to `validation_interactive`, verify:

1. The current Waymax/LatentDriver dataloader can read `validation_interactive` files.
2. Preprocessing can create map, route, and intention labels for `validation_interactive`.
3. Scenario IDs emitted during simulation match WOMD-Reasoning `sid`.
4. The same scenario ID can be found in CausalAgents labels.
5. Reactive simulation works on this split without assumptions tied to regular validation.

If any check fails, start with regular validation plus interaction buckets, then add `validation_interactive` support later.

## 36. Scenario Bucket Construction Order

Construct buckets in this order:

1. Metadata buckets from WOMD and WOMD-Reasoning.
2. Causal-agent buckets from CausalAgents.
3. Geometry/density buckets from scenario proto.
4. Failure buckets from planner outcomes.

Do not define primary buckets from planner failures alone. That would leak outcome information into the evaluation design.

Failure buckets are allowed for diagnosis after the primary comparison.

## 37. Causal Label Consensus Rules

CausalAgents has multiple labelers. Define three causal sets:

```text
causal_union = agent selected by at least 1 labeler
causal_majority = agent selected by at least 3 labelers
causal_unanimous = agent selected by all labelers
```

Use `causal_union` for conservative safety analysis.

Use `causal_majority` for main quantitative claims.

Use `causal_unanimous` for high-confidence qualitative examples.

Report which consensus rule is used.

## 38. Handling Missing Or Partial Metadata

Not every scenario may have all metadata.

Define metadata availability groups:

| Group | Meaning |
| --- | --- |
| `full_metadata` | Has WOMD-Reasoning and CausalAgents |
| `reasoning_only` | Has WOMD-Reasoning only |
| `causal_only` | Has CausalAgents only |
| `scenario_only` | Has only raw WOMD metadata |

Main causal-semantic claims should use `full_metadata`.

Broader aggregate results can use all scenarios, but must report metadata coverage.

## 39. Threats To Validity

| Threat | Why it matters | Mitigation |
| --- | --- | --- |
| Metadata labels are imperfect | Reasoning and causal labels may contain noise | Use bucket-level trends, not single-label absolute truth |
| Causal labels are subjective | Labelers may disagree | Report union and majority variants |
| Validation-interactive is biased toward interactions | Results may not generalize to normal driving | Later evaluate regular validation |
| Simulator agents are imperfect | Closed-loop behavior depends on reactive model | Compare reactive and non-reactive modes |
| Method may overfit to labels | It may exploit metadata unavailable at deployment | Separate evaluation-only labels from planner inputs unless explicitly studying privileged methods |
| Small bucket sizes | High variance | Minimum N thresholds and confidence intervals |

## 40. Deployment-Realism Boundary

Decide whether causal-semantic metadata is used only for evaluation or also for the planner method.

Two possible settings:

### Evaluation-only metadata

The planner does not see WOMD-Reasoning or CausalAgents labels.

Use labels only for analysis.

This is the default setting for the current fixed 10-shard `validation_interactive` rapid-prototyping benchmark.

This is more deployment-realistic.

### Privileged-method metadata

The planner uses causal or reasoning labels during selection.

This may be acceptable as an oracle study, but must be labeled clearly as privileged.

Recommended path:

1. Start with evaluation-only metadata.
2. Use it to identify failures.
3. Design a deployable proxy method that estimates causal pressure from observable scene features.
4. Compare proxy method against an oracle upper bound using true causal labels.

## 41. Oracle And Proxy Experiments

For method development, separate oracle and deployable versions.

| Version | Uses ground-truth metadata? | Purpose |
| --- | --- | --- |
| `oracle_causal_selector` | Yes | Upper bound and hypothesis test |
| `proxy_causal_selector` | No | Deployable method |
| `default_selector` | No | Baseline |

If the oracle works but the proxy fails, the issue is causal-pressure estimation.

If the oracle also fails, the proposed causal-selection idea may be weak.

## 42. Non-Reactive Comparison Role

Reactive evaluation is primary, but non-reactive evaluation is still useful.

Use it to distinguish:

```text
ego-policy improvement
vs
interaction-feedback improvement
```

If a method improves reactive but not non-reactive:

```text
The method likely helps interactive feedback behavior.
```

If it improves non-reactive but not reactive:

```text
The method may match replay better but fail under interaction.
```

If it improves both:

```text
The method likely improves general ego behavior.
```

## 43. Compute Budget Plan

Use compute in stages.

| Stage | Planners | Data | Purpose |
| --- | --- | --- | --- |
| `S0` | IDM only | 1 shard | Pipeline sanity |
| `S1` | IDM + LatentDriver | 10 shards | Plain WOMD rapid-prototyping comparison |
| `S2` | Metadata join overlay | same 10 shards | Attach WOMD-Reasoning and CausalAgents after rollout for diagnostic insight |
| `S3` | LatentDriver + proxy method | 10 shards | Main rapid-prototyping evidence once overlay diagnostics suggest a method |
| `S4` | IDM + LatentDriver + proxy method | 20 shards | GPU request / thesis committee evidence |
| `S5` | Full suite | full split | Final result |

Do not spend full-validation compute before the pilot shows a meaningful signal.

## 44. Paper-Style Contribution Statement

A strong final contribution could be stated as:

```text
We propose a causal-semantic closed-loop evaluation protocol for autonomous driving planners by joining WOMD-Reasoning interaction semantics with CausalAgents human causal labels. Using this protocol, we show that aggregate planner metrics hide systematic failures under high causal interaction pressure. We then introduce a risk-aware candidate selection method for latent multi-hypothesis planners and demonstrate improved safety-progress tradeoff over LatentDriver in interaction-heavy validation scenarios.
```

This statement has two parts:

1. Evaluation methodology.
2. Planner method.

Both parts must be supported experimentally.
