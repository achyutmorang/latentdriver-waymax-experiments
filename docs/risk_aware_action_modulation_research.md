# Research Note: Post-Hoc Risk-Aware Action Modulation for Pretrained Driving Planners

## 1. Purpose

This note investigates a lightweight, planner-agnostic method that sits on top of a pretrained driving planner and modulates its output action at inference time based on predicted future interaction risk.

The target use case is the current repo setup:

- frozen pretrained planners,
- Waymax closed-loop simulation,
- fixed 10-shard `validation_interactive` rapid-prototyping subset,
- causal-semantic evaluation attached after rollout,
- no retraining of the base planner.

The concrete research question is:

> Can a post-hoc risk-aware action modulation layer improve the safety-progress tradeoff of a frozen planner under high interaction pressure without retraining the planner backbone?

This method is meant to be portable across:

- LatentDriver,
- PlanT as currently wired in this repo,
- and future planners that produce a single ego action per step.

## 2. Repo-Grounded Feasibility

### 2.1 Current action interface

In this repo, both LatentDriver and the PlanT baseline are evaluated under the same local delta waypoint action space:

- `external/LatentDriver/configs/method/latentdriver.yaml`
- `external/LatentDriver/configs/method/planT.yaml`

Both use:

```yaml
action_space:
  dynamic_type: waypoint
  action_ranges: [[-0.14, 6], [-0.35, 0.35], [-0.15, 0.15]]
```

So the action is a 3D local delta:

```text
[dx, dy, dyaw]
```

This is important because a planner-agnostic modulator only needs to wrap that action tensor.

### 2.2 Exact insertion point

The current inference path is:

1. planner predicts action in:
   - `external/LatentDriver/simulator/engines/ltd_simulator.py`
2. predicted action is assigned to `control_action`
3. environment steps with:
   - `self.env.step(control_action, show_global=True)`

The minimal intervention point is therefore between:

```python
action = self.model.get_predictions(...)
control_action = action
obs, obs_dict, rew, done, info = self.env.step(control_action, show_global=True)
```

This is the correct place to insert a post-hoc modulator.

### 2.3 Why modulation is more portable than reranking

LatentDriver exposes candidate diversity in this repo, but PlanT does not.

So:

- candidate reranking is a strong LatentDriver-specific path,
- action modulation is the stronger planner-agnostic path.

This makes modulation the right first cross-planner innovation.

## 3. What the Literature Suggests

The most relevant literature does not point to one exact method. It points to a family of successful ideas:

1. learn or estimate short-horizon risk at runtime,
2. minimally modify the policy output rather than replacing the whole planner,
3. preserve performance when risk is low,
4. intervene more strongly when safety risk is high.

### 3.1 Safety layers in continuous control

Dalal et al., ["Safe Exploration in Continuous Action Spaces"](https://arxiv.org/abs/1801.08757), add a safety layer that corrects actions analytically per state using a learned linearized constraint model.

The key transferable idea is not the exact RL setup. It is this:

- keep the policy,
- predict whether an action is unsafe,
- apply a minimal corrective transformation to the action.

That maps directly onto our setting.

### 3.2 Learned safety critics

Srinivasan et al., ["Learning to be Safe: Deep RL with a Safety Critic"](https://arxiv.org/abs/2010.14603), show that a learned safety signal can constrain behavior in new environments and reduce safety incidents.

The transferable idea here is:

- separate task competence from safety competence,
- use a small learned risk head to modulate a stronger base controller.

This is highly relevant for a frozen-pretrained-planner regime.

### 3.3 Safety filters and fallback layers for driving

Vitelli et al., ["SafetyNet: Safe planning for real-world self-driving vehicles using machine-learned policies"](https://arxiv.org/abs/2109.13602), deploy a learned planner together with a simple rule-based fallback layer that performs sanity checks like collision avoidance and physical feasibility.

Their central empirical result is directly aligned with this project:

- keep the ML planner,
- add a lightweight safety layer,
- reduce planner-only collisions substantially.

This paper is the strongest real-world evidence that a small post-hoc safeguard can matter.

### 3.4 Control Barrier Function based safety filters

Ames et al., ["Control Barrier Function Based Quadratic Programs for Safety Critical Systems"](https://arxiv.org/abs/1609.06408), and Xiao et al., ["Rule-based Optimal Control for Autonomous Driving"](https://arxiv.org/abs/2101.05709), formalize the idea of minimally altering control to stay inside a safe set.

The transferable ideas are:

- safety should be minimally invasive,
- performance and safety should be mediated explicitly,
- the safety layer can be viewed as a separate optimization problem.

For our first implementation, we likely do not want a full CBF-QP stack. But the design principle is exactly right.

### 3.5 Predictive runtime monitoring

Yoon and Sankaranarayanan, ["Predictive Runtime Monitoring for Mobile Robots using Logic-Based Bayesian Intent Inference"](https://arxiv.org/abs/2108.01227), forecast future states online to detect impending violations.

Peddi and Bezzo, ["Interpretable Run-Time Prediction and Planning in Co-Robotic Environments"](https://arxiv.org/abs/2109.03893), use an interpretable runtime monitor that predicts interference and plans corrective behavior.

The transferable idea is:

- do not wait for collision or overlap to happen,
- predict a short-horizon failure likelihood and intervene before the event.

This strongly supports a future-risk predictor instead of a purely reactive brake layer.

### 3.6 Confidence-aware prediction under interaction uncertainty

Fisac et al., ["Probabilistically Safe Robot Planning with Confidence-Based Human Predictions"](https://arxiv.org/abs/1806.00109), explicitly reason about uncertainty in future agent behavior and increase conservatism when predictive confidence degrades.

This is relevant because in high interaction pressure scenes, a fixed aggressiveness threshold is often wrong. The safety layer should become more conservative when uncertainty or interaction density rises.

### 3.7 Why inference-time modification is an active research direction

Recent planner work continues to reinforce the value of inference-time safety shaping:

- Zheng et al., ["Diffusion-Based Planning for Autonomous Driving with Flexible Guidance"](https://arxiv.org/abs/2501.15564), use guidance at inference to balance objectives.
- Yao et al., ["ReflexDiffusion"](https://arxiv.org/abs/2601.09377), show that inference-time trajectory adjustment can improve safety-critical regimes without retraining the full planner.

These are diffusion-specific, not directly reusable here. But they support the broader claim that inference-time control shaping is a serious planner research direction, not a hack.

## 4. Why This Fits Our Evaluation Setup

Waymax is specifically designed for large-scale interactive, accelerator-friendly closed-loop simulation, and its authors explicitly highlight multi-agent interactions and the risk of planners overfitting to simulated agents:

- Gulino et al., ["Waymax"](https://arxiv.org/abs/2310.08710)

LatentDriver itself explicitly models multiple probabilistic decisions before deriving a deterministic control output:

- Xiao et al., ["Learning Multiple Probabilistic Decisions from Latent World Model in Autonomous Driving"](https://arxiv.org/abs/2409.15730)

PlanT is object-centric and fast, but in this repo it currently exposes a single final action:

- Renz et al., ["PlanT"](https://arxiv.org/abs/2210.14222)

nuPlan emphasizes that the right evaluation target is closed-loop behavior in diverse reactive scenarios rather than only open-loop trajectory matching:

- Karnchanachari et al., ["Towards learning-based planning: The nuPlan benchmark for real-world autonomous driving"](https://arxiv.org/abs/2403.04133)

Therefore, for this repo:

- a safety-progress wrapper is methodologically aligned,
- closed-loop evaluation is the right place to test it,
- and a planner-agnostic action modulator is more transferable than a planner-specific candidate reranker.

## 5. Causal-Semantic Evaluation Constraint

Our current protocol keeps planner inputs plain WOMD and attaches WOMD-Reasoning plus CausalAgents only after rollout for evaluation.

That means the first fair method should **not** consume causal-semantic labels at inference time.

So the modulator should use only runtime-available proxy features such as:

- local interaction density,
- minimum TTC,
- relative speed,
- overlap risk,
- curvature or turn severity,
- route-following context,
- ego speed and yaw change,
- planner action magnitude.

Then, after rollout, we use causal-semantic metadata to ask:

> Does the modulator help more in high-causal-pressure regimes?

This is the correct fairness boundary for the first paper.

## 6. Design Space

### Option A: Heuristic TTC brake layer

Mechanism:

- compute TTC-like proxy from current state and proposed action,
- if TTC below threshold, uniformly scale the action down.

Pros:

- simple,
- no training,
- very low engineering cost.

Cons:

- easy to over-brake,
- ignores map and route context,
- no learned calibration,
- may hurt progress badly in merges and turns.

Use:

- required baseline,
- not likely the best final method.

### Option B: Learned risk predictor plus analytic scaling

Mechanism:

- predict future risk from current state plus proposed action,
- convert predicted risk into a scale factor,
- apply scale factor to the ego action.

Pros:

- planner-agnostic,
- lightweight,
- cheap at inference,
- learns when to intervene instead of using only hand-tuned TTC thresholds.

Cons:

- requires an offline training set for the risk head,
- needs carefully designed labels.

Use:

- recommended main method.

### Option C: Short-horizon ghost-rollout selector over action scales

Mechanism:

- for each step, evaluate several scaled versions of the proposed action through a short simulator lookahead,
- choose the best one under an explicit risk-progress objective.

Pros:

- no learned model required,
- strong and interpretable,
- can generate labels for Option B.

Cons:

- more expensive,
- less elegant for deployment,
- may be too slow if used online over long horizons.

Use:

- excellent oracle or upper-bound baseline,
- best source of offline supervision for Option B.

### Option D: LatentDriver candidate reranking

Mechanism:

- rerank candidate actions before native LatentDriver selection.

Pros:

- very strong for LatentDriver,
- directly aligns with multimodal latent decisions.

Cons:

- not planner-agnostic,
- not available for PlanT in current repo wiring.

Use:

- later LatentDriver-specific ablation,
- not the first general method.

## 7. Recommended Method

The best first method is:

```text
Learned risk predictor + analytic action scaling
```

This yields a lightweight, planner-agnostic wrapper.

### 7.1 Proposed name

Use a descriptive working name first:

```text
Risk-Aware Sensitivity Modulator
RASM
```

The name is not important. The interface is.

### 7.2 Core interface

At each step:

```text
state_t, planner_action_t -> risk_features_t -> risk_head -> scale_t -> modulated_action_t
```

Then:

```text
env.step(modulated_action_t)
```

### 7.3 Output form

Start with a scalar scale:

```text
s_t in [s_min, 1]
```

and:

```text
a'_t = s_t * a_t
```

This is the simplest and most portable version.

Later ablation:

- longitudinal-only scaling,
- longitudinal plus lateral paired scaling,
- heading-preserving scaling.

### 7.4 Why scalar first

A scalar modulator:

- is compatible with both LatentDriver and PlanT immediately,
- preserves the action direction,
- minimizes intervention complexity,
- is easier to interpret in CS-SP analysis.

## 8. Recommended Inputs

The first fair version should use only information available online from the simulator state and planner output.

### 8.1 Ego features

- current ego speed,
- current ego yaw and yaw rate,
- current ego valid flag,
- recent action history,
- proposed action `[dx, dy, dyaw]`,
- proposed action norm,
- route heading mismatch if accessible.

### 8.2 Interaction features

For top `N` nearest valid agents in ego-centric coordinates:

- relative position `(x_rel, y_rel)`,
- relative velocity `(vx_rel, vy_rel)`,
- relative heading,
- distance,
- closing speed,
- object type if available,
- whether agent lies in front cone of ego.

### 8.3 Risk proxies

- minimum approximate TTC,
- count of nearby agents under distance threshold,
- weighted interaction density,
- minimum predicted clearance,
- near-overlap flag from one-step ghost rollout,
- offroad proximity proxy if accessible.

### 8.4 Optional planner-specific features

These should be optional, not required:

- LatentDriver mode entropy,
- candidate disagreement,
- world-model uncertainty,
- PlanT attention-derived relevance if later exposed.

Do not make the first version depend on them.

## 9. Risk Targets

The modulator should predict short-horizon risk, not only current-state danger.

Recommended targets:

### 9.1 Binary event targets

- collision within `H` steps,
- offroad within `H` steps,
- severe TTC violation within `H` steps.

### 9.2 Continuous targets

- minimum TTC over horizon,
- minimum clearance over horizon,
- interaction density over horizon,
- progress retained over horizon.

### 9.3 Horizon choice

Start with:

```text
H = 3 to 5 simulation steps
```

Reason:

- long enough to detect imminent interaction failures,
- short enough for cheap ghost rollouts,
- consistent with a modulation layer rather than a full replanner.

## 10. Label Generation Strategy

This is the most important engineering choice.

The cleanest approach is to generate offline labels using simulator-consistent short-horizon ghost rollouts.

### 10.1 For each encountered state-action pair

Given:

- current simulator state `s_t`,
- planner proposal `a_t`,

construct a small discrete set of scale candidates:

```text
S = {0.25, 0.5, 0.75, 1.0}
```

For each scale `s`:

```text
a_t^(s) = s * a_t
```

Run a short ghost rollout from the current state and compute:

- collision in horizon,
- offroad in horizon,
- min TTC,
- progress retained.

### 10.2 Oracle objective for training labels

Define:

```text
J(s) = w_collision * collision_H
     + w_offroad * offroad_H
     + w_ttc * softplus(ttc_safe - min_ttc_H)
     + w_progress * progress_loss_H
```

Choose:

```text
s_star = argmin_s J(s)
```

Then train the modulator to predict `s_star` or the associated risk profile.

This is attractive because:

- labels are simulator-consistent,
- no manual labeling required,
- planner remains frozen,
- we can use the 10-shard pilot to bootstrap the dataset.

## 11. Model Formulation

### 11.1 Phase 1: risk-head plus analytic scaler

Train a tiny MLP:

```text
f_theta(z_t) -> r_t
```

where `z_t` is the feature vector.

Then convert risk to scale:

```text
s_t = clip(sigmoid(alpha - beta * r_t), s_min, 1.0)
```

This separates:

- risk estimation,
- action scaling policy.

It is easier to debug than direct scale regression.

### 11.2 Phase 2: direct scale predictor

Train:

```text
g_theta(z_t) -> s_t
```

directly against `s_star`.

This may work better eventually, but it is harder to interpret.

### 11.3 Recommended first training objective

Use a multi-task loss:

```text
L = lambda_cls * BCE(collision_H_hat, collision_H)
  + lambda_ttc * SmoothL1(min_ttc_H_hat, min_ttc_H)
  + lambda_scale * SmoothL1(scale_hat, s_star)
```

If simplifying:

- first predict risk only,
- derive scale analytically.

## 12. Action Parameterization Details

Because the action is local delta waypoint:

```text
[dx, dy, dyaw]
```

uniform scaling is the first correct implementation:

```text
[dx', dy', dyaw'] = s * [dx, dy, dyaw]
```

This usually means:

- shorter movement,
- smaller lateral offset,
- smaller heading change.

### 12.1 Risk of uniform scaling

Uniform scaling may under-turn in sharp bends and intersections.

That is why the next ablation should be:

```text
dx'   = s_long * dx
dy'   = s_shape * dy
dyaw' = s_shape * dyaw
```

with:

```text
s_shape = gamma + (1 - gamma) * s_long
```

for some small `gamma > 0`.

This preserves more turning structure while still slowing the vehicle.

### 12.2 Recommendation

Implement scalar first.

Only add anisotropic scaling if scalar modulation improves safety but hurts turn completion too much.

## 13. Evaluation Plan

### 13.1 Core comparisons

Run on the fixed 10-shard plain WOMD `validation_interactive` pilot:

1. IDM
2. LatentDriver default
3. PlanT default
4. heuristic TTC scaler
5. learned RASM

Then attach WOMD-Reasoning and CausalAgents after rollout.

### 13.2 Main metrics

- collision rate,
- offroad rate,
- progress rate,
- BaseScore,
- Balanced CS-SP,
- intervention rate,
- mean scale,
- fraction of steps with strong intervention,
- progress-retention ratio relative to unmodified planner.

### 13.3 Main bucket analysis

Focus on:

- `causal_high`,
- `causal_reasoning_overlap`,
- `reasoning_rule`,
- `dense_scene`,
- `intersection_or_turn`,
- `near_causal_agent`.

### 13.4 Main hypothesis

The right hypothesis is not:

> The modulator improves average score everywhere.

It is:

> The modulator reduces failure rates in high interaction pressure regimes with acceptable progress loss, improving the safety-progress Pareto tradeoff.

## 14. Required Baselines and Ablations

These are necessary to prevent weak claims.

### 14.1 Required baselines

- planner default output,
- constant global scale `s = 0.75`,
- constant global scale `s = 0.5`,
- heuristic TTC-only scale rule,
- learned RASM.

### 14.2 Important ablations

- scalar vs anisotropic scaling,
- no density feature,
- no TTC feature,
- no ghost-rollout labels, direct heuristics only,
- LatentDriver-only optional uncertainty feature,
- oracle causal-semantic feature version as privileged ablation only.

### 14.3 Why these matter

Without constant-scale and TTC-only baselines, a learned modulator can look better than it really is.

## 15. Failure Modes

### 15.1 Over-conservatism

The modulator may reduce collisions simply by barely moving.

Mitigation:

- report progress retention,
- compare against constant scaling baselines,
- inspect CS-SP plus Pareto plots,
- require improvement beyond trivial braking.

### 15.2 Proxy mismatch

Short-horizon TTC may miss future offroad or negotiated merge failures.

Mitigation:

- add ghost-rollout overlap and offroad targets,
- use short-horizon simulator-consistent labels.

### 15.3 Planner mismatch

A modulator trained mostly on LatentDriver actions may not transfer cleanly to PlanT actions.

Mitigation:

- train on mixed planner outputs,
- or train planner-specific tiny heads on shared feature backbone.

### 15.4 Turn degradation

Uniform scaling may cause under-turning or route loss in curved scenarios.

Mitigation:

- analyze `intersection_or_turn` bucket separately,
- add anisotropic scaling ablation.

### 15.5 Oracle leakage

If causal-semantic labels are used at inference in the main experiment, the result becomes a privileged setting.

Mitigation:

- keep the main method proxy-only,
- use causal-semantic labels only in analysis or oracle ablations.

## 16. What This Adds Beyond Candidate Reranking

Reranking is valuable for LatentDriver, but it does not generalize to PlanT in the current repo.

The modulator adds:

- planner-agnostic applicability,
- compatibility with single-action planners,
- a clean runtime-safety interpretation,
- a direct connection to safety-layer and fallback-controller literature.

This makes it the stronger cross-model method contribution.

## 17. Recommended Implementation Phases

### Phase A: analytic baseline

Implement:

- top-N neighbor extraction,
- TTC estimator,
- density estimator,
- scalar modulation rule.

No learning yet.

### Phase B: ghost-rollout label generator

Implement:

- short-horizon ghost rollout for scaled actions,
- `s_star` target generation,
- offline training dataset dump.

### Phase C: learned modulator

Implement:

- lightweight feature encoder,
- risk head,
- analytic scale mapping,
- runtime wrapper in `ltd_simulator.py`.

### Phase D: causal-semantic evaluation

After rollouts:

- join WOMD-Reasoning and CausalAgents,
- compute bucketed CS-SP,
- identify where the modulator helps most.

### Phase E: LatentDriver-specific hybrid

If the general modulator works:

- combine candidate reranking with action modulation,
- compare selector-only vs modulator-only vs hybrid.

## 18. Recommended First Claim

The strongest defensible first claim is:

> A lightweight post-hoc risk-aware action modulation layer can improve the safety-progress tradeoff of frozen pretrained planners in interactive closed-loop simulation, particularly in scenarios with high causal interaction pressure, without retraining the planner backbone.

This is stronger and more portable than a claim tied only to LatentDriver candidate reranking.

## 19. Practical Recommendation

For this repo, the correct next implementation target is:

1. a planner-agnostic scalar action modulator,
2. trained or calibrated from short-horizon simulator-consistent risk labels,
3. evaluated on the fixed 10-shard `validation_interactive` pilot,
4. analyzed with post-rollout causal-semantic buckets.

If only one path should be built first, build:

```text
heuristic TTC+density modulator -> learned risk head -> causal-semantic bucket analysis
```

not:

```text
full fine-tuning of the pretrained planner
```

## 20. References

- Dalal et al., ["Safe Exploration in Continuous Action Spaces"](https://arxiv.org/abs/1801.08757)
- Ames et al., ["Control Barrier Function Based Quadratic Programs for Safety Critical Systems"](https://arxiv.org/abs/1609.06408)
- Fisac et al., ["Probabilistically Safe Robot Planning with Confidence-Based Human Predictions"](https://arxiv.org/abs/1806.00109)
- Xiao et al., ["Rule-based Optimal Control for Autonomous Driving"](https://arxiv.org/abs/2101.05709)
- Yoon and Sankaranarayanan, ["Predictive Runtime Monitoring for Mobile Robots using Logic-Based Bayesian Intent Inference"](https://arxiv.org/abs/2108.01227)
- Peddi and Bezzo, ["Interpretable Run-Time Prediction and Planning in Co-Robotic Environments"](https://arxiv.org/abs/2109.03893)
- Vitelli et al., ["SafetyNet: Safe planning for real-world self-driving vehicles using machine-learned policies"](https://arxiv.org/abs/2109.13602)
- Gulino et al., ["Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research"](https://arxiv.org/abs/2310.08710)
- Ettinger et al., ["Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset"](https://arxiv.org/abs/2104.10133)
- Renz et al., ["PlanT: Explainable Planning Transformers via Object-Level Representations"](https://arxiv.org/abs/2210.14222)
- Li et al., ["WOMD-Reasoning"](https://arxiv.org/abs/2407.04281)
- Refaat et al., ["CausalAgents"](https://arxiv.org/abs/2207.03586)
- Xiao et al., ["Learning Multiple Probabilistic Decisions from Latent World Model in Autonomous Driving"](https://arxiv.org/abs/2409.15730)
- Xiao et al., ["EasyChauffeur: A Baseline Advancing Simplicity and Efficiency on Waymax"](https://arxiv.org/abs/2408.16375)
- Karnchanachari et al., ["Towards learning-based planning: The nuPlan benchmark for real-world autonomous driving"](https://arxiv.org/abs/2403.04133)
- Zheng et al., ["Diffusion-Based Planning for Autonomous Driving with Flexible Guidance"](https://arxiv.org/abs/2501.15564)
- Yao et al., ["ReflexDiffusion"](https://arxiv.org/abs/2601.09377)
