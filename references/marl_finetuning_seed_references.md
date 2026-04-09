# MARL Fine-Tuning Seed References

This directory collects papers and code references that are directly relevant to extending the repo beyond pure replication into:

- candidate-level evaluation,
- reranking over multiple futures,
- lightweight scorer adaptation,
- group-relative fine-tuning,
- and interaction-aware multi-agent reasoning.

## Local paper bundle

Directory:

- [`references/papers/marl_finetuning_seed/`](./papers/marl_finetuning_seed)

## Most directly actionable for this repo

1. **RIFT: Group-Relative RL Fine-Tuning for Realistic and Controllable Traffic Simulation**
   - local: [`2025_rift_group_relative_rl_fine_tuning_for_realistic_and_controllable_traffic_simulation.pdf`](./papers/marl_finetuning_seed/2025_rift_group_relative_rl_fine_tuning_for_realistic_and_controllable_traffic_simulation.pdf)
   - paper: [arXiv](https://arxiv.org/abs/2505.03344)
   - code: [GitHub](https://github.com/mr-d-self-driving/RIFT)
   - why it matters:
     - freeze pretrained generator,
     - evaluate multiple rollout candidates,
     - compute group-relative advantage,
     - improve the scorer/selection layer rather than retraining everything first.

2. **DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving**
   - local: [`2024_diffusiondrive_truncated_diffusion_model_for_end_to_end_autonomous_driving.pdf`](./papers/marl_finetuning_seed/2024_diffusiondrive_truncated_diffusion_model_for_end_to_end_autonomous_driving.pdf)
   - paper: [arXiv](https://arxiv.org/abs/2411.15139)
   - code: [GitHub](https://github.com/hustvl/DiffusionDrive)
   - why it matters:
     - multi-candidate trajectory generation,
     - low-step diffusion for real-time candidate sampling,
     - useful as a reference for candidate diversity and generation cost.

3. **DiffusionDrive-GRPO**
   - local code note: [`diffusiondrive_grpo_README.md`](./code_refs/diffusiondrive_grpo_README.md)
   - code: [GitHub](https://github.com/Ryannn-ry63/DiffusionDrive-GRPO)
   - why it matters:
     - freeze most of the model,
     - keep a frozen reference policy,
     - compute group-relative advantages from planner rewards,
     - optimize a small trainable component with KL regularization.

4. **Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving**
   - local: [`2025_poutine_vlt_pretraining_and_rl_post_training_for_end_to_end_autonomous_driving.pdf`](./papers/marl_finetuning_seed/2025_poutine_vlt_pretraining_and_rl_post_training_for_end_to_end_autonomous_driving.pdf)
   - paper: [arXiv](https://arxiv.org/abs/2506.11234)
   - why it matters:
     - post-training rather than from-scratch training,
     - lightweight RL adaptation on top of pretrained driving models.

5. **WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving**
   - local: [`2025_worldrft_latent_world_model_planning_with_reinforcement_fine_tuning_for_autonomous_driving.pdf`](./papers/marl_finetuning_seed/2025_worldrft_latent_world_model_planning_with_reinforcement_fine_tuning_for_autonomous_driving.pdf)
   - paper: [arXiv](https://arxiv.org/abs/2512.19133)
   - why it matters:
     - latent world model planning,
     - reinforcement fine-tuning on top of a pretrained planning stack.

6. **RoaD: Rollouts as Demonstrations for Closed-Loop Supervised Fine-Tuning of Autonomous Driving Policies**
   - local: [`2025_road_rollouts_as_demonstrations_for_closed_loop_supervised_fine_tuning.pdf`](./papers/marl_finetuning_seed/2025_road_rollouts_as_demonstrations_for_closed_loop_supervised_fine_tuning.pdf)
   - paper: [arXiv](https://arxiv.org/abs/2512.01993)
   - why it matters:
     - strong non-RL comparison point,
     - closed-loop improvement using rollout-derived demonstrations.

## Interaction-aware MARL references

7. **COIN: Collaborative Interaction-Aware Multi-Agent Reinforcement Learning for Self-Driving Systems**
   - local: [`2026_coin_collaborative_interaction_aware_multi_agent_reinforcement_learning_for_self_driving_systems.pdf`](./papers/marl_finetuning_seed/2026_coin_collaborative_interaction_aware_multi_agent_reinforcement_learning_for_self_driving_systems.pdf)
   - paper: [arXiv](https://arxiv.org/abs/2603.24931)
   - why it matters:
     - explicit interaction-aware MARL,
     - centralized multi-agent credit assignment,
     - useful if we later move beyond ego-only reranking.

8. **Socially-Attentive Policy Optimization in Multi-Agent Self-Driving System**
   - local: [`2023_sapo_socially_attentive_policy_optimization_in_multi_agent_self_driving_system.pdf`](./papers/marl_finetuning_seed/2023_sapo_socially_attentive_policy_optimization_in_multi_agent_self_driving_system.pdf)
   - paper: [PMLR](https://proceedings.mlr.press/v205/dai23a.html)
   - why it matters:
     - attention over the most interactive agents,
     - social preference integration.

9. **A Two-stage Based Social Preference Recognition in Multi-Agent Autonomous Driving System**
   - local: [`2023_two_stage_social_preference_recognition_in_multi_agent_autonomous_driving_system.pdf`](./papers/marl_finetuning_seed/2023_two_stage_social_preference_recognition_in_multi_agent_autonomous_driving_system.pdf)
   - paper: [arXiv](https://arxiv.org/abs/2310.03303)
   - why it matters:
     - latent social style modeling,
     - interaction heterogeneity across nearby agents.

10. **COMA: Counterfactual Multi-Agent Policy Gradients**
    - local: [`2017_coma_counterfactual_multi_agent_policy_gradients.pdf`](./papers/marl_finetuning_seed/2017_coma_counterfactual_multi_agent_policy_gradients.pdf)
    - paper: [arXiv](https://arxiv.org/abs/1705.08926)
    - why it matters:
      - foundational counterfactual credit assignment,
      - still useful as a theoretical template for “which agent caused what”.

## Supporting simulator and benchmark papers

11. **SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving**
    - local: [`2020_smarts_scalable_multi_agent_reinforcement_learning_training_school_for_autonomous_driving.pdf`](./papers/marl_finetuning_seed/2020_smarts_scalable_multi_agent_reinforcement_learning_training_school_for_autonomous_driving.pdf)
    - paper: [arXiv](https://arxiv.org/abs/2010.09776)
    - why it matters:
      - benchmark philosophy and training-school framing for driving MARL.

12. **ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling**
    - local: [`2023_scenarionet_open_source_platform_for_large_scale_traffic_scenario_simulation_and_modeling.pdf`](./papers/marl_finetuning_seed/2023_scenarionet_open_source_platform_for_large_scale_traffic_scenario_simulation_and_modeling.pdf)
    - paper: [arXiv](https://arxiv.org/abs/2306.12241)
    - why it matters:
      - scenario-centric training and evaluation substrate,
      - relevant for portability across simulation contracts.

## Reading order

If the goal is to extend **LatentDriver + Waymax** with minimal compute:

1. RIFT
2. DiffusionDrive-GRPO
3. Poutine
4. WorldRFT
5. RoaD
6. COIN
7. SAPO

## What to extract from this bundle

- **RIFT / DiffusionDrive-GRPO / Poutine / WorldRFT**
  - freeze most of the model,
  - use candidate-level rewards,
  - keep a frozen reference policy,
  - adapt only a small planning/scoring component.

- **Plan-R1 / Diffusion-Planner / RIFT** together
  - use explicit evaluator modules and grouped candidate comparison.

- **COIN / SAPO / social-preference papers**
  - focus on critical interacting agents,
  - avoid scoring all agents equally.

- **RoaD**
  - keep a strong non-RL baseline in scope.
