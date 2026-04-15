from __future__ import annotations

import numpy as np

from .base import ModulationDecision, RiskFeatureBatch
from .config import ModulationConfig


class HeuristicActionModulator:
    def __init__(self, config: ModulationConfig) -> None:
        if config.mode != "heuristic":
            raise ValueError(f"HeuristicActionModulator requires mode='heuristic', got {config.mode!r}")
        self.config = config

    def decide(self, features: RiskFeatureBatch) -> ModulationDecision:
        cfg = self.config
        finite_ttc = np.where(np.isfinite(features.min_ttc_seconds), features.min_ttc_seconds, np.inf)
        risk_ttc = np.clip(
            (cfg.ttc_threshold_seconds - finite_ttc)
            / max(cfg.ttc_threshold_seconds - cfg.ttc_hard_stop_seconds, 1e-6),
            0.0,
            1.0,
        )
        risk_distance = np.clip(
            (cfg.distance_threshold_meters - features.min_distance_meters)
            / max(cfg.distance_threshold_meters, 1e-6),
            0.0,
            1.0,
        )
        risk_density = np.clip(features.interaction_density / max(cfg.density_saturation, 1e-6), 0.0, 1.0)
        risk_overlap = np.clip(features.overlap_risk_meters / max(cfg.overlap_scale_meters, 1e-6), 0.0, 1.0)

        context_risk = np.maximum.reduce([risk_ttc, risk_distance, risk_overlap, 0.5 * risk_density])
        risk_action = np.clip(features.action_norm / max(cfg.action_norm_saturation, 1e-6), 0.0, 1.0) * context_risk

        weighted = (
            cfg.weight_ttc * risk_ttc
            + cfg.weight_distance * risk_distance
            + cfg.weight_density * risk_density
            + cfg.weight_overlap * risk_overlap
            + cfg.weight_action * risk_action
        )
        total_weight = (
            cfg.weight_ttc
            + cfg.weight_distance
            + cfg.weight_density
            + cfg.weight_overlap
            + cfg.weight_action
        )
        risk_score = np.clip(weighted / max(total_weight, 1e-6), 0.0, 1.0)
        risk_score = np.where(features.ego_present, risk_score, 0.0)
        scale = 1.0 - (1.0 - cfg.min_scale) * risk_score
        scale = np.where(features.ego_present, np.clip(scale, cfg.min_scale, 1.0), 1.0).astype(np.float32)
        return ModulationDecision(
            scale=scale,
            risk_score=risk_score.astype(np.float32),
            components={
                "ttc": risk_ttc.astype(np.float32),
                "distance": risk_distance.astype(np.float32),
                "density": risk_density.astype(np.float32),
                "overlap": risk_overlap.astype(np.float32),
                "action": risk_action.astype(np.float32),
                "context": context_risk.astype(np.float32),
            },
        )

    def modulate(self, action: np.ndarray, features: RiskFeatureBatch) -> tuple[np.ndarray, ModulationDecision]:
        decision = self.decide(features)
        modulated = np.asarray(action, dtype=np.float32) * decision.scale[:, np.newaxis]
        return modulated.astype(np.float32, copy=False), decision
