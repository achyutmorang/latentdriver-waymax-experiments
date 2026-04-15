from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class RiskFeatureBatch:
    scenario_ids: np.ndarray
    ego_present: np.ndarray
    ego_speed_mps: np.ndarray
    action_norm: np.ndarray
    min_distance_meters: np.ndarray
    min_ttc_seconds: np.ndarray
    interaction_density: np.ndarray
    overlap_risk_meters: np.ndarray
    valid_neighbor_count: np.ndarray

    def size(self) -> int:
        return int(self.scenario_ids.shape[0])


@dataclass(frozen=True)
class ModulationDecision:
    scale: np.ndarray
    risk_score: np.ndarray
    components: Dict[str, np.ndarray]

    @property
    def intervention_mask(self) -> np.ndarray:
        return self.scale < (1.0 - 1e-6)
