from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ModulationConfig, load_modulation_config_from_env
from .features import extract_risk_features
from .heuristic import HeuristicActionModulator


class ActionModulationRuntime:
    def __init__(self, *, config: ModulationConfig, batch_dims: Sequence[int]) -> None:
        self.config = config
        self.batch_dims = tuple(int(v) for v in batch_dims)
        self._modulator = HeuristicActionModulator(config)
        self._steps = 0
        self._intervened_env_steps = 0
        self._total_env_steps = 0
        self._sum_scale = 0.0
        self._min_scale_seen = 1.0
        self._max_risk_seen = 0.0
        self._trace_path = Path(config.trace_path).expanduser() if config.trace_path else None

    def apply(self, *, action: np.ndarray, current_state: object, batch_index: int, step_index: int) -> np.ndarray:
        features = extract_risk_features(
            current_state,
            action,
            batch_dims=self.batch_dims,
            interaction_radius_meters=self.config.interaction_radius_meters,
        )
        modulated_action, decision = self._modulator.modulate(action, features)
        self._steps += 1
        env_count = int(decision.scale.shape[0])
        self._total_env_steps += env_count
        self._intervened_env_steps += int(np.count_nonzero(decision.intervention_mask))
        self._sum_scale += float(np.sum(decision.scale))
        self._min_scale_seen = min(self._min_scale_seen, float(np.min(decision.scale)))
        self._max_risk_seen = max(self._max_risk_seen, float(np.max(decision.risk_score)))
        if self._trace_path is not None:
            self._append_trace(
                batch_index=batch_index,
                step_index=step_index,
                features=features,
                decision=decision,
                planner_action=np.asarray(action, dtype=np.float32),
                modulated_action=modulated_action,
            )
        return modulated_action

    def summary(self) -> dict[str, float | int]:
        mean_scale = self._sum_scale / self._total_env_steps if self._total_env_steps else 1.0
        intervention_rate = self._intervened_env_steps / self._total_env_steps if self._total_env_steps else 0.0
        return {
            "mode": self.config.mode,
            "steps": self._steps,
            "env_steps": self._total_env_steps,
            "intervened_env_steps": self._intervened_env_steps,
            "intervention_rate": round(intervention_rate, 6),
            "mean_scale": round(mean_scale, 6),
            "min_scale_seen": round(self._min_scale_seen, 6),
            "max_risk_seen": round(self._max_risk_seen, 6),
        }

    def format_summary(self, *, prefix: str = "[action-modulation]") -> str:
        summary = self.summary()
        return (
            f"{prefix} mode={summary['mode']} steps={summary['steps']} env_steps={summary['env_steps']} "
            f"intervened={summary['intervened_env_steps']} intervention_rate={summary['intervention_rate']:.3f} "
            f"mean_scale={summary['mean_scale']:.3f} min_scale={summary['min_scale_seen']:.3f} "
            f"max_risk={summary['max_risk_seen']:.3f}"
        )

    def _append_trace(
        self,
        *,
        batch_index: int,
        step_index: int,
        features: object,
        decision: object,
        planner_action: np.ndarray,
        modulated_action: np.ndarray,
    ) -> None:
        assert self._trace_path is not None
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "batch_index": int(batch_index),
            "step_index": int(step_index),
            "scenario_ids": features.scenario_ids.tolist(),
            "ego_present": features.ego_present.astype(bool).tolist(),
            "ego_speed_mps": _rounded_list(features.ego_speed_mps),
            "action_norm": _rounded_list(features.action_norm),
            "min_distance_meters": _rounded_list(features.min_distance_meters),
            "min_ttc_seconds": _rounded_list(features.min_ttc_seconds),
            "interaction_density": _rounded_list(features.interaction_density),
            "overlap_risk_meters": _rounded_list(features.overlap_risk_meters),
            "valid_neighbor_count": features.valid_neighbor_count.astype(int).tolist(),
            "scale": _rounded_list(decision.scale),
            "risk_score": _rounded_list(decision.risk_score),
            "risk_components": {name: _rounded_list(values) for name, values in decision.components.items()},
            "planner_action": np.asarray(planner_action, dtype=np.float32).round(6).tolist(),
            "modulated_action": np.asarray(modulated_action, dtype=np.float32).round(6).tolist(),
        }
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _rounded_list(values: np.ndarray) -> list[float | None]:
    output: list[float | None] = []
    for value in np.asarray(values, dtype=np.float32).tolist():
        if value is None or not np.isfinite(value):
            output.append(None)
        else:
            output.append(round(float(value), 6))
    return output


def build_action_modulation_runtime_from_env(batch_dims: Sequence[int]) -> ActionModulationRuntime | None:
    config = load_modulation_config_from_env()
    if not config.enabled:
        return None
    return ActionModulationRuntime(config=config, batch_dims=batch_dims)
