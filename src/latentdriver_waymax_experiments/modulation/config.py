from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Mapping


MODULATION_MODE_ENV = "LATENTDRIVER_ACTION_MODULATION"
MODULATION_PREFIX = "LATENTDRIVER_ACTION_MODULATION_"


@dataclass(frozen=True)
class ModulationConfig:
    mode: str = "disabled"
    min_scale: float = 0.35
    ttc_threshold_seconds: float = 3.0
    ttc_hard_stop_seconds: float = 0.75
    distance_threshold_meters: float = 8.0
    interaction_radius_meters: float = 15.0
    density_saturation: float = 6.0
    action_norm_saturation: float = 4.0
    overlap_scale_meters: float = 1.0
    min_closing_speed_mps: float = 0.1
    weight_ttc: float = 0.45
    weight_distance: float = 0.2
    weight_density: float = 0.15
    weight_overlap: float = 0.15
    weight_action: float = 0.05
    trace_path: str | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_float(name: str, default: float, environ: Mapping[str, str]) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _normalize_mode(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        return "disabled"
    if value in _TRUE_VALUES:
        return "heuristic"
    if value in _FALSE_VALUES:
        return "disabled"
    if value not in {"disabled", "heuristic"}:
        raise ValueError(
            f"Unsupported {MODULATION_MODE_ENV}={raw!r}. Supported values: disabled, heuristic."
        )
    return value


def _validate_config(config: ModulationConfig) -> ModulationConfig:
    if not 0.0 < config.min_scale <= 1.0:
        raise ValueError("LATENTDRIVER_ACTION_MODULATION_MIN_SCALE must be in (0, 1].")
    if config.ttc_hard_stop_seconds <= 0:
        raise ValueError("LATENTDRIVER_ACTION_MODULATION_TTC_HARD_STOP_SECONDS must be positive.")
    if config.ttc_threshold_seconds <= config.ttc_hard_stop_seconds:
        raise ValueError(
            "LATENTDRIVER_ACTION_MODULATION_TTC_THRESHOLD_SECONDS must be greater than the hard-stop TTC."
        )
    if config.distance_threshold_meters <= 0:
        raise ValueError("LATENTDRIVER_ACTION_MODULATION_DISTANCE_THRESHOLD_METERS must be positive.")
    if config.interaction_radius_meters <= 0:
        raise ValueError("LATENTDRIVER_ACTION_MODULATION_INTERACTION_RADIUS_METERS must be positive.")
    if config.density_saturation <= 0:
        raise ValueError("LATENTDRIVER_ACTION_MODULATION_DENSITY_SATURATION must be positive.")
    if config.action_norm_saturation <= 0:
        raise ValueError("LATENTDRIVER_ACTION_MODULATION_ACTION_NORM_SATURATION must be positive.")
    if config.overlap_scale_meters <= 0:
        raise ValueError("LATENTDRIVER_ACTION_MODULATION_OVERLAP_SCALE_METERS must be positive.")
    if config.min_closing_speed_mps <= 0:
        raise ValueError("LATENTDRIVER_ACTION_MODULATION_MIN_CLOSING_SPEED_MPS must be positive.")
    weight_sum = (
        config.weight_ttc
        + config.weight_distance
        + config.weight_density
        + config.weight_overlap
        + config.weight_action
    )
    if weight_sum <= 0:
        raise ValueError("At least one modulation risk weight must be positive.")
    return config


def load_modulation_config_from_env(environ: Mapping[str, str] | None = None) -> ModulationConfig:
    env = os.environ if environ is None else environ
    config = ModulationConfig(
        mode=_normalize_mode(env.get(MODULATION_MODE_ENV, "disabled")),
        min_scale=_parse_float(f"{MODULATION_PREFIX}MIN_SCALE", 0.35, env),
        ttc_threshold_seconds=_parse_float(f"{MODULATION_PREFIX}TTC_THRESHOLD_SECONDS", 3.0, env),
        ttc_hard_stop_seconds=_parse_float(f"{MODULATION_PREFIX}TTC_HARD_STOP_SECONDS", 0.75, env),
        distance_threshold_meters=_parse_float(f"{MODULATION_PREFIX}DISTANCE_THRESHOLD_METERS", 8.0, env),
        interaction_radius_meters=_parse_float(f"{MODULATION_PREFIX}INTERACTION_RADIUS_METERS", 15.0, env),
        density_saturation=_parse_float(f"{MODULATION_PREFIX}DENSITY_SATURATION", 6.0, env),
        action_norm_saturation=_parse_float(f"{MODULATION_PREFIX}ACTION_NORM_SATURATION", 4.0, env),
        overlap_scale_meters=_parse_float(f"{MODULATION_PREFIX}OVERLAP_SCALE_METERS", 1.0, env),
        min_closing_speed_mps=_parse_float(f"{MODULATION_PREFIX}MIN_CLOSING_SPEED_MPS", 0.1, env),
        weight_ttc=_parse_float(f"{MODULATION_PREFIX}WEIGHT_TTC", 0.45, env),
        weight_distance=_parse_float(f"{MODULATION_PREFIX}WEIGHT_DISTANCE", 0.2, env),
        weight_density=_parse_float(f"{MODULATION_PREFIX}WEIGHT_DENSITY", 0.15, env),
        weight_overlap=_parse_float(f"{MODULATION_PREFIX}WEIGHT_OVERLAP", 0.15, env),
        weight_action=_parse_float(f"{MODULATION_PREFIX}WEIGHT_ACTION", 0.05, env),
        trace_path=(env.get(f"{MODULATION_PREFIX}TRACE_PATH", "").strip() or None),
    )
    return _validate_config(config)


def collect_modulation_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    keys = [key for key in env if key == MODULATION_MODE_ENV or key.startswith(MODULATION_PREFIX)]
    return {key: env[key] for key in sorted(keys)}
