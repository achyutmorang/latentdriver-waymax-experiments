from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import RiskFeatureBatch


def _flatten_env_axes(value: object, batch_dims: Sequence[int]) -> np.ndarray:
    arr = np.asarray(value)
    if len(batch_dims) < 2:
        raise ValueError(f"Expected batch_dims with at least 2 entries, got {batch_dims!r}")
    num_devices = int(batch_dims[0])
    batch_size = int(batch_dims[1])
    flat_envs = num_devices * batch_size
    if arr.ndim >= 2 and arr.shape[0] == num_devices and arr.shape[1] == batch_size:
        return arr.reshape(flat_envs, *arr.shape[2:])
    if arr.ndim >= 1 and arr.shape[0] == flat_envs:
        return arr
    raise ValueError(
        f"Unable to flatten env axes for array with shape={arr.shape!r} and batch_dims={tuple(batch_dims)!r}."
    )


def _squeeze_single_time_axis(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def _finite_min(values: np.ndarray, mask: np.ndarray, *, default: float) -> np.ndarray:
    safe_values = np.where(mask, values, np.inf)
    mins = safe_values.min(axis=1)
    return np.where(np.any(mask, axis=1), mins, float(default))


def extract_risk_features(
    current_state: object,
    planner_action: np.ndarray,
    *,
    batch_dims: Sequence[int],
    interaction_radius_meters: float,
) -> RiskFeatureBatch:
    action = np.asarray(planner_action, dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"Expected planner_action to have shape [num_envs, action_dim], got {action.shape!r}")

    trajectory = current_state.current_sim_trajectory
    object_metadata = current_state.object_metadata

    scenario_ids = _flatten_env_axes(current_state._scenario_id, batch_dims).reshape(-1)
    x = _squeeze_single_time_axis(_flatten_env_axes(trajectory.x, batch_dims)).astype(np.float32, copy=False)
    y = _squeeze_single_time_axis(_flatten_env_axes(trajectory.y, batch_dims)).astype(np.float32, copy=False)
    vel_x = _squeeze_single_time_axis(_flatten_env_axes(trajectory.vel_x, batch_dims)).astype(np.float32, copy=False)
    vel_y = _squeeze_single_time_axis(_flatten_env_axes(trajectory.vel_y, batch_dims)).astype(np.float32, copy=False)
    length = _squeeze_single_time_axis(_flatten_env_axes(trajectory.length, batch_dims)).astype(np.float32, copy=False)
    width = _squeeze_single_time_axis(_flatten_env_axes(trajectory.width, batch_dims)).astype(np.float32, copy=False)
    valid = _squeeze_single_time_axis(_flatten_env_axes(trajectory.valid, batch_dims)).astype(bool, copy=False)
    is_sdc = _flatten_env_axes(object_metadata.is_sdc, batch_dims).astype(bool, copy=False)

    if x.ndim != 2:
        raise ValueError(f"Expected flattened x positions to have shape [num_envs, num_objects], got {x.shape!r}")
    if action.shape[0] != x.shape[0]:
        raise ValueError(
            f"Planner action env count {action.shape[0]} does not match state env count {x.shape[0]}."
        )

    ego_present = np.any(is_sdc, axis=1)
    ego_index = np.argmax(is_sdc, axis=1)
    row_index = np.arange(x.shape[0], dtype=np.int64)

    ego_x = x[row_index, ego_index]
    ego_y = y[row_index, ego_index]
    ego_vel_x = vel_x[row_index, ego_index]
    ego_vel_y = vel_y[row_index, ego_index]
    ego_length = length[row_index, ego_index]
    ego_width = width[row_index, ego_index]
    ego_valid = valid[row_index, ego_index]
    ego_present = np.logical_and(ego_present, ego_valid)

    rel_x = x - ego_x[:, np.newaxis]
    rel_y = y - ego_y[:, np.newaxis]
    rel_vel_x = vel_x - ego_vel_x[:, np.newaxis]
    rel_vel_y = vel_y - ego_vel_y[:, np.newaxis]
    distance = np.sqrt(rel_x**2 + rel_y**2)

    neighbor_mask = np.logical_and(valid, np.logical_not(is_sdc))
    min_distance = _finite_min(distance, neighbor_mask, default=np.inf)
    local_density_mask = np.logical_and(neighbor_mask, distance <= float(interaction_radius_meters))
    interaction_density = np.sum(local_density_mask, axis=1).astype(np.float32)

    distance_safe = np.maximum(distance, 1e-3)
    closing_speed = -((rel_x * rel_vel_x) + (rel_y * rel_vel_y)) / distance_safe
    ttc = np.where(np.logical_and(neighbor_mask, closing_speed > 0.0), distance / np.maximum(closing_speed, 1e-3), np.inf)
    min_ttc = _finite_min(ttc, neighbor_mask, default=np.inf)

    ego_radius = 0.5 * np.sqrt(ego_length**2 + ego_width**2)
    other_radius = 0.5 * np.sqrt(length**2 + width**2)
    clearance = distance - (ego_radius[:, np.newaxis] + other_radius)
    min_clearance = _finite_min(clearance, neighbor_mask, default=np.inf)
    overlap_risk = np.maximum(-min_clearance, 0.0)

    action_norm = np.linalg.norm(action, axis=1)
    ego_speed = np.sqrt(ego_vel_x**2 + ego_vel_y**2)

    return RiskFeatureBatch(
        scenario_ids=scenario_ids.astype(np.int64, copy=False),
        ego_present=ego_present.astype(bool, copy=False),
        ego_speed_mps=ego_speed.astype(np.float32, copy=False),
        action_norm=action_norm.astype(np.float32, copy=False),
        min_distance_meters=min_distance.astype(np.float32, copy=False),
        min_ttc_seconds=min_ttc.astype(np.float32, copy=False),
        interaction_density=interaction_density,
        overlap_risk_meters=overlap_risk.astype(np.float32, copy=False),
        valid_neighbor_count=np.sum(neighbor_mask, axis=1).astype(np.int32),
    )
