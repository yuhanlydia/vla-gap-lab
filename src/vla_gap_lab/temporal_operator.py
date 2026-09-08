"""Small, frozen action-space interventions for the Storage-Dynamics Gate."""

from __future__ import annotations

import numpy as np


def velocity_action_gain(
    action_y_std: float,
    velocity_y_std: float,
    *,
    fraction: float = 0.25,
) -> float:
    """Scale velocity into 25% of one observed train-set action std by default."""
    if action_y_std <= 0 or velocity_y_std <= 0:
        raise ValueError("action and velocity standard deviations must be positive")
    if fraction <= 0:
        raise ValueError("fraction must be positive")
    return float(fraction * action_y_std / velocity_y_std)


def apply_velocity_operator(
    action: np.ndarray,
    *,
    velocity_y: float,
    gain: float,
) -> np.ndarray:
    """Add a fixed velocity-derived correction to only the EE y translation."""
    adjusted = np.array(action, dtype=np.float32, copy=True)
    if adjusted.shape[-1] != 7:
        raise ValueError(f"expected 7D pd_ee_delta_pose action, got {adjusted.shape}")
    adjusted[..., 1] += float(gain) * float(velocity_y)
    return np.clip(adjusted, -1.0, 1.0)


def apply_memory_velocity_operator(
    memory: np.ndarray,
    direction: np.ndarray,
    *,
    standardized_velocity: float,
    gain: float,
    token_stride: int = 8,
) -> np.ndarray:
    """Inject a fixed-norm velocity direction into retained memory tokens."""
    adjusted = np.array(memory, dtype=np.float32, copy=True)
    direction = np.asarray(direction, dtype=np.float32)
    if adjusted.ndim != 3 or direction.ndim != 2:
        raise ValueError("memory must be [batch,tokens,dim] and direction must be [retained,dim]")
    if adjusted.shape[0] != 1 or adjusted.shape[1] < direction.shape[0] * token_stride:
        raise ValueError("memory shape is incompatible with the retained-token direction")
    if adjusted.shape[-1] != direction.shape[-1]:
        raise ValueError("memory and direction hidden dimensions must match")
    adjusted[:, ::token_stride, :] += (
        float(gain) * float(standardized_velocity) * direction[None, :, :]
    )
    return adjusted
