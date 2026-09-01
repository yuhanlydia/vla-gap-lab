"""Cross-embodiment probes, orthogonal transport, and state distillation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_orthogonal_transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit row-vector mapping P such that ``source @ P ~= target``."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must share shape [N, D]")
    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return (u @ vt).astype(np.float32)


def transport(source: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return source @ mapping


def dual_ridge_predict(
    train_source: np.ndarray,
    train_target: np.ndarray,
    test_source: np.ndarray,
    *,
    alpha: float,
    standardize: bool = True,
) -> np.ndarray:
    """Predict a high-dimensional target via an N×N ridge solve.

    This is equivalent to multi-output ridge with an intercept, but avoids a
    potentially enormous feature-by-target coefficient matrix when N is small.
    """
    train_source = np.asarray(train_source, dtype=np.float64)
    train_target = np.asarray(train_target, dtype=np.float64)
    test_source = np.asarray(test_source, dtype=np.float64)
    source_mean = train_source.mean(axis=0)
    target_mean = train_target.mean(axis=0)
    source_scale = train_source.std(axis=0) if standardize else np.ones(train_source.shape[1])
    source_scale = np.where(source_scale > 1e-12, source_scale, 1.0)
    train_x = (train_source - source_mean) / source_scale
    test_x = (test_source - source_mean) / source_scale
    train_y = train_target - target_mean
    kernel = train_x @ train_x.T
    dual = np.linalg.solve(kernel + alpha * np.eye(len(kernel)), train_y)
    return (test_x @ train_x.T @ dual + target_mean).astype(np.float32)


def state_distillation_loss(
    action_loss: torch.Tensor,
    source_state: torch.Tensor,
    target_state: torch.Tensor,
    projector: torch.nn.Module,
    weight: float,
) -> torch.Tensor:
    return action_loss + weight * F.mse_loss(projector(source_state), target_state.detach())


def shared_state_gate(
    cross_phase_accuracy: np.ndarray,
    embodiment_accuracy: np.ndarray,
    late_layer_index: int = -1,
    phase_threshold: float = 0.8,
    separation_margin: float = 0.1,
) -> dict:
    if cross_phase_accuracy.shape != embodiment_accuracy.shape:
        raise ValueError("accuracy arrays must have identical shapes")
    late_embodiment = embodiment_accuracy[late_layer_index]
    eligible = np.flatnonzero(
        (cross_phase_accuracy > phase_threshold)
        & (embodiment_accuracy + separation_margin < late_embodiment)
    )
    return {
        "passed": bool(len(eligible)),
        "eligible_layer_indices": eligible.tolist(),
        "phase_threshold": phase_threshold,
        "separation_margin": separation_margin,
        "late_embodiment_accuracy": float(late_embodiment),
    }
