"""Causal latent interventions and utilization metrics."""

from __future__ import annotations

import torch


def normalized_probe_direction(weight: torch.Tensor, output_index: int) -> torch.Tensor:
    direction = weight[output_index]
    return direction / direction.norm().clamp_min(torch.finfo(direction.dtype).eps)


def intervene(hidden: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    return hidden + alpha * direction.to(device=hidden.device, dtype=hidden.dtype)


def control_utilization_ratio(
    base_action: torch.Tensor,
    changed_action: torch.Tensor,
    base_probe: torch.Tensor,
    changed_probe: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    action_change = (changed_action - base_action).flatten(1).norm(dim=1)
    probe_change = (changed_probe - base_probe).flatten(1).norm(dim=1)
    return action_change / probe_change.clamp_min(eps)

