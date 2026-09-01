"""Explicit adapter around OpenVLA-OFT hidden-state outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def action_token_mask(
    current_action_mask: torch.Tensor, next_actions_mask: torch.Tensor
) -> torch.Tensor:
    if current_action_mask.shape != next_actions_mask.shape:
        raise ValueError("current and next action masks must have identical shapes")
    if current_action_mask.ndim != 2:
        raise ValueError("action masks must have shape [B, text_tokens]")
    return current_action_mask.bool() | next_actions_mask.bool()


def extract_layer_action_states(
    hidden_states: Sequence[torch.Tensor],
    layers: Sequence[int],
    num_patches: int,
    current_action_mask: torch.Tensor,
    next_actions_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean-pooled ``[B, L, D]`` states at OFT action positions."""
    mask = action_token_mask(current_action_mask, next_actions_mask)
    selected = []
    for layer in layers:
        if layer < 0 or layer >= len(hidden_states):
            raise IndexError(f"layer {layer} outside [0, {len(hidden_states) - 1}]")
        text = hidden_states[layer][:, num_patches:-1]
        if text.shape[:2] != mask.shape:
            raise ValueError(
                f"layer text shape {tuple(text.shape[:2])} does not match mask {tuple(mask.shape)}"
            )
        weight = mask.to(text.dtype).unsqueeze(-1)
        pooled = (text * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)
        selected.append(pooled)
    return torch.stack(selected, dim=1)


def assert_action_chunk(actions: torch.Tensor, horizon: int = 8, action_dim: int = 7) -> None:
    if tuple(actions.shape[-2:]) != (horizon, action_dim):
        raise ValueError(
            f"expected action chunk [..., {horizon}, {action_dim}], got {tuple(actions.shape)}"
        )
