"""Minimal low-rank latent-to-action control path."""

from __future__ import annotations

import torch
from torch import nn


class ControlSkip(nn.Module):
    """Adds ``beta * B(A(pool(h_l)))`` to a frozen base action chunk."""

    def __init__(self, hidden_dim: int, action_horizon: int = 8, action_dim: int = 7, rank: int = 8):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, action_horizon * action_dim, bias=False)
        self.beta = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.down.weight, std=hidden_dim**-0.5)
        nn.init.zeros_(self.up.weight)

    @staticmethod
    def pool(hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B, D] or [B, T, D]")
        if mask is None:
            return hidden.mean(dim=1)
        weight = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)

    def forward(self, base_action: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        expected = (self.action_horizon, self.action_dim)
        if tuple(base_action.shape[-2:]) != expected:
            raise ValueError(f"base_action must end in {expected}")
        pooled = self.pool(hidden, mask)
        residual = self.up(self.down(pooled)).view(-1, *expected)
        return base_action + self.beta * residual


def freeze_except_controlskip(model: nn.Module, control_skip: ControlSkip) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in control_skip.parameters():
        parameter.requires_grad_(True)

