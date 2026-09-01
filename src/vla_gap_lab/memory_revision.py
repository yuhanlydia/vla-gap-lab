"""Interventions and minimal refresh mechanism for the Persistence-Revision gap."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

MemoryUpdater = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def memory_intervention(
    mode: str,
    previous: torch.Tensor,
    evidence: torch.Tensor,
    initial: torch.Tensor,
    updater: MemoryUpdater,
) -> torch.Tensor:
    """Apply normal, freeze, or diagnostic oracle-refresh update."""
    if mode == "normal":
        return updater(previous, evidence)
    if mode == "freeze":
        return previous
    if mode == "oracle_refresh":
        return updater(initial, evidence)
    raise ValueError("mode must be normal, freeze, or oracle_refresh")


def memory_inertia(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(before.flatten(1), after.flatten(1), dim=1)


def revision_gate(
    normal_tracking_sr: float,
    oracle_tracking_sr: float,
    minimum_gain_pp: float = 10.0,
) -> dict:
    gain_pp = 100.0 * (oracle_tracking_sr - normal_tracking_sr)
    return {
        "passed": gain_pp >= minimum_gain_pp,
        "oracle_gain_pp": gain_pp,
        "threshold_pp": minimum_gain_pp,
    }


class ConflictAdaptiveRefresh(nn.Module):
    """Blend recurrent and fresh candidates using learned memory/evidence conflict."""

    def __init__(self, memory_dim: int, evidence_dim: int, projection_dim: int = 64):
        super().__init__()
        self.memory_projection = nn.Linear(memory_dim, projection_dim, bias=False)
        self.evidence_projection = nn.Linear(evidence_dim, projection_dim, bias=False)
        self.gate = nn.Linear(1, 1)

    def forward(
        self,
        previous: torch.Tensor,
        evidence: torch.Tensor,
        recurrent_candidate: torch.Tensor,
        fresh_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory_summary = previous.mean(dim=1) if previous.ndim == 3 else previous
        evidence_summary = evidence.mean(dim=1) if evidence.ndim == 3 else evidence
        conflict = 1 - F.cosine_similarity(
            self.memory_projection(memory_summary),
            self.evidence_projection(evidence_summary),
            dim=-1,
        )
        alpha = torch.sigmoid(self.gate(conflict[:, None]))
        while alpha.ndim < recurrent_candidate.ndim:
            alpha = alpha.unsqueeze(-1)
        memory = (1 - alpha) * recurrent_candidate + alpha * fresh_candidate
        return memory, alpha.flatten(1)[:, 0], conflict
