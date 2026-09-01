import torch

from vla_gap_lab.memory_revision import (
    ConflictAdaptiveRefresh,
    memory_inertia,
    memory_intervention,
    revision_gate,
)


def updater(memory, evidence):
    return memory + evidence


def test_memory_interventions_are_causally_distinct():
    previous, evidence, initial = torch.tensor([[2.0]]), torch.tensor([[3.0]]), torch.zeros(1, 1)
    assert memory_intervention("normal", previous, evidence, initial, updater).item() == 5
    assert memory_intervention("freeze", previous, evidence, initial, updater).item() == 2
    assert memory_intervention("oracle_refresh", previous, evidence, initial, updater).item() == 3


def test_inertia_and_gate():
    torch.testing.assert_close(memory_inertia(torch.ones(1, 2), torch.ones(1, 2)), torch.ones(1))
    assert revision_gate(0.4, 0.51)["passed"] is True


def test_refresh_is_convex_blend():
    module = ConflictAdaptiveRefresh(4, 5, 3)
    previous, evidence = torch.randn(2, 8, 4), torch.randn(2, 9, 5)
    recurrent, fresh = torch.zeros(2, 8, 4), torch.ones(2, 8, 4)
    output, alpha, _ = module(previous, evidence, recurrent, fresh)
    assert torch.all((alpha > 0) & (alpha < 1))
    torch.testing.assert_close(output[:, 0, 0], alpha)
