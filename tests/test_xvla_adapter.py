import pytest
import torch
from torch import nn

from vla_gap_lab.xvla_adapter import capture_xvla_action_layers


class ToyXVLA(nn.Module):
    num_actions = 3

    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def generate_actions(self, domain_id, proprio, steps, **inputs):
        value = torch.ones(len(proprio), 5, 4)
        for block in self.transformer.blocks:
            value = block(value)
        return value[:, :3]


def test_xvla_layer_capture_and_bounds():
    model = ToyXVLA()
    action, states = capture_xvla_action_layers(
        model, {}, domain_id=2, proprio=torch.zeros(2, 4), layers=[0, 2]
    )
    assert action.shape == (2, 3, 4)
    assert states.shape == (2, 2, 4)
    with pytest.raises(IndexError):
        capture_xvla_action_layers(model, {}, domain_id=2, proprio=torch.zeros(2, 4), layers=[3])
