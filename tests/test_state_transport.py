import numpy as np
import torch

from vla_gap_lab.state_transport import (
    fit_orthogonal_transport,
    shared_state_gate,
    state_distillation_loss,
)


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(200, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    target = source @ q
    mapping = fit_orthogonal_transport(source, target)
    np.testing.assert_allclose(source @ mapping, target, atol=1e-5)


def test_shared_state_gate_detects_mid_layer_crossover():
    result = shared_state_gate(np.array([0.5, 0.85, 0.82]), np.array([0.6, 0.55, 0.9]))
    assert result["passed"] is True
    assert result["eligible_layer_indices"] == [1]


def test_distillation_detaches_teacher_target():
    source = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3, requires_grad=True)
    loss = state_distillation_loss(torch.tensor(1.0), source, target, torch.nn.Identity(), 0.5)
    loss.backward()
    assert source.grad is not None
    assert target.grad is None

