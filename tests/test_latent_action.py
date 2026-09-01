import numpy as np
import torch

from vla_gap_lab.cache import HiddenCache
from vla_gap_lab.controlskip import ControlSkip
from vla_gap_lab.interventions import control_utilization_ratio
from vla_gap_lab.metrics import latent_action_gate
from vla_gap_lab.probes import fit_layerwise_ridge


def synthetic_cache() -> HiddenCache:
    rng = np.random.default_rng(4)
    n, layers, hidden, action_horizon, action_dim = 160, 3, 12, 2, 3
    clean = rng.normal(size=(n, layers, hidden)).astype(np.float32)
    mapping = rng.normal(size=(hidden, action_horizon * action_dim))
    actions = (clean[:, 1] @ mapping).reshape(n, action_horizon, action_dim).astype(np.float32)
    shifted = clean.copy()
    shifted[:, 1] += rng.normal(scale=0.02, size=shifted[:, 1].shape)
    shifted[:, 0] = rng.normal(size=shifted[:, 0].shape)
    return HiddenCache(clean, shifted, actions, np.array([4, 8, 12]), np.arange(n), np.array(["noise"] * n))


def test_layerwise_probe_finds_retained_layer(tmp_path):
    cache = synthetic_cache()
    path = tmp_path / "cache.npz"
    cache.save(path)
    results = fit_layerwise_ridge(HiddenCache.load(path), alpha=0.01)
    assert results[1].clean.r2 > 0.99
    assert results[1].shifted.r2 > 0.98


def test_gate_requires_interaction_of_both_conditions():
    passed = latent_action_gate(np.array([0.8]), np.array([0.72]), 0.6)
    failed = latent_action_gate(np.array([0.8]), np.array([0.72]), 0.8)
    assert passed["passed"] is True
    assert failed["passed"] is False


def test_controlskip_is_identity_at_initialization():
    module = ControlSkip(hidden_dim=12, action_horizon=2, action_dim=3, rank=4)
    action = torch.randn(5, 2, 3)
    hidden = torch.randn(5, 7, 12)
    torch.testing.assert_close(module(action, hidden), action)


def test_control_utilization_ratio():
    zeros = torch.zeros(2, 1)
    ratio = control_utilization_ratio(zeros, torch.ones_like(zeros), zeros, 2 * torch.ones_like(zeros))
    torch.testing.assert_close(ratio, torch.full((2,), 0.5))

