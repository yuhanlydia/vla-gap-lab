import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vla_gap_lab.state_transport import (
    dual_ridge_predict,
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


def test_dual_ridge_matches_sklearn_multioutput_prediction():
    rng = np.random.default_rng(9)
    train_x = rng.normal(size=(12, 30))
    train_y = rng.normal(size=(12, 20))
    test_x = rng.normal(size=(4, 30))
    expected = make_pipeline(StandardScaler(), Ridge(alpha=7.0)).fit(train_x, train_y)
    actual = dual_ridge_predict(train_x, train_y, test_x, alpha=7.0, standardize=True)
    np.testing.assert_allclose(actual, expected.predict(test_x), rtol=1e-5, atol=1e-5)


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
