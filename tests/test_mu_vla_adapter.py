import numpy as np
import torch

from vla_gap_lab.mu_vla_adapter import normalize_bounds_q99, strip_ddp_prefix


def test_ddp_prefix_and_q99_normalization():
    state = {"module.weight": torch.ones(2), "bias": torch.zeros(1)}
    assert set(strip_ddp_prefix(state)) == {"weight", "bias"}
    stats = {"q01": [0.0, -1.0], "q99": [2.0, 1.0]}
    np.testing.assert_allclose(
        normalize_bounds_q99(np.array([1.0, 1.0]), stats), [0.0, 1.0], atol=1e-7
    )
