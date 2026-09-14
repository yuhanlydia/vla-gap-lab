import numpy as np
import torch

from vla_gap_lab.state_motion import (
    motion_compression_decision,
    pool_projector_tokens,
    temporal_visual_features,
)


def test_temporal_visual_features_current_and_pair():
    tokens = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    current, idx_current = temporal_visual_features(tokens, mode="current", lag=1)
    pair, idx_pair = temporal_visual_features(tokens, mode="pair", lag=1)
    assert current.shape == (4, 6)
    np.testing.assert_array_equal(idx_current, [0, 1, 2, 3])
    assert pair.shape == (3, 12)
    np.testing.assert_array_equal(idx_pair, [1, 2, 3])
    np.testing.assert_array_equal(pair[0, :6], tokens[0].reshape(-1))
    np.testing.assert_array_equal(pair[0, 6:], tokens[1].reshape(-1))


def test_pool_projector_tokens_preserves_views_and_grid():
    x = torch.arange(1 * 32 * 2, dtype=torch.float32).reshape(1, 32, 2)
    pooled = pool_projector_tokens(x, num_views=2, grid=2)
    assert pooled.shape == (8, 2)
    expected = x[0, :16].reshape(4, 4, 2)[:2, :2].mean((0, 1))
    torch.testing.assert_close(pooled[0], expected)


def test_motion_compression_decision_requires_observability_and_compression_gap():
    good = motion_compression_decision(
        medium_current={"velocity_y_r2_median": 0.20},
        medium_pair={"velocity_y_r2_median": 0.72},
        medium_memory={"velocity_y_r2_median": 0.35, "position_r2_mean_median": 0.78},
        fast_pair={"velocity_y_r2_median": 0.66},
        fast_memory={"velocity_y_r2_median": 0.44, "position_r2_mean_median": 0.70},
    )
    assert good["go_icassp"] is True

    bad = motion_compression_decision(
        medium_current={"velocity_y_r2_median": 0.20},
        medium_pair={"velocity_y_r2_median": 0.42},
        medium_memory={"velocity_y_r2_median": 0.35, "position_r2_mean_median": 0.78},
        fast_pair={"velocity_y_r2_median": 0.50},
        fast_memory={"velocity_y_r2_median": 0.44, "position_r2_mean_median": 0.70},
    )
    assert bad["go_icassp"] is False
    assert "medium_pair_velocity_ge_0_60" in bad["failed_conditions"]
