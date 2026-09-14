import numpy as np

from vla_gap_lab.state_motion import (
    fit_mlp_probe,
    make_row_mask,
    paper_decision,
    select_memory_tokens,
    summarize_split_metrics,
)


def test_select_memory_tokens_full_and_stride8():
    memory = np.arange(2 * 64 * 3, dtype=np.float32).reshape(2, 64, 3)
    full = select_memory_tokens(memory, "full64")
    stride = select_memory_tokens(memory, "stride8")
    assert full.shape == (2, 64 * 3)
    assert stride.shape == (2, 8 * 3)
    np.testing.assert_array_equal(stride.reshape(2, 8, 3), memory[:, ::8, :])


def test_make_row_mask_applies_min_step_and_contact_filter():
    step = np.array([0, 1, 2, 3, 4])
    reached = np.array([0, 0, 0, 1, 0], dtype=np.float32)
    primary = make_row_mask(step, reached, min_step=2, pre_contact_only=True)
    all_steps = make_row_mask(step, reached, min_step=2, pre_contact_only=False)
    np.testing.assert_array_equal(primary, [False, False, True, False, True])
    np.testing.assert_array_equal(all_steps, [False, False, True, True, True])


def test_fit_mlp_probe_returns_per_dim_metrics():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(240, 12)).astype(np.float32)
    y = np.stack([x[:, 0] - 0.5 * x[:, 1], 0.7 * x[:, 2] + x[:, 3]], axis=1)
    model, metrics = fit_mlp_probe(
        x[:160], y[:160], x[160:200], y[160:200], x[200:], y[200:], seed=3
    )
    pred = model.predict(x[200:])
    assert pred.shape == (40, 2)
    assert len(metrics["test_r2_per_dim"]) == 2
    assert metrics["test_r2_mean"] > 0.5


def test_summarize_split_metrics_uses_median_and_iqr():
    rows = [
        {"position_r2_mean": 0.7, "velocity_y_r2": 0.1},
        {"position_r2_mean": 0.8, "velocity_y_r2": 0.2},
        {"position_r2_mean": 0.9, "velocity_y_r2": 0.3},
    ]
    summary = summarize_split_metrics(rows)
    assert summary["position_r2_mean_median"] == 0.8
    assert summary["velocity_y_r2_median"] == 0.2
    assert np.isclose(summary["state_motion_gap_median"], 0.6)


def test_paper_decision_enforces_frozen_kill_rule():
    good = paper_decision(
        {"position_r2_mean_median": 0.80, "velocity_y_r2_median": 0.30, "state_motion_gap_median": 0.50},
        {"position_r2_mean_median": 0.75, "velocity_y_r2_median": 0.35, "state_motion_gap_median": 0.40},
        {"position_r2_mean_median": 0.70, "velocity_y_r2_median": 0.40, "state_motion_gap_median": 0.30},
        {"position_r2_mean_median": 0.68, "velocity_y_r2_median": 0.45, "state_motion_gap_median": 0.23},
    )
    assert good["go_icassp"] is True
    killed = paper_decision(
        {"position_r2_mean_median": 0.80, "velocity_y_r2_median": 0.30, "state_motion_gap_median": 0.50},
        {"position_r2_mean_median": 0.75, "velocity_y_r2_median": 0.65, "state_motion_gap_median": 0.10},
        {"position_r2_mean_median": 0.70, "velocity_y_r2_median": 0.40, "state_motion_gap_median": 0.30},
        {"position_r2_mean_median": 0.68, "velocity_y_r2_median": 0.45, "state_motion_gap_median": 0.23},
    )
    assert killed["go_icassp"] is False
    assert "mlp_velocity_y_below_0_60" in killed["failed_conditions"]
