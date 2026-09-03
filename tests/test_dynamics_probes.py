import numpy as np

from vla_gap_lab.dynamics_probes import select_ridge_regression, split_episode_ids


def test_split_episode_ids_is_disjoint_and_reproducible():
    groups = np.arange(40)
    first = split_episode_ids(groups, train=24, dev=8, test=8, seed=5)
    second = split_episode_ids(groups, train=24, dev=8, test=8, seed=5)
    assert first == second
    train, dev, test = first
    assert len(train) == 24 and len(dev) == 8 and len(test) == 8
    assert set(train).isdisjoint(dev)
    assert set(train).isdisjoint(test)
    assert set(dev).isdisjoint(test)


def test_select_ridge_regression_recovers_linear_multioutput_signal():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(240, 12))
    weights = rng.normal(size=(12, 2))
    targets = features @ weights + rng.normal(scale=0.01, size=(240, 2))
    model, report = select_ridge_regression(
        features[:140],
        targets[:140],
        features[140:190],
        targets[140:190],
        features[190:],
        targets[190:],
        alphas=(0.01, 0.1, 1.0),
    )
    assert report["test_r2_mean"] > 0.99
    assert len(report["test_r2_per_dim"]) == 2
    assert report["alpha"] in {0.01, 0.1, 1.0}
    assert model is not None
