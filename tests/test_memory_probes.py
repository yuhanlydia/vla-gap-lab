import numpy as np

from vla_gap_lab.memory_probes import grouped_ridge_classification


def test_grouped_ridge_classification_decodes_signal_without_group_leakage():
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(12), 6)
    labels_by_group = np.arange(12) % 3
    labels = np.repeat(labels_by_group, 6)
    prototypes = np.eye(3, 8)
    features = prototypes[labels] + rng.normal(scale=0.05, size=(len(labels), 8))

    result = grouped_ridge_classification(features, labels, groups, alpha=1.0, folds=4)

    assert result["samples"] == len(labels)
    assert result["episodes"] == 12
    assert result["balanced_accuracy"] > 0.95
    low, high = result["balanced_accuracy_cluster_bootstrap_95_ci"]
    assert low > 0.9
    assert high <= 1.0
