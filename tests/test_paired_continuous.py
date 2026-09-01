import numpy as np
import pytest

from vla_gap_lab.paired_continuous import paired_cluster_mean_difference


def test_paired_cluster_bootstrap_preserves_clustered_difference():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    shifted = reference + np.array([0.5, 0.5, 1.0, 1.0])
    clusters = np.array(["a", "a", "b", "b"])
    result = paired_cluster_mean_difference(
        reference, shifted, clusters, bootstrap_samples=500, seed=3
    )
    assert result["paired_mean_difference"] == 0.75
    low, high = result["cluster_bootstrap_95_ci"]
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(1.0)
