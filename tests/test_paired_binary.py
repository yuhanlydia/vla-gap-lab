import math

import pytest

from vla_gap_lab.paired_binary import paired_binary_summary


def test_paired_binary_summary_matches_exact_pilot_counts():
    clean = [True, False, True, True, True, True, True, True, True, True]
    shifted = [False, True, False, True, True, False, True, True, False, False]

    result = paired_binary_summary(clean, shifted, bootstrap_samples=1_000, seed=4)

    assert result["reference_success_rate"] == 0.9
    assert result["shifted_success_rate"] == 0.5
    assert result["control_retention"] == pytest.approx(5 / 9)
    assert result["reference_only_successes"] == 5
    assert result["shifted_only_successes"] == 1
    assert result["mcnemar_exact_two_sided_p"] == 0.21875
    low, high = result["paired_bootstrap_95_ci"]
    assert low <= -0.4 <= high


def test_paired_binary_summary_handles_no_reference_successes():
    result = paired_binary_summary([False, False], [False, True], bootstrap_samples=10)
    assert math.isnan(result["control_retention"])


@pytest.mark.parametrize("left,right", [([], []), ([True], []), ([[True]], [[True]])])
def test_paired_binary_summary_rejects_invalid_shapes(left, right):
    with pytest.raises(ValueError):
        paired_binary_summary(left, right)
