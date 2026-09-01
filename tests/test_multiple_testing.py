from __future__ import annotations

import pytest

from vla_gap_lab.multiple_testing import bonferroni, holm


def test_bonferroni_and_holm_preserve_original_order() -> None:
    values = [0.04, 0.01, 0.2]
    assert bonferroni(values) == pytest.approx([0.12, 0.03, 0.6])
    assert holm(values) == pytest.approx([0.08, 0.03, 0.2])


def test_holm_is_monotone_when_sorted_and_validates_input() -> None:
    adjusted = holm([0.03, 0.01, 0.02, 0.9])
    ordered = [adjusted[index] for index in [1, 2, 0, 3]]
    assert ordered == sorted(ordered)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        holm([1.1])
