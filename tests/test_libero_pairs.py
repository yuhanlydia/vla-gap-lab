import json

import numpy as np
import pytest

from vla_gap_lab.libero_pairs import (
    demonstration_instruction,
    evenly_spaced_indices,
    find_demonstration,
    load_perturbation,
    require_action_preserving,
)


def test_catalog_resolution_and_category_guard(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "suite": [
            {"id": 7, "name": "visual", "category": "Light Conditions", "difficulty_level": 2},
                    {"id": 8, "name": "robot", "category": "Robot Position", "difficulty_level": 3},
                ]
            }
        )
    )
    visual = load_perturbation(path, "suite", 7)
    require_action_preserving(visual)
    with pytest.raises(ValueError, match="not action-preserving"):
        require_action_preserving(load_perturbation(path, "suite", 8))


def test_evenly_spaced_indices_are_deterministic_and_unique():
    actual = evenly_spaced_indices(length=20, count=4, margin=2)
    np.testing.assert_array_equal(actual, [2, 7, 12, 17])
    assert len(np.unique(actual)) == len(actual)


def test_demo_fallback_uses_longest_original_prefix(tmp_path):
    short = tmp_path / "pick_black_bowl_demo.hdf5"
    exact = tmp_path / "pick_black_bowl_on_plate_demo.hdf5"
    short.touch()
    exact.touch()
    found = find_demonstration(
        tmp_path,
        "pick black bowl on plate table 10",
        variant_name="pick_black_bowl_on_plate_table_10",
    )
    assert found == exact
    assert demonstration_instruction(found) == "pick black bowl on plate"
