import pytest

from vla_gap_lab.libero_language import (
    canonical_instruction_from_name,
    filename_derived_instruction,
)


def test_camera_filename_metadata_is_removed_from_canonical_instruction():
    name = "pick_up_the_bowl_view_0_0_100_2_354_initstate_0"
    assert canonical_instruction_from_name(name, "Camera Viewpoints") == "pick up the bowl"
    assert filename_derived_instruction(name).endswith("view 0 0 100 2 354 initstate 0")


@pytest.mark.parametrize(
    "category,name",
    [
        ("Background Textures", "pick_up_bowl_table_7"),
        ("Light Conditions", "pick_up_bowl_light_3"),
        ("Objects Layout", "pick_up_bowl_add_12"),
        ("Robot Initial States", "pick_up_bowl_view_0_0_100_0_0_initstate_2"),
        ("Sensor Noise", "pick_up_bowl_view_0_0_100_0_0_initstate_0_noise_3"),
    ],
)
def test_non_language_categories_strip_their_suffix(category, name):
    assert canonical_instruction_from_name(name, category) == "pick up bowl"


def test_language_category_requires_bddl_parser():
    with pytest.raises(ValueError):
        canonical_instruction_from_name("pick_up_bowl_language_1", "Language Instructions")
