"""Audit helpers for LIBERO-Plus task-name-derived instructions."""

_CATEGORY_MARKERS = {
    "Background Textures": ("_table_", "_tb_"),
    "Camera Viewpoints": ("_view_",),
    "Light Conditions": ("_light_",),
    "Objects Layout": ("_add_", "_level"),
    "Robot Initial States": ("_view_",),
    "Sensor Noise": ("_view_",),
}


def canonical_instruction_from_name(task_name: str, category: str) -> str:
    """Recover the base instruction for non-language perturbation task names."""
    if category == "Language Instructions":
        raise ValueError("language perturbations require parsing their BDDL instruction")
    try:
        markers = _CATEGORY_MARKERS[category]
    except KeyError as error:
        raise ValueError(f"unsupported perturbation category: {category}") from error
    marker = next((candidate for candidate in markers if candidate in task_name), None)
    if marker is None:
        raise ValueError(f"expected one of {markers!r} in task name")
    return task_name.split(marker, 1)[0].replace("_", " ")


def filename_derived_instruction(task_name: str) -> str:
    """Reproduce LIBERO-Plus's non-language `grab_language_from_filename` path."""
    return task_name.replace("_", " ")
