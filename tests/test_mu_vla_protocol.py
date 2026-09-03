import numpy as np
import torch

from vla_gap_lab.mu_vla_protocol import (
    MU_VLA_TOKENIZERS_VERSION,
    MU_VLA_TRANSFORMERS_COMMIT,
    MU_VLA_TRANSFORMERS_REPO,
    MU_VLA_TRANSFORMERS_VERSION,
    clip_mikasa_action,
    mu_vla_runtime_issues,
    prepare_training_matched_image,
)


def test_training_matched_image_resizes_and_center_crops():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :8, 0] = 255
    image[:, -8:, 2] = 255
    no_crop = np.asarray(prepare_training_matched_image(image, center_crop=False))
    cropped = np.asarray(prepare_training_matched_image(image, center_crop=True))
    assert no_crop.shape == (224, 224, 3)
    assert cropped.shape == (224, 224, 3)
    assert cropped[:, :8, 0].mean() < no_crop[:, :8, 0].mean()
    assert cropped[:, -8:, 2].mean() < no_crop[:, -8:, 2].mean()


def test_training_matched_image_rejects_non_rgb_uint8():
    bad = np.zeros((128, 128), dtype=np.uint8)
    try:
        prepare_training_matched_image(bad)
    except ValueError as error:
        assert "uint8 HxWx3" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_action_clipping_matches_mikasa_official_eval():
    array = np.array([[-1.7, -0.2, 0.5, 1.8]], dtype=np.float32)
    np.testing.assert_allclose(
        clip_mikasa_action(array),
        np.array([[-1.0, -0.2, 0.5, 1.0]], dtype=np.float32),
    )
    tensor = torch.tensor([[-1.7, -0.2, 0.5, 1.8]])
    torch.testing.assert_close(
        clip_mikasa_action(tensor), torch.tensor([[-1.0, -0.2, 0.5, 1.0]])
    )


def test_runtime_accepts_only_exact_memory_aware_transformers_fork():
    valid = {
        "transformers_version": MU_VLA_TRANSFORMERS_VERSION,
        "tokenizers_version": MU_VLA_TOKENIZERS_VERSION,
        "transformers_url": MU_VLA_TRANSFORMERS_REPO,
        "transformers_commit": MU_VLA_TRANSFORMERS_COMMIT,
        "transformers_requested_revision": MU_VLA_TRANSFORMERS_COMMIT,
    }
    assert mu_vla_runtime_issues(valid) == []

    wrong_fork = {
        **valid,
        "transformers_url": "https://github.com/moojink/transformers-openvla-oft.git",
        "transformers_commit": "deadbeef",
    }
    issues = mu_vla_runtime_issues(wrong_fork)
    assert any("memory-aware mu-VLA fork" in issue for issue in issues)
    assert any(MU_VLA_TRANSFORMERS_COMMIT in issue for issue in issues)
