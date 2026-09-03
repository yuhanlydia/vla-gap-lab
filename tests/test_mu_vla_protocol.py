import numpy as np

from vla_gap_lab.mu_vla_protocol import prepare_training_matched_image


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
