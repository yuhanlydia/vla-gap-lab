"""Training-matched preprocessing and inference wrapper for released mu-VLA checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .mu_vla_adapter import MuVLAPolicy, normalize_bounds_q99

OPENVLA_IMAGE_SIZE = 224
CENTER_CROP_SCALE = 0.9


def prepare_training_matched_image(
    image: np.ndarray, *, center_crop: bool = True
) -> Image.Image:
    """Match official mu-VLA resize + image-augmentation center-crop geometry.

    Published mu-VLA MIKASA checkpoints were trained with image augmentation.
    The upstream evaluator first resizes each 128px camera to 224px with a
    Lanczos filter, then applies the same centered 0.9-area crop used by the
    OpenVLA-OFT evaluation stack.  The legacy local adapter delegated resizing
    to the Hugging Face processor and skipped this crop, which is a protocol
    mismatch.  This helper makes the geometry explicit without importing the
    upstream TensorFlow evaluation stack.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("image must be uint8 HxWx3")
    prepared = Image.fromarray(array, mode="RGB").resize(
        (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    if center_crop:
        side = OPENVLA_IMAGE_SIZE * float(np.sqrt(CENTER_CROP_SCALE))
        offset = (OPENVLA_IMAGE_SIZE - side) / 2.0
        prepared = prepared.crop(
            (
                offset,
                offset,
                OPENVLA_IMAGE_SIZE - offset,
                OPENVLA_IMAGE_SIZE - offset,
            )
        ).resize(
            (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
    return prepared


class ProtocolMatchedMuVLAPolicy(MuVLAPolicy):
    """MuVLAPolicy with the released training/evaluation image geometry."""

    def __init__(self, *args: Any, center_crop: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.center_crop = bool(center_crop)

    def _inputs(self, obs: dict[str, Any]) -> tuple[Any, np.ndarray]:
        rgb = obs["rgb"]
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)[0]
        prompt = (
            f"In: What action should the robot take to "
            f"{self.instruction.lower()}?\nOut:"
        )
        processed = [
            prepare_training_matched_image(
                np.asarray(rgb[..., start : start + 3], dtype=np.uint8),
                center_crop=self.center_crop,
            )
            for start in (0, 3)
        ]
        inputs = [
            self.processor(prompt, image).to("cuda", dtype=torch.bfloat16)
            for image in processed
        ]
        inputs[0]["pixel_values"] = torch.cat(
            [inputs[0]["pixel_values"], inputs[1]["pixel_values"]], dim=1
        )
        proprio = obs["proprio"]
        if hasattr(proprio, "detach"):
            proprio = proprio.detach().cpu().numpy()
        proprio = np.asarray(proprio)[0]
        return inputs[0], normalize_bounds_q99(proprio, self.stats["proprio"])
