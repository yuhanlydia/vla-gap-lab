"""Training-matched preprocessing and inference wrapper for released mu-VLA checkpoints."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any

import numpy as np
import torch
from PIL import Image

from .mu_vla_adapter import MuVLAPolicy, normalize_bounds_q99

OPENVLA_IMAGE_SIZE = 224
CENTER_CROP_SCALE = 0.9
MU_VLA_TRANSFORMERS_VERSION = "4.40.1"
MU_VLA_TOKENIZERS_VERSION = "0.19.1"
MU_VLA_TRANSFORMERS_REPO = "https://github.com/CognitiveAISystems/transformers-mu-openvla-oft.git"
MU_VLA_TRANSFORMERS_COMMIT = "9dbc09f574912a45dd0d71354c035e3c37bcce9e"


def _normalize_vcs_url(url: str | None) -> str:
    value = (url or "").strip().lower().rstrip("/")
    return value[:-4] if value.endswith(".git") else value


def transformers_runtime_provenance() -> dict[str, str | None]:
    """Return installed Transformers/Tokenizers versions and VCS provenance.

    mu-VLA relies on a memory-aware Transformers fork. The package version is
    intentionally still 4.40.1, so version checking alone cannot distinguish
    the correct fork from upstream or OpenVLA-OFT's non-memory fork. PEP 610
    ``direct_url.json`` records the VCS source and resolved commit.
    """
    import tokenizers
    import transformers

    url = None
    commit = None
    requested_revision = None
    try:
        direct_url_text = distribution("transformers").read_text("direct_url.json")
    except PackageNotFoundError:
        direct_url_text = None
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            direct_url = {}
        url = direct_url.get("url")
        vcs_info = direct_url.get("vcs_info") or {}
        commit = vcs_info.get("commit_id")
        requested_revision = vcs_info.get("requested_revision")
    return {
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "transformers_url": url,
        "transformers_commit": commit,
        "transformers_requested_revision": requested_revision,
    }


def mu_vla_runtime_issues(provenance: dict[str, str | None]) -> list[str]:
    """Explain why a runtime is not the released mu-VLA inference stack."""
    issues = []
    if provenance.get("transformers_version") != MU_VLA_TRANSFORMERS_VERSION:
        issues.append(
            "transformers version must be "
            f"{MU_VLA_TRANSFORMERS_VERSION}, got {provenance.get('transformers_version')!r}"
        )
    if provenance.get("tokenizers_version") != MU_VLA_TOKENIZERS_VERSION:
        issues.append(
            "tokenizers version must be "
            f"{MU_VLA_TOKENIZERS_VERSION}, got {provenance.get('tokenizers_version')!r}"
        )
    expected_repo = _normalize_vcs_url(MU_VLA_TRANSFORMERS_REPO)
    actual_repo = _normalize_vcs_url(provenance.get("transformers_url"))
    if actual_repo != expected_repo:
        issues.append(
            "transformers must come from the memory-aware mu-VLA fork "
            f"{MU_VLA_TRANSFORMERS_REPO}, got {provenance.get('transformers_url')!r}"
        )
    if provenance.get("transformers_commit") != MU_VLA_TRANSFORMERS_COMMIT:
        issues.append(
            "transformers fork commit must be "
            f"{MU_VLA_TRANSFORMERS_COMMIT}, got {provenance.get('transformers_commit')!r}"
        )
    return issues


def assert_mu_vla_runtime() -> dict[str, str | None]:
    """Fail fast when the installed attention stack cannot reproduce mu-VLA."""
    provenance = transformers_runtime_provenance()
    issues = mu_vla_runtime_issues(provenance)
    if issues:
        rendered = "\n - ".join(issues)
        raise RuntimeError(
            "incompatible mu-VLA runtime; reinstall requirements/track2-extra.txt:\n - "
            + rendered
        )
    return provenance


def clip_mikasa_action(action: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Match the released MIKASA evaluator's final action-space clipping."""
    if torch.is_tensor(action):
        return torch.clamp(action, -1.0, 1.0)
    return np.clip(np.asarray(action), -1.0, 1.0)


def prepare_training_matched_image(
    image: np.ndarray, *, center_crop: bool = True
) -> Image.Image:
    """Match official mu-VLA resize + image-augmentation center-crop geometry.

    Published mu-VLA MIKASA checkpoints were trained with image augmentation.
    The upstream evaluator first resizes each 128px camera to 224px with a
    Lanczos filter, then applies the same centered 0.9-area crop used by the
    OpenVLA-OFT evaluation stack. The legacy local adapter delegated resizing
    to the Hugging Face processor and skipped this crop, which is a protocol
    mismatch. This helper makes the geometry explicit without importing the
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
    """MuVLAPolicy with the released runtime and training/evaluation geometry."""

    def __init__(
        self,
        *args: Any,
        center_crop: bool = True,
        validate_runtime: bool = True,
        **kwargs: Any,
    ) -> None:
        self.runtime_provenance = assert_mu_vla_runtime() if validate_runtime else None
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

    @torch.inference_mode()
    def forward(self, obs: dict[str, Any]) -> torch.Tensor:
        return clip_mikasa_action(super().forward(obs))
