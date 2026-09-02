"""Identity-preserving causal edits for opaque recurrent VLA memory."""

from __future__ import annotations

import numpy as np
import torch


def identity_orthogonal_projector(identity_weights: torch.Tensor) -> torch.Tensor:
    """Project vectors onto the orthogonal complement of identity directions.

    ``identity_weights`` is expected to contain one linear-probe row per class
    in a shared latent space.  The pseudoinverse makes the construction robust
    to linearly dependent class rows without inventing an identity/token
    factorization in the recurrent memory tensor.
    """
    weights = torch.as_tensor(identity_weights)
    if weights.ndim != 2:
        raise ValueError("identity_weights must be a rank-2 matrix")
    dimension = weights.shape[1]
    identity = torch.eye(dimension, dtype=weights.dtype, device=weights.device)
    return identity - torch.linalg.pinv(weights) @ weights


def _unit_vector(value: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value)
    if not bool(norm > eps):
        raise ValueError("projected edit direction is numerically zero")
    return value / norm


def identity_preserving_slot_direction(
    identity_weights: torch.Tensor,
    slot_weights: torch.Tensor,
    *,
    target_index: int,
    predicted_index: int,
) -> torch.Tensor:
    """Return a unit slot-correction direction orthogonal to identity probes."""
    slot = torch.as_tensor(slot_weights)
    if slot.ndim != 2:
        raise ValueError("slot_weights must be a rank-2 matrix")
    if slot.shape[1] != identity_weights.shape[1]:
        raise ValueError("identity and slot probes must share the latent dimension")
    raw = slot[target_index] - slot[predicted_index]
    projected = identity_orthogonal_projector(identity_weights) @ raw
    return _unit_vector(projected)


def random_identity_orthogonal_direction(
    identity_weights: torch.Tensor,
    *,
    dimension: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a unit random control direction with the same identity constraint."""
    weights = torch.as_tensor(identity_weights)
    latent_dim = weights.shape[1]
    if dimension is not None and dimension != latent_dim:
        raise ValueError("dimension must match identity_weights.shape[1]")
    random = torch.randn(
        latent_dim, dtype=weights.dtype, device=weights.device, generator=generator
    )
    projected = identity_orthogonal_projector(weights) @ random
    return _unit_vector(projected)


class IdentitySlotEditor:
    """Apply privileged slot edits in a learned PCA space while preserving identity.

    The editor does not assume that recurrent memory tokens have semantic roles.
    Instead it edits only the token subset used to fit held-out linear probes,
    projects a slot-correction direction away from the identity-probe row space,
    and maps the resulting delta back through orthonormal PCA components.
    """

    def __init__(
        self,
        *,
        token_indices: torch.Tensor,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
        identity_weights: torch.Tensor,
        identity_bias: torch.Tensor,
        identity_classes: torch.Tensor,
        slot_weights: torch.Tensor,
        slot_bias: torch.Tensor,
        slot_classes: torch.Tensor,
        edit_norm: float,
    ) -> None:
        self.token_indices = torch.as_tensor(token_indices, dtype=torch.long)
        self.pca_mean = torch.as_tensor(pca_mean, dtype=torch.float32)
        self.pca_components = torch.as_tensor(pca_components, dtype=torch.float32)
        self.identity_weights = torch.as_tensor(identity_weights, dtype=torch.float32)
        self.identity_bias = torch.as_tensor(identity_bias, dtype=torch.float32)
        self.identity_classes = torch.as_tensor(identity_classes)
        self.slot_weights = torch.as_tensor(slot_weights, dtype=torch.float32)
        self.slot_bias = torch.as_tensor(slot_bias, dtype=torch.float32)
        self.slot_classes = torch.as_tensor(slot_classes)
        self.edit_norm = float(edit_norm)
        if self.pca_components.ndim != 2:
            raise ValueError("pca_components must be rank 2")
        if self.pca_mean.numel() != self.pca_components.shape[1]:
            raise ValueError("pca_mean must match PCA input dimension")
        latent_dim = self.pca_components.shape[0]
        if self.identity_weights.shape[1] != latent_dim or self.slot_weights.shape[1] != latent_dim:
            raise ValueError("probe weights must match PCA latent dimension")
        if self.edit_norm < 0:
            raise ValueError("edit_norm must be non-negative")

    @classmethod
    def from_npz(cls, path: str) -> IdentitySlotEditor:
        data = np.load(path, allow_pickle=False)
        return cls(
            token_indices=torch.from_numpy(data["token_indices"]),
            pca_mean=torch.from_numpy(data["pca_mean"]),
            pca_components=torch.from_numpy(data["pca_components"]),
            identity_weights=torch.from_numpy(data["identity_weights"]),
            identity_bias=torch.from_numpy(data["identity_bias"]),
            identity_classes=torch.from_numpy(data["identity_classes"]),
            slot_weights=torch.from_numpy(data["slot_weights"]),
            slot_bias=torch.from_numpy(data["slot_bias"]),
            slot_classes=torch.from_numpy(data["slot_classes"]),
            edit_norm=float(data["edit_norm"]),
        )

    def encode(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError("memory must have shape (batch, tokens, hidden_dim)")
        indices = self.token_indices.to(memory.device)
        selected = memory.index_select(1, indices).float().flatten(1)
        mean = self.pca_mean.to(selected.device)
        components = self.pca_components.to(selected.device)
        if selected.shape[1] != mean.numel():
            raise ValueError("memory/token shape does not match fitted editor")
        return (selected - mean) @ components.T

    def identity_logits(self, memory: torch.Tensor) -> torch.Tensor:
        z = self.encode(memory)
        weights = self.identity_weights.to(z.device)
        bias = self.identity_bias.to(z.device)
        return z @ weights.T + bias

    def slot_logits(self, memory: torch.Tensor) -> torch.Tensor:
        z = self.encode(memory)
        weights = self.slot_weights.to(z.device)
        bias = self.slot_bias.to(z.device)
        return z @ weights.T + bias

    @staticmethod
    def _class_index(classes: torch.Tensor, value: int) -> int:
        matches = torch.nonzero(classes.cpu() == int(value), as_tuple=True)[0]
        if len(matches) != 1:
            raise ValueError(f"class {value} is not uniquely represented in editor artifact")
        return int(matches[0])

    def edit(
        self,
        memory: torch.Tensor,
        *,
        target_slot: int,
        mode: str = "ipsi",
        predicted_slot: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
        if memory.shape[0] != 1:
            raise ValueError("privileged editor currently supports batch size 1")
        z = self.encode(memory)
        slot_weights = self.slot_weights.to(z.device)
        identity_weights = self.identity_weights.to(z.device)
        classes = self.slot_classes.to(z.device)
        target_index = self._class_index(classes, target_slot)
        if predicted_slot is None:
            predicted_index = int(torch.argmax(self.slot_logits(memory)[0]).item())
            predicted_slot = int(classes[predicted_index].item())
        else:
            predicted_index = self._class_index(classes, predicted_slot)
        slot_logits_before = self.slot_logits(memory)[0]
        identity_logits_before = self.identity_logits(memory)[0]
        identity_index_before = int(torch.argmax(identity_logits_before).item())
        identity_class_before = int(self.identity_classes[identity_index_before].item())
        target_margin_before = float(
            (slot_logits_before[target_index] - slot_logits_before[predicted_index]).item()
        )
        if target_index == predicted_index or self.edit_norm == 0:
            return memory.clone(), {
                "target_slot": int(target_slot),
                "predicted_slot": int(predicted_slot),
                "predicted_slot_after": int(predicted_slot),
                "predicted_identity": identity_class_before,
                "predicted_identity_after": identity_class_before,
                "identity_logit_shift_l2": 0.0,
                "slot_target_margin_before": target_margin_before,
                "slot_target_margin_after": target_margin_before,
                "latent_edit_norm": 0.0,
                "skipped": True,
            }
        if mode == "ipsi":
            direction = identity_preserving_slot_direction(
                identity_weights,
                slot_weights,
                target_index=target_index,
                predicted_index=predicted_index,
            )
        elif mode == "slot_only":
            direction = _unit_vector(slot_weights[target_index] - slot_weights[predicted_index])
        elif mode == "random_orthogonal":
            direction = random_identity_orthogonal_direction(
                identity_weights, generator=generator
            )
        else:
            raise ValueError("mode must be ipsi, slot_only, or random_orthogonal")
        delta_z = direction * self.edit_norm
        components = self.pca_components.to(z.device)
        delta_raw = delta_z @ components
        indices = self.token_indices.to(memory.device)
        edited = memory.clone()
        selected_shape = edited.index_select(1, indices).shape
        delta_tokens = delta_raw.reshape(selected_shape).to(
            device=memory.device, dtype=memory.dtype
        )
        edited[:, indices, :] = edited[:, indices, :] + delta_tokens
        slot_logits_after = self.slot_logits(edited)[0]
        identity_logits_after = self.identity_logits(edited)[0]
        predicted_after_index = int(torch.argmax(slot_logits_after).item())
        identity_after_index = int(torch.argmax(identity_logits_after).item())
        return edited, {
            "target_slot": int(target_slot),
            "predicted_slot": int(predicted_slot),
            "predicted_slot_after": int(self.slot_classes[predicted_after_index].item()),
            "predicted_identity": identity_class_before,
            "predicted_identity_after": int(self.identity_classes[identity_after_index].item()),
            "identity_logit_shift_l2": float(
                torch.linalg.vector_norm(identity_logits_after - identity_logits_before).item()
            ),
            "slot_target_margin_before": target_margin_before,
            "slot_target_margin_after": float(
                (slot_logits_after[target_index] - slot_logits_after[predicted_index]).item()
            ),
            "latent_edit_norm": float(torch.linalg.vector_norm(delta_z).item()),
            "skipped": False,
        }
