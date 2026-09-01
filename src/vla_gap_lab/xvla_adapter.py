"""Layer capture utilities for X-VLA cross-embodiment diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def pool_token_features(tokens: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool token sequences without silently discarding token heterogeneity."""
    if pooling == "mean":
        return tokens.mean(dim=1)
    if pooling == "summary":
        return torch.stack(
            [tokens.mean(dim=1), tokens.std(dim=1), tokens[:, 0], tokens[:, -1]], dim=1
        )
    raise ValueError("pooling must be 'mean' or 'summary'")


def _tensor_output(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"hook output has unsupported type {type(output).__name__}")


@torch.inference_mode()
def capture_xvla_action_layers(
    model,
    model_inputs: dict[str, torch.Tensor],
    *,
    domain_id: int,
    proprio: torch.Tensor,
    layers: Sequence[int],
    steps: int = 1,
    pooling: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate actions and return mean-pooled action tokens at selected blocks.

    Returned features have shape ``[B, len(layers), H]``. When flow matching
    uses multiple denoising steps, captures from the final step are retained.
    """
    depth = len(model.transformer.blocks)
    if any(layer < 0 or layer >= depth for layer in layers):
        raise IndexError(f"layers must be within [0, {depth - 1}]")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:

        def hook(_module, _inputs, output, layer=layer):
            captured[layer] = _tensor_output(output)[:, : model.num_actions].detach()

        handles.append(model.transformer.blocks[layer].register_forward_hook(hook))
    try:
        batch = proprio.shape[0]
        action = model.generate_actions(
            **model_inputs,
            domain_id=torch.full((batch,), domain_id, device=proprio.device, dtype=torch.long),
            proprio=proprio,
            steps=steps,
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError("not every requested X-VLA layer executed")
    features = torch.stack(
        [pool_token_features(captured[layer], pooling) for layer in layers], dim=1
    )
    return action, features


@torch.inference_mode()
def capture_xvla_joint_layers(
    model,
    model_inputs: dict[str, torch.Tensor],
    *,
    domain_id: int,
    proprio: torch.Tensor,
    vlm_layers: Sequence[int],
    action_layers: Sequence[int],
    steps: int = 1,
    pooling: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture Florence encoder and domain-conditioned action-stack layers."""
    encoder = model.vlm.language_model.model.encoder.layers
    if any(layer < 0 or layer >= len(encoder) for layer in vlm_layers):
        raise IndexError("VLM layer out of bounds")
    vlm_captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in vlm_layers:

        def hook(_module, _inputs, output, layer=layer):
            vlm_captured[layer] = _tensor_output(output).detach()

        handles.append(encoder[layer].register_forward_hook(hook))
    try:
        action, action_features = capture_xvla_action_layers(
            model,
            model_inputs,
            domain_id=domain_id,
            proprio=proprio,
            layers=action_layers,
            steps=steps,
            pooling=pooling,
        )
    finally:
        for handle in handles:
            handle.remove()
    vlm_features = torch.stack(
        [pool_token_features(vlm_captured[layer], pooling) for layer in vlm_layers], dim=1
    )
    return action, vlm_features, action_features
