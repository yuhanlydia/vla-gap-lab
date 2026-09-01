"""Layer capture utilities for X-VLA cross-embodiment diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import torch


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
    features = torch.stack([captured[layer].mean(dim=1) for layer in layers], dim=1)
    return action, features
