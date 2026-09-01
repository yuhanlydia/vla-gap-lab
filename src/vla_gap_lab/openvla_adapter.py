"""Explicit adapter around OpenVLA-OFT hidden-state outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def slice_action_token_layers(
    hidden_states: Sequence[torch.Tensor],
    layers: Sequence[int],
    action_start: int,
    action_tokens: int = 56,
) -> torch.Tensor:
    """Return ``[B, L, action_tokens, D]`` from inference-time hidden states."""
    selected = []
    action_stop = action_start + action_tokens
    for layer in layers:
        if layer < 0 or layer >= len(hidden_states):
            raise IndexError(f"layer {layer} outside [0, {len(hidden_states) - 1}]")
        state = hidden_states[layer]
        if action_stop > state.shape[1]:
            raise ValueError(f"action slice [{action_start}:{action_stop}] exceeds sequence")
        selected.append(state[:, action_start:action_stop])
    return torch.stack(selected, dim=1)


def forward_openvla_action_layers(model, inputs, layers: Sequence[int]) -> torch.Tensor:
    """Run the OFT regression path and capture requested action-token layers."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    if not torch.all(input_ids[:, -1] == 29871):
        empty = torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat((input_ids, empty), dim=1)
    prompt_tokens = input_ids.shape[-1] - 1
    labels = torch.full_like(input_ids, -100)
    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)
    embeddings = model.get_input_embeddings()(input_ids)
    action_mask = model._process_action_masks(labels)
    language = embeddings[~action_mask].reshape(embeddings.shape[0], -1, embeddings.shape[2])
    patches = model._process_vision_features(pixel_values, language, False)
    embeddings = embeddings * ~action_mask.unsqueeze(-1)
    multimodal, multimodal_mask = model._build_multimodal_attention(
        embeddings, patches, attention_mask
    )
    output = model.language_model(
        input_ids=None,
        attention_mask=multimodal_mask,
        inputs_embeds=multimodal,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    num_patches = model.vision_backbone.get_num_patches()
    return slice_action_token_layers(output.hidden_states, layers, num_patches + prompt_tokens)


def action_token_mask(
    current_action_mask: torch.Tensor, next_actions_mask: torch.Tensor
) -> torch.Tensor:
    if current_action_mask.shape != next_actions_mask.shape:
        raise ValueError("current and next action masks must have identical shapes")
    if current_action_mask.ndim != 2:
        raise ValueError("action masks must have shape [B, text_tokens]")
    return current_action_mask.bool() | next_actions_mask.bool()


def extract_layer_action_states(
    hidden_states: Sequence[torch.Tensor],
    layers: Sequence[int],
    num_patches: int,
    current_action_mask: torch.Tensor,
    next_actions_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean-pooled ``[B, L, D]`` states at OFT action positions."""
    mask = action_token_mask(current_action_mask, next_actions_mask)
    selected = []
    for layer in layers:
        if layer < 0 or layer >= len(hidden_states):
            raise IndexError(f"layer {layer} outside [0, {len(hidden_states) - 1}]")
        text = hidden_states[layer][:, num_patches:-1]
        if text.shape[:2] != mask.shape:
            raise ValueError(
                f"layer text shape {tuple(text.shape[:2])} does not match mask {tuple(mask.shape)}"
            )
        weight = mask.to(text.dtype).unsqueeze(-1)
        pooled = (text * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)
        selected.append(pooled)
    return torch.stack(selected, dim=1)


def assert_action_chunk(actions: torch.Tensor, horizon: int = 8, action_dim: int = 7) -> None:
    if tuple(actions.shape[-2:]) != (horizon, action_dim):
        raise ValueError(
            f"expected action chunk [..., {horizon}, {action_dim}], got {tuple(actions.shape)}"
        )
