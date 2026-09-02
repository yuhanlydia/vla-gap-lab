import torch

from vla_gap_lab.identity_slot_editor import (
    IdentitySlotEditor,
    identity_orthogonal_projector,
    identity_preserving_slot_direction,
    random_identity_orthogonal_direction,
)


def test_identity_preserving_slot_direction_removes_identity_component():
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    slot = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    direction = identity_preserving_slot_direction(
        identity, slot, target_index=1, predicted_index=2
    )
    torch.testing.assert_close(identity @ direction, torch.zeros(2), atol=1e-6, rtol=0)
    assert torch.dot(slot[1] - slot[2], direction) > 0
    torch.testing.assert_close(torch.linalg.vector_norm(direction), torch.tensor(1.0))


def test_identity_orthogonal_projector_is_symmetric_and_idempotent():
    identity = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    projector = identity_orthogonal_projector(identity)
    torch.testing.assert_close(projector, projector.T, atol=1e-6, rtol=0)
    torch.testing.assert_close(projector @ projector, projector, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(identity @ projector, torch.zeros_like(identity), atol=1e-5, rtol=0)


def test_random_control_is_identity_orthogonal_and_norm_matched():
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    generator = torch.Generator().manual_seed(7)
    direction = random_identity_orthogonal_direction(
        identity, dimension=3, generator=generator
    )
    torch.testing.assert_close(identity @ direction, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(direction), torch.tensor(1.0))


def test_editor_changes_only_selected_tokens_and_preserves_identity_logits():
    editor = IdentitySlotEditor(
        token_indices=torch.tensor([0, 2]),
        pca_mean=torch.zeros(4),
        pca_components=torch.eye(4),
        identity_weights=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        identity_bias=torch.zeros(2),
        identity_classes=torch.tensor([0, 1]),
        slot_weights=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 2.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        slot_bias=torch.zeros(3),
        slot_classes=torch.tensor([0, 1, 2]),
        edit_norm=0.75,
    )
    memory = torch.zeros(1, 4, 2)
    identity_before = editor.identity_logits(memory)
    slot_before = editor.slot_logits(memory)
    edited, stats = editor.edit(memory, target_slot=1, mode="ipsi", predicted_slot=2)
    identity_after = editor.identity_logits(edited)
    slot_after = editor.slot_logits(edited)

    torch.testing.assert_close(edited[:, 1], memory[:, 1])
    torch.testing.assert_close(edited[:, 3], memory[:, 3])
    torch.testing.assert_close(identity_after, identity_before, atol=1e-6, rtol=0)
    assert (slot_after[0, 1] - slot_after[0, 2]) > (slot_before[0, 1] - slot_before[0, 2])
    assert stats["predicted_slot"] == 2
    assert stats["target_slot"] == 1
    assert abs(stats["latent_edit_norm"] - 0.75) < 1e-6


def test_editor_random_control_is_deterministic_for_generator_seed():
    editor = IdentitySlotEditor(
        token_indices=torch.tensor([0]),
        pca_mean=torch.zeros(2),
        pca_components=torch.eye(2),
        identity_weights=torch.tensor([[1.0, 0.0]]),
        identity_bias=torch.zeros(1),
        identity_classes=torch.tensor([0]),
        slot_weights=torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        slot_bias=torch.zeros(2),
        slot_classes=torch.tensor([0, 1]),
        edit_norm=0.5,
    )
    memory = torch.zeros(1, 2, 2)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a, _ = editor.edit(memory, target_slot=1, mode="random_orthogonal", generator=g1)
    b, _ = editor.edit(memory, target_slot=1, mode="random_orthogonal", generator=g2)
    torch.testing.assert_close(a, b)
