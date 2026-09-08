from pathlib import Path

import numpy as np
import torch

from scripts.eval_mu_vla_temporal_operator import resolve_episodes_dir
from vla_gap_lab.mu_vla_protocol import (
    MU_VLA_TOKENIZERS_VERSION,
    MU_VLA_TRANSFORMERS_COMMIT,
    MU_VLA_TRANSFORMERS_REPO,
    MU_VLA_TRANSFORMERS_VERSION,
    clip_mikasa_action,
    mu_vla_runtime_issues,
    prepare_training_matched_image,
    step_mikasa_env,
)
from vla_gap_lab.temporal_operator import (
    apply_memory_velocity_operator,
    apply_velocity_operator,
    velocity_action_gain,
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


def test_mikasa_step_flushes_gpu_pose_updates_with_render():
    class RenderSynchronizedEnv:
        def __init__(self):
            self.events = []
            self.transition = ("obs", "reward", "terminated", "truncated", "info")

        def step(self, action):
            self.events.append(("step", action))
            return self.transition

        def render(self):
            self.events.append(("render", None))

    env = RenderSynchronizedEnv()
    transition = step_mikasa_env(env, "action")

    assert transition == env.transition
    assert env.events == [("step", "action"), ("render", None)]


def test_mikasa_rollout_entry_points_use_synchronized_step_helper():
    repository = Path(__file__).resolve().parents[1]
    entry_points = (
        "scripts/eval_mu_vla_protocol.py",
        "scripts/collect_mu_vla_dynamics_trajectory.py",
        "scripts/collect_mu_vla_memory_trajectory.py",
        "scripts/eval_mu_vla_intervention.py",
        "scripts/eval_mu_vla_identity_slot_intervention.py",
    )

    for relative_path in entry_points:
        source = (repository / relative_path).read_text()
        assert "step_mikasa_env(" in source, relative_path
        assert "env.step(" not in source, relative_path


def test_velocity_action_gain_uses_train_scale_without_tuning():
    assert velocity_action_gain(0.4, 0.8, fraction=0.25) == 0.125


def test_velocity_operator_changes_only_y_translation_and_clips():
    action = np.array([[0.2, 0.95, -0.1, 0.3, 0.4, 0.5, -0.2]], dtype=np.float32)
    adjusted = apply_velocity_operator(action, velocity_y=1.0, gain=0.2)
    np.testing.assert_allclose(adjusted[0, [0, 2, 3, 4, 5, 6]], action[0, [0, 2, 3, 4, 5, 6]])
    assert adjusted[0, 1] == 1.0
    assert np.shares_memory(adjusted, action) is False


def test_velocity_operator_rejects_non_seven_dimensional_actions():
    with np.testing.assert_raises(ValueError):
        apply_velocity_operator(np.zeros((1, 6), dtype=np.float32), velocity_y=1.0, gain=0.2)


def test_memory_velocity_operator_changes_only_retained_tokens():
    memory = np.zeros((1, 16, 3), dtype=np.float32)
    direction = np.ones((2, 3), dtype=np.float32)
    adjusted = apply_memory_velocity_operator(
        memory, direction, standardized_velocity=2.0, gain=0.5
    )
    np.testing.assert_allclose(adjusted[0, ::8], 1.0)
    np.testing.assert_allclose(adjusted[0, 1::8], 0.0)


def test_temporal_operator_resolves_source_data_next_to_probe_report():
    report = Path("/repo/artifacts/reports/probe.json")
    assert resolve_episodes_dir(report) == Path(
        "/repo/artifacts/mikasa/intercept_medium_k2_dynamics_n40"
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
