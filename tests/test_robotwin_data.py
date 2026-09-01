import numpy as np

from vla_gap_lab.robotwin_data import pose_gripper_to_ee6d


def test_robotwin_proprio_conversion_identity_quaternions():
    pose = np.array([[1, 2, 3, 0, 0, 0, 1]], dtype=float)
    result = pose_gripper_to_ee6d(pose, np.array([[0.25]]), pose, np.array([[0.75]]))
    assert result.shape == (1, 20)
    np.testing.assert_allclose(result[0, :3], [1, 2, 3])
    np.testing.assert_allclose(result[0, 9], 0.5)
    np.testing.assert_allclose(result[0, 19], -0.5)
