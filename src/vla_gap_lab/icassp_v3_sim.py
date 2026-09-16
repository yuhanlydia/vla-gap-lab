"""Read-only simulator annotation; this module never loads or changes a VLA.

APIs checked against mani-skill/ManiSkill v3.0.0b15. The collector must still
pass a GPU smoke test on the user's pinned MIKASA runtime before a long run.
"""
from __future__ import annotations

import numpy as np

from .icassp_v3 import visible_pixel_count


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ContactRecorder:
    """Latch any robot-link/ball force at EVERY physics substep, not just frame end."""
    def __init__(self, base, threshold=1e-3):
        if not callable(getattr(base.scene, 'get_pairwise_contact_forces', None)):
            raise RuntimeError('missing physical contact API; do not use reached_status as a substitute')
        self.links = base.agent.robot.get_links()
        if not self.links or not callable(getattr(base, '_after_simulation_step', None)):
            raise RuntimeError('missing robot links or physics-substep contact hook')
        self.base, self.threshold = base, float(threshold)
        if self.threshold <= 0:
            raise ValueError('contact threshold must be positive')
        self.seen, self.peak_force, self.samples = False, 0., 0
        self.original = None

    def __enter__(self):
        self.original = self.base._after_simulation_step
        def callback():
            self.original()
            for link in self.links:
                force = as_numpy(self.base.scene.get_pairwise_contact_forces(link, self.base.ball))
                if force.shape != (1, 3) or not np.isfinite(force).all():
                    raise RuntimeError('invalid physical contact force sample')
                magnitude = float(np.linalg.norm(force[0]))
                self.peak_force = max(self.peak_force, magnitude)
                self.seen |= magnitude > self.threshold
            self.samples += 1
        self.base._after_simulation_step = callback
        return self

    def __exit__(self, *exc):
        self.base._after_simulation_step = self.original


def ball_pixel_counts(base):
    """Read already captured policy-camera segmentation, without recapturing RGB."""
    sensors = base._sensors
    names = ('base_camera', 'hand_camera')
    if any(name not in sensors for name in names):
        raise RuntimeError('expected base_camera and hand_camera policy sensors')
    ids = as_numpy(base.ball.per_scene_id).reshape(-1)
    if len(ids) != 1 or int(ids[0]) <= 0:
        raise RuntimeError('missing unambiguous ball segmentation ID')
    counts = []
    for name in names:
        sensor = sensors[name]
        obs = sensor.get_obs(rgb=False, depth=False, position=False, segmentation=True)
        if 'segmentation' not in obs:
            raise RuntimeError('shader does not expose segmentation; cannot certify visibility')
        counts.append(visible_pixel_count(as_numpy(obs['segmentation']), int(ids[0])))
    return np.asarray(counts, dtype=np.int32)
