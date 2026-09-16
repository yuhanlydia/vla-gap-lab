#!/usr/bin/env python3
"""Replay saved v2 actions, verify traces, add small v3 contact/visibility sidecars.

No model/checkpoint is loaded. The original NPZ and all v2 reports are read-only.
Replay drift blocks annotation; it never silently relabels a different rollout.
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np

from vla_gap_lab.icassp_v3 import TASKS, check_replay_array, config_digest, digest_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/memory_revision/icassp_v3.json')
    parser.add_argument('--limit', type=int, default=None, help='Use 2 for runtime smoke; no probes/claims.')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    protocol_hash = config_digest(config)
    files = sorted(args.episodes_dir.glob('episode_*.npz'))
    if len(files) != config['episodes']:
        raise ValueError(f'expected {config["episodes"]} v2 source episodes, found {len(files)}')
    if args.output_dir.resolve() == args.episodes_dir.resolve():
        raise ValueError('annotations must not overwrite source episodes')
    if args.limit is not None and not 1 <= args.limit <= len(files):
        raise ValueError('--limit is outside source episode range')
    if version('mani_skill') != config['annotation']['maniskill_version']:
        raise RuntimeError('install the pinned MIKASA runtime; ManiSkill version mismatch')
    mikasa = ROOT/'external/MIKASA-Robo'
    actual_commit = subprocess.check_output(['git', '-C', str(mikasa), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != config['annotation']['mikasa_commit']:
        raise RuntimeError('MIKASA source commit mismatch; do not relabel with a different simulator')

    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers
    import torch
    from vla_gap_lab.dynamics_io import load_episode_npz, save_episode_npz_atomic
    from vla_gap_lab.icassp_v3_sim import ContactRecorder, as_numpy, ball_pixel_counts
    from vla_gap_lab.mu_vla_protocol import step_mikasa_env

    env = gym.make(args.task, num_envs=1, obs_mode='rgb', control_mode='pd_ee_delta_pose',
                   reward_mode='normalized_dense', render_mode='rgb_array', sim_backend='gpu')
    env = apply_mikasa_vla_wrappers(env, include_overlays=False)
    base = env.unwrapped
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for episode_id, source in enumerate(files[:args.limit]):
            arrays, meta = load_episode_npz(source)
            if meta.get('schema_version') != 2 or meta.get('study') != 'icassp_motion_compression':
                raise ValueError(f'{source}: not the v2 motion-compression cache')
            if meta.get('task') != args.task or meta.get('episode') != episode_id:
                raise ValueError(f'{source}: task/episode mismatch')
            if meta.get('simulator_step_sync') != 'render_after_step':
                raise ValueError('uncertified source simulator synchronization')
            steps = np.asarray(arrays['step'])
            if not np.array_equal(steps, np.arange(len(steps))):
                raise ValueError('source must contain all contiguous steps, not filtered rows')
            source_hash = digest_file(source)
            destination = args.output_dir/source.name
            expected = {'schema_version': 3, 'study': 'icassp_v3_annotations', 'episode': episode_id,
                        'seed': meta['seed'], 'task': args.task, 'source_sha256': source_hash,
                        'protocol_hash': protocol_hash, 'source_filename': source.name}
            if destination.exists():
                if not args.resume:
                    raise ValueError(f'{destination} exists; pass --resume')
                _, saved_meta = load_episode_npz(destination)
                if any(saved_meta.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'{destination}: incompatible annotation provenance')
                print(json.dumps({'episode': episode_id, 'status': 'validated_resume'}), flush=True)
                continue
            env.reset(seed=int(meta['seed']))
            contacts, pixels, forces, samples = [], [], [], []
            maximum_errors = {}
            with ContactRecorder(base, config['annotation']['contact_threshold_newton']) as recorder:
                for t in range(len(steps)):
                    current = {'ball_position_xy': as_numpy(base.ball.pose.p)[0, :2],
                               'ball_velocity_xy': as_numpy(base.ball.linear_velocity)[0, :2],
                               'goal_position_xy': as_numpy(base.goal_region.pose.p)[0, :2],
                               'tcp_position_xy': as_numpy(base.agent.tcp.pose.p)[0, :2]}
                    for key, observed in current.items():
                        err = check_replay_array(arrays[key][t], observed, name=f'{source.name}/{key}/t{t}',
                                                 atol=config['annotation']['replay_atol'])
                        maximum_errors[key] = max(maximum_errors.get(key, 0.), err)
                    contacts.append(recorder.seen)
                    forces.append(recorder.peak_force)
                    samples.append(recorder.samples)
                    pixels.append(ball_pixel_counts(base))
                    action = np.asarray(arrays['action'][t], dtype=np.float32)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError('source contains invalid executed action')
                    _, reward, terminated, truncated, info = step_mikasa_env(
                        env, torch.as_tensor(action[None], device=base.device))
                    check_replay_array(np.asarray(arrays['reward'][t]).reshape(1), as_numpy(reward).reshape(1),
                                       name=f'reward/t{t}', atol=config['annotation']['reward_atol'])
                    success = bool(as_numpy(info['success']).reshape(-1)[0])
                    if success != bool(arrays['success'][t]):
                        raise ValueError(f'replay success mismatch at {source.name}/t{t}')
                    done = bool(as_numpy(terminated).reshape(-1)[0]) or bool(as_numpy(truncated).reshape(-1)[0])
                    if done and t != len(steps) - 1:
                        raise ValueError('replay terminated earlier than the source trajectory')
                if recorder.samples == 0:
                    raise RuntimeError('physics contact hook never ran; do not emit fake no-contact labels')
            save_episode_npz_atomic(destination, {
                'step': steps.astype(np.int32),
                'robot_contact_seen': np.asarray(contacts, dtype=np.int8),
                'ball_visible_pixels': np.asarray(pixels, dtype=np.int32),
                'peak_robot_ball_force_so_far': np.asarray(forces, dtype=np.float32),
                'contact_samples_so_far': np.asarray(samples, dtype=np.int32),
            }, {**expected, 'maximum_replay_errors': maximum_errors,
                'validation': 'saved_xy_state_reward_success_trace',
                'limitations': 'v2 has no saved RGB hash or full joint state; image identity is not independently proven',
                'mani_skill_version': version('mani_skill'), 'mikasa_commit': actual_commit})
            print(json.dumps({'episode': episode_id, 'status': 'annotated',
                              'contact_seen': bool(any(contacts)), 'max_errors': maximum_errors}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
