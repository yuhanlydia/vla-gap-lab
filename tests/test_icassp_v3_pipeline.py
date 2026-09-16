import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v3_probe', ROOT/'scripts/probe_icassp_v3.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
from vla_gap_lab.icassp_v3 import PRIMARY_MASK, config_digest, digest_file


def fixtures(tmp_path):
    cfg = json.loads((ROOT/'configs/memory_revision/icassp_v3.json').read_text())
    cfg['episodes'] = 10
    cfg['probe'].update(train_episodes=6, dev_episodes=2, test_episodes=2,
                        token_pca_dim=2, sample_pca_dim=6, mlp_max_epochs=3,
                        mlp_patience=2, min_rows_per_split=4)
    source, annotation = tmp_path/'source', tmp_path/'annotation'
    source.mkdir(); annotation.mkdir()
    rng = np.random.default_rng(7)
    for ep in range(10):
        n = 10
        position = rng.normal(size=(n, 2))
        velocity = rng.normal(size=(n, 2))
        memory = rng.normal(size=(n, 64, 4))
        memory[:, :, :2] += position[:, None]
        visual = rng.normal(size=(n, 8, 4))
        arrays = dict(memory_after=memory.astype(np.float16), visual_tokens=visual.astype(np.float16),
                      step=np.arange(n), ball_position_xy=position, ball_velocity_xy=velocity)
        meta = dict(schema_version=2, study='icassp_motion_compression',
                    task='InterceptMedium-VLA-v0', episode=ep, seed=100+ep, success_once=ep%2==0,
                    checkpoint='synthetic-test-checkpoint', precision='4bit',
                    visual_source='same_mu_vla_projector_current_step', visual_pool_grid=2,
                    preprocess='official_224_center_crop_0.9', simulator_step_sync='render_after_step')
        path = source/f'episode_{ep:04d}_seed_{100+ep}.npz'
        np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(meta)))
        ameta = {**meta, 'schema_version': 3, 'protocol_hash': config_digest(cfg),
                 'source_sha256': digest_file(path), 'validation': 'SYNTHETIC_TEST_ONLY'}
        np.savez_compressed(annotation/path.name, step=np.arange(n),
                            robot_contact_seen=np.zeros(n, dtype=np.int8),
                            ball_visible_pixels=np.ones((n, 2), dtype=np.int32)*20,
                            metadata=np.asarray(json.dumps(ameta)))
    return cfg, source, annotation


def test_small_end_to_end_pipeline_checks_all_cells_pairing_and_resume(tmp_path):
    cfg, source, annotation = fixtures(tmp_path)
    output = tmp_path/'report.json'
    report = probe.run_task(source, annotation, 'InterceptMedium-VLA-v0', cfg, output)
    assert report['status'] == 'complete'
    assert report['canonical_protocol'] is False
    assert len(report['cells']) == 45
    for split in range(5):
        primary = [r for r in report['cells'] if r['split_seed'] == split and r['mask'] == PRIMARY_MASK]
        assert len({r['row_signature'] for r in primary}) == 1
        assert all(r['rows_per_split'] == {'train': 48, 'dev': 16, 'test': 16} for r in primary)
    before = output.read_bytes()
    result = probe.run_task(source, annotation, 'InterceptMedium-VLA-v0', cfg, output, resume=True)
    assert result == report
    assert output.read_bytes() == before


def test_annotations_reject_altered_cache_instead_of_reusing_old_masks(tmp_path):
    cfg, source, annotation = fixtures(tmp_path)
    path = sorted(source.glob('*.npz'))[0]
    arrays, meta = probe.load(path)
    arrays['visual_tokens'][0, 0, 0] += 1
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError, match='another source'):
        probe.validate_sources(sorted(source.glob('*.npz')), annotation, cfg, 'InterceptMedium-VLA-v0')


@pytest.mark.parametrize('field,value', [('checkpoint', 'different-model'), ('visual_pool_grid', 4)])
def test_mixed_source_protocol_is_rejected_even_with_matching_sidecar_hash(tmp_path, field, value):
    cfg, source, annotation = fixtures(tmp_path)
    path = sorted(source.glob('*.npz'))[1]
    arrays, meta = probe.load(path)
    meta[field] = value
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(meta)))
    a, am = probe.load(annotation/path.name)
    am['source_sha256'] = digest_file(path)
    np.savez_compressed(annotation/path.name, **a, metadata=np.asarray(json.dumps(am)))
    with pytest.raises(ValueError, match='source protocol'):
        probe.validate_sources(sorted(source.glob('*.npz')), annotation, cfg, 'InterceptMedium-VLA-v0')
