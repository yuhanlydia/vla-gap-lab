import importlib

import numpy as np
import pytest


def api():
    try:
        return importlib.import_module('vla_gap_lab.icassp_v3')
    except ModuleNotFoundError:
        pytest.fail('corrected ICASSP v3 implementation is missing')


def test_mask_uses_sticky_physical_contact_and_shared_visible_history():
    m = api()
    step = np.arange(8)
    contact = np.array([0, 0, 0, 0, 1, 0, 0, 0], bool)
    pixels = np.full((8, 2), 10)
    pixels[2] = 0
    rows = m.valid_rows(step, contact, pixels, lags=(0, 1), min_step=2)
    np.testing.assert_array_equal(rows, np.zeros(8, bool))
    # Contact is a sticky exclusion, even if a later frame loses contact.
    pixels[:] = 10
    rows = m.valid_rows(step, contact, pixels, lags=(0, 1), min_step=2)
    np.testing.assert_array_equal(np.flatnonzero(rows), [2, 3])


def test_visibility_requires_one_common_camera_not_camera_hopping():
    m = api()
    pixels = np.array([[10, 0], [0, 10], [10, 0], [10, 0]])
    rows = m.valid_rows(np.arange(4), np.zeros(4, bool), pixels, lags=(0, 1), min_step=2)
    np.testing.assert_array_equal(np.flatnonzero(rows), [3])


def test_lag_alignment_rejects_missing_steps_and_does_not_drop_based_on_velocity():
    m = api()
    step = np.array([0, 1, 3, 4, 5])
    rows = m.valid_rows(step, np.zeros(5, bool), np.ones((5, 2)) * 9, lags=(0, 1), min_step=2)
    np.testing.assert_array_equal(np.flatnonzero(rows), [3, 4])


def test_replay_fails_on_drift_instead_of_certifying_old_features():
    m = api()
    assert m.check_replay_array(np.ones(3), np.ones(3), name='position', atol=1e-4) == 0
    with pytest.raises(ValueError, match='replay'):
        m.check_replay_array(np.ones(3), np.ones(3) + .01, name='position', atol=1e-4)
    with pytest.raises(ValueError, match='finite'):
        m.check_replay_array(np.ones(3), [1, float('nan'), 1], name='position', atol=1e-4)


def toy():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(160, 6))
    y = np.column_stack((.002 * (x[:, 0] + x[:, 1]) + 8, 20 * x[:, 2] - 30))
    return x[:96], y[:96], x[96:128], y[96:128], x[128:], y[128:]


def test_mlp_targets_scaled_on_train_and_predict_inverse_transforms():
    m = api()
    trainx, trainy, devx, devy, testx, testy = toy()
    model, report = m.fit_probe(trainx, trainy, devx, devy, testx, testy,
                                probe='mlp', seed=1, max_epochs=180, patience=25)
    np.testing.assert_allclose(model.y_scaler.mean_, trainy.mean(axis=0))
    assert model.predict(testx).shape == testy.shape
    assert report['target_standardization'] == 'train_only'
    assert report['validation'] == 'explicit_episode_dev'
    assert report['test']['r2_mean'] > .60
    assert len(report['training']['dev_loss_curve']) == report['training']['epochs_run']


def test_mlp_is_equivariant_to_target_units():
    m = api()
    tx, ty, dx, dy, xx, yy = toy()
    kwargs = dict(probe='mlp', seed=4, max_epochs=60, patience=12)
    a, _ = m.fit_probe(tx, ty, dx, dy, xx, yy, **kwargs)
    scale, offset = np.array([1000., .001]), np.array([3., -5.])
    b, _ = m.fit_probe(tx, ty * scale + offset, dx, dy * scale + offset,
                       xx, yy * scale + offset, **kwargs)
    np.testing.assert_allclose((b.predict(xx) - offset) / scale, a.predict(xx), rtol=1e-5, atol=1e-5)


def test_test_labels_cannot_select_epoch():
    m = api()
    tx, ty, dx, dy, xx, yy = toy()
    kwargs = dict(probe='mlp', seed=7, max_epochs=40, patience=8)
    a, ar = m.fit_probe(tx, ty, dx, dy, xx, yy, **kwargs)
    b, br = m.fit_probe(tx, ty, dx, dy, xx, yy[::-1], **kwargs)
    np.testing.assert_allclose(a.predict(xx), b.predict(xx))
    assert ar['training']['best_epoch'] == br['training']['best_epoch']


def test_epoch_limit_is_not_reported_as_convergence():
    m = api()
    _, report = m.fit_probe(*toy(), probe='mlp', seed=0, max_epochs=1, patience=10)
    assert report['training']['converged'] is False
    assert report['training']['stop_reason'] == 'max_epochs'


def test_constant_target_axis_is_undefined_not_fake_zero_or_one():
    m = api()
    y = np.column_stack((np.zeros(5), np.arange(5)))
    r = m.regression_metrics(y, y)
    assert r['r2_per_dim'][0] is None
    assert r['r2_per_dim'][1] == 1.
    assert r['r2_mean'] is None


def make_report(task, probe_values=None, converged=True):
    vals = probe_values or {'visual_current': (.8, .1), 'visual_pair': (.8, .8), 'memory': (.8, .3)}
    cells = []
    for seed in range(5):
        for probe in ('ridge', 'mlp'):
            for rep, (p, v) in vals.items():
                cells.append({'split_seed': seed, 'representation': rep, 'probe': probe,
                              'mask': 'visible_pre_robot_contact', 'lag': 1 if rep == 'visual_pair' else 0,
                              'position_r2_mean': p, 'velocity_y_r2': v,
                              'fit_converged': converged if probe == 'mlp' else True,
                              'primary_eligible': True, 'row_signature': f'rows{seed}'})
    return {'schema_version': 3, 'task': task, 'cells': cells, 'protocol_hash': 'same', 'status': 'complete'}


def test_gate_reads_ridge_and_mlp_consistency_separately():
    m = api()
    medium, fast = make_report('InterceptMedium-VLA-v0'), make_report('InterceptFast-VLA-v0')
    for r in (medium, fast):
        for c in r['cells']:
            if c['probe'] == 'mlp':
                c['velocity_y_r2'] = -3.
                c['fit_converged'] = False
    result = m.evaluate_claim(medium, fast)
    assert result['primary_probe'] == 'ridge'
    assert result['ridge_gate_passes'] is True
    assert result['status'] == 'inconclusive_mlp'
    assert result['go_icassp'] is False


def test_valid_negative_ridge_cannot_be_rescued_by_positive_mlp():
    m = api()
    medium, fast = make_report('InterceptMedium-VLA-v0'), make_report('InterceptFast-VLA-v0')
    for c in medium['cells']:
        if c['probe'] == 'ridge' and c['representation'] == 'visual_pair':
            c['velocity_y_r2'] = .01
    result = m.evaluate_claim(medium, fast)
    assert result['status'] == 'not_supported'
    assert result['go_icassp'] is False


def test_gate_rejects_mixed_versions_incomplete_splits_or_unmatched_rows():
    m = api()
    medium, fast = make_report('InterceptMedium-VLA-v0'), make_report('InterceptFast-VLA-v0')
    fast['schema_version'] = 2
    with pytest.raises(ValueError, match='version'):
        m.evaluate_claim(medium, fast)
    fast['schema_version'] = 3
    fast['cells'].pop()
    with pytest.raises(ValueError, match='split'):
        m.evaluate_claim(medium, fast)
    fast = make_report('InterceptFast-VLA-v0')
    fast['cells'][0]['row_signature'] = 'wrong'
    with pytest.raises(ValueError, match='paired'):
        m.evaluate_claim(medium, fast)


def test_segmentation_counts_crop_not_color_or_world_coordinate():
    m = api()
    seg = np.zeros((1, 128, 128, 1), np.int32)
    seg[0, 50:56, 50:56] = 17
    assert m.visible_pixel_count(seg, 17) > 0
    seg[:] = 0
    seg[0, :, 0] = 17
    assert m.visible_pixel_count(seg, 17) == 0
    with pytest.raises(ValueError):
        m.visible_pixel_count(seg, 0)
