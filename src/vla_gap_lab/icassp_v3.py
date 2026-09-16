"""Corrected, versioned ICASSP diagnostics. No VLA weights are trained here.

v2 remains reproducible at f4e4ced. Never pool its outputs with v3.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PRIMARY_MASK = 'visible_pre_robot_contact'
TASKS = ('InterceptMedium-VLA-v0', 'InterceptFast-VLA-v0')


def digest_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def visible_pixel_count(segmentation, actor_id: int) -> int:
    """Count actor pixels within the SAME 224px/0.9-area crop as policy RGB.

    Nearest-neighbour resampling preserves categorical segmentation IDs. This
    detects occlusion and image support, not merely a world-coordinate box.
    """
    seg = np.asarray(segmentation)
    if actor_id <= 0 or seg.shape != (1, 128, 128, 1):
        raise ValueError('expected positive actor ID and segmentation [1,128,128,1]')
    mask = (seg[0, :, :, 0] == actor_id).astype(np.uint8)
    image = Image.fromarray(mask).resize((224, 224), Image.Resampling.NEAREST)
    offset = (224 - 224 * np.sqrt(.9)) / 2
    image = image.crop((offset, offset, 224 - offset, 224 - offset))
    image = image.resize((224, 224), Image.Resampling.NEAREST)
    return int(np.count_nonzero(np.asarray(image)))


def valid_rows(step, robot_contact, visible_pixels, *, lags=(0, 1, 2), min_step=2,
               min_pixels=4, pre_contact=True) -> np.ndarray:
    """A common paired mask, never conditioned on target velocity or success.

    Contact is latched from physics substeps. All lag endpoints must be visible
    in at least ONE common camera. Initial/missing history is not zero-padded.
    """
    steps = np.asarray(step)
    contact = np.asarray(robot_contact)
    pixels = np.asarray(visible_pixels)
    n = len(steps)
    if steps.ndim != 1 or contact.shape != (n,) or pixels.shape != (n, 2):
        raise ValueError('step/contact/pixel shapes do not align')
    if not np.isfinite(pixels).all() or np.any(pixels < 0):
        raise ValueError('pixel counts must be finite and non-negative')
    if not np.isin(contact, [0, 1]).all():
        raise ValueError('contact labels must be binary')
    if not lags or any(int(k) != k or k < 0 for k in lags) or 0 not in lags:
        raise ValueError('lags must include 0 and contain nonnegative integers')
    if np.any(np.diff(steps) <= 0):
        raise ValueError('steps must be strictly increasing')
    current = np.arange(n)
    supported = steps >= min_step
    common_view = np.ones((n, 2), bool)
    seen_contact = np.maximum.accumulate(contact.astype(bool))
    if pre_contact:
        supported &= ~seen_contact
    for lag in lags:
        previous = np.maximum(current - lag, 0)
        supported &= (current >= lag) & (steps - steps[previous] == lag)
        common_view &= pixels[previous] >= min_pixels
    return supported & common_view.any(axis=1)


def check_replay_array(expected, actual, *, name: str, atol: float) -> float:
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape:
        raise ValueError(f'replay {name}: shape mismatch {expected.shape} != {actual.shape}')
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError(f'replay {name}: non-finite values')
    error = float(np.max(np.abs(expected.astype(float) - actual.astype(float))))
    if error > atol:
        raise ValueError(f'replay {name}: drift {error:.6g} > {atol}; old features cannot be certified')
    return error


def regression_metrics(y, prediction) -> dict:
    true, pred = np.asarray(y, float), np.asarray(prediction, float)
    if true.ndim != 2 or true.shape != pred.shape or len(true) < 2:
        raise ValueError('regression targets must be aligned 2D arrays with >=2 rows')
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError('regression arrays must be finite')
    residual = np.sum((true - pred) ** 2, axis=0)
    total = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    variance = np.var(true, axis=0)
    # Undefined R2 is never converted to a deceptively valid zero or one.
    r2 = [float(1 - a / b) if b > 1e-20 else None for a, b in zip(residual, total)]
    return {'r2_mean': float(np.mean(r2)) if all(v is not None for v in r2) else None,
            'r2_per_dim': r2, 'rmse_per_dim': np.sqrt(np.mean((true - pred) ** 2, axis=0)).tolist(),
            'mae_per_dim': np.mean(np.abs(true - pred), axis=0).tolist(),
            'target_variance_per_dim': variance.tolist(), 'rows': len(true)}


@dataclass
class ScaledProbe:
    estimator: object
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    def predict(self, x):
        pred = np.asarray(self.estimator.predict(self.x_scaler.transform(x)))
        return self.y_scaler.inverse_transform(pred.reshape(len(x), -1))


def fit_probe(train_x, train_y, dev_x, dev_y, test_x, test_y, *, probe: str,
              seed=0, max_epochs=500, patience=25, batch_size=256, tol=1e-4,
              alphas=(.1, 1., 10., 100.)):
    """Fit with train-only X/Y scaling and explicit whole-episode dev stopping.

    Test labels only enter final scoring. A max-epoch stop is NOT convergence.
    MLP capacity/optimizer match v2; no target-dependent hyperparameter search.
    """
    arrays = [np.asarray(v, np.float64) for v in (train_x, train_y, dev_x, dev_y, test_x, test_y)]
    tx, ty, dx, dy, xx, yy = arrays
    if any(a.ndim != 2 or not len(a) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('probe arrays must be nonempty finite matrices')
    if any(len(x) != len(y) for x, y in ((tx, ty), (dx, dy), (xx, yy))):
        raise ValueError('X/Y lengths differ')
    if len({tx.shape[1], dx.shape[1], xx.shape[1]}) != 1 or len({ty.shape[1], dy.shape[1], yy.shape[1]}) != 1:
        raise ValueError('train/dev/test dimensions differ')
    if probe not in {'ridge', 'mlp'} or min(max_epochs, patience, batch_size) < 1:
        raise ValueError('invalid probe or training settings')
    sx, sy = StandardScaler().fit(tx), StandardScaler().fit(ty)
    xt, xd = sx.transform(tx), sx.transform(dx)
    yt, yd = sy.transform(ty), sy.transform(dy)
    training = {'converged': True, 'stop_reason': 'closed_form', 'epochs_run': 0}
    if probe == 'ridge':
        best = None
        for alpha in alphas:
            if alpha <= 0:
                raise ValueError('ridge alphas must be positive')
            candidate = Ridge(alpha=alpha, solver='cholesky').fit(xt, yt)
            loss = float(np.mean((candidate.predict(xd) - yd) ** 2))
            if best is None or loss < best[0]:
                best = (loss, candidate, float(alpha))
        estimator = best[1]
        training['alpha'] = best[2]
        training['selection'] = 'dev_standardized_mse'
    else:
        estimator = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                                 solver='adam', alpha=1e-4, batch_size=min(batch_size, len(xt)),
                                 learning_rate_init=1e-3, early_stopping=False,
                                 max_iter=1, random_state=seed, shuffle=True)
        best_loss, reference_loss, stale, best_epoch = np.inf, np.inf, 0, 0
        best_model = None
        train_curve, dev_curve = [], []
        converged = False
        for epoch in range(1, max_epochs + 1):
            estimator.partial_fit(xt, yt)
            loss = float(np.mean((estimator.predict(xd) - yd) ** 2))
            train_loss = float(estimator.loss_)
            if not np.isfinite(loss) or not np.isfinite(train_loss):
                raise ValueError('non-finite MLP optimization loss')
            train_curve.append(train_loss)
            dev_curve.append(loss)
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best_model = copy.deepcopy(estimator)
            if loss < reference_loss - tol:
                reference_loss, stale = loss, 0
            else:
                stale += 1
            if stale >= patience:
                converged = True
                break
        estimator = best_model
        training = {'converged': converged,
                    'stop_reason': 'dev_plateau' if converged else 'max_epochs',
                    'epochs_run': epoch, 'best_epoch': best_epoch,
                    'best_dev_standardized_mse': best_loss,
                    'train_loss_curve': train_curve, 'dev_loss_curve': dev_curve,
                    'batch_size': min(batch_size, len(xt)), 'max_epochs': max_epochs,
                    'patience': patience, 'tol': tol, 'seed': seed,
                    'hidden_layer_sizes': [128, 64], 'optimizer': 'adam'}
    model = ScaledProbe(estimator, sx, sy)
    report = {'probe': probe, 'target_standardization': 'train_only',
              'validation': 'explicit_episode_dev', 'training': training,
              'x_mean': sx.mean_.tolist(), 'y_mean': sy.mean_.tolist(),
              'y_scale': sy.scale_.tolist(), 'dev': regression_metrics(dy, model.predict(dx)),
              'test': regression_metrics(yy, model.predict(xx)),
              'test_train_mean_baseline': regression_metrics(yy, np.tile(sy.mean_, (len(yy), 1)))}
    return model, report


def _primary_by_seed(report, probe):
    selected = {}
    for rep in ('visual_current', 'visual_pair', 'memory'):
        rows = [c for c in report['cells'] if c['representation'] == rep
                and c['probe'] == probe and c['mask'] == PRIMARY_MASK
                and c['lag'] == (1 if rep == 'visual_pair' else 0)]
        if sorted(r['split_seed'] for r in rows) != list(range(5)):
            raise ValueError(f'need exactly five unique splits for {probe}/{rep}')
        selected[rep] = sorted(rows, key=lambda r: r['split_seed'])
    for seed in range(5):
        if len({selected[r][seed]['row_signature'] for r in selected}) != 1:
            raise ValueError('primary representations do not use paired rows')
    return selected


def evaluate_claim(medium: dict, fast: dict) -> dict:
    """Ridge is the primary gate; MLP is a separately labelled consistency test.

    Revision after the v2 audit is NOT an untouched preregistered experiment.
    These thresholds reproduce the six v2 effect-size requirements, not a
    significance test or an acceptance prediction.
    """
    for report, task in zip((medium, fast), TASKS):
        if report.get('schema_version') != 3:
            raise ValueError('do not mix v2/v3 report versions')
        if report.get('task') != task or report.get('status') != 'complete':
            raise ValueError('wrong task or incomplete report')
    if medium['protocol_hash'] != fast['protocol_hash']:
        raise ValueError('protocol hashes differ')
    ridge = [_primary_by_seed(r, 'ridge') for r in (medium, fast)]
    mlp = [_primary_by_seed(r, 'mlp') for r in (medium, fast)]
    for a, b in zip(ridge, mlp):
        for rep in a:
            for seed in range(5):
                if a[rep][seed]['row_signature'] != b[rep][seed]['row_signature']:
                    raise ValueError('probe families do not use paired rows')

    def finite(rows):
        return all(c.get('primary_eligible', False)
                   and all(c[k] is not None and np.isfinite(c[k]) for k in ('position_r2_mean', 'velocity_y_r2'))
                   for group in rows for values in group.values() for c in values)

    def vector(g, rep, key):
        return np.asarray([c[key] for c in g[rep]], float)

    def checks(groups):
        m, f = groups
        pair_m = vector(m, 'visual_pair', 'velocity_y_r2')
        values = {
            'medium_pair_velocity_ge_0_60': (float(np.median(pair_m)), .60, '>='),
            'medium_pair_gain_over_current_ge_0_20': (float(np.median(pair_m - vector(m, 'visual_current', 'velocity_y_r2'))), .20, '>='),
            'medium_pair_gain_over_memory_ge_0_15': (float(np.median(pair_m - vector(m, 'memory', 'velocity_y_r2'))), .15, '>='),
            'medium_memory_position_ge_0_65': (float(np.median(vector(m, 'memory', 'position_r2_mean'))), .65, '>='),
            'fast_pair_gt_memory': (float(np.median(vector(f, 'visual_pair', 'velocity_y_r2') - vector(f, 'memory', 'velocity_y_r2'))), 0., '>'),
            'fast_memory_position_ge_0_60': (float(np.median(vector(f, 'memory', 'position_r2_mean'))), .60, '>='),
        }
        return {k: {'value': v, 'threshold': t, 'comparison': op,
                    'passes': bool(v >= t if op == '>=' else v > t)} for k, (v, t, op) in values.items()}

    result = {'primary_probe': 'ridge', 'go_icassp': False,
              'caution': 'Post-audit correction on existing cohorts; not independent confirmation.',
              'ridge_gate_passes': False, 'status': 'inconclusive_data'}
    if not finite(ridge):
        return result
    result['ridge_conditions'] = checks(ridge)
    result['ridge_gate_passes'] = all(c['passes'] for c in result['ridge_conditions'].values())
    mlp_ok = finite(mlp) and all(c.get('fit_converged', False) for g in mlp for rows in g.values() for c in rows)
    result['mlp_all_primary_fits_converged'] = bool(mlp_ok)
    # Report all primary MLP values even if unconverged; never silently drop them.
    result['mlp_primary'] = mlp
    if not result['ridge_gate_passes']:
        result['status'] = 'not_supported'
        return result
    if not mlp_ok:
        result['status'] = 'inconclusive_mlp'
        return result
    result['mlp_conditions'] = checks(mlp)
    consistency = all(c['passes'] for c in result['mlp_conditions'].values())
    result['status'] = 'review_candidate' if consistency else 'inconclusive_probe_disagreement'
    result['go_icassp'] = bool(consistency)
    return result
