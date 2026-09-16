#!/usr/bin/env python3
"""Corrected v3 probes: physical masks, explicit dev stopping, paired row IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA

from vla_gap_lab.icassp_v3 import (PRIMARY_MASK, TASKS, config_digest, digest_file,
                                   fit_probe, valid_rows)

ROOT = Path(__file__).resolve().parents[1]
CELLS = tuple((r, PRIMARY_MASK, p, 1 if r == 'visual_pair' else 0)
              for p in ('ridge', 'mlp') for r in ('visual_current', 'visual_pair', 'memory')) + (
    ('memory_stride8', PRIMARY_MASK, 'ridge', 0),
    ('memory', 'visible_all_contact_states', 'ridge', 0),
    ('visual_pair', PRIMARY_MASK, 'ridge', 2),
)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        arrays = {key: data[key] for key in data.files if key != 'metadata'}
    return arrays, meta


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_masks(arrays, annotations, cfg):
    kwargs = dict(lags=tuple(cfg['mask']['common_visible_lags']),
                  min_step=cfg['mask']['min_step'], min_pixels=cfg['mask']['min_visible_pixels'])
    primary = valid_rows(arrays['step'], annotations['robot_contact_seen'],
                         annotations['ball_visible_pixels'], **kwargs)
    all_contact = valid_rows(arrays['step'], annotations['robot_contact_seen'],
                             annotations['ball_visible_pixels'], pre_contact=False, **kwargs)
    return {PRIMARY_MASK: primary, 'visible_all_contact_states': all_contact}


def validate_sources(files, annotation_dir, cfg, task):
    protocol_hash = config_digest(cfg)
    manifest, masks, labels = [], {}, {}
    seen_seeds = set()
    checkpoint = None
    for episode, source in enumerate(files):
        arrays, meta = load(source)
        annotations, ameta = load(annotation_dir/source.name)
        if meta.get('schema_version') != 2 or meta.get('study') != 'icassp_motion_compression':
            raise ValueError('expected unmodified v2 visual+memory source cache')
        if meta.get('task') != task or meta.get('episode') != episode or meta.get('seed') in seen_seeds:
            raise ValueError('task, episode order, or unique seed contract failed')
        seen_seeds.add(meta['seed'])
        source_contract = {
            'precision': '4bit', 'visual_source': 'same_mu_vla_projector_current_step',
            'visual_pool_grid': 2, 'preprocess': 'official_224_center_crop_0.9',
            'simulator_step_sync': 'render_after_step',
        }
        if any(meta.get(key) != value for key, value in source_contract.items()):
            raise ValueError('source protocol differs from the fixed v2 collection contract')
        if not isinstance(meta.get('checkpoint'), str) or not meta['checkpoint']:
            raise ValueError('source protocol has no checkpoint identity')
        if checkpoint is not None and meta['checkpoint'] != checkpoint:
            raise ValueError('source protocol mixes checkpoints')
        checkpoint = meta['checkpoint']
        source_hash = digest_file(source)
        if ameta.get('schema_version') != 3 or ameta.get('protocol_hash') != protocol_hash:
            raise ValueError('annotation version/protocol mismatch')
        if (ameta.get('source_sha256') != source_hash or ameta.get('task') != task
                or ameta.get('seed') != meta['seed'] or ameta.get('episode') != episode):
            raise ValueError('annotation belongs to another source/seed/task')
        if not np.array_equal(arrays['step'], annotations['step']):
            raise ValueError('source/annotation time mismatch')
        length = len(arrays['step'])
        if arrays['memory_after'].shape[:2] != (length, 64) or arrays['visual_tokens'].shape[:2] != (length, 8):
            raise ValueError('expected full64 memory and 8 same-projector visual tokens')
        if not np.array_equal(arrays['step'], np.arange(length)):
            raise ValueError('must preserve contiguous source trajectories before masking')
        masks[episode] = make_masks(arrays, annotations, cfg)
        labels[episode] = (arrays['ball_position_xy'], arrays['ball_velocity_xy'], arrays['step'])
        manifest.append({'episode': episode, 'seed': meta['seed'], 'file': source.name,
                         'sha256': source_hash, 'annotation_sha256': digest_file(annotation_dir/source.name),
                         'total_rows': length,
                         'primary_rows': int(masks[episode][PRIMARY_MASK].sum()),
                         'visible_rows': int(masks[episode]['visible_all_contact_states'].sum()),
                         'success_once': bool(meta.get('success_once', False)),
                         'annotation_validation': ameta.get('validation'),
                         'checkpoint': checkpoint, 'collection_protocol': source_contract})
    return manifest, masks, labels


def token_pca(files, train_ids, masks, *, key, mask_name, dimension, batch=512):
    """Actually bound partial_fit batch size; constructor batch_size is not enough."""
    reducer = IncrementalPCA(n_components=dimension)
    pending = None
    for episode in train_ids:
        # Include visible history endpoints of retained TRAIN rows, not dev/test.
        support = np.flatnonzero(masks[episode][mask_name])
        support = np.unique(np.concatenate([support, support - 1, support - 2]))
        support = support[support >= 0]
        if not len(support):
            continue
        with np.load(files[episode], allow_pickle=False) as data:
            value = data[key][support].astype(np.float32)
        tokens = value.reshape(-1, value.shape[-1])
        for start in range(0, len(tokens), batch):
            part = tokens[start:start + batch]
            pending = part if pending is None else np.concatenate([pending, part])
            while len(pending) >= batch + dimension:
                reducer.partial_fit(pending[:batch])
                pending = pending[batch:]
    if pending is None or len(pending) < dimension:
        raise ValueError('insufficient valid train tokens for fixed PCA')
    reducer.partial_fit(pending)
    return reducer


def compress_sources(files, reducer, key):
    output = []
    for source in files:
        with np.load(source, allow_pickle=False) as data:
            values = data[key].astype(np.float32)
        flat = values.reshape(-1, values.shape[-1])
        z = np.concatenate([reducer.transform(flat[i:i+512]) for i in range(0, len(flat), 512)])
        output.append(z.reshape(values.shape[0], values.shape[1], -1).astype(np.float32))
    return output


def dataset(memory, visual, masks, labels, *, rep, mask_name, lag):
    xs, ps, vs, groups, identifiers = [], [], [], [], []
    for episode in range(len(memory)):
        indices = np.flatnonzero(masks[episode][mask_name])
        if not len(indices):
            continue
        if rep == 'memory':
            x = memory[episode][indices].reshape(len(indices), -1)
        elif rep == 'memory_stride8':
            x = memory[episode][indices, ::8].reshape(len(indices), -1)
        elif rep == 'visual_current':
            x = visual[episode][indices].reshape(len(indices), -1)
        elif rep == 'visual_pair':
            a = visual[episode][indices - lag].reshape(len(indices), -1)
            b = visual[episode][indices].reshape(len(indices), -1)
            x = np.concatenate([a, b], axis=1)
        else:
            raise ValueError(rep)
        xs.append(x)
        ps.append(labels[episode][0][indices])
        vs.append(labels[episode][1][indices])
        groups.append(np.full(len(indices), episode))
        identifiers.extend((episode, int(t)) for t in indices)
    if not xs:
        raise ValueError('no physically valid visible paired rows; do not relax masks based on scores')
    return np.concatenate(xs), np.concatenate(ps), np.concatenate(vs), np.concatenate(groups), identifiers


def aggregate(cells):
    output = {}
    for rep, mask, probe, lag in CELLS:
        rows = [r for r in cells if (r['representation'], r['mask'], r['probe'], r['lag']) == (rep, mask, probe, lag)]
        stats = {'completed_splits': len(rows), 'all_fits_converged': all(r['fit_converged'] for r in rows)}
        for key in ('position_r2_mean', 'velocity_y_r2'):
            values = [r[key] for r in rows]
            usable = len(values) == 5 and all(v is not None and np.isfinite(v) for v in values)
            stats[key + '_median'] = float(np.median(values)) if usable else None
            stats[key + '_iqr'] = [float(x) for x in np.quantile(values, [.25, .75])] if usable else None
        output[f'{rep}__{mask}__{probe}__lag{lag}'] = stats
    return output


def run_task(episodes_dir, annotation_dir, task, cfg, output, *, resume=False):
    files = sorted(episodes_dir.glob('episode_*.npz'))
    if len(files) != cfg['episodes']:
        raise ValueError('source episode count does not match protocol')
    manifest, masks, labels = validate_sources(files, annotation_dir, cfg, task)
    p = cfg['probe']
    phash = config_digest(cfg)
    default_config = ROOT/'configs/memory_revision/icassp_v3.json'
    canonical = phash == config_digest(json.loads(default_config.read_text()))
    report = {'schema_version': 3, 'study': 'icassp_motion_compression_v3',
              'task': task, 'status': 'running', 'protocol': cfg, 'protocol_hash': phash,
              'canonical_protocol': canonical, 'source_manifest': manifest,
              'success_rate': float(np.mean([r['success_once'] for r in manifest])), 'cells': [],
              'caution': 'Existing cohorts already inspected in v2; repeated splits are not independent replications.'}
    if output.exists():
        if not resume:
            raise ValueError('output exists; pass --resume or use a new versioned output')
        previous = json.loads(output.read_text())
        if any(previous.get(k) != report[k] for k in ('schema_version', 'task', 'protocol_hash', 'source_manifest')):
            raise ValueError('report resume provenance mismatch')
        report = previous
        if report['status'] == 'complete':
            return report
    complete_cells = {(r['split_seed'], r['representation'], r['mask'], r['probe'], r['lag']) for r in report['cells']}
    write_json(output, report)
    for split_seed in p['split_seeds']:
        order = np.random.default_rng(split_seed).permutation(len(files))
        ntrain, ndev, ntest = p['train_episodes'], p['dev_episodes'], p['test_episodes']
        if ntrain + ndev + ntest != len(files):
            raise ValueError('split counts must sum to all episodes')
        split_ids = {'train': order[:ntrain].tolist(), 'dev': order[ntrain:ntrain+ndev].tolist(),
                     'test': order[ntrain+ndev:].tolist()}
        compressed_cache = {}
        for rep, mask_name, probe, lag in CELLS:
            cell_key = (split_seed, rep, mask_name, probe, lag)
            if cell_key in complete_cells:
                continue
            if mask_name not in compressed_cache:
                reduced = []
                for key in ('memory_after', 'visual_tokens'):
                    reducer = token_pca(files, split_ids['train'], masks, key=key, mask_name=mask_name,
                                        dimension=p['token_pca_dim'])
                    reduced.append(compress_sources(files, reducer, key))
                compressed_cache[mask_name] = reduced
            memory, visual = compressed_cache[mask_name]
            x, position, velocity, group, row_ids = dataset(memory, visual, masks, labels,
                                                            rep=rep, mask_name=mask_name, lag=lag)
            selection = {k: np.isin(group, ids) for k, ids in split_ids.items()}
            counts = {k: int(mask.sum()) for k, mask in selection.items()}
            coverage = {k: len(np.unique(group[mask])) for k, mask in selection.items()}
            if min(counts.values()) < 2:
                raise ValueError(f'{cell_key}: fewer than two rows in a split; block instead of fabricating R2')
            dimension = min(p['sample_pca_dim'], counts['train'] - 1, x.shape[1])
            reducer = PCA(n_components=dimension, svd_solver='randomized', random_state=split_seed)
            reducer.fit(x[selection['train']])
            z = {k: reducer.transform(x[mask]) for k, mask in selection.items()}
            metrics = {}
            for target, y in (('position', position), ('velocity', velocity)):
                _, metrics[target] = fit_probe(z['train'], y[selection['train']], z['dev'], y[selection['dev']],
                                               z['test'], y[selection['test']], probe=probe, seed=split_seed,
                                               max_epochs=p['mlp_max_epochs'], patience=p['mlp_patience'],
                                               batch_size=p['mlp_batch_size'], tol=p['mlp_tol'],
                                               alphas=tuple(p['ridge_alphas']))
            eligible = all(counts[k] >= p['min_rows_per_split']
                           and coverage[k] >= np.ceil(len(split_ids[k]) * p['min_episode_coverage']) for k in counts)
            cell = {'split_seed': split_seed, 'representation': rep, 'mask': mask_name,
                    'probe': probe, 'lag': lag, 'split_episode_ids': split_ids,
                    'rows_per_split': counts, 'episodes_per_split_retained': coverage,
                    'row_signature': hashlib.sha256(json.dumps(row_ids).encode()).hexdigest(),
                    'primary_eligible': bool(eligible), 'sample_pca_dim': dimension,
                    'sample_pca_variance': float(reducer.explained_variance_ratio_.sum()),
                    'position_r2_mean': metrics['position']['test']['r2_mean'],
                    'velocity_y_r2': metrics['velocity']['test']['r2_per_dim'][1],
                    'fit_converged': all(m['training']['converged'] for m in metrics.values()),
                    **metrics}
            report['cells'].append(cell)
            # Each cell is checkpointed; 7B features and completed fits are not rerun.
            write_json(output, report)
            print(json.dumps({k: cell[k] for k in ('split_seed', 'representation', 'probe', 'mask',
                                                  'position_r2_mean', 'velocity_y_r2', 'fit_converged')}), flush=True)
    report['aggregates'] = aggregate(report['cells'])
    report['status'] = 'complete'
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-dir', type=Path, required=True)
    parser.add_argument('--annotations-dir', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/memory_revision/icassp_v3.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['schema_version'] != 3:
        raise ValueError('expected v3 config')
    run_task(args.episodes_dir, args.annotations_dir, args.task, config, args.output, resume=args.resume)


if __name__ == '__main__':
    main()
