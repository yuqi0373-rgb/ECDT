from pathlib import Path

import numpy as np
import torch

from .evidence import RELATIONS, edge_attributes


def load_graph(path):
    with np.load(path, allow_pickle=False) as source:
        required = ('x', 'coords', 'is_real', 'spatial_membership', 'morphology_membership') + RELATIONS
        missing = set(required) - set(source.files)
        if missing:
            raise ValueError(f'Missing graph arrays: {sorted(missing)}')
        result = {key: source[key].copy() for key in required if key not in RELATIONS}
        result['edges'] = {name: source[name].copy() for name in RELATIONS}
        for key in ('barcodes', 'source_row_indices', 'raw_panel_gene_names', 'target'):
            if key in source.files:
                result[key] = source[key].copy()
    x, real = result['x'], result['is_real']
    if x.ndim != 2 or x.shape[1] != 1024 or not np.isfinite(x).all():
        raise ValueError('x must contain finite 1024-dimensional frozen features')
    if real.dtype != np.bool_ or real.shape != (len(x),) or not real.any():
        raise ValueError('is_real must identify the real expression spots')
    if result['coords'].shape != (len(x), 2):
        raise ValueError('coords must have shape (nodes, 2)')
    for key in ('spatial_membership', 'morphology_membership'):
        if result[key].shape != (int(real.sum()), 5) or not np.isfinite(result[key]).all():
            raise ValueError(f'Invalid {key}')
    for name, edge in result['edges'].items():
        if edge.ndim != 2 or edge.shape[0] != 2 or not np.issubdtype(edge.dtype, np.integer):
            raise ValueError(f'Invalid edge indices: {name}')
        if edge.size and (edge.min() < 0 or edge.max() >= len(x)):
            raise ValueError(f'Out-of-range node index: {name}')
    return result


def load_target(path, record):
    target = np.load(path, allow_pickle=False)
    count = int(record['is_real'].sum())
    if target.ndim != 2 or target.shape[1] != 250:
        raise ValueError('Expression target must have 250 columns in the checkpoint gene order')
    if len(target) != count:
        if 'source_row_indices' not in record:
            raise ValueError('Non-aligned targets require source_row_indices')
        target = target[record['source_row_indices']]
    if target.shape != (count, 250) or not np.isfinite(target).all():
        raise ValueError('Expression targets must align with real nodes')
    return target.astype(np.float32)


def fit_normalization(records):
    if not records:
        raise ValueError('Training graphs are required to fit normalization')
    features = np.concatenate([r['x'][r['is_real']] for r in records])
    return dict(mean=features.mean(0), std=features.std(0).clip(.01))


def load_normalization(path):
    with np.load(path, allow_pickle=False) as source:
        norm = {k: source[k].copy() for k in ('mean', 'std')}
    if any(v.shape != (1024,) or not np.isfinite(v).all() for v in norm.values()) or not (norm['std'] > 0).all():
        raise ValueError('Invalid frozen-feature normalization')
    return norm


def prepare_graph(record, normalization, device='cpu'):
    x = record['x']
    mean, std = normalization['mean'], normalization['std']
    data = dict(x=torch.tensor((x - mean) / std, device=device),
                is_real=torch.tensor(record['is_real'], device=device),
                spatial_membership=torch.tensor(record['spatial_membership'], device=device),
                morphology_membership=torch.tensor(record['morphology_membership'], device=device),
                edge_metadata={k: torch.tensor(v, dtype=torch.long, device=device) for k, v in record['edges'].items()})
    data['edge_metadata']['_evidence'] = edge_attributes(
        torch.as_tensor(x, device=device), torch.as_tensor(record['coords'], device=device),
        data['is_real'], data['spatial_membership'], data['morphology_membership'], data['edge_metadata'])
    return data
