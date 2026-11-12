import argparse
from pathlib import Path

import numpy as np

from ecdt.graph import build_hierarchy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=3927)
    args = parser.parse_args()
    with np.load(args.features, allow_pickle=False) as source:
        record = {key: source[key].copy() for key in ('x', 'coords', 'is_real', 'barcodes')}
        for key in ('source_row_indices', 'raw_panel_gene_names', 'target'):
            if key in source.files:
                record[key] = source[key].copy()
    real, x, coords = record['is_real'], record['x'], record['coords']
    count = int(real.sum())
    if x.shape != (len(real), 1024) or coords.shape != (len(real), 2) or record['barcodes'].shape != real.shape:
        parser.error('Expected aligned x, coords, is_real and barcodes arrays')
    if real.dtype != np.bool_ or not count or not real[:count].all() or real[count:].any():
        parser.error('Real nodes must come first, followed by any pseudo nodes')
    if not np.isfinite(x).all() or not np.isfinite(coords).all():
        parser.error('Features and coordinates must be finite')
    graph = build_hierarchy(x[:count], coords[:count], record['barcodes'][:count], coords[count:], args.seed)
    record.update(spatial_membership=graph['spatial_membership'], morphology_membership=graph['morphology_membership'])
    record.update(graph['edges'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **record)
    print(args.output)


if __name__ == '__main__':
    main()
