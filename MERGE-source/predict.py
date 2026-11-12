import argparse
from pathlib import Path

import numpy as np
import torch

from ecdt import load_model
from ecdt.data import load_graph, load_normalization, prepare_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('prediction.npz'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, args.device)
    record = load_graph(args.graph)
    data = prepare_graph(record, load_normalization(args.normalization), args.device)
    with torch.inference_mode():
        prediction = model(**data)['prediction'].cpu().numpy()
    if prediction.shape != (int(record['is_real'].sum()), 250) or not np.isfinite(prediction).all():
        raise RuntimeError('Invalid expression prediction')
    arrays = dict(prediction=prediction, coordinates=record['coords'][record['is_real']])
    if 'barcodes' in record:
        arrays['barcodes'] = record['barcodes'][record['is_real']]
    if 'raw_panel_gene_names' in record:
        arrays['gene_names'] = record['raw_panel_gene_names']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f'{args.output}: {prediction.shape[0]} spots x {prediction.shape[1]} genes')


if __name__ == '__main__':
    main()
