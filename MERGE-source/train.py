import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ecdt import ECDT, save_weights
from ecdt.data import load_graph, load_target, fit_normalization, prepare_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-graphs', nargs='+', type=Path, required=True)
    parser.add_argument('--val-graphs', nargs='+', type=Path, required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs'))
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=3927)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.weight_decay < 0 or args.threads < 1:
        parser.error('Invalid training configuration')
    if set(p.resolve() for p in args.train_graphs) & set(p.resolve() for p in args.val_graphs):
        parser.error('Training and validation graphs must be disjoint')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    raw = {path: load_graph(path) for path in args.train_graphs + args.val_graphs}
    for path, record in raw.items():
        record['target'] = load_target(args.target_root / (path.stem + '.npy'), record)
    norm = fit_normalization([raw[path] for path in args.train_graphs])
    data = {path: prepare_graph(record, norm, args.device) for path, record in raw.items()}
    target = {path: torch.tensor(record['target'], device=args.device) for path, record in raw.items()}
    model = ECDT().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'normalization.npz', **norm)
    best_mse = float('inf')
    device_type = torch.device(args.device).type
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for path in random.sample(args.train_graphs, len(args.train_graphs)):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == 'cuda'):
                prediction = model(**data[path])['prediction']
                loss = (prediction - target[path]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        squared, elements = 0., 0
        with torch.inference_mode():
            for path in args.val_graphs:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == 'cuda'):
                    prediction = model(**data[path])['prediction']
                error = prediction - target[path]
                if not torch.isfinite(error).all():
                    raise RuntimeError('Nonfinite validation prediction')
                squared += float(error.square().sum())
                elements += error.numel()
        mse = squared / elements
        if mse < best_mse:
            best_mse = mse
            save_weights(model, args.output / 'model.ckpt')
        save_weights(model, args.output / 'final.ckpt')
        print(f'{epoch + 1}/{args.epochs} train_mse={np.mean(losses):.6f} val_mse={mse:.6f}', flush=True)


if __name__ == '__main__':
    main()
