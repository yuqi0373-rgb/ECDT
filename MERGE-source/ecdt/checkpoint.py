from pathlib import Path

import torch

from .model import ECDT


def load_model(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and isinstance(state.get('model'), dict):
        state = state['model']
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a model tensor state dictionary')
    model = ECDT()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, temporary)
    temporary.replace(path)
