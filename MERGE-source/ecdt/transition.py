import torch

def bounded_update(z, message, kappa, valid):
    u = message.float()
    base = z.float()
    capacity = kappa.float() * (base.norm(dim=-1, keepdim=True) + 1e-08) + 1e-08
    delta = u / (1 + u.norm(dim=-1, keepdim=True) / capacity)
    delta = delta * valid[:, None]
    return (base + delta, delta, capacity)

def transition(z, message, kappa, valid, is_real, batch=None):
    updated, delta, capacity = bounded_update(z, message, kappa, valid)
    labels = torch.zeros(len(z), dtype=torch.long, device=z.device) if batch is None else batch
    delta = delta.clone()
    for label in labels.unique():
        active = (labels == label) & is_real & valid
        if not bool(active.any()):
            continue
        current = delta[active]
        centered = current - current.mean(0, keepdim=True)
        scale = torch.minimum(z.new_tensor(1.0, dtype=torch.float32), (capacity[active] / centered.norm(dim=-1, keepdim=True).clamp_min(1e-12)).amin())
        delta[active] = centered * scale
    updated = z.float() + delta
    return (updated, delta, capacity)
