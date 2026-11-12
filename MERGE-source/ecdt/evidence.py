import torch
RELATIONS = ('spatial_local', 'spatial_cluster_star', 'morphology_cluster_star', 'spatial_representative_shortcut', 'morphology_representative_shortcut', 'cross_representative_shortcut', 'pseudo_context')
ATTRIBUTE_DIM = 12

def edge_attributes(x, coordinates, is_real, ps, pm, edges, batch=None):
    n = len(x)
    device = x.device
    full_s = x.new_zeros((n, ps.shape[-1]))
    full_m = x.new_zeros((n, pm.shape[-1]))
    full_s[is_real] = ps.to(x)
    full_m[is_real] = pm.to(x)
    local = edges.get('spatial_local', torch.empty(2, 0, dtype=torch.long, device=device))
    if coordinates is not None:
        valid = torch.isfinite(coordinates).all(-1)
        c = torch.nan_to_num(coordinates.float())
        distances = (c[local[0]] - c[local[1]]).norm(dim=-1)
        positive = distances[(distances > 0) & valid[local[0]] & valid[local[1]]]
        scale = c.new_ones(n)
        labels = torch.zeros(n, dtype=torch.long, device=device) if batch is None else batch
        for label in labels.unique():
            chosen = (distances > 0) & valid[local[0]] & valid[local[1]] & (labels[local[1]] == label)
            if bool(chosen.any()):
                scale[labels == label] = distances[chosen].median()
    else:
        valid = torch.zeros(n, dtype=torch.bool, device=device)
        c = x.new_zeros((n, 2))
        scale = x.new_ones(n)
    xf = x.float()
    norm = xf.norm(dim=-1)
    out = {}
    for name in RELATIONS:
        src, dst = edges[name]
        cv = valid[src] & valid[dst]
        mv = is_real[src] & is_real[dst]
        offset = (c[src] - c[dst]) / scale[dst, None].clamp_min(1e-06)
        offset = offset.clamp(-20, 20) * cv[:, None]
        cosine = (xf[src] * xf[dst]).sum(-1) / (norm[src] * norm[dst]).clamp_min(1e-08)
        fv = (norm[src] > 0) & (norm[dst] > 0)
        out[name] = torch.cat([offset, offset.norm(dim=-1, keepdim=True), cv[:, None], (cosine * fv)[:, None], fv[:, None], (full_s[src] * full_s[dst]).sum(-1, keepdim=True), (full_m[src] * full_m[dst]).sum(-1, keepdim=True), mv[:, None], (~is_real[src])[:, None], (~is_real[dst])[:, None], mv[:, None]], -1).float().detach()
    return out
