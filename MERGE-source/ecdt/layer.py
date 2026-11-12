import torch
from torch import nn
from .evidence import RELATIONS, ATTRIBUTE_DIM
from .transport import DifferenceGroup
from .transition import transition
from .reduction import segment_sum

class ECDTLayer(nn.Module):

    def __init__(self, node_dim=2048, width=128, task_dim=64, output_dim=250, kappa_max=0.5):
        super().__init__()
        self.kappa_max = kappa_max
        self.down = nn.Linear(node_dim, width, bias=False)
        self.up = nn.Linear(width, node_dim, bias=False)
        nn.init.normal_(self.up.weight, std=0.002)
        self.groups = nn.ModuleList([DifferenceGroup(width, task_dim) for _ in RELATIONS])
        self.task_projection = nn.Linear(output_dim, task_dim, bias=False)
        self.calibrator = nn.Sequential(nn.Linear(task_dim + 8, 48), nn.GELU(), nn.Linear(48, 1, bias=False))
        self.group_bias = nn.Parameter(torch.zeros(len(RELATIONS)))
        self.negative_strength = nn.Linear(task_dim, len(RELATIONS))
        nn.init.constant_(self.negative_strength.bias, -2.0)
        self.budget = nn.Sequential(nn.Linear(task_dim + 8, 48), nn.GELU(), nn.Linear(48, 1))
        nn.init.constant_(self.budget[-1].bias, -2.0)

    def forward(self, z, edges, attrs, decoder, is_real, batch=None):
        n = len(z)
        v = self.down(z)
        task = torch.tanh(self.task_projection(decoder(z)).float())
        pos = []
        neg = []
        summaries = []
        masks = []
        for group, name in zip(self.groups, RELATIONS):
            edge = edges[name]
            src, dst = edge
            count = z.new_zeros(n, dtype=torch.float32)
            count.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
            valid = count > 0
            masks.append(valid)
            p, m = group(v, task, edge, attrs[name])
            pos.append(p)
            neg.append(m)
            sums = z.new_zeros((n, 3), dtype=torch.float32)
            a = attrs[name].float()
            values = torch.stack([a[:, 4], a[:, 6] - a[:, 7], (v[src] - v[dst]).float().norm(dim=-1)], -1)
            sums = segment_sum(values, dst, n)
            sums = sums / count[:, None].clamp_min(1)
            pn = p.float().norm(dim=-1)
            mn = m.float().norm(dim=-1)
            direction = (p.float() * v.float()).sum(-1) / (pn * v.float().norm(dim=-1)).clamp_min(1e-08)
            signed_cos = (p.float() * m.float()).sum(-1) / (pn * mn).clamp_min(1e-08)
            summaries.append(torch.cat([count.log1p()[:, None], sums, pn[:, None], mn[:, None], direction[:, None], signed_cos[:, None]], -1))
        mask = torch.stack(masks, 1)
        any_valid = mask.any(1)
        summary = torch.stack(summaries, 1)
        condition = torch.cat([task[:, None, :].expand(-1, len(RELATIONS), -1), summary.to(task)], -1)
        score = self.calibrator(condition).squeeze(-1).float() + self.group_bias
        score = score.masked_fill(~mask, -1000000000.0)
        pi = score.softmax(-1) * mask
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-12)
        lam = torch.sigmoid(self.negative_strength(task).float())
        selected = (summary * pi[:, :, None]).sum(1)
        kappa = self.kappa_max * torch.sigmoid(self.budget(torch.cat([task, selected.to(task)], -1)).float())
        p = torch.stack(pos, 1)
        m = torch.stack(neg, 1)
        message = ((p.float() - lam[:, :, None] * m.float()) * pi[:, :, None]).sum(1)
        candidate = self.up(message.to(v))
        updated, delta, capacity = transition(z, candidate, kappa, any_valid, is_real, batch)
        stats = {}
        return updated
