import torch
from torch import nn
from .evidence import ATTRIBUTE_DIM
from .reduction import segment_softmax, segment_sum

class DifferenceGroup(nn.Module):

    def __init__(self, width, task_dim):
        super().__init__()
        self.value_positive = nn.Linear(width, width, bias=False)
        self.value_negative = nn.Linear(width, width, bias=False)
        self.positive = nn.Sequential(nn.Linear(2 * task_dim + ATTRIBUTE_DIM, 48), nn.GELU(), nn.Linear(48, 1, bias=False))
        self.negative = nn.Sequential(nn.Linear(2 * task_dim + ATTRIBUTE_DIM, 48), nn.GELU(), nn.Linear(48, 1, bias=False))

    def forward(self, v, task, edge, attributes):
        src, dst = edge
        n = len(v)
        diff = (v[src].float() - v[dst].float()).to(v)
        evidence = torch.cat([task[dst], task[src] - task[dst], attributes.to(task)], -1)
        ap = segment_softmax(self.positive(evidence).squeeze(-1), dst, n)
        an = segment_softmax(self.negative(evidence).squeeze(-1), dst, n)
        positive = segment_sum(ap[:, None] * self.value_positive(diff).float(), dst, n)
        negative = torch.zeros((n, v.shape[1]), device=v.device, dtype=torch.float32)
        negative = segment_sum(an[:, None] * self.value_negative(diff).float(), dst, n)
        return (positive, negative)
