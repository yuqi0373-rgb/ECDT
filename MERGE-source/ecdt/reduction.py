import torch

def segment_sum(values, dst, n):
    order = dst.argsort(stable=True)
    lengths = torch.bincount(dst, minlength=n)
    return torch.segment_reduce(values.float()[order], reduce='sum', lengths=lengths)

def segment_softmax(score, dst, n):
    score = score.float()
    order = dst.argsort(stable=True)
    lengths = torch.bincount(dst, minlength=n)
    maxima = torch.segment_reduce(score[order], reduce='max', lengths=lengths)
    exp = (score - maxima[dst]).exp()
    den = segment_sum(exp, dst, n)
    return exp / den[dst].clamp_min(1e-12)
