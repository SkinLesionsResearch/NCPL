import torch


def get_idx(input, cond):
    zero = torch.zeros_like(input)
    one = torch.ones_like(input)
    idx = torch.nonzero(torch.where(cond, one, zero))
    return torch.unique(idx[:, 0]), torch.unique(idx[:, 1])