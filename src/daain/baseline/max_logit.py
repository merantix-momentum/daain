import torch

from daain.utils.array_utils import bmean_over_voxels, t2np


def max_logit(out, topk=0, agg=bmean_over_voxels):
    # the maximal value for each pixel 512 x 1014
    t = torch.max(out, dim=1)[0].flatten()
    if topk > 0:
        return agg(t2np(torch.topk(t, k=topk)[0]))
    else:
        return agg(t2np(t))


class MaxLogit:
    def __init__(self, topk=0, agg=bmean_over_voxels):
        self.topk = topk
        self.agg = agg

    def __call__(self, data):
        return max_logit(data, topk=self.topk, agg=self.agg)
