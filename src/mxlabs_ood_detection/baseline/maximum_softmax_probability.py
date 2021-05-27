import torch
import torch.nn.functional as F

from mxlabs_ood_detection.utils.array_utils import bmean_over_voxels


def maximum_softmax_probability(out, topk=0, agg=bmean_over_voxels):
    """Assuming single-model evaluation and no access to other anomalies or test-time adaptation,
    the maximum softmax probability (MSP) is the state-of-the-art multi-class out-of-distribution
    detection method. However, we show that the MSP is problematic for large-scale datasets with many
    classes including ImageNet-1K and Places365"""
    t = torch.max(F.softmax(out, dim=1), dim=1)[0]  # the maximal value for each pixel 512 x 1014
    if topk > 0:
        return agg(torch.topk(t, k=topk)[0])
    else:
        return agg(t)


class MaximumSoftmaxProbability:
    def __init__(self, topk=0, agg=bmean_over_voxels):
        self.topk = topk
        self.agg = agg

    def __call__(self, data):
        return maximum_softmax_probability(data, topk=self.topk, agg=self.agg)
