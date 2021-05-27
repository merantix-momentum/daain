from math import exp

import torch
from FrEIA.modules import GLOWCouplingBlock

from mxlabs_ood_detection.utils.array_utils import btranspose
from mxlabs_ood_detection.utils.pytorch_utils import to_device


class FixedDistanceAttention(torch.nn.Module):
    """Fixed distance attention layer. Expects that the input is a tuple of (features, keys).

    Given that the keys are expected to be the pairwise distances a double selection is necessary to get the correct
    values (column- and row-wise).

    The attention layer then corresponds to the "default" matrix attention layer:
        (Q x K) x V^T
    """

    def __init__(self, pairwise_distances):
        super().__init__()

        self.pairwise_distance = torch.Tensor(pairwise_distances)
        self.n_dims = pairwise_distances.shape[0]

    def forward(self, input_tuple):
        (X, keys) = input_tuple  # X: [b x f], keys: [b x f x f / 2]
        if self.pairwise_distance.device != X.device:
            self.pairwise_distance = to_device(self.pairwise_distance, X.is_cuda)

        # funky stuff... may need to check that again
        broadcasted_pd = self.pairwise_distance.expand(keys.shape[0], *self.pairwise_distance.shape)

        # now do a double selection (since not samples are present)
        # K^T x Q x K
        weights = torch.bmm(btranspose(keys), torch.bmm(broadcasted_pd, keys))

        # flatten the output
        out = btranspose(torch.bmm(weights, btranspose(torch.unsqueeze(X, 1)))).view(X.shape[0], -1)

        return out, keys


class FixedPositionAttentionGlowCouplingBlock(GLOWCouplingBlock):
    """A coupling block that uses a fixed positional encoding for the attention layer.

    It works by assuming that the input's structure is [b x (1 + pos) x n]. Whereas:
        - `b` stands for the batch size.
        - `pos` stands for the positional encoding, in this case only cartesian coordinates in 3d where tested
        - `n` stands for the number of features (activations in this case)

    The rest is the same as any other GLOW type coupling block. But the subnets have to be functions that take the
    features and the keys as input.
    """

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=5.0):
        super().__init__(dims_in, dims_c, subnet_constructor, clamp)

        channels = dims_in[0][1]
        self.ndims = len(dims_in[0]) - 1

        self.split_len1 = channels // 2
        self.split_len2 = channels - self.split_len1

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        assert all(
            [tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]
        ), f"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."

        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2 * 2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1 * 2)

    def forward(self, x, c=[], rev=False):
        _x, keys = torch.split(x[0], [1, x[0].shape[1] - 1], dim=1)
        _x = _x.view(_x.shape[0], -1)

        x1, x2 = (_x.narrow(1, 0, self.split_len1), _x.narrow(1, self.split_len1, self.split_len2))

        keys1, keys2 = torch.split(keys, [self.split_len1, self.split_len2], dim=2)

        if rev:  # names of x and y are swapped!
            r1, _ = self.s1((x1, keys1))
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2, _ = self.s2((y2, keys2))
            s2, t2 = r2[:, : self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = -torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) - torch.sum(
                self.log_e(s2), dim=tuple(range(1, self.ndims + 1))
            )
        else:
            r2, _ = self.s2((x2, keys2))
            s2, t2 = r2[:, : self.split_len1], r2[:, self.split_len1:]
            # print(f"s2 shape: {s2.shape}, x1 {x1.shape}, t2 {t2.shape}, r2 {r2.shape}")
            y1 = self.e(s2) * x1 + t2

            r1, _ = self.s1((y1, keys1))
            s1, t1 = r1[:, : self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims + 1))) + torch.sum(
                self.log_e(s2), dim=tuple(range(1, self.ndims + 1))
            )

        res = torch.cat((y1, y2), 1)
        res = torch.cat((torch.unsqueeze(res, 1), keys), 1)

        return [res]