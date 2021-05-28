import torch
from torch import exp
from torch import nn as nn


class CausalCouplingBlock(nn.Module):
    def __init__(self, dims_in, dims_c=None, subnet_constructor=None, clamp=5.0):
        super().__init__()

        if dims_c is None:
            dims_c = []

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        if any([dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]):
            raise ValueError("Dimensions of input and one or more conditions don't agree.")

        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2)
        self.t1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.t2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1)

    def e(self, s):
        # return torch.exp(torch.clamp(s, -self.clamp, self.clamp))
        # return (self.max_s-self.min_s) * torch.sigmoid(s) + self.min_s
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        """log of the nonlinear function e"""
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, c=None, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1), x[0].narrow(1, self.split_len1, self.split_len2))

        if rev:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            s1, t1 = self.s1(x1_c), self.t1(x1_c)
            y2 = (x2 - t1) / self.e(s1)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            s2, t2 = self.s2(y2_c), self.t2(y2_c)
            y1 = (x1 - t2) / self.e(s2)
        else:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            s2, t2 = self.s2(x2_c), self.t2(x2_c)
            y1 = self.e(s2) * x1 + t2
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            s1, t1 = self.s1(y1_c), self.t1(y1_c)
            y2 = self.e(s1) * x2 + t1
            self.last_s = [s1, s2]

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1), x[0].narrow(1, self.split_len1, self.split_len2))

        if rev:
            jac1 = -torch.sum(self.log_e(self.last_s[0]), dim=tuple(range(1, self.ndims + 1)))
            jac2 = -torch.sum(self.log_e(self.last_s[1]), dim=tuple(range(1, self.ndims + 1)))
        else:
            jac1 = torch.sum(self.log_e(self.last_s[0]), dim=tuple(range(1, self.ndims + 1)))
            jac2 = torch.sum(self.log_e(self.last_s[1]), dim=tuple(range(1, self.ndims + 1)))

        return jac1 + jac2

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims
