from enum import Enum

from torch import nn as nn

from daain.model.normalising_flow.coupling_blocks.attention_blocks import ApplyModuleOnSplit
from daain.model.normalising_flow.coupling_blocks.attention_blocks.fixed_distance_attention import (
    FixedDistanceAttention,
)
from daain.model.normalising_flow.coupling_blocks.attention_blocks.self_attention import (
    SelfAttentionLayerSAGAN,
)


class CouplingBlockType(Enum):
    GLOW_1x1_CONV = "glow_1x1_conv"
    GLOW_1x1_CONV_v2 = "glow_1x1_conv_v2"
    GLOW_DEPTH_CONV = "glow_depth_conv"
    GLOW_ATTENTION = "glow_attention"
    GLOW_LINEAR = "glow_linear"
    GLOW_POS_ATTENTION = "glow_positional_attention"
    IRESNET = "iresnet"
    RNVP_1x1_CONV = "rnvp_1x1_conv"
    GIN_LINEAR = "gin_linear"
    CONDITIONAL = "conditional"
    GLOW_1x1_CONV_GIN = "glow_1x1_conv_gin"

    def is_1d(self):
        return self in [
            CouplingBlockType.GLOW_LINEAR,
            CouplingBlockType.GLOW_POS_ATTENTION,
            CouplingBlockType.CONDITIONAL,
            CouplingBlockType.GIN_LINEAR,
        ]

    def use_permutation_layer(self):
        # the GLOW_POS_ATTENTION adds the coordinates to the data as additional channels. the permutation is thus
        # done a bit differently for these type of models.
        return self not in [CouplingBlockType.GLOW_POS_ATTENTION]

    def use_cuda(self):
        return not self == CouplingBlockType.GLOW_LINEAR


def min_factor_to_be_larger(a, b):
    f = 1
    while f * a < b:
        f += 1

    return f, f * a


def subnet_linear(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, min(c_in, c_out // 2)), nn.ReLU(), nn.Linear(min(c_in, c_out // 2), c_out),)


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 10), nn.ReLU(), nn.Linear(10, c_out))


def subnet_conv_1x1(c_in, c_out):
    # TODO adjust this according to the input size, should always be larger than the given input
    return nn.Sequential(nn.Conv2d(c_in, 256, 1), nn.ReLU(), nn.Conv2d(256, c_out, 1))


# def subnet_conv_1x1_v2(c_in, c_out, num_channels):
#    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1), nn.ReLU())


def subnet_depth_conv(c_in, c_out, num_channels):
    k, out_depth = min_factor_to_be_larger(c_in, c_out)
    return nn.Sequential(
        nn.Conv2d(c_in, out_depth, 3, groups=c_in, padding=1), nn.ReLU(), nn.Conv2d(out_depth, c_out, 3, padding=1)
    )


def subnet_attention(c_in, c_out, n_attention_blocks=3):
    return nn.Sequential(
        *[SelfAttentionLayerSAGAN(c_in) for _ in range(n_attention_blocks)],
        # nn.ReLU(),
        nn.Conv2d(c_in, c_out, 1),  # to bring it up to the required dimension
        nn.BatchNorm2d(c_out)
    )


def subnet_dist_attention(c_in, c_out, pairwise_distances):
    return nn.Sequential(
        FixedDistanceAttention(pairwise_distances=pairwise_distances),
        ApplyModuleOnSplit(nn.Linear(c_in, c_out)),
        ApplyModuleOnSplit(nn.ReLU()),
    )
