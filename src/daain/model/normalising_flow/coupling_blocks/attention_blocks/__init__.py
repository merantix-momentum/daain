from torch import nn as nn


class ApplyModuleOnSplit(nn.Module):
    """This module applies a given function only to the values and not the keys.
    This is useful in the attention layers with fixed positions."""

    def __init__(self, activation_function):
        super(ApplyModuleOnSplit, self).__init__()
        self.fn = activation_function

    def forward(self, data):
        values, keys = data
        data = self.fn(values)
        return data, keys
