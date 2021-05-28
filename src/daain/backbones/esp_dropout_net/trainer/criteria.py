import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    This file defines a cross entropy loss for 2D images
    """

    def __init__(self, weight=None, ignore_index=None):
        """
        :param weight: 1D weight vector to deal with the class-imbalance
        """
        super().__init__()

        self.loss = nn.NLLLoss(torch.Tensor(weight), ignore_index=ignore_index)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)
