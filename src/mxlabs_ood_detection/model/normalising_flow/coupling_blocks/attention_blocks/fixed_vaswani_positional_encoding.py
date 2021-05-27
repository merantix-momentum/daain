import torch
from torch import nn as nn


class FixedVaswaniPositionalEncoding(nn.Module):
    def __init__(self, n_hiddens, max_len=1000):
        super(FixedVaswaniPositionalEncoding, self).__init__()

        self.pos_encoding = torch.zeros((1, max_len, n_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, n_hiddens, 2, dtype=torch.float32) / n_hiddens
        )

        self.pos_encoding[:, :, 0::2] = torch.sin(X)
        self.pos_encoding[:, :, 1::2] = torch.cos(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X + self.pos_encoding[:, : X.shape[1], :].to(X.device)
