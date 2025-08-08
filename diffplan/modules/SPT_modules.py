import math

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model=64):
        super().__init__()
        pe = torch.zeros(1, d_model, max_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(max_len) / d_model)
        ).unsqueeze(1)
        pe[0, 0::2, :] = torch.sin(torch.matmul(div_term, position))
        pe[0, 1::2, :] = torch.cos(torch.matmul(div_term, position))
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return x


class CNNEncoding(nn.Module):
    def __init__(self, in_dim=2, out_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=out_dim, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=2)

    def init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.conv1.weight.data, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight.data, nonlinearity="relu")

    def forward(self, x) -> Tensor:
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.flatten(x)

        return x
