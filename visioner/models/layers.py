import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=None):
        super(AdaptiveConcatPool2d).__init__()
        self.output_size = output_size or 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(self.output_size)
        self.max_pooling = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        poolings = (self.avg_pooling(x), self.max_pooling(x))
        return torch.cat(poolings, 1)


class Flatten(nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)
