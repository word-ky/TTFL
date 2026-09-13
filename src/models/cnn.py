import torch
from torch import nn


class DigitCNN(nn.Module):
    """Small baseline; optional explicit channel states after three spatial blocks."""
    channels = (32, 64, 128)

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                          nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))
            for cin, cout in zip((3, 32, 64), self.channels)
        ])
        self.head = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x, state=None):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if state is not None:
                gamma, beta = state[2*i:2*i+2]
                x = (1 + gamma[None, :, None, None]) * x + beta[None, :, None, None]
        return self.head(x.flatten(1))
