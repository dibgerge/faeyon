"""U-Net with recall-symbol skip connections."""

from __future__ import annotations

import torch
from torch import nn

from faeyon import F, FaeList, materialize, R, X, faek


def conv_block(c_in: int, c_out: int) -> nn.Module:
    return (
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        >> nn.BatchNorm2d(c_out)
        >> nn.ReLU()
        >> nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        >> nn.BatchNorm2d(c_out)
        >> nn.ReLU()
    )


def down(channels: int) -> nn.Module:
    return nn.MaxPool2d(2)


def up(c_in: int, c_out: int) -> nn.Module:
    return nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)


def build_unet(in_channels: int = 1, out_channels: int = 1, base: int = 64) -> nn.Module:
    with faek:
        return materialize(
            conv_block(in_channels, base) % "e1"
            >> down(base)
            >> conv_block(base, base * 2) % "e2"
            >> down(base * 2)
            >> conv_block(base * 2, base * 4)
            >> up(base * 4, base * 2)
            >> F(torch.cat, FaeList([X, R["e2"]]), dim=1)
            >> conv_block(base * 4, base * 2)
            >> up(base * 2, base)
            >> F(torch.cat, FaeList([X, R["e1"]]), dim=1)
            >> conv_block(base * 2, base)
            >> nn.Conv2d(base, out_channels, 1)
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 64) -> None:
        super().__init__()
        self.model = build_unet(in_channels=in_channels, out_channels=out_channels, base=base)

    def forward(self, x):
        return self.model(x)
