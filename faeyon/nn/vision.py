"""Vision building blocks shared by YOLO/FPN-style models."""

from __future__ import annotations

from torch import nn

from faeyon import X, faek


def conv_bn_act(c_in: int, c_out: int, k: int = 3, stride: int = 1) -> nn.Module:
    with faek:
        return (
            nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False)
            >> nn.BatchNorm2d(c_out)
            >> nn.SiLU()
        )


def stem_(c_in: int, c_out: int) -> nn.Module:
    return conv_bn_act(c_in, c_out, k=3, stride=2)


def csp_stage(c_in: int, c_out: int, n: int = 1) -> nn.Module:
    """Lightweight CSP-like stage: downsample then ``n`` residual bottleneck convs."""
    with faek:
        block = conv_bn_act(c_out, c_out)
        body = block if n <= 1 else (block >> (n - 1))
        return conv_bn_act(c_in, c_out, stride=2) >> body


def fuse(c_in: int, c_out: int) -> nn.Module:
    return conv_bn_act(c_in, c_out, k=1)


def upsample(scale: int = 2) -> nn.Module:
    return nn.Upsample(scale_factor=scale, mode="nearest")


def detect_head(channels: int, num_anchors: int, num_classes: int) -> nn.Module:
    out = num_anchors * (5 + num_classes)
    with faek:
        return conv_bn_act(channels, channels) >> nn.Conv2d(channels, out, 1)
