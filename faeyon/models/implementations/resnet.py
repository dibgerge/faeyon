"""ResNet-50 as a single bottleneck template expanded over a stage table."""

from __future__ import annotations

from torch import nn

from faeyon import materialize, I, X, faek
from faeyon.modifiers import IF


def resnet50_blocks() -> list[tuple[int, int, int, int]]:
    """Shape schedule: one (c_in, c_mid, c_out, stride) row per bottleneck."""
    return [
        (a if i == 0 else b, mid, b, s if i == 0 else 1)
        for a, mid, b, n, s in [
            (64, 64, 256, 3, 1),
            (256, 128, 512, 4, 2),
            (512, 256, 1024, 6, 2),
            (1024, 512, 2048, 3, 2),
        ]
        for i in range(n)
    ]


def build_resnet50(num_classes: int = 1000) -> nn.Module:
    """
    ResNet-50 via one DelayedModule bottleneck template + ``>> blocks`` table
    (design Version B). Requires ``faek`` while building.
    """
    blocks = resnet50_blocks()
    c_in, c_mid, c_out, stride = I[0], I[1], I[2], I[3]

    with faek:
        bottleneck = (
            (
                nn.Conv2d(c_in, c_mid, 1, bias=False)
                >> nn.BatchNorm2d(c_mid)
                >> nn.ReLU()
                >> nn.Conv2d(c_mid, c_mid, 3, stride=stride, padding=1, bias=False)
                >> nn.BatchNorm2d(c_mid)
                >> nn.ReLU()
                >> nn.Conv2d(c_mid, c_out, 1, bias=False)
                >> nn.BatchNorm2d(c_out)
            )
            + IF(
                (c_in == c_out) & (stride == 1),
                X,
                else_=(
                    nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
                    >> nn.BatchNorm2d(c_out)
                ),
            )
            >> nn.ReLU()
        ) % "layer"

        return materialize(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            >> nn.BatchNorm2d(64)
            >> nn.ReLU()
            >> nn.MaxPool2d(3, stride=2, padding=1)
            >> (bottleneck >> blocks)
            >> nn.AdaptiveAvgPool2d(1)
            >> X.flatten(1)
            >> nn.Linear(2048, num_classes) % "fc"
        )


class ResNet50(nn.Module):
    """Callable wrapper around the Faeyon ResNet-50 expression."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.model = build_resnet50(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
