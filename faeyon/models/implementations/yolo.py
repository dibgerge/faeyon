"""YOLO-style backbone + FPN multi-scale heads (cookbook shape)."""

from __future__ import annotations

import torch
from torch import nn

from faeyon import F, FaeDict, FaeList, materialize, R, X, faek
from faeyon.nn.vision import csp_stage, detect_head, fuse, stem_, upsample


def build_yolo(
    num_classes: int = 80,
    num_anchors: int = 3,
    in_channels: int = 3,
) -> nn.Module:
    with faek:
        return materialize(
            stem_(in_channels, 32)
            >> csp_stage(32, 64, n=1)
            >> csp_stage(64, 128, n=3) % "c3"
            >> csp_stage(128, 256, n=3) % "c4"
            >> csp_stage(256, 512, n=1) % "c5"
            >> FaeDict(
                {
                    "p5": X,
                    "p4": F(torch.cat, FaeList([R["c4"], upsample()(X)]), dim=1)
                    >> fuse(768, 256),
                }
            )
            >> FaeDict(
                {
                    "p5": X["p5"],
                    "p4": X["p4"],
                    "p3": F(torch.cat, FaeList([R["c3"], upsample()(X["p4"])]), dim=1)
                    >> fuse(384, 128),
                }
            )
            >> FaeDict(
                {
                    "small": X["p3"] >> detect_head(128, num_anchors, num_classes),
                    "medium": X["p4"] >> detect_head(256, num_anchors, num_classes),
                    "large": X["p5"] >> detect_head(512, num_anchors, num_classes),
                }
            )
        )


class YOLO(nn.Module):
    def __init__(self, num_classes: int = 80, num_anchors: int = 3) -> None:
        super().__init__()
        self.model = build_yolo(num_classes=num_classes, num_anchors=num_anchors)

    def forward(self, x):
        return self.model(x)
