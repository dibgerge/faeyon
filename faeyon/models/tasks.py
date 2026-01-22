import math
import torch
from typing import Optional
from torch import nn
from faeyon import X


class Task(nn.Module):
    pass


class ClassifyTask(Task):
    dropout: nn.Dropout
    pooling: X

    def __init__(
        self, 
        num_hidden: int, 
        num_labels: int, 
        dropout: Optional[float] = None, 
        pooling: Optional[int] = None,
        bias: bool = True
    ) -> None:
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = X

        if isinstance(pooling, int):
            self.pooling = X[..., pooling, :]
        else:
            self.pooling = X

        self.classifier = nn.Linear(num_hidden, num_labels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The shape is (B, T, D) or (B, D). If pooling is specified, the `T` dimension will be 
        reduced accordingly.
        """
        return x >> self.pooling >> self.dropout >> self.classifier


class PatchedImageDecoder(nn.Module):
    """
    ignore_first is used to remove the "cls" token in ViT like models.
    """
    preproc: X

    def __init__(
        self,
        hidden_size: int,
        encoder_stride: int,
        num_channels: int,
        kernel_size: int = 1,
        ignore_first: bool = True,
        bias: bool = True
     ):
        super().__init__()
        out_channels = encoder_stride ** 2 * num_channels

        self.conv = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias
        )
        self.shuffle = nn.PixelShuffle(encoder_stride)

        if ignore_first:
            self.preproc = X[..., 1:, :]
        else:
            self.preproc = X

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has shape (B, T, D)
        """
        batch_size, length, num_channels = x.shape
        size = math.floor(length ** 0.5)
        return (
            x 
            >> self.preproc 
            >> X.permute(0, 2, 1)
            >> X.reshape(batch_size, num_channels, size, size)
            >> self.conv 
            >> self.shuffle 
        )
