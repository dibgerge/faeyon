"""Multi-task trunk with shared embedding and typed heads."""

from __future__ import annotations

import torch
from torch import nn

from faeyon import A, F, FaeDict, FaeList, materialize, X, faek


def image_encoder(embed: int) -> nn.Module:
    with faek:
        return (
            nn.Conv2d(3, 32, 3, stride=2, padding=1)
            >> nn.ReLU()
            >> nn.AdaptiveAvgPool2d(1)
            >> X.flatten(1)
            >> nn.Linear(32, embed)
        )


def text_encoder(vocab: int, embed: int) -> nn.Module:
    with faek:
        return nn.Embedding(vocab, embed) >> X.mean(dim=1)


def build_multitask(
    embed: int = 128,
    n_meta: int = 8,
    n_categories: int = 100,
    vocab_size: int = 30_000,
) -> nn.Module:
    with faek:
        return materialize(
            FaeDict(
                {
                    "img": A["image"] >> image_encoder(embed) % "image_encoder",
                    "txt": A["text"] >> text_encoder(vocab_size, embed) % "text_encoder",
                }
            )
            >> F(torch.cat, FaeList([X["img"], X["txt"]]), dim=-1)
            >> (nn.Linear(2 * embed, embed) >> nn.ReLU()) % "trunk"
            >> FaeDict(
                {
                    "category": X >> nn.Linear(embed, n_categories) % "category_head",
                    "price": (
                        F(torch.cat, FaeList([X, A["meta"]]), dim=-1)
                        >> nn.Linear(embed + n_meta, 1)
                    )
                    % "price_head",
                    "quality": (nn.Dropout(0.1) >> nn.Linear(embed, 1)) % "quality_head",
                }
            )
        )


class MultiTask(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = build_multitask(**kwargs)

    def forward(self, image, text, meta):
        from faeyon import Input

        return self.model(Input(image=image, text=text, meta=meta))
