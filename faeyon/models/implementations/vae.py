"""Variational autoencoder cookbook expression."""

from __future__ import annotations

import torch
from torch import nn

from faeyon import F, FaeDict, materialize, X, faek


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def build_vae(input_dim: int = 784, hidden: int = 400, latent: int = 20) -> nn.Module:
    with faek:
        encoder = nn.Linear(input_dim, hidden) >> nn.ReLU()
        decoder = (
            nn.Linear(latent, hidden)
            >> nn.ReLU()
            >> nn.Linear(hidden, input_dim)
            >> nn.Sigmoid()
        )
        return materialize(
            encoder
            >> FaeDict(
                {
                    "mu": nn.Linear(hidden, latent)(X),
                    "logvar": nn.Linear(hidden, latent)(X),
                }
            )
            >> FaeDict(
                {
                    "recon": F(reparameterize, X["mu"], X["logvar"]) >> decoder,
                    "mu": X["mu"],
                    "logvar": X["logvar"],
                }
            )
        )


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden: int = 400, latent: int = 20) -> None:
        super().__init__()
        self.model = build_vae(input_dim=input_dim, hidden=hidden, latent=latent)

    def forward(self, x):
        return self.model(x)
