"""Mixture-of-experts and related feed-forward modules."""

from __future__ import annotations

import torch
from torch import nn


class SwiGLU(nn.Module):
    """SwiGLU FFN: ``out = W3(W1(x) * silu(W2(x)))``."""

    def __init__(self, hidden: int, intermediate: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=bias)
        self.w2 = nn.Linear(hidden, intermediate, bias=bias)
        self.w3 = nn.Linear(intermediate, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from faeyon.magic.faek import faek

        call = faek.module__call__
        return call(self.w3, call(self.w1, x) * torch.nn.functional.silu(call(self.w2, x)))


class MoE(nn.Module):
    """
    Sparse top-k mixture of SwiGLU experts.

    Config is ints so ``clone()`` / ``>> num_layers`` create independent weights.
    """

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        n_experts: int,
        top_k: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        self.top_k = top_k
        self.router = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(hidden, intermediate, bias=bias) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Submodule calls must bypass faek's delayed ``nn.Module.__call__``.
        from faeyon.magic.faek import faek

        call = faek.module__call__
        orig = x.shape
        flat = x.reshape(-1, orig[-1])
        logits = call(self.router, flat)
        k = min(self.top_k, logits.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = flat.new_zeros(flat.shape)
        for slot in range(k):
            expert_ids = top_idx[:, slot]
            weights = top_probs[:, slot]
            for e, expert in enumerate(self.experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                out[mask] = out[mask] + weights[mask].unsqueeze(-1) * call(expert, flat[mask])
        return out.reshape(orig)
