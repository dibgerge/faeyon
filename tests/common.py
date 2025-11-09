from __future__ import annotations
import torch
from torch import nn
from typing import Optional
from faeyon import X, Op


class ConstantLayer(nn.Module):
    """
    A layer that multiplies the input by a constant value. Allows for predictable outputs during testing.
    """
    def __init__(
        self, 
        size: int | tuple[int, ...], 
        value: int | float,
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            size = (size,)
        self.value = value
        self.weight = nn.Parameter(torch.ones(*size, dtype=dtype) * value, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class BasicModel(nn.Module):
    """
    A model for testing purposes, which includes a embedding layer and two linear layers.
    """
    def __init__(
        self, 
        num_inputs: int = 5,
        embedding: Optional[nn.Embedding] = None,
        num_hidden: int = 10, 
        num_outputs: int = 3
    ) -> None:
        super().__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = Op(X)

        self.layer1 = nn.Linear(num_inputs, num_hidden)
        self.layer2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        return x >> self.embedding >> self.layer1 >> self.layer2

    def is_similar(self, other: BasicModel) -> bool:
        """ Same model configuration, but does not have to be the same paramters/weights."""
        conditions = [
            self.layer1.in_features == other.layer1.in_features,
            self.layer1.out_features == other.layer1.out_features,
            self.layer2.in_features == other.layer2.in_features,
            self.layer2.out_features == other.layer2.out_features,
            type(self.embedding) == type(other.embedding),
        ]

        if isinstance(self.embedding, nn.Embedding) and isinstance(other.embedding, nn.Embedding):
            conditions.extend([
                self.embedding.num_embeddings == other.embedding.num_embeddings,
                self.embedding.embedding_dim == other.embedding.embedding_dim,
                self.embedding.padding_idx == other.embedding.padding_idx,
                self.embedding.max_norm == other.embedding.max_norm,
                self.embedding.norm_type == other.embedding.norm_type,
                self.embedding.scale_grad_by_freq == other.embedding.scale_grad_by_freq,
                self.embedding.sparse == other.embedding.sparse,
            ])
        return all(conditions)

    def __eq__(self, other: BasicModel) -> bool:
        """ 
        Check that the two models are the same, including the state dictionary (weights).
        """
        state1 = self.state_dict() 
        state2 = other.state_dict()

        if set(state1.keys()) != set(state2.keys()):
            return False

        for key in state1:
            if not torch.allclose(state1[key], state2[key]):
                return False

        return True
