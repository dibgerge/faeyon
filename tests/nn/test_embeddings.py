import torch
from faeyon.nn import PosInterpEmbedding
import pytest


class TestPosInterpEmbedding:

    def test_init(self):
        embedding = PosInterpEmbedding(
            size=8,
            embedding_dim=4,
            interpolate="nearest",
        )
        ans = embedding((8,))
        assert ans.shape == (1, 4, 8)
        
