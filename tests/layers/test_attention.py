import pytest
import torch

from faeyon.layers import AdditiveAttention


class TestAdditiveAttention:
    def test_forward_batched_single_size(self):
        """ 
        Check that the output shape is correct when the input tensors have the same size
        and they are batched.
        """
        batch_size = 2
        embed_size = 10
        Tk = 3
        Tq = 6

        keys = torch.randn(batch_size, Tk, embed_size)
        queries = torch.randn(batch_size, Tq, embed_size)
        values = torch.randn(batch_size, Tk, embed_size)
        attention = AdditiveAttention(embed_size)
        output = attention(queries, keys, values)
        assert output.shape == (batch_size, Tq, embed_size)
