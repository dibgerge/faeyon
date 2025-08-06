import torch
from faeyon.nn import TokenizedMask
import pytest


class TestTokenizedMask:
    def test_forward_enabled(self):
        mask = torch.tensor([
            [
                [True, True, True],
                [True, True, True],
                [True, True, True]
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False]
            ]
        ])
        embedding_dim = 4
        batch_size = 2

        tokenized_mask = TokenizedMask(embedding_dim)
        torch.nn.init.ones_(tokenized_mask.tokens)
        x = torch.ones(batch_size, embedding_dim, 3, 3)
        out = tokenized_mask(x, mask)
        assert out.shape ==  (batch_size, embedding_dim, 3, 3)

        expected = torch.ones(batch_size, embedding_dim, 3, 3)
        torch.testing.assert_close(out, expected)

    def test_forward_disabled(self):
        embedding_dim = 4
        tokenized_mask = TokenizedMask(embedding_dim, enabled=False)
        x = torch.ones(2, embedding_dim, 3, 3)
        out = tokenized_mask(x)
        torch.testing.assert_close(out, x)

    def test_forward_disabled_with_mask_error(self):
        """
        If mask is disabled, but received a mask argument in forward, we have error 
        """
        embedding_dim = 4
        tokenized_mask = TokenizedMask(embedding_dim, enabled=False)
        x = torch.ones(2, embedding_dim, 3, 3)
        mask = torch.randn(2, 3, 3) > 0.4

        with pytest.raises(ValueError):
            tokenized_mask(x, mask)
