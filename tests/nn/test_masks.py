import torch
from faeyon.nn import TokenizedMask, head_to_attn_mask
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


@pytest.mark.parametrize("head_mask, num_layers", [
    (torch.randn(4, 5), None),
    (torch.randn(4, 5), 4),
    (torch.randn(5), 4),
    (torch.randn(5), None),
])
def test_head_to_attn_mask(head_mask, num_layers):
    batch_size = 2
    src_len = 3
    tgt_len = 3
    heads = 5
    res = head_to_attn_mask(head_mask, batch_size, src_len, tgt_len, num_layers)
    expected = torch.zeros(batch_size, src_len, tgt_len)
    
    if num_layers is None:
        if len(head_mask.shape) == 2:
            expected_shape = (head_mask.shape[0], batch_size, heads, src_len, tgt_len)
            diff = res[0, :, 0, :, :] - head_mask[0, 0]
        else:
            expected_shape = (batch_size, heads, src_len, tgt_len)
            diff = res[:, 0, :, :] - head_mask[0]
    else:
        expected_shape = (num_layers, batch_size, heads, src_len, tgt_len)
        if len(head_mask.shape) == 2:
            diff = res[0, :, 0, :, :] - head_mask[0, 0]
        else:
            diff = res[0, :, 0, :, :] - head_mask[0]
    
    assert res.shape == expected_shape
    torch.testing.assert_close(diff, expected)


def test_head_to_attn_mask_error():
    # head_mask first dim does not match num_layers
    with pytest.raises(ValueError):
        head_mask = torch.randn(4, 5)
        head_to_attn_mask(head_mask, batch_size=2, src_len=3, tgt_len=3, num_layers=1)
    
    # head_mask shape is not right
    with pytest.raises(ValueError):
        head_mask = torch.randn(4, 5, 4)
        head_to_attn_mask(head_mask, batch_size=2, src_len=3, tgt_len=3, num_layers=4)