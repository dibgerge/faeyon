import pytest
import torch

from faeyon.nn import AdditiveAttention, MultiHeadAttention


class TestAdditiveAttention:
    def test_raises_error_on_init(self):
        """
        Must raise value error if any of the specified sizes of inputs is negative.
        """
        with pytest.raises(ValueError):
            AdditiveAttention(embed_size=-20)

        with pytest.raises(ValueError):
            AdditiveAttention(embed_size=20, key_size=-20)

        with pytest.raises(ValueError):
            AdditiveAttention(embed_size=20, key_size=20, query_size=-20)

        with pytest.raises(ValueError):
            AdditiveAttention(embed_size=20, key_size=20, query_size=20, value_size=-20)

    @pytest.mark.parametrize(
        "batch_size, embed_size, key_size, query_size, value_size",
        [
            (0, 10, None, None, None),
            (0, 10, 10, 10, 10),
            (0, 10, 9, 8, 7),
            (2, 10, None, None, None),
            (2, 10, 10, 10, 10),
            (2, 10, 9, 8, 7),
        ]
    )
    def test_forward_output_size(
        self,
        batch_size: int,
        embed_size: int,
        key_size: int,
        query_size: int,
        value_size: int
    ):
        """
        Check that the output shape is correct when the input tensors have different sizes
        and they are batched.
        """
        Tk, Tq = 3, 6

        if batch_size == 0:
            keys = torch.randn(Tk, key_size or embed_size)
            queries = torch.randn(Tq, query_size or embed_size)
            values = torch.randn(Tk, value_size or embed_size)

            expected_shape = (Tq, value_size or embed_size)
        else:
            keys = torch.randn(batch_size, Tk, key_size or embed_size)
            queries = torch.randn(batch_size, Tq, query_size or embed_size)
            values = torch.randn(batch_size, Tk, value_size or embed_size)

            expected_shape = (batch_size, Tq, value_size or embed_size)

        args = {"embed_size": embed_size}
        if key_size is not None:
            args["key_size"] = key_size
        if query_size is not None:
            args["query_size"] = query_size
        if value_size is not None:
            args["value_size"] = value_size

        attention = AdditiveAttention(**args)

        output = attention(queries, keys, values)
        assert output.shape == expected_shape


class TestMultiHeadAttention:
    def test_usage(self):
        fattn = MultiHeadAttention(embed_size=4, num_heads=2)
        x = torch.rand(2, 5, 4)
        y = fattn(x, x, x)
