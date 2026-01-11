from faeyon.nn import InterpolatedPosEmbedding
import pytest


class TestPosInterpEmbedding:

    @pytest.mark.parametrize(
        "size, input_size, expected_shape",
        [
            (8, (8,), (1, 4, 8)),
            ((8,), (11,), (1, 4, 11)),
            ((8, 8), (8, 8), (1, 4, 8, 8)),
            ((8, 8), (11, 12), (1, 4, 11, 12)),
        ]
    )
    def test_forward_usage(self, size, input_size, expected_shape):
        embedding = InterpolatedPosEmbedding(size=size, embedding_dim=4, interpolate="nearest")
        ans = embedding(input_size)
        assert ans.shape == expected_shape
        
    def test_forward_raises_value_error(self):
        """ Input size must have same number of dimensions as embedding size. """
        embedding = InterpolatedPosEmbedding(size=8, embedding_dim=4, interpolate="nearest")
        with pytest.raises(ValueError):
            embedding((9, 10))

    def test_interpolate_none_ok(self):
        """ Interpolate can be None if input size matches embedding size. """
        embedding = InterpolatedPosEmbedding(size=8, embedding_dim=4, interpolate=None)
        assert embedding((8,)).shape == (1, 4, 8)

    def test_interpolate_none_error(self):
        """ Interpolate can't be None if input size doesn't match embedding size. """
        embedding = InterpolatedPosEmbedding(size=8, embedding_dim=4, interpolate=None)
        with pytest.raises(ValueError):
            embedding((9,))

    @pytest.mark.parametrize(
        "size, input_size, expected_shape",
        [
            (8, (8,), (1, 4, 9)),
            ((8,), (11,), (1, 4, 12)),
            ((8, 8), (8, 8), (1, 4, 65)),
            ((8, 8), (11, 12), (1, 4, 133)),
        ]
    )
    def test_non_positional(self, size, input_size, expected_shape):
        embedding = InterpolatedPosEmbedding(size=size, embedding_dim=4, non_positional=1, interpolate="nearest")
        assert embedding(input_size).shape == expected_shape
